"""Triton Capacity routing, selected-row scan, reverse scan and online replay.

Dense scans and canonical geometric routes parallelize across row tiles and
batch lanes. Arbitrary caller-supplied routes retain a per-batch fallback that
preserves dependencies when rows change slot order. Row tiles bound live
registers; sparse execution loads only routed rows, including eligibility.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None


def available():
    return triton is not None and torch.cuda.is_available()


if triton is not None:
    @triton.jit
    def _operand(value, MODE: tl.constexpr):
        if MODE == 1:
            return value.to(tl.float16).to(value.dtype)
        elif MODE == 2:
            return value.to(tl.bfloat16).to(value.dtype)
        else:
            return value

    @triton.jit
    def _address_kernel(X, P, OUT, T: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
                        L: tl.constexpr, A: tl.constexpr, BD: tl.constexpr,
                        BN: tl.constexpr, BA: tl.constexpr):
        token = tl.program_id(0)
        d = tl.arange(0, BD)
        n = tl.arange(0, BN)
        a = tl.arange(0, BA)
        xx = tl.load(X + token * D + d, d < D, 0)
        bucket = tl.full((), 0, tl.int64)
        for level in range(L):
            # Projection rows are tiled so flat addressing does not require N*D registers.
            best = tl.full((), -float("inf"), xx.dtype)
            digit = tl.full((), 0, tl.int64)
            for start in range(tl.cdiv(N, BN)):
                row = start * BN + n
                pp = tl.load(P + (level * N + row[:, None]) * D + d[None, :],
                             (row[:, None] < N) & (d[None, :] < D), 0)
                scores = tl.sum(pp * xx[None, :], 1)
                scores = tl.where(row < N, scores, -float("inf"))
                tile_best = tl.max(scores, 0)
                tile_digit = tl.min(tl.where(scores == tile_best, row, 2147483647), 0)
                replace = tile_best > best
                digit = tl.where(replace, tile_digit, digit)
                best = tl.maximum(best, tile_best)
            bucket = bucket * N + digit
        tl.store(OUT + token * A + a, bucket * A + a, a < A)

    @triton.jit
    def _forward_kernel(X, K, BIAS, R, V, GAIN, Z, IDX, Y, ACT, OLD, STATES,
                        T: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                        S: tl.constexpr, A: tl.constexpr, SPARSE: tl.constexpr,
                        BD: tl.constexpr, BS: tl.constexpr, RA: tl.constexpr,
                        PARALLEL: tl.constexpr, MODE: tl.constexpr):
        b = tl.program_id(0)
        ar = tl.arange(0, RA)
        d = tl.arange(0, BD)
        start_first = tl.program_id(1) if PARALLEL else 0
        start_end = start_first+1 if PARALLEL else tl.cdiv(A, RA)
        for t in range(T):
            xt = tl.load(X + (b*T+t)*D+d, d<D, 0)
            for start in range(start_first, start_end):
                a = start*RA+ar
                if SPARSE:
                    row = tl.load(IDX+(b*T+t)*A+a, a<A, 0)
                else:
                    row = a
                kk = tl.load(K+row[:,None]*D+d[None,:], (a[:,None]<A)&(d[None,:]<D), 0)
                kk = _operand(kk.to(xt.dtype),MODE)
                bias = _operand(tl.load(BIAS+row, a<A, 0).to(xt.dtype),MODE)
                r = tl.load(R+row, a<A, 0).to(xt.dtype)
                rho = 1/(1+libdevice.exp(-r))
                act = _operand(libdevice.tanh(_operand(tl.sum(kk*xt[None,:], 1)+bias,MODE)),MODE)
                prev = tl.load(Z+b*C+row, a<A, 0).to(xt.dtype)
                state = rho*prev+(1-rho)*act
                tl.store(Z+b*C+row, state, a<A)
                q = (b*T+t)*A+a
                tl.store(ACT+q, act, a<A)
                tl.store(OLD+q, prev, a<A)
                tl.store(STATES+q, state, a<A)

    @triton.jit
    def _readout_kernel(STATES, IDX, V, GAIN, Y, S: tl.constexpr, A: tl.constexpr,
                        SPARSE: tl.constexpr, RS: tl.constexpr, RA: tl.constexpr,
                        MODE: tl.constexpr):
        token = tl.program_id(0)
        s = tl.program_id(1)*RS+tl.arange(0,RS)
        ar = tl.arange(0,RA)
        gain = tl.load(GAIN).to(STATES.dtype.element_ty)
        total = tl.full((RS,),0,STATES.dtype.element_ty)
        # One writer per output element, fixed ascending row-tile order.
        for start in range(tl.cdiv(A,RA)):
            a = start*RA+ar
            row = tl.load(IDX+token*A+a,a<A,0) if SPARSE else a
            state = _operand(tl.load(STATES+token*A+a,a<A,0),MODE)
            value = _operand(tl.load(V+row[:,None]*S+s[None,:],
                             (a[:,None]<A)&(s[None,:]<S),0).to(total.dtype),MODE)
            total += tl.sum(state[:,None]*value,0)
        tl.store(Y+token*S+s,gain*total,s<S)

    @triton.jit
    def _input_vjp_kernel(CREDIT, IDX, K, DX, D: tl.constexpr, A: tl.constexpr,
                          SPARSE: tl.constexpr, RD: tl.constexpr, RA: tl.constexpr,
                          MODE: tl.constexpr):
        token = tl.program_id(0)
        d = tl.program_id(1)*RD+tl.arange(0,RD)
        ar = tl.arange(0,RA)
        total = tl.full((RD,),0,CREDIT.dtype.element_ty)
        for start in range(tl.cdiv(A,RA)):
            a = start*RA+ar
            row = tl.load(IDX+token*A+a,a<A,0) if SPARSE else a
            credit = tl.load(CREDIT+token*A+a,a<A,0)
            key = _operand(tl.load(K+row[:,None]*D+d[None,:],
                           (a[:,None]<A)&(d[None,:]<D),0).to(total.dtype),MODE)
            total += tl.sum(credit[:,None]*key,0)
        tl.store(DX+token*D+d,total,d<D)

    @triton.jit
    def _backward_kernel(X, K, R, V, GAIN, IDX, LOCAL, ACT, OLD, STATES, DY,
                         EK, EB, ER, DX, DK, DB, DR, DV, DG, DZ,
                         T: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                         S: tl.constexpr, A: tl.constexpr, SPARSE: tl.constexpr,
                         ONLINE: tl.constexpr, BD: tl.constexpr, BS: tl.constexpr,
                         RA: tl.constexpr, PARALLEL: tl.constexpr, MODE: tl.constexpr,
                         ROWS: tl.constexpr, PRIVATE_ROWS: tl.constexpr,
                         GROUPS: tl.constexpr, EVENT_ROWS: tl.constexpr):
        b = tl.program_id(0)
        ar = tl.arange(0,RA)
        d = tl.arange(0,BD)
        s = tl.arange(0,BS)
        gain = tl.load(GAIN).to(X.dtype.element_ty)
        gg = tl.full((),0,gain.dtype)
        start_first = tl.program_id(1) if PARALLEL else 0
        start_end = start_first+1 if PARALLEL else tl.cdiv(A, RA)
        if not SPARSE and not ONLINE:
            dense_row = start_first*RA+ar
            key_total = tl.full((RA,BD),0,X.dtype.element_ty)
            bias_total = tl.full((RA,),0,X.dtype.element_ty)
            retention_total = tl.full((RA,),0,X.dtype.element_ty)
            value_total = tl.full((RA,BS),0,X.dtype.element_ty)
            reverse_carry = tl.load(DZ+b*C+dense_row,dense_row<C,0)
        for ti in range(T):
            if ONLINE:
                t = ti
            else:
                t = T-1-ti
            xt = tl.load(X+(b*T+t)*D+d,d<D,0)
            cot = tl.load(DY+(b*T+t)*S+s,s<S,0)
            for start in range(start_first, start_end):
                a = start*RA+ar
                q = (b*T+t)*A+a
                if SPARSE:
                    row = tl.load(IDX+q,a<A,0)
                    local = tl.load(LOCAL+q,a<A,0)
                else:
                    row = a
                    local = a
                if EVENT_ROWS:
                    owner = q
                else:
                    owner = b*ROWS+local if PRIVATE_ROWS else local
                logit = tl.load(R+row,a<A,0).to(xt.dtype)
                rho = 1/(1+libdevice.exp(-logit))
                act = tl.load(ACT+q,a<A,0)
                prev = tl.load(OLD+q,a<A,0)
                state = tl.load(STATES+q,a<A,0)
                state = _operand(state,MODE)
                slope = (1-rho)*(1-act*act)
                vv = _operand(tl.load(V+row[:,None]*S+s[None,:],(a[:,None]<A)&(s[None,:]<S),0).to(xt.dtype),MODE)
                gz = gain*tl.sum(cot[None,:]*vv,1)
                gg += tl.sum(tl.where(a<A,state*tl.sum(cot[None,:]*vv,1),0),0)
                value_ptr = DV+owner[:,None]*S+s[None,:]
                value_mask = (a[:,None]<A)&(s[None,:]<S)
                dv = gain*state[:,None]*cot[None,:]
                if SPARSE or ONLINE:
                    old_value_grad = tl.load(value_ptr,value_mask,0)
                    tl.store(value_ptr,old_value_grad+dv,value_mask)
                else:
                    value_total += dv
                if ONLINE:
                    ek = tl.load(EK+(b*C+row[:,None])*D+d[None,:],(a[:,None]<A)&(d[None,:]<D),0)
                    eb = tl.load(EB+b*C+row,a<A,0)
                    er = tl.load(ER+b*C+row,a<A,0)
                    ek = rho[:,None]*ek+slope[:,None]*xt[None,:]
                    eb = rho*eb+slope
                    er = rho*er+rho*(1-rho)*(prev-act)
                    tl.store(EK+(b*C+row[:,None])*D+d[None,:],ek,(a[:,None]<A)&(d[None,:]<D))
                    tl.store(EB+b*C+row,eb,a<A)
                    tl.store(ER+b*C+row,er,a<A)
                    dk = gz[:,None]*ek
                    db = gz*eb
                    dr = gz*er
                else:
                    if SPARSE:
                        gz += tl.load(DZ+b*C+row,a<A,0)
                        tl.store(DZ+b*C+row,gz*rho,a<A)
                    else:
                        # Dense tiles own the same rows for the entire scan.
                        # Keep the recurrence in SSA registers: reading back
                        # a scalar global store through the next iteration's
                        # cross-warp layout can observe stale carry values.
                        gz += reverse_carry
                        reverse_carry = gz*rho
                    dk = (gz*slope)[:,None]*xt[None,:]
                    db = gz*slope
                    dr = gz*rho*(1-rho)*(prev-act)
                # DX is the selected-row scalar credit tape, reduced by a
                # separate single-writer input VJP kernel after this scan.
                tl.store(DX+q,gz*slope,a<A)
                key_ptr = DK+owner[:,None]*D+d[None,:]
                key_mask = (a[:,None]<A)&(d[None,:]<D)
                if SPARSE or ONLINE:
                    tl.store(key_ptr,tl.load(key_ptr,key_mask,0)+dk,key_mask)
                    tl.store(DB+owner,tl.load(DB+owner,a<A,0)+db,a<A)
                    tl.store(DR+owner,tl.load(DR+owner,a<A,0)+dr,a<A)
                else:
                    # Dense row owners are constant across time. Register
                    # accumulation avoids cross-warp global read-after-write
                    # hazards and per-token gradient-table memory traffic.
                    key_total += dk
                    bias_total += db
                    retention_total += dr
        if not SPARSE and not ONLINE:
            owner = b*ROWS+dense_row
            tl.store(DK+owner[:,None]*D+d[None,:],key_total,(dense_row[:,None]<C)&(d[None,:]<D))
            tl.store(DB+owner,bias_total,dense_row<C)
            tl.store(DR+owner,retention_total,dense_row<C)
            tl.store(DV+owner[:,None]*S+s[None,:],value_total,(dense_row[:,None]<C)&(s[None,:]<S))
            tl.store(DZ+b*C+dense_row,reverse_carry,dense_row<C)
        group = tl.program_id(1) if PARALLEL else 0
        tl.store(DG+b*GROUPS+group,gg)

    @triton.jit
    def _row_reduce_kernel(SORTED, ORDER, VALUES, OUT, N: tl.constexpr,
                           WIDTH: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr):
        position = tl.program_id(0)*BR+tl.arange(0,BR)
        d = tl.program_id(1)*BD+tl.arange(0,BD)
        row = tl.load(SORTED+position,position<N,-1)
        previous = tl.load(SORTED+position-1,(position>0)&(position<N),-1)
        head = (position<N)&((position==0)|(previous!=row))
        if tl.sum(head.to(tl.int32),0)>0:
            total = tl.full((BR,BD),0,VALUES.dtype.element_ty)
            cursor = position
            active = head
            # Stable sorting retains each row's original event order. Every
            # output has one writer and accumulates duplicates in that order.
            while tl.sum(active.to(tl.int32),0)>0:
                source = tl.load(ORDER+cursor,active,0)
                value = tl.load(VALUES+source[:,None]*WIDTH+d[None,:],
                                active[:,None]&(d[None,:]<WIDTH),0)
                total += value
                cursor += 1
                next_row = tl.load(SORTED+cursor,cursor<N,-1)
                active = head&(cursor<N)&(next_row==row)
            tl.store(OUT+row[:,None]*WIDTH+d[None,:],total,
                     head[:,None]&(d[None,:]<WIDTH))


def address(x, projection, active):
    b,t,d = x.shape
    levels = projection.shape[0] if projection.ndim == 3 else 1
    n = projection.shape[-2]
    out = torch.empty((b,t,active),device=x.device,dtype=torch.long)
    if b*t:
        _address_kernel[(b*t,)](x,projection,out,t,d,n,levels,active,
                               triton.next_power_of_2(d),4,triton.next_power_of_2(active),
                               enable_fp_fusion=False)
    return out


def forward(x,key,bias,logit,value,gain,features,indices,sparse,stable_slots=False,operand_mode=0):
    b,t,d = x.shape
    c,s = value.shape
    a = indices.shape[2] if sparse else c
    parallel = not sparse or stable_slots
    ra = 1 if sparse else 4
    out = x.new_zeros((b,t,s))
    acts,old,states = [x.new_empty((b,t,a)) for _ in range(3)]
    if b and t:
        grid = (b,triton.cdiv(a,ra)) if parallel else (b,1)
        _forward_kernel[grid](x,key,bias,logit,value,gain,features,indices,out,acts,old,states,
                              t,d,c,s,a,sparse,triton.next_power_of_2(d),triton.next_power_of_2(s),ra,parallel,operand_mode,
                              enable_fp_fusion=False)
        _readout_kernel[(b*t,triton.cdiv(s,64))](states,indices,value,gain,out,s,a,sparse,
                                               64,32,operand_mode,enable_fp_fusion=False)
    return [out,acts,old,states]


def backward(x,key,logit,value,gain,indices,local,rows,acts,old,states,gout,gfinal,
             ek,eb,er,sparse,online,stable_slots=False,operand_mode=0):
    b,t,d = x.shape
    c,s = value.shape
    a = acts.shape[2]
    gx = torch.zeros_like(x)
    # Eager online row slots can repeat between batch lanes: give each lane
    # private ownership. BPTT/deferred slots are unique selected events, so they
    # need no extra batch dimension. Dense rows use private batch ownership.
    # Never multiply a nearly-all-unique sparse event table by batch size.
    # Choose the smaller of per-batch touched rows and direct event slots.
    event_rows = sparse and rows < indices.numel() and b*rows > indices.numel()
    private_rows = not sparse or (rows < indices.numel() and not event_rows)
    owners = indices.numel() if event_rows else (b*rows if private_rows else rows)
    gk,gb,gr,gv = [x.new_zeros(shape) for shape in ((owners,d),(owners,),(owners,),(owners,s))]
    gp = x.new_empty(0) if online else gfinal.clone()
    parallel = not sparse or stable_slots
    ra = 1 if sparse else 4
    groups = triton.cdiv(a,ra) if parallel else 1
    gain_parts = x.new_zeros((b,groups))
    credit = x.new_empty((b,t,a))
    if b and t:
        grid = (b,triton.cdiv(a,ra)) if parallel else (b,1)
        _backward_kernel[grid](x,key,logit,value,gain,indices,local,acts,old,states,gout,
                               ek,eb,er,credit,gk,gb,gr,gv,gain_parts,gp,t,d,c,s,a,sparse,online,
                               triton.next_power_of_2(d),triton.next_power_of_2(s),ra,parallel,operand_mode,
                               rows,private_rows,groups,event_rows,
                               enable_fp_fusion=False)
        _input_vjp_kernel[(b*t,triton.cdiv(d,64))](credit,indices,key,gx,d,a,sparse,
                                                64,32,operand_mode,enable_fp_fusion=False)
    if private_rows:
        gk = gk.reshape(b,rows,d).sum(0)
        gb = gb.reshape(b,rows).sum(0)
        gr = gr.reshape(b,rows).sum(0)
        gv = gv.reshape(b,rows,s).sum(0)
    elif event_rows:
        event_map = local.reshape(-1)
        gk = row_reduce(event_map,gk,rows)
        gb = row_reduce(event_map,gb,rows)
        gr = row_reduce(event_map,gr,rows)
        gv = row_reduce(event_map,gv,rows)
    gg = gain_parts.sum()
    return [gx,gk,gb,gr,gv,gg,gp]


def row_reduce(indices, values, row_count):
    """Reduce selected event gradients in a fixed order without float atomics.

    The output is the only row-count-sized allocation. Sorting and reduction
    work scale with the supplied events; unused Capacity rows are not visited.
    Shapes and allocations are static, including while CUDA graphs record.
    """
    indices = indices.reshape(-1).contiguous()
    values = values.contiguous()
    width = 1
    for dimension in values.shape[1:]:
        width *= dimension
    out = values.new_zeros((row_count,*values.shape[1:]))
    if indices.numel() and width:
        sorted_indices, order = torch.sort(indices,stable=True)
        _row_reduce_kernel[(triton.cdiv(indices.numel(),8),triton.cdiv(width,64))](
            sorted_indices,order,values,out,indices.numel(),width,8,64,
            enable_fp_fusion=False)
    return out
