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
        if not SPARSE:
            # Dense row owners are constant across time: accumulate gradients
            # in registers (BPTT and online). Scalar global read-after-write
            # through a different cross-warp layout can observe stale values.
            dense_row = start_first*RA+ar
            key_total = tl.full((RA,BD),0,X.dtype.element_ty)
            bias_total = tl.full((RA,),0,X.dtype.element_ty)
            retention_total = tl.full((RA,),0,X.dtype.element_ty)
            value_total = tl.full((RA,BS),0,X.dtype.element_ty)
        if not SPARSE and not ONLINE:
            reverse_carry = tl.load(DZ+b*C+dense_row,dense_row<C,0)
        if not SPARSE and ONLINE:
            # Dense online traces are owned by this program for the whole
            # scan: carry them in registers and write them once at the end. A
            # per-step global store/reload raced under Triton 3.2 (dbias
            # differed run to run).
            dense_mask = (dense_row[:,None]<C)&(d[None,:]<D)
            ek_reg = tl.load(EK+(b*C+dense_row[:,None])*D+d[None,:],dense_mask,0)
            eb_reg = tl.load(EB+b*C+dense_row,dense_row<C,0)
            er_reg = tl.load(ER+b*C+dense_row,dense_row<C,0)
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
                if SPARSE:
                    old_value_grad = tl.load(value_ptr,value_mask,0)
                    tl.store(value_ptr,old_value_grad+dv,value_mask)
                else:
                    value_total += dv
                if ONLINE:
                    if SPARSE:
                        ek = tl.load(EK+(b*C+row[:,None])*D+d[None,:],(a[:,None]<A)&(d[None,:]<D),0)
                        eb = tl.load(EB+b*C+row,a<A,0)
                        er = tl.load(ER+b*C+row,a<A,0)
                    else:
                        ek = ek_reg
                        eb = eb_reg
                        er = er_reg
                    ek = rho[:,None]*ek+slope[:,None]*xt[None,:]
                    eb = rho*eb+slope
                    er = rho*er+rho*(1-rho)*(prev-act)
                    if SPARSE:
                        tl.store(EK+(b*C+row[:,None])*D+d[None,:],ek,(a[:,None]<A)&(d[None,:]<D))
                        tl.store(EB+b*C+row,eb,a<A)
                        tl.store(ER+b*C+row,er,a<A)
                    else:
                        ek_reg = ek
                        eb_reg = eb
                        er_reg = er
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
                if SPARSE:
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
        if not SPARSE:
            owner = b*ROWS+dense_row
            tl.store(DK+owner[:,None]*D+d[None,:],key_total,(dense_row[:,None]<C)&(d[None,:]<D))
            tl.store(DB+owner,bias_total,dense_row<C)
            tl.store(DR+owner,retention_total,dense_row<C)
            tl.store(DV+owner[:,None]*S+s[None,:],value_total,(dense_row[:,None]<C)&(s[None,:]<S))
        if not SPARSE and not ONLINE:
            tl.store(DZ+b*C+dense_row,reverse_carry,dense_row<C)
        if not SPARSE and ONLINE:
            tl.store(EK+(b*C+dense_row[:,None])*D+d[None,:],ek_reg,dense_mask)
            tl.store(EB+b*C+dense_row,eb_reg,dense_row<C)
            tl.store(ER+b*C+dense_row,er_reg,dense_row<C)
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


if triton is not None:
    @triton.jit
    def _row_outer_kernel(SORTED, ORDER, CREDIT, X, OUT, N: tl.constexpr, A: tl.constexpr,
                          D: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr):
        # OUT[row] = sum over selected events e of that row of CREDIT[e] * X[token(e)].
        # Stable sort keeps event order; each output row has a single writer.
        position = tl.program_id(0)*BR+tl.arange(0,BR)
        d = tl.program_id(1)*BD+tl.arange(0,BD)
        row = tl.load(SORTED+position,position<N,-1)
        previous = tl.load(SORTED+position-1,(position>0)&(position<N),-1)
        head = (position<N)&((position==0)|(previous!=row))
        if tl.sum(head.to(tl.int32),0)>0:
            total = tl.full((BR,BD),0,CREDIT.dtype.element_ty)
            cursor = position
            active = head
            while tl.sum(active.to(tl.int32),0)>0:
                event = tl.load(ORDER+cursor,active,0)
                credit = tl.load(CREDIT+event,active,0)
                xv = tl.load(X+(event//A)[:,None]*D+d[None,:],active[:,None]&(d[None,:]<D),0)
                total += credit[:,None]*xv
                cursor += 1
                next_row = tl.load(SORTED+cursor,cursor<N,-1)
                active = head&(cursor<N)&(next_row==row)
            tl.store(OUT+row[:,None]*D+d[None,:],total,head[:,None]&(d[None,:]<D))


def row_outer_reduce(indices, credit, x, row_count):
    """Key gradient per Capacity row from the selected-event credit tape.

    Deterministic segmented reduction; only routed rows are visited and the
    only row-count-sized allocation is the output.
    """
    b, t, a = indices.shape
    d = x.shape[-1]
    flat = indices.reshape(-1).contiguous()
    out = x.new_zeros((row_count, d))
    if flat.numel() and d:
        ordered, order = torch.sort(flat, stable=True)
        _row_outer_kernel[(triton.cdiv(flat.numel(),8),triton.cdiv(d,64))](
            ordered,order,credit.reshape(-1).contiguous(),x.reshape(b*t,d).contiguous(),out,
            flat.numel(),a,d,8,64,enable_fp_fusion=False)
    return out


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
                               enable_fp_fusion=False,num_stages=1)
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
    if sparse and not online:
        # The scan's in-kernel key accumulation disagrees with the reference
        # for sparse BPTT at D>=16, so dK is rebuilt from the scalar credit
        # tape (which matches it) in the layout the caller expects:
        # row totals [C,D], or one slot per selected event [B*T*A,D].
        if rows == c:
            gk = row_outer_reduce(indices, credit, x, c)
        else:
            token = torch.arange(b*t, device=x.device).repeat_interleave(a)
            gk = credit.reshape(-1, 1)*x.reshape(b*t, d).index_select(0, token)
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


# ---------------------------------------------------------------------------
# Grouped BPTT path.
#
# Every heavy Capacity operation is independent of the recurrence. A canonical
# route sends each token to one contiguous bucket of A rows (dense execution is
# a single bucket of all C rows), so tokens are sorted by bucket and the
# projections, readout and every backward product become grouped GEMMs over
# bucket segments. Only the scalar retention recurrence remains sequential.
# All shapes, grids and allocations are fixed by (B, T, A, C): data-dependent
# segment bounds are read on device, so the path is CUDA-graph capturable.
# Reductions run in a fixed order without atomics: results are reproducible.
# ---------------------------------------------------------------------------

GROUP_BM = 128
GROUP_BN = 128
# Scalar retention scans: lanes per program and warps (latency-bound; small
# blocks put more independent programs in flight).
SCAN_BA = 128
SCAN_WARPS = 4
GROUP_BK = 16
# Launch shapes per grouped GEMM (measured on A40; BK fixes summation order).
GROUP_CONFIG = {"nt": {"BN": 128, "num_warps": 4, "num_stages": 3},
                "nn": {"BN": 128, "num_warps": 4, "num_stages": 3},
                "tn": {"BM": 128, "BN": 128, "num_warps": 8, "num_stages": 2}}


if triton is not None:
    @triton.jit
    def _grouped_nt_kernel(IN, W, PERM, TILE_G, TILE_LO, TILE_HI, OUT, SCALE,
                           KD: tl.constexpr, N: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                           MODE_W: tl.constexpr, HAS_SCALE: tl.constexpr, LOWP: tl.constexpr,
                           IP: tl.constexpr):
        # OUT[token, a] = sum_k IN[token, k] * W[g*N + a, k] for the tokens of bucket g.
        # LOWP: fp16 (1) / bf16 (2) operands with FP32 accumulation (autocast policy).
        # 1D grid, column blocks fastest: consecutive programs share the token
        # tile, and consecutive tiles of a bucket share its weights, in L2.
        blocks: tl.constexpr = (N+BN-1)//BN
        tile = tl.program_id(0)//blocks
        column = tl.program_id(0) % blocks
        g = tl.load(TILE_G+tile)
        lo = tl.load(TILE_LO+tile)
        hi = tl.load(TILE_HI+tile)
        if lo < hi:
            pos = lo+tl.arange(0, BM)
            pm = pos < hi
            token = tl.load(PERM+pos, pm, 0)
            n = column*BN+tl.arange(0, BN)
            nm = n < N
            row = g*N+n
            acc = tl.zeros((BM, BN), tl.float32)
            for k0 in range(0, KD, BK):
                k = k0+tl.arange(0, BK)
                km = k < KD
                a = tl.load(IN+token[:, None]*KD+k[None, :], pm[:, None] & km[None, :], 0)
                w = tl.load(W+row[:, None]*KD+k[None, :], nm[:, None] & km[None, :], 0)
                if LOWP == 2:
                    acc += tl.dot(a.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
                elif LOWP == 1:
                    acc += tl.dot(a.to(tl.float16), tl.trans(w.to(tl.float16)))
                else:
                    acc += tl.dot(a.to(tl.float32), tl.trans(_operand(w.to(tl.float32), MODE_W)),
                                  input_precision=IP)
            if HAS_SCALE:
                acc = acc*tl.load(SCALE).to(tl.float32)
            tl.store(OUT+token[:, None]*N+n[None, :], acc, pm[:, None] & nm[None, :])

    @triton.jit
    def _grouped_nn_kernel(P, W, PERM, TILE_G, TILE_LO, TILE_HI, OUT, SCALE,
                           A: tl.constexpr, N: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                           MODE_P: tl.constexpr, MODE_W: tl.constexpr, HAS_SCALE: tl.constexpr,
                           LOWP: tl.constexpr, IP: tl.constexpr):
        # OUT[token, n] = sum_a P[token, a] * W[g*A + a, n] for the tokens of bucket g.
        blocks: tl.constexpr = (N+BN-1)//BN
        tile = tl.program_id(0)//blocks
        column = tl.program_id(0) % blocks
        g = tl.load(TILE_G+tile)
        lo = tl.load(TILE_LO+tile)
        hi = tl.load(TILE_HI+tile)
        if lo < hi:
            pos = lo+tl.arange(0, BM)
            pm = pos < hi
            token = tl.load(PERM+pos, pm, 0)
            n = column*BN+tl.arange(0, BN)
            nm = n < N
            acc = tl.zeros((BM, BN), tl.float32)
            for k0 in range(0, A, BK):
                k = k0+tl.arange(0, BK)
                km = k < A
                p = tl.load(P+token[:, None]*A+k[None, :], pm[:, None] & km[None, :], 0)
                w = tl.load(W+(g*A+k)[:, None]*N+n[None, :], km[:, None] & nm[None, :], 0)
                if LOWP == 2:
                    acc += tl.dot(p.to(tl.bfloat16), w.to(tl.bfloat16))
                elif LOWP == 1:
                    acc += tl.dot(p.to(tl.float16), w.to(tl.float16))
                else:
                    acc += tl.dot(_operand(p.to(tl.float32), MODE_P), _operand(w.to(tl.float32), MODE_W),
                                  input_precision=IP)
            if HAS_SCALE:
                acc = acc*tl.load(SCALE).to(tl.float32)
            tl.store(OUT+token[:, None]*N+n[None, :], acc, pm[:, None] & nm[None, :])

    @triton.jit
    def _grouped_tn_kernel(P, IN, PERM, OFF, OUT, SCALE, CARRY, TRACE,
                           A: tl.constexpr, N: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                           MODE_P: tl.constexpr, HAS_SCALE: tl.constexpr,
                           SEQS: tl.constexpr, C: tl.constexpr, LOWP: tl.constexpr,
                           IP: tl.constexpr):
        # OUT[g*A + a, n] = sum over bucket-g tokens of P[token, a] * IN[token, n].
        # One writer per output element; tokens are summed in sorted order.
        # 1D grid: column blocks, then row blocks, then buckets (L2 reuse).
        nblocks: tl.constexpr = (N+BN-1)//BN
        ablocks: tl.constexpr = (A+BM-1)//BM
        g = tl.program_id(0)//(nblocks*ablocks)
        a = (tl.program_id(0)//nblocks % ablocks)*BM+tl.arange(0, BM)
        am = a < A
        n = (tl.program_id(0) % nblocks)*BN+tl.arange(0, BN)
        nm = n < N
        lo = tl.load(OFF+g)
        hi = tl.load(OFF+g+1)
        acc = tl.zeros((BM, BN), tl.float32)
        for p0 in range(lo, hi, BK):
            pos = p0+tl.arange(0, BK)
            pm = pos < hi
            token = tl.load(PERM+pos, pm, 0)
            p = tl.load(P+token[:, None]*A+a[None, :], pm[:, None] & am[None, :], 0)
            x = tl.load(IN+token[:, None]*N+n[None, :], pm[:, None] & nm[None, :], 0)
            if LOWP == 2:
                acc += tl.dot(tl.trans(p.to(tl.bfloat16)), x.to(tl.bfloat16))
            elif LOWP == 1:
                acc += tl.dot(tl.trans(p.to(tl.float16)), x.to(tl.float16))
            else:
                acc += tl.dot(tl.trans(_operand(p.to(tl.float32), MODE_P)), x.to(tl.float32),
                              input_precision=IP)
        if HAS_SCALE:
            acc = acc*tl.load(SCALE).to(tl.float32)
        # Online carried-trace term: OUT[row] += sum_b CARRY[b,row] * TRACE[b,row].
        for b in range(SEQS):
            carry = tl.load(CARRY+b*C+g*A+a, am, 0)
            trace = tl.load(TRACE+(b*C+g*A+a)[:, None]*N+n[None, :], am[:, None] & nm[None, :], 0)
            acc += carry[:, None]*trace
        tl.store(OUT+(g*A+a)[:, None]*N+n[None, :], acc, am[:, None] & nm[None, :])

    @triton.jit
    def _grouped_colsum_kernel(P, Q, PERM, OFF, OUTP, OUTQ, A: tl.constexpr,
                               BA: tl.constexpr, BK: tl.constexpr):
        # OUTP[g*A + a] = sum over bucket-g tokens of P[token, a] (likewise Q).
        g = tl.program_id(0)
        a = tl.program_id(1)*BA+tl.arange(0, BA)
        am = a < A
        lo = tl.load(OFF+g)
        hi = tl.load(OFF+g+1)
        sp = tl.zeros((BA,), tl.float32)
        sq = tl.zeros((BA,), tl.float32)
        for p0 in range(lo, hi, BK):
            pos = p0+tl.arange(0, BK)
            pm = pos < hi
            token = tl.load(PERM+pos, pm, 0)
            mask = pm[:, None] & am[None, :]
            sp += tl.sum(tl.load(P+token[:, None]*A+a[None, :], mask, 0), 0)
            sq += tl.sum(tl.load(Q+token[:, None]*A+a[None, :], mask, 0), 0)
        tl.store(OUTP+g*A+a, sp, am)
        tl.store(OUTQ+g*A+a, sq, am)

    @triton.jit
    def _grouped_scan_kernel(PRE, BUCKET, BIAS, R, Z, ACT, OLD, STATES,
                             T: tl.constexpr, C: tl.constexpr, A: tl.constexpr,
                             BA: tl.constexpr, MODE: tl.constexpr):
        # Scalar retention recurrence: lane (b, a) owns rows {g*A + a} of sequence b.
        b = tl.program_id(0)
        a = tl.program_id(1)*BA+tl.arange(0, BA)
        am = a < A
        for t in range(T):
            g = tl.load(BUCKET+b*T+t)
            row = g*A+a
            q = (b*T+t)*A+a
            bias = _operand(tl.load(BIAS+row, am, 0).to(tl.float32), MODE)
            act = _operand(libdevice.tanh(_operand(tl.load(PRE+q, am, 0)+bias, MODE)), MODE)
            rho = 1/(1+libdevice.exp(-tl.load(R+row, am, 0).to(tl.float32)))
            prev = tl.load(Z+b*C+row, am, 0)
            state = rho*prev+(1-rho)*act
            tl.store(Z+b*C+row, state, am)
            tl.store(ACT+q, act, am)
            tl.store(OLD+q, prev, am)
            tl.store(STATES+q, state, am)

    @triton.jit
    def _grouped_reverse_kernel(U, GAIN, BUCKET, R, ACT, OLD, DZ, CREDIT, DRE,
                                T: tl.constexpr, C: tl.constexpr, A: tl.constexpr,
                                BA: tl.constexpr):
        # Reverse retention recurrence; DZ enters as d(final) and leaves as d(initial).
        b = tl.program_id(0)
        a = tl.program_id(1)*BA+tl.arange(0, BA)
        am = a < A
        gain = tl.load(GAIN).to(tl.float32)
        for ti in range(T):
            t = T-1-ti
            g = tl.load(BUCKET+b*T+t)
            row = g*A+a
            q = (b*T+t)*A+a
            rho = 1/(1+libdevice.exp(-tl.load(R+row, am, 0).to(tl.float32)))
            act = tl.load(ACT+q, am, 0)
            prev = tl.load(OLD+q, am, 0)
            gz = gain*tl.load(U+q, am, 0)+tl.load(DZ+b*C+row, am, 0)
            tl.store(DZ+b*C+row, gz*rho, am)
            tl.store(CREDIT+q, gz*((1-rho)*(1-act*act)), am)
            tl.store(DRE+q, gz*rho*(1-rho)*(prev-act), am)


if triton is not None:
    @triton.jit
    def _grouped_online_reverse_kernel(U, GAIN, BUCKET, R, ACT, OLD, Q, M, SB, SR,
                                       W, CI, DRE, ES, T: tl.constexpr, C: tl.constexpr,
                                       A: tl.constexpr, BA: tl.constexpr):
        # Online credit within one call. With q_i = sum_{j>=i} gz_j rho^(j-i):
        #   sum_j gz_j ek_j = c0 ek_0 + sum_i q_i s_i x_i, c0 = rho q_first (left in Q),
        # and the carried traces become rho^n e_0 + sum_i rho^(n-i) (event input)_i.
        b = tl.program_id(0)
        a = tl.program_id(1)*BA+tl.arange(0, BA)
        am = a < A
        gain = tl.load(GAIN).to(tl.float32)
        for ti in range(T):
            t = T-1-ti
            g = tl.load(BUCKET+b*T+t)
            row = g*A+a
            q = (b*T+t)*A+a
            rho = 1/(1+libdevice.exp(-tl.load(R+row, am, 0).to(tl.float32)))
            act = tl.load(ACT+q, am, 0)
            prev = tl.load(OLD+q, am, 0)
            slope = (1-rho)*(1-act*act)
            h = rho*(1-rho)*(prev-act)
            gz = gain*tl.load(U+q, am, 0)
            total = gz+tl.load(Q+b*C+row, am, 0)
            tl.store(Q+b*C+row, total*rho, am)
            tl.store(W+q, total*slope, am)
            tl.store(CI+q, gz*slope, am)
            tl.store(DRE+q, total*h, am)
            m = tl.load(M+b*C+row, am, 0)
            tl.store(ES+q, m*slope, am)
            tl.store(SB+b*C+row, tl.load(SB+b*C+row, am, 0)+m*slope, am)
            tl.store(SR+b*C+row, tl.load(SR+b*C+row, am, 0)+m*h, am)
            tl.store(M+b*C+row, m*rho, am)

    @triton.jit
    def _online_trace_kernel(ES, X, PERM, OFF, M, EK, G: tl.constexpr, A: tl.constexpr,
                             D: tl.constexpr, C: tl.constexpr, BM: tl.constexpr,
                             BN: tl.constexpr, BK: tl.constexpr, LOWP: tl.constexpr,
                             IP: tl.constexpr):
        # EK[b, g*A+a] <- M[b,row] EK[b,row] + sum over (b,g) tokens of ES[token,a] X[token].
        # Rows a sequence did not route to keep M == 1 and are not rewritten.
        nblocks: tl.constexpr = (D+BN-1)//BN
        ablocks: tl.constexpr = (A+BM-1)//BM
        segment = tl.program_id(0)//(nblocks*ablocks)
        b = segment//G
        g = segment % G
        lo = tl.load(OFF+segment)
        hi = tl.load(OFF+segment+1)
        if lo < hi:
            a = (tl.program_id(0)//nblocks % ablocks)*BM+tl.arange(0, BM)
            am = a < A
            n = (tl.program_id(0) % nblocks)*BN+tl.arange(0, BN)
            nm = n < D
            acc = tl.zeros((BM, BN), tl.float32)
            for p0 in range(lo, hi, BK):
                pos = p0+tl.arange(0, BK)
                pm = pos < hi
                token = tl.load(PERM+pos, pm, 0)
                p = tl.load(ES+token[:, None]*A+a[None, :], pm[:, None] & am[None, :], 0)
                x = tl.load(X+token[:, None]*D+n[None, :], pm[:, None] & nm[None, :], 0)
                if LOWP == 2:
                    acc += tl.dot(tl.trans(p.to(tl.bfloat16)), x.to(tl.bfloat16))
                elif LOWP == 1:
                    acc += tl.dot(tl.trans(p.to(tl.float16)), x.to(tl.float16))
                else:
                    acc += tl.dot(tl.trans(p), x, input_precision=IP)
            row = b*C+g*A+a
            mask = am[:, None] & nm[None, :]
            trace = tl.load(EK+row[:, None]*D+n[None, :], mask, 0)
            decay = tl.load(M+row, am, 1)
            tl.store(EK+row[:, None]*D+n[None, :], decay[:, None]*trace+acc, mask)


def grouped_supported(x, sparse, online):
    """The grouped path: CUDA with FP32 accumulation (BPTT and online)."""
    return x.is_cuda and x.dtype == torch.float32


def _sequence_segments(bucket, b, t, groups):
    """Tokens sorted by (sequence, bucket): segment offsets for per-stream traces."""
    key = bucket + torch.arange(b, device=bucket.device).repeat_interleave(t)*groups
    ordered, perm = torch.sort(key, stable=True)
    off = torch.searchsorted(ordered, torch.arange(b*groups+1, device=bucket.device, dtype=torch.long))
    return perm.contiguous(), off.contiguous()


def _groups(indices, b, t, a, c, sparse):
    """Bucket sort and static-shape tile tables (no host synchronization)."""
    device = indices.device
    groups = c//a
    n = b*t
    if sparse:
        bucket = torch.div(indices[..., 0], a, rounding_mode="floor").reshape(-1).contiguous()
    else:
        bucket = torch.zeros(n, device=device, dtype=torch.long)
    ordered, perm = torch.sort(bucket, stable=True)
    off = torch.searchsorted(ordered, torch.arange(groups+1, device=device, dtype=torch.long))
    counts = off[1:]-off[:-1]
    tiles = (counts+GROUP_BM-1)//GROUP_BM
    last = torch.cumsum(tiles, 0)
    tile = torch.arange(triton.cdiv(n, GROUP_BM)+groups, device=device, dtype=torch.long)
    owner = torch.searchsorted(last, tile, right=True)
    valid = owner < groups
    owner = owner.clamp(max=groups-1)
    lo = off[owner]+(tile-(last[owner]-tiles[owner]))*GROUP_BM
    hi = torch.minimum(lo+GROUP_BM, off[owner+1])
    lo = torch.where(valid, lo, 0)
    hi = torch.where(valid, hi, 0)
    return {"bucket": bucket, "perm": perm.contiguous(), "off": off.contiguous(),
            "tile_g": owner.contiguous(), "tile_lo": lo.contiguous(), "tile_hi": hi.contiguous(),
            "groups": groups}


def _input_precision():
    """FP32 GEMM precision from PyTorch's global policy (as cuBLAS honours it):
    ``highest`` keeps IEEE FP32; ``high``/``medium`` allow TF32 tensor cores."""
    return "ieee" if torch.get_float32_matmul_precision() == "highest" else "tf32"


def _nt(inp, weight, grp, n_out, scale=None, mode_w=0, lowp=0):
    width = inp.shape[2]
    out = inp.new_empty((*inp.shape[:2], n_out))
    if inp.shape[0]*inp.shape[1]:
        cfg = GROUP_CONFIG["nt"]
        grid = (grp["tile_g"].numel()*triton.cdiv(n_out, cfg["BN"]),)
        _grouped_nt_kernel[grid](inp, weight, grp["perm"], grp["tile_g"], grp["tile_lo"], grp["tile_hi"],
                                 out, scale if scale is not None else out, width, n_out,
                                 GROUP_BM, cfg["BN"], GROUP_BK, mode_w, scale is not None, lowp,
                                 _input_precision(), num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return out


def _nn(p, weight, grp, n_out, scale=None, mode_p=0, mode_w=0, lowp=0):
    a = p.shape[2]
    out = p.new_empty((*p.shape[:2], n_out))
    if p.shape[0]*p.shape[1]:
        cfg = GROUP_CONFIG["nn"]
        grid = (grp["tile_g"].numel()*triton.cdiv(n_out, cfg["BN"]),)
        _grouped_nn_kernel[grid](p, weight, grp["perm"], grp["tile_g"], grp["tile_lo"], grp["tile_hi"],
                                 out, scale if scale is not None else out, a, n_out,
                                 GROUP_BM, cfg["BN"], GROUP_BK, mode_p, mode_w, scale is not None, lowp,
                                 _input_precision(), num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return out


def _tn(p, inp, grp, scale=None, mode_p=0, carry=None, trace=None, lowp=0):
    a, width = p.shape[2], inp.shape[2]
    rows = grp["groups"]*a
    out = p.new_empty((rows, width))
    cfg = GROUP_CONFIG["tn"]
    grid = (grp["groups"]*triton.cdiv(a, cfg["BM"])*triton.cdiv(width, cfg["BN"]),)
    seqs = 0 if carry is None else carry.shape[0]
    _grouped_tn_kernel[grid](p, inp, grp["perm"], grp["off"], out, scale if scale is not None else out,
                             carry if carry is not None else out, trace if trace is not None else out,
                             a, width, cfg["BM"], cfg["BN"], GROUP_BK, mode_p, scale is not None,
                             seqs, rows, lowp, _input_precision(),
                             num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return out


_LOW = {1: torch.float16, 2: torch.bfloat16}


def _low(tensor, operand_mode):
    """16-bit operand copy for autocast modes (round-to-nearest-even); FP32 otherwise."""
    return tensor.to(_LOW[operand_mode]) if operand_mode else tensor


def grouped_forward(x, key, bias, logit, value, gain, features, indices, sparse, operand_mode=0):
    """Ordinary-BPTT forward. ``features`` is updated in place to the final carry."""
    b, t, d = x.shape
    c, s = value.shape
    a = indices.shape[2] if sparse else c
    grp = _groups(indices, b, t, a, c, sparse)
    acts, old, states = [features.new_empty((b, t, a)) for _ in range(3)]
    if b and t:
        # Operand modes round x (autocast input), key, state and value to the
        # operand dtype, so these two products run exactly on tensor cores.
        pre = _nt(_low(x, operand_mode), _low(key, operand_mode), grp, a, lowp=operand_mode)
        _grouped_scan_kernel[(b, triton.cdiv(a, SCAN_BA))](pre, grp["bucket"], bias, logit, features,
                                                      acts, old, states, t, c, a, SCAN_BA, operand_mode,
                                                      num_warps=SCAN_WARPS)
        out = _nn(states, _low(value, operand_mode), grp, s, gain, lowp=operand_mode)
    else:
        out = features.new_zeros((b, t, s))
    return [out, acts, old, states]


def grouped_backward(x, key, logit, value, gain, indices, acts, old, states, gout, gfinal,
                     sparse, operand_mode=0):
    """Ordinary-BPTT backward: dense parameter gradients summed over every event."""
    b, t, d = x.shape
    c, s = value.shape
    a = acts.shape[2]
    grp = _groups(indices, b, t, a, c, sparse)
    gp = gfinal.clone()
    if not (b and t):
        return [states.new_zeros(x.shape), states.new_zeros((c, d)), states.new_zeros((c,)), states.new_zeros((c,)),
                states.new_zeros((c, s)), states.new_zeros(()), gp]
    # Operand modes follow autocast: every GEMM takes 16-bit operands with FP32
    # accumulation; the scans, reductions and carried state stay FP32.
    x_op = _low(x, operand_mode)
    gout_op = _low(gout, operand_mode) if operand_mode else gout.to(torch.float32)
    key_op, value_op = _low(key, operand_mode), _low(value, operand_mode)
    u = _nt(gout_op, value_op, grp, a, lowp=operand_mode)
    op_states = states
    if operand_mode == 1:
        op_states = states.to(torch.float16).to(states.dtype)
    elif operand_mode == 2:
        op_states = states.to(torch.bfloat16).to(states.dtype)
    gg = (op_states*u).sum()
    credit, dre = states.new_empty((b, t, a)), states.new_empty((b, t, a))
    _grouped_reverse_kernel[(b, triton.cdiv(a, SCAN_BA))](u, gain, grp["bucket"], logit, acts, old, gp,
                                                     credit, dre, t, c, a, SCAN_BA, num_warps=SCAN_WARPS)
    gv = _tn(states, gout_op, grp, gain, lowp=operand_mode)
    gk = _tn(credit, x_op, grp, lowp=operand_mode)
    gb, gr = states.new_empty((c,)), states.new_empty((c,))
    _grouped_colsum_kernel[(grp["groups"], triton.cdiv(a, 128))](credit, dre, grp["perm"], grp["off"],
                                                                gb, gr, a, 128, 32)
    gx = _nn(credit, key_op, grp, d, lowp=operand_mode)
    return [gx, gk, gb, gr, gv, gg, gp]


def grouped_online_backward(x, key, logit, value, gain, indices, acts, old, states, gout,
                            ek, eb, er, sparse, operand_mode=0):
    """Online credit for one call; eligibility traces ``ek, eb, er`` advance in place.

    Matches the per-event online kernels: parameter gradients are
    sum_j gz_j * e_j over every routed event, input credit is immediate.
    Returns dense parameter-gradient totals over the C rows.
    """
    b, t, d = x.shape
    c, s = value.shape
    a = acts.shape[2]
    if not (b and t):
        return [states.new_zeros(x.shape), states.new_zeros((c, d)), states.new_zeros((c,)), states.new_zeros((c,)),
                states.new_zeros((c, s)), states.new_zeros(())]
    grp = _groups(indices, b, t, a, c, sparse)
    x_op = _low(x, operand_mode)
    gout_op = _low(gout, operand_mode) if operand_mode else gout.to(torch.float32)
    key_op, value_op = _low(key, operand_mode), _low(value, operand_mode)
    u = _nt(gout_op, value_op, grp, a, lowp=operand_mode)
    op_states = states
    if operand_mode == 1:
        op_states = states.to(torch.float16).to(states.dtype)
    elif operand_mode == 2:
        op_states = states.to(torch.bfloat16).to(states.dtype)
    gg = (op_states*u).sum()
    carry = states.new_zeros((b, c))
    decay = states.new_ones((b, c))
    bias_in, retention_in = states.new_zeros((b, c)), states.new_zeros((b, c))
    w, credit, dre, es = [states.new_empty((b, t, a)) for _ in range(4)]
    _grouped_online_reverse_kernel[(b, triton.cdiv(a, SCAN_BA))](
        u, gain, grp["bucket"], logit, acts, old, carry, decay, bias_in, retention_in,
        w, credit, dre, es, t, c, a, SCAN_BA, num_warps=SCAN_WARPS)
    gv = _tn(states, gout_op, grp, gain, lowp=operand_mode)
    # dK reads the incoming traces, so it runs before the traces advance.
    gk = _tn(w, x_op, grp, carry=carry, trace=ek, lowp=operand_mode)
    gb, gr = states.new_empty((c,)), states.new_empty((c,))
    _grouped_colsum_kernel[(grp["groups"], triton.cdiv(a, 128))](w, dre, grp["perm"], grp["off"],
                                                                gb, gr, a, 128, 32)
    gb += (carry*eb).sum(0)
    gr += (carry*er).sum(0)
    gx = _nn(credit, key_op, grp, d, lowp=operand_mode)
    perm, off = _sequence_segments(grp["bucket"], b, t, grp["groups"])
    cfg = GROUP_CONFIG["tn"]
    _online_trace_kernel[(b*grp["groups"]*triton.cdiv(a, cfg["BM"])*triton.cdiv(d, cfg["BN"]),)](
        es, x_op, perm, off, decay, ek, grp["groups"], a, d, c, cfg["BM"], cfg["BN"], GROUP_BK,
        operand_mode, _input_precision(), num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    eb.mul_(decay).add_(bias_in)
    er.mul_(decay).add_(retention_in)
    return [gx, gk, gb, gr, gv, gg]
