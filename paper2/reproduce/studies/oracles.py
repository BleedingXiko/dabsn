"""Exact mechanism oracles for the B-channel credit (FP64 reference equations).

Each experiment asserts its identity to stated tolerance and returns its measured
values. Experiments: allocated storage, single-step derivatives, partition
invariance, the scalar counterexample, the omitted-influence residual identity,
delayed transport and learning interventions, terminal-Jacobian separation in
both directions, and the carried credit operator K.
"""
from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F
from dabsn.temporal_credit import CoreEligibilityState, initial_core_eligibility
from dabsn.temporal_credit import CORE_LOCAL_PARAMETER_NAMES, CORE_GLOBAL_PARAMETER_NAMES

import reference
from reference import core, signals, credit, flat, rel, maxdiff, tc, dataclasses
from credit import embedded_operator

NEEDS_NATIVE = False
FP64_REFERENCE = True  # reference equations in float64; the compute policy does not apply
DEVICE = "cpu"  # FP64 reference equations; the result does not depend on the accelerator


def matrix_mismatch(c,actual,expected):
    """Name the largest state/parameter-coordinate discrepancy."""
    difference=(actual-expected).detach().abs()
    index=int(difference.reshape(-1).argmax())
    row,column=divmod(index,difference.shape[1])
    offset=0
    for name,parameter in c.named_parameters():
        if column<offset+parameter.numel():
            within=column-offset
            coordinates=[]
            for width in reversed(parameter.shape):
                coordinates.append(within%width); within//=width
            return {'state_channel':row//3,
                    'state_component':('budget','energy','saturation')[row%3],
                    'parameter':name,'parameter_index':list(reversed(coordinates)),
                    'actual':float(actual[row,column].detach()),
                    'expected':float(expected[row,column].detach()),
                    'absolute_error':float(difference[row,column])}
        offset+=parameter.numel()
    raise AssertionError('Unmapped parameter coordinate')

def storage():
    rows=[]
    for i,s,b in [(1,1,1),(2,3,1),(3,2,2),(4,6,2),(6,4,1),(8,8,1)]:
        c=core(inputs=i,state=s)
        e=tc.initial_core_eligibility(c,b,device='cpu',dtype=torch.float64)
        p=sum(p.numel() for p in c.parameters()); n=sum(q.numel() for q in e.tensors())
        assert p==2*s*i+2*s*s+16*s+4
        assert n==b*(2*s*i+2*s*s+60*s)
        rows.append({'input':i,'state':s,'batch':b,'parameters':p,'allocated_eligibility_scalars':n})
    formula=[]
    for s in (16,64,256,1024,4096):
        p=4*s*s+16*s+4; n=4*s*s+60*s
        formula.append({'state':s,'parameters':p,'eligibility_scalars':n,'full_influence_scalars':3*s*p,
                        'eligibility_mib_fp32':4*n/2**20,'full_influence_gib_fp32':4*3*s*p/2**30,
                        'ratio':3*s*p/n,'kind':'analytical count; no large allocation'})
    return {'allocation_checks':rows,'scaling':formula}

def partitions():
    rows=[]
    for seed in range(3):
        for dtype in (torch.float64,torch.float32):
            c=core(seed,2,3,dtype)
            torch.manual_seed(100+seed)
            x=torch.randn(2,16,2,dtype=dtype)*.3
            r=c.initial_state(2,dtype=dtype)
            e=tc.initial_core_eligibility(c,2,device='cpu',dtype=dtype)
            out,_=tc.core_online_forward_reference(c,x,r,e)
            v=signals(out,200+seed)
            ref=credit(c,x,v,[16]); refg=flat(c,ref[1])
            for name,ch in [('one_step',[1]*16),('irregular',[3,1,5,2,5])]:
                got=credit(c,x,v,ch)
                row={'seed':seed,'dtype':str(dtype),'partition':name,'chunks':ch,
                     'credit_relative_l2':rel(flat(c,got[1]),refg),
                     'trace_relative_l2':rel(torch.cat([q.reshape(-1) for q in got[0].tensors()]),torch.cat([q.reshape(-1) for q in ref[0].tensors()])),
                     'input_vjp_max_abs':maxdiff(got[2],ref[2]),
                     'forward_max_abs':maxdiff(got[4],ref[4]),
                     'final_state_max_abs':max(maxdiff(a,b) for a,b in zip(got[3],ref[3]))}
                assert row['credit_relative_l2'] < (1e-11 if dtype==torch.float64 else 2e-5)
                assert row['trace_relative_l2'] < (1e-11 if dtype==torch.float64 else 2e-5)
                for key in ('input_vjp_max_abs','forward_max_abs','final_state_max_abs'):
                    assert row[key] < (1e-11 if dtype==torch.float64 else 2e-5)
                rows.append(row)
    return {'config':{'input':2,'state':3,'batch':2,'time':16,'seeds':[0,1,2],
                      'signals':'all eight emitted signals, independently prescribed',
                      'parameters_fixed':True},'rows':rows}

def exact_single_step():
    rows=[]
    for seed in range(3):
        c=core(seed,2,3); torch.manual_seed(300+seed)
        x=(torch.randn(2,1,2,dtype=torch.float64)*.2).requires_grad_()
        r=(torch.rand(2,3,dtype=x.dtype)*.05,
           .7+torch.rand(2,3,dtype=x.dtype)*.1,
           .1+torch.rand(2,3,dtype=x.dtype)*.05)
        e=tc.initial_core_eligibility(c,2,device='cpu',dtype=x.dtype)
        out,tape=tc.core_online_forward_reference(c,x.detach(),r,e)
        v=signals(out,400+seed)
        raw=c._reference_forward_from_state(x,initial_state=r,return_writes=True,return_cocktail=True)
        vals=(raw[0][...,:3],raw[0][...,3:],*raw[1:])
        # Canonical order: trajectory, novelty, plasticity, expression, write, energy, saturation.
        loss=sum((q*getattr(v,f.name)).sum() for q,f in zip(vals,dataclasses.fields(v)))
        expected=torch.autograd.grad(loss,(*c.parameters(),x),allow_unused=True)
        _,g,gx=tc.core_online_replay_reference(c,tape,v,e)
        expectedg=torch.cat([(torch.zeros_like(p) if d is None else d).reshape(-1) for p,d in zip(c.parameters(),expected[:-1])])
        err=maxdiff(flat(c,g),expectedg); inerr=maxdiff(gx,expected[-1])
        assert err<1e-11 and inerr<1e-11
        rows.append({'seed':seed,'parameter_vjp_max_abs':err,'input_vjp_max_abs':inerr})
    return {'config':{'input':2,'state':3,'batch':2,'time':1,'dtype':'float64','signals':'all eight'},'rows':rows}

def scalar_counterexample():
    c=core(inputs=1,state=1)
    with torch.no_grad():
        for p in c.parameters(): p.zero_()
        c.A.weight.fill_(1); c.beta.fill_(math.log(3)/2)
        c.logit_alpha.fill_(0); c.log_lambda.fill_(math.log(math.expm1(1)))
        c.log_kappa.fill_(math.log(math.expm1(.1))); c.logit_recover.fill_(math.log(.1/.9))
        c.k_y.fill_(1)
    x=torch.tensor([[[1.],[0.]]],dtype=torch.float64)
    r=c.initial_state(1,dtype=x.dtype)
    output=c._reference_forward_from_state(x,initial_state=r)
    derivative=torch.autograd.grad(output[0][0,-1,1],(c.W.weight,c.log_kappa))
    exact=derivative[0].item(); local_exact=derivative[1].item()
    v=tc.CoreSignalGrads(next_budget=torch.tensor([[[0.],[1.]]],dtype=x.dtype))
    online_grads=credit(c,x,v,[1,1])[1]
    online=online_grads[c.W.weight].item()
    local_online=online_grads[c.log_kappa].item()
    local_formula=-(1-math.exp(-.1))/8
    assert abs(local_exact-local_formula)<1e-12
    assert abs(local_online-local_formula)<1e-12
    negative=x.clone(); negative[:,0].mul_(-1)
    negative_out=c._reference_forward_from_state(negative,initial_state=r)
    assert maxdiff(output[0],negative_out[0])<1e-12
    negative_exact=torch.autograd.grad(negative_out[0][0,-1,1],c.W.weight)[0].item()
    negative_online=credit(c,negative,v,[1,1])[1][c.W.weight].item()
    assert abs(negative_exact+exact)<1e-12 and abs(negative_online+online)<1e-12
    predicted=-1/160
    eps=1e-5
    with torch.no_grad():
        c.W.weight.fill_(eps)
        plus=c._reference_forward_from_state(x,initial_state=r)[0][0,-1,1].item()
        c.W.weight.fill_(-eps)
        minus=c._reference_forward_from_state(x,initial_state=r)[0][0,-1,1].item()
        c.W.weight.zero_()
    finite_difference=(plus-minus)/(2*eps)
    assert abs(finite_difference-exact)<1e-8
    assert abs((exact-online)-predicted)<1e-12
    assert abs(exact-27/64)<1e-12 and abs(online-137/320)<1e-12
    return {'beta':math.log(3)/2,'bptt':exact,'online':online,'difference':exact-online,
            'finite_difference':finite_difference,'finite_difference_epsilon':eps,
            'analytic_difference':predicted,'absolute_identity_error':abs(exact-online-predicted),
            'local_control_exact':local_exact,'local_control_online':local_online,
            'local_control_formula':local_formula,'negative_input_exact':negative_exact,
            'negative_input_online':negative_online,'identical_forward_max_abs':maxdiff(output[0],negative_out[0])}

def embed(c,e):
    """Conceptual 3S x P matrix only for microscopic derivative checks."""
    s=c.hidden_dim; named=list(c.named_parameters()); pcount=sum(p.numel() for _,p in named)
    h=torch.zeros(3*s,pcount,dtype=torch.float64); offset=0
    for name,p in named:
        if name in ('W.weight','A.weight','Wg.weight','Ug.weight'):
            q=getattr(e,{'W.weight':'w_budget','A.weight':'a_budget','Wg.weight':'wg_budget','Ug.weight':'ug_budget'}[name])[0]
            for i in range(s): h[3*i,offset+i*p.shape[1]:offset+(i+1)*p.shape[1]]=q[i]
        elif name in tc.CORE_LOCAL_PARAMETER_NAMES:
            k=tc.CORE_LOCAL_PARAMETER_NAMES.index(name)
            for i in range(s): h[3*i:3*i+3,offset+i]=e.local[0,i,:,k]
        elif name in tc.CORE_GLOBAL_PARAMETER_NAMES:
            k=tc.CORE_GLOBAL_PARAMETER_NAMES.index(name); h[:,offset]=e.global_[0,:,:,k].reshape(-1)
        else: raise AssertionError(name)
        offset+=p.numel()
    return h

def project_mask(c):
    e=tc.initial_core_eligibility(c,1,device='cpu',dtype=torch.float64)
    e=tc.CoreEligibilityState(*(torch.ones_like(q) for q in e.tensors()))
    return embed(c,e)

def mathematical_state_step(c,x,state):
    """Independent one-step mathematical reference for the residual identity.

    Eligibility uses the mathematical hardsigmoid slope 1/6. PyTorch's
    F.hardsigmoid backward instead rounds that constant to float32 even for
    float64 tensors. An explicit clamp/division avoids comparing those two
    different numerical derivative conventions. Forward values are separately
    checked against the framework; no framework function is patched.
    """
    budget,energy,saturation=state
    y=torch.tanh(c.W(x)+budget)
    expression=F.linear(y,c.A.weight)
    gate=torch.clamp((c.Wg(x)+F.linear(y,c.Ug.weight)+3.0)/6.0,0.0,1.0)
    novelty=torch.tanh((expression-budget).abs())
    decay=torch.sigmoid(c.logit_saturation_decay)
    next_saturation=decay*saturation+(1.0-decay)*novelty*(1.0-energy)
    effective=novelty*(1.0-torch.sigmoid(c.logit_saturation_suppress)*next_saturation)
    k_signal=(c.k_s*gate+c.k_y*y+c.k_b*torch.tanh(budget)
              +c.k_n*effective+c.k_bias+c.k_saturation*next_saturation)
    r_signal=(c.r_s*gate+c.r_y*y+c.r_b*torch.tanh(budget)
              +c.r_n*effective+c.r_bias+c.r_saturation*next_saturation)
    cost=F.softplus(c.log_kappa)*torch.exp(0.5*torch.tanh(k_signal))
    recovery=torch.clamp(torch.sigmoid(c.logit_recover)*torch.exp(0.5*torch.tanh(r_signal)),0.0,1.0)
    plasticity=gate*energy
    next_budget=(1.0-torch.sigmoid(c.logit_alpha))*budget+c.beta+F.softplus(c.log_lambda)*(plasticity*expression)
    next_energy=torch.clamp(energy+recovery*(1.0-energy)-cost*plasticity,0.0,1.0)
    return next_budget,next_energy,next_saturation

def residual_identity():
    rows=[]
    for seed in range(3):
        c=core(seed,2,2); torch.manual_seed(500+seed)
        x=torch.randn(1,3,2,dtype=torch.float64)*.2
        r=(torch.rand(1,2,dtype=x.dtype)*.03,
           .75+torch.rand(1,2,dtype=x.dtype)*.03,
           .1+torch.rand(1,2,dtype=x.dtype)*.03)
        e=tc.initial_core_eligibility(c,1,device='cpu',dtype=x.dtype)
        mask=project_mask(c); full=torch.zeros_like(mask); hat=full.clone(); discrepancy=full.clone()
        for t in range(3):
            statevec=torch.stack(r,dim=-1).reshape(-1).detach().requires_grad_()
            rr=tuple(statevec.reshape(1,2,3)[...,k] for k in range(3))
            last=mathematical_state_step(c,x[:,t],rr)
            f=torch.stack(last,dim=-1).reshape(-1)
            js=[]; bs=[]
            for component in f:
                ds=torch.autograd.grad(component,(statevec,*c.parameters()),retain_graph=True,allow_unused=True)
                js.append(ds[0]); bs.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(c.parameters(),ds[1:])]))
            j=torch.stack(js); b=torch.stack(bs)
            d=torch.zeros_like(j)
            for k in range(2): d[3*k:3*k+3,3*k:3*k+3]=j[3*k:3*k+3,3*k:3*k+3]
            out,tape=tc.core_online_forward_reference(c,x[:,t:t+1],r,e)
            forwarderr=max(maxdiff(a,b) for a,b in zip(last,out.final_state))
            assert forwarderr<1e-12, f'Mathematical/framework forward mismatch: {forwarderr}'
            e,_,_=tc.core_online_replay_reference(c,tape,tc.CoreSignalGrads(),e)
            newhat=embed(c,e)
            predicted=mask*(d@hat+b)
            residual=(j-d)@hat+(1-mask)*(d@hat+b)
            full=j@full+b; discrepancy=j@discrepancy+residual
            ruleerr=maxdiff(newhat,predicted); residerr=maxdiff(full-newhat,discrepancy)
            if not (ruleerr<1e-11 and residerr<1e-11):
                details={'seed':seed,'time':t,'tolerance':1e-11,
                         'projected_rule_max_abs':ruleerr,
                         'residual_identity_max_abs':residerr,
                         'rule_mismatch':matrix_mismatch(c,newhat,predicted),
                         'residual_mismatch':matrix_mismatch(c,full-newhat,discrepancy),
                         'scope':'implemented E versus its declared recurrence; E=full Jacobian is not required'}
                print(json.dumps({'residual_diagnostic':details}),flush=True)
                raise AssertionError(json.dumps(details,sort_keys=True))
            rows.append({'seed':seed,'time':t,'projected_rule_max_abs':ruleerr,
                         'mathematical_forward_max_abs':forwarderr,
                         'residual_identity_max_abs':residerr,'full_vs_online_relative_l2':rel(newhat,full),
                         'cross_channel_residual_norm':float(torch.linalg.vector_norm((j-d)@hat)),
                         'discarded_coordinates_norm':float(torch.linalg.vector_norm((1-mask)*(d@hat+b)))})
            hat=newhat; r=out.final_state
    return {'config':{'input':2,'state':2,'batch':1,'time':3,'dtype':'float64',
                      'derivative_reference':'explicit mathematical hardsigmoid; slope 1/6 in float64'},'rows':rows}

def task_core(seed):
    c=core(seed,1,4)
    with torch.no_grad():
        c.W.weight.zero_(); c.Wg.weight.zero_(); c.Wg.bias.fill_(-.5)
        c.A.weight.copy_(.5*torch.eye(4,dtype=torch.float64)+.04*torch.randn(4,4,dtype=torch.float64))
        c.Ug.weight.mul_(.1)
        c.logit_alpha.fill_(-4); c.log_lambda.fill_(math.log(math.expm1(1)))
        c.beta.zero_()
    return c

def delayed_pass(c,condition,length=8,gradient=True):
    x=torch.zeros(2,length,1,dtype=torch.float64); x[0,0,0]=-1; x[1,0,0]=1
    labels=torch.tensor([-1.,1.],dtype=torch.float64)
    r=c.initial_state(2,dtype=x.dtype); e=tc.initial_core_eligibility(c,2,device='cpu',dtype=x.dtype)
    # A fixed readout ensures learning can only open the W input pathway.
    head=torch.ones(4,dtype=x.dtype)/2
    if condition=='bptt':
        out=c._reference_forward_from_state(x,initial_state=r)
        logits=out[0][:,-1,4:]@head
        loss=F.softplus(-labels*logits).mean()
        g=torch.autograd.grad(loss,c.W.weight)[0] if gradient else torch.zeros_like(c.W.weight)
    else:
        tapes=[]; states=[]
        for start in range(0,length,2):
            out,tape=tc.core_online_forward_reference(c,x[:,start:start+2],r,e)
            r=out.final_state
            if start+2==length:
                logits=r[0]@head
                loss=F.softplus(-labels*logits).mean()
                dl=-labels*torch.sigmoid(-labels*logits)/2
                v=torch.zeros(2,2,4,dtype=x.dtype); v[:,-1]=dl[:,None]*head
                sig=tc.CoreSignalGrads(next_budget=v)
            else: sig=tc.CoreSignalGrads()
            e,gs,_=tc.core_online_replay_reference(c,tape,sig,e)
            if start+2==length: g=gs[c.W.weight]
            elif condition=='reset': e=tc.initial_core_eligibility(c,2,device='cpu',dtype=x.dtype)
            elif condition=='swap_once' and start==0:
                e=tc.CoreEligibilityState(*(q.flip(0).contiguous() for q in e.tensors()))
    # Ties at exactly zero receive half-credit, not a class-dependent argmax.
    margin=labels*logits
    accuracy=float(((margin>0).double()+.5*(margin==0).double()).mean())
    return float(loss.detach()),accuracy,g,logits.detach()

def delayed_learning():
    rows=[]
    for seed in range(3):
        for condition in ('carry','reset','swap_once','bptt'):
            c=task_core(seed)
            initial=c.W.weight.detach().clone()
            for step in range(41):
                loss,acc,g,_=delayed_pass(c,condition)
                assert math.isfinite(loss) and torch.isfinite(g).all()
                if step in (0,1,2,5,10,20,40):
                    rows.append({'seed':seed,'condition':condition,'step':step,'loss':loss,'accuracy':acc,
                                 'w_norm':float(c.W.weight.detach().norm()),'w_credit_norm':float(g.norm()),
                                 'w_change':float((c.W.weight.detach()-initial).norm())})
                if step<40:
                    with torch.no_grad(): c.W.weight.add_(g,alpha=-.5)
            if condition=='reset':
                assert torch.equal(c.W.weight,initial) and acc==.5
    return {'config':{'input':1,'state':4,'batch':2,'time':8,'chunk':2,'seeds':[0,1,2],
                      'updates':40,'optimizer':'plain SGD on W only','lr':.5,'momentum':0,'weight_decay':0,
                      'readout':'fixed ones / 2 on final budget','task':'exhaustive delayed binary input at t=0',
                      'initial_W':'zero','initial_Wg':'zero','loss':'mean binary logistic loss',
                      'data_support':'two deterministic sequences; mechanism intervention, no population benchmark'},'rows':rows}

def delayed_transport():
    rows=[]
    for seed in range(3):
        for length in (4,8,16,32):
            c=task_core(seed)
            with torch.no_grad(): c.W.weight.fill_(.15)
            ref=None
            for condition in ('carry','reset','swap_once','bptt'):
                loss,acc,g,logits=delayed_pass(c,condition,length)
                if ref is None: ref=logits
                assert maxdiff(logits,ref)<1e-12
                assert float(logits[1]-logits[0])>0, 'forward class information was not retained'
                if condition=='carry': assert float(g.norm())>1e-12, 'carried credit vanished'
                if condition=='reset': assert torch.count_nonzero(g)==0
                rows.append({'seed':seed,'length':length,'boundaries':length//2-1,'condition':condition,
                             'w_credit_norm':float(g.norm()),'loss':loss,'accuracy':acc,
                             'class_logit_gap':float(logits[1]-logits[0]),'forward_control_max_abs':maxdiff(logits,ref)})
    return {'config':{'input':1,'state':4,'batch':2,'chunk':2,'initial_W':.15,'seeds':[0,1,2],
                      'optimizer_updates':0,'purpose':'preserve nonzero information while intervening only on credit'},'rows':rows}

def jacobian_separation():
    """Full terminal Jacobian collisions; every parameter and trace is checked."""
    c=core(0,1,1)
    bstar=math.log(3)/2; nstar=math.tanh(bstar-.5)
    chi=1/math.cosh(.5)**2
    kappa=.5*math.exp(-.5*math.tanh(.5))
    with torch.no_grad():
        for p in c.parameters(): p.zero_()
        c.A.weight.fill_(1)
        c.k_y.fill_(1)
        c.beta.fill_(bstar/2-1/8)
        c.log_lambda.fill_(math.log(math.expm1(1)))
        c.log_kappa.fill_(math.log(math.expm1(kappa)))
        c.logit_recover.fill_(math.log(1/3))
    # Constants, not differentiable functions of the live parameters.
    initial=tuple(torch.tensor([[v]],dtype=torch.float64) for v in (bstar,.5,nstar/2))
    histories={'zero':[0.,0.,0.,0.],
               'same_H':[1.,-27/16,15/16+3*chi/256,-11/64-3*chi/512],
               'same_E':[0.,0.,1.,-11/16]}
    records={}; tensors={}
    for name,values in histories.items():
        x=torch.tensor(values,dtype=torch.float64).reshape(1,4,1)
        outputs,last=c._reference_forward_from_state(x,initial_state=initial,
                                                     return_cocktail=True,return_final_state=True)
        final=torch.stack(last,dim=-1).reshape(-1)
        rows=[]
        for component in final:
            derivatives=torch.autograd.grad(component,tuple(c.parameters()),retain_graph=True,allow_unused=True)
            rows.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                                   for p,g in zip(c.parameters(),derivatives)]))
        H=torch.stack(rows)
        e=tc.initial_core_eligibility(c,1,device='cpu',dtype=torch.float64)
        out,tape=tc.core_online_forward_reference(c,x,initial,e)
        e,_,_=tc.core_online_replay_reference(c,tape,tc.CoreSignalGrads(),e)
        E=torch.cat([v.reshape(-1) for v in e.tensors()])
        tensors[name]=(H,E,final.detach(),outputs[0].detach(),e)
    base=tensors['zero']; offset=0
    for name,p in c.named_parameters():
        if name=='W.weight': w_column=offset
        offset+=p.numel()
    for name in ('same_H','same_E'):
        H,E,final,trajectory,e=tensors[name]
        records[name]={'full_jacobian_max_abs_difference':maxdiff(H,base[0]),
                       'all_eligibility_max_abs_difference':maxdiff(E,base[1]),
                       'terminal_state_max_abs_difference':maxdiff(final,base[2]),
                       'trajectory_max_abs_difference':maxdiff(trajectory,base[3]),
                       'W_budget_jacobian_difference':float(H[0,w_column]-base[0][0,w_column]),
                       'W_eligibility_difference':float(e.w_budget-base[4].w_budget)}
        assert maxdiff(final,base[2])<1e-12 and maxdiff(trajectory,base[3])<1e-12
    assert records['same_H']['full_jacobian_max_abs_difference']<1e-11
    assert abs(records['same_H']['W_eligibility_difference']-27*chi/65536)<1e-12
    assert records['same_E']['all_eligibility_max_abs_difference']<1e-11
    assert abs(records['same_E']['W_budget_jacobian_difference']+3*chi/256)<1e-12
    return {'histories':histories,'comparisons':records,'initial_state':[bstar,.5,nstar/2],
            'initial_state_contract':'prescribed constant, non-default, no parameter derivative',
            'jacobian_shape':list(base[0].shape),'parameter_names':[n for n,_ in c.named_parameters()],
            'scope':'terminal recurrent-state Jacobian; all parameters; fixed histories during differentiation'}


def operator():
    """The carried credit operator K: readout, Gram block structure, locality, family scaling."""
    return {"cases":[operator_case(seed) for seed in (0,1,2)]}


def operator_case(seed):
    c=reference.core(seed,inputs=2,state=3,dtype=torch.float64)
    b=2; s=3
    torch.manual_seed(8011+seed)
    prefix=torch.randn(b,6,2,dtype=torch.float64)*.13
    state=c.initial_state(b,dtype=torch.float64)
    eligibility=initial_core_eligibility(c,b,device="cpu",dtype=torch.float64)
    forward,tape=tc.core_online_forward_reference(c,prefix,state,eligibility)
    eligibility,_,_=tc.core_online_replay_reference(c,tape,
        tc.CoreSignalGrads(),eligibility)
    state=forward.final_state
    K,owners,names=embedded_operator(c,eligibility)
    assert K.shape==(3*b*s,sum(p.numel() for p in c.parameters()))
    x=torch.randn(b,1,2,dtype=torch.float64)*.11
    continuation,tape=tc.core_online_forward_reference(c,x,state,eligibility)
    signals=reference.signals(continuation,9100+seed)
    _,with_trace,_=tc.core_online_replay_reference(c,tape,signals,eligibility)
    empty=initial_core_eligibility(c,b,device="cpu",dtype=torch.float64)
    _,without_trace,_=tc.core_online_replay_reference(c,tape,signals,empty)
    historical=reference.flat(c,with_trace)-reference.flat(c,without_trace)

    state_vector=torch.stack(state,dim=-1).detach().requires_grad_()
    differentiable_state=tuple(state_vector[...,k] for k in range(3))
    ordinary,_=c._reference_forward_from_state(x,initial_state=differentiable_state,
        return_cocktail=True,return_writes=True,return_final_state=True)
    fields=(ordinary[0][...,:s],ordinary[0][...,s:],*ordinary[1:])
    weighted=sum((value*getattr(signals,field.name)).sum()
                 for value,field in zip(fields,dataclasses.fields(signals)))
    incoming=torch.autograd.grad(weighted,state_vector)[0].reshape(-1)
    predicted=K.T@incoming
    readout_error=reference.rel(historical,predicted)
    assert readout_error<1e-11

    local_columns=[i for i,owner in enumerate(owners) if owner!="shared"]
    shared_columns=[i for i,owner in enumerate(owners) if owner=="shared"]
    assert len(shared_columns)==4
    block=torch.zeros(3*b*s,3*b*s,dtype=K.dtype)
    for channel in range(s):
        coordinates=[lane*3*s+3*channel+k for lane in range(b) for k in range(3)]
        owned=[i for i,owner in enumerate(owners) if owner==channel]
        Ki=K[coordinates][:,owned]
        block[torch.tensor(coordinates)[:,None],torch.tensor(coordinates)[None,:]]=Ki@Ki.T
    G=K[:,shared_columns]
    gram_error=reference.rel(K@K.T,block+G@G.T)
    assert gram_error<1e-12
    nonlocal_support=max(float(K[[lane*3*s+3*other+k
                                   for lane in range(b) for k in range(3)],column].abs().max())
                         for column,owner in enumerate(owners) if owner!="shared"
                         for other in range(s) if other!=owner)
    assert nonlocal_support==0

    scaled=CoreEligibilityState(-2*eligibility.w_budget,eligibility.a_budget,
        eligibility.wg_budget,eligibility.ug_budget,eligibility.local,eligibility.global_)
    after,tape_after=tc.core_online_forward_reference(c,x,state,scaled)
    assert reference.maxdiff(after.trajectory,continuation.trajectory)==0
    _,scaled_credit,_=tc.core_online_replay_reference(c,tape_after,signals,scaled)
    delta=reference.flat(c,scaled_credit)-reference.flat(c,with_trace)
    expected=torch.zeros_like(delta)
    for column,name in enumerate(names):
        if name.startswith("W.weight["):
            expected[column]=-3*historical[column]
    intervention_error=reference.rel(delta,expected)
    assert intervention_error<1e-11

    torch.manual_seed(12000+seed)
    isotropic=torch.randn(1024,K.shape[0],dtype=K.dtype)
    credit_samples=isotropic@K
    predicted_variance=K.square().sum(0)
    empirical_variance=credit_samples.square().mean(0)
    positive=predicted_variance>1e-14
    rms_ratio=(empirical_variance[positive]/predicted_variance[positive]).sqrt()
    return {"seed":seed,"shape":list(K.shape),"local_columns":len(local_columns),
            "shared_columns":len(shared_columns),"operator_relative_l2_error":readout_error,
            "gram_relative_l2_error":gram_error,"nonlocal_support_max_abs":nonlocal_support,
            "family_scale_relative_l2_error":intervention_error,
            "isotropic_rms_ratio_mean":float(rms_ratio.mean()),
            "isotropic_rms_ratio_min":float(rms_ratio.min()),
            "isotropic_rms_ratio_max":float(rms_ratio.max()),
            "signal_covariance":"identity in expectation; 1024 samples for descriptive RMS estimate",
            "scope":"historical parameter credit at one boundary; exact current-step credit subtracted"}


EXPERIMENTS={'storage':storage,'single_step':exact_single_step,'partition':partitions,
             'counterexample':scalar_counterexample,'residual':residual_identity,
             'transport':delayed_transport,'learning':delayed_learning,
             'jacobian_separation':jacobian_separation,'operator':operator}


def jobs(protocol):
    return [{"experiment":name} for name in protocol["experiments"]]


def identifier(job):
    return job["experiment"]


def run(job, ctx):
    return EXPERIMENTS[job["experiment"]]()
