"""FP64 reference helpers shared by the exact studies.

These call the framework's reference equations directly (``core_online_forward_reference``,
``core_online_replay_reference``, ``_reference_forward_from_state``). Registered kernels
are checked against them in the preflight study.
"""
from __future__ import annotations

import dataclasses
import math

import harness  # noqa: F401  (verifies the framework lock first)
import torch
import torch.nn.functional as F
from dabsn.core import DABSNCore
import dabsn.temporal_credit as tc

torch.set_num_threads(1)


def core(seed=0,inputs=2,state=3,dtype=torch.float64):
    torch.manual_seed(seed)
    c=DABSNCore(inputs,state).to(dtype=dtype)
    # Exercise nonzero controls and nontrivial recurrent operating points.
    with torch.no_grad():
        c.Wg.bias.fill_(-.5)
        for name in ('k_s','k_y','k_b','k_n','k_saturation','r_s','r_y','r_b','r_n','r_saturation'):
            getattr(c,name).uniform_(-.12,.12)
    return c

def signals(result,seed):
    torch.manual_seed(seed)
    b,t,s=result.novelty.shape
    return tc.CoreSignalGrads(**{f.name:torch.randn(b,t,s,dtype=result.novelty.dtype)
                                for f in dataclasses.fields(tc.CoreSignalGrads)})

def credit(c,x,v,chunks,initial=None,intervention=None):
    r=c.initial_state(x.shape[0],dtype=x.dtype)
    e=tc.initial_core_eligibility(c,x.shape[0],device='cpu',dtype=x.dtype)
    if initial is not None: e=initial
    gradients={p:torch.zeros_like(p) for p in c.parameters()}
    input_grad=[]; offset=0; values=[]
    for index,width in enumerate(chunks):
        out,tape=tc.core_online_forward_reference(c,x[:,offset:offset+width],r,e)
        sv=tc.CoreSignalGrads(**{f.name:(None if getattr(v,f.name) is None else getattr(v,f.name)[:,offset:offset+width])
                                for f in dataclasses.fields(v)})
        e,g,gx=tc.core_online_replay_reference(c,tape,sv,e)
        for p,d in g.items(): gradients[p]+=d
        input_grad.append(gx); values.append(out.trajectory)
        r=out.final_state; offset+=width
        if intervention == 'reset' and index < len(chunks)-1:
            e=tc.initial_core_eligibility(c,x.shape[0],device='cpu',dtype=x.dtype)
        elif intervention == 'swap_once' and index == 0 and len(chunks)>1:
            e=tc.CoreEligibilityState(*(q.flip(0).contiguous() for q in e.tensors()))
    assert offset==x.shape[1]
    return e,gradients,torch.cat(input_grad,1),r,torch.cat(values,1)

def flat(c,g): return torch.cat([g.get(p,torch.zeros_like(p)).reshape(-1) for p in c.parameters()])
def rel(a,b): return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30))
def maxdiff(a,b): return float((a-b).detach().abs().max())
