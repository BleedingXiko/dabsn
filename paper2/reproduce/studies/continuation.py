"""Objective-conditioned continuation (loss habituation): E steers behavior after the objective is removed.

Native-defined E continuation with objective-blind readout and causal edits.

The framework reference path executes all trace families. A fixed reader encodes
a target family; continuation gets no objective argument or external target.
This restriction investigates objective information, not a universal loss-free
learner. CPU/CUDA are selected explicitly and use the same FP64 equations.
"""
from __future__ import annotations

import copy
import math
import random

import torch

import reference
from credit import embedded_operator

NEEDS_NATIVE = False
FP64_REFERENCE = True  # reference equations in float64; the compute policy does not apply


PROTOCOL = None
KY_INDEX = reference.tc.CORE_LOCAL_PARAMETER_NAMES.index('k_y')


def new_core(device):
    c = reference.core(0,inputs=2,state=2,dtype=torch.float64).to(device)
    with torch.no_grad():
        for p in c.parameters():
            p.zero_()
        c.A.weight.copy_(torch.tensor(PROTOCOL['forward']['A'],device=device,dtype=torch.float64))
        c.log_lambda.fill_(math.log(math.expm1(1.0)))
        c.log_kappa.fill_(math.log(math.expm1(.1)))
        c.logit_recover.fill_(math.log(.1/.9))
    return c


def zero_e(c):
    return reference.tc.initial_core_eligibility(c,1,device=c.W.weight.device,
                                                dtype=torch.float64)


@torch.no_grad()
def advance(c,state,e,x):
    tensor = torch.tensor(x,device=c.W.weight.device,dtype=torch.float64).reshape(1,1,2)
    forward,tape = reference.tc.core_online_forward_reference(c,tensor,state,e)
    after,_,_ = reference.tc.core_online_replay_reference(c,tape,
                                      reference.tc.CoreSignalGrads(),e)
    return forward.final_state,after


def target_family(x):
    return (.12*(x[0]+.5*x[1]),.12*(.5*x[0]+x[1]))


def orientation(e):
    value = -float(e.local[0,:,1,KY_INDEX].sum())
    return 0.0 if value == 0 else math.copysign(1.0,value)


@torch.no_grad()
def act(c,e,prediction,desired):
    ew = e.w_budget[0]
    residual = prediction.reshape(2)-torch.tensor(desired,device=ew.device,dtype=ew.dtype)
    displacement = -residual[:,None]*ew/(.05+ew.square().sum(-1,keepdim=True))
    norm = float(displacement.norm())
    if not math.isfinite(norm):
        raise FloatingPointError('Nonfinite action')
    c.W.weight.add_(displacement*min(1.0,.05/max(norm,1e-30)))


def condition(seed,objective,device):
    c = new_core(device)
    state = c.initial_state(1,device=device,dtype=torch.float64)
    e = zero_e(c)
    rng = random.Random(7001+seed)
    for _ in range(PROTOCOL['conditioning_steps']):
        x = [rng.uniform(.5,1.5),rng.uniform(.5,1.5)]
        state,e = advance(c,state,e,x)
        act(c,e,state[0],[objective*v for v in target_family(x)])
    return c,state,e


def edited(c,e,control,other):
    if control == 'erased':
        return zero_e(c)
    if control == 'exchanged':
        return other
    if control == 'k_y_erased':
        local = e.local.clone()
        local[:,:,:,KY_INDEX] = 0
        return reference.tc.CoreEligibilityState(e.w_budget,e.a_budget,e.wg_budget,
                                                  e.ug_budget,local,e.global_)
    return e


def continuation(seed,c,e):
    # No objective/target argument. The common reader retains its encoded family.
    state = c.initial_state(1,device=c.W.weight.device,dtype=torch.float64)
    with torch.no_grad():
        c.W.weight.zero_()
    rng = random.Random(90001+seed)
    predictions,inputs,orientations = [],[],[]
    for _ in range(PROTOCOL['continuation_steps']):
        x = [rng.uniform(.25,1.75),rng.uniform(.25,1.75)]
        choice = orientation(e)
        state,e = advance(c,state,e,x)
        predictions.append(state[0][0].tolist()); inputs.append(x); orientations.append(choice)
        act(c,e,state[0],[choice*v for v in target_family(x)])
    return predictions,inputs,orientations


def one(seed,device):
    conditioned = {z:condition(seed,z,device) for z in PROTOCOL['objectives']}
    rows = []
    for z in PROTOCOL['objectives']:
        for control in PROTOCOL['controls']:
            core = copy.deepcopy(conditioned[z][0])
            e = edited(core,conditioned[z][2],control,conditioned[-z][2])
            initial_orientation = orientation(e)
            predictions,inputs,orientations = continuation(seed,core,e)
            # The evaluator, separate from continuation, knows the objective.
            errors = [sum((p[i]-z*target_family(x)[i])**2 for i in range(2))/2
                      for p,x in zip(predictions,inputs)]
            if not all(math.isfinite(v) for v in errors):
                raise FloatingPointError('Nonfinite continuation error')
            half = len(errors)//2
            rows.append({'seed':seed,'objective':z,'control':control,
                'initial_decoded_orientation':initial_orientation,
                'orientation_correct_fraction':sum(v == z for v in orientations)/len(orientations),
                'pre_action_mse_first_half':sum(errors[:half])/half,
                'pre_action_mse_last_half':sum(errors[half:])/half,
                'pre_action_mse_by_event':errors,
                'zero_prediction_mse':sum(sum(v*v for v in target_family(x))/2
                                         for x in inputs)/len(inputs)})
    return rows

def jobs(protocol):
    return [{"seed":seed} for seed in protocol["seeds"]]


def identifier(job):
    return f"s{job['seed']}"


def run(job, ctx):
    global PROTOCOL
    PROTOCOL=ctx.protocol
    return {"rows":one(job["seed"],ctx.device)}
