"""Objective directions and update cadence: held-out directions and trace-derived action amplitude.

Calibration uses separate histories and objective directions. The continuation
function receives neither the ground-truth direction nor external targets.
The fixed trained reader still encodes the task family. All causal handoffs
match parameters and predictive state; the evaluator alone sees the objective.
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


from studies import continuation as base

PROTOCOL = None


def direction(angle):
    return (math.cos(math.pi*angle),math.sin(math.pi*angle))


def features(e):
    return torch.cat((-e.local[0,:,1,base.KY_INDEX],
        torch.ones(1,device=e.device,dtype=e.dtype)))


@torch.no_grad()
def column_scale(c,e):
    matrix,_,_ = embedded_operator(c,e)
    return float((matrix.square().sum(0)/matrix.shape[0]).sqrt().mean())


@torch.no_grad()
def action(c,e,prediction,desired,gain=1.):
    ew = e.w_budget[0]
    residual = prediction.reshape(2)-torch.tensor(desired,device=e.device,dtype=e.dtype)
    displacement = -gain*residual[:,None]*ew/(.05+ew.square().sum(-1,keepdim=True))
    norm = float(displacement.norm())
    if not math.isfinite(norm):
        raise FloatingPointError('Nonfinite direction action')
    factor = min(1.,.05/max(norm,1e-30))
    c.W.weight.add_(displacement*factor)
    return {'pre_cap_action_l2':norm,'action_l2':norm*factor,'cap_applied':factor<1}


def condition(seed,q,device):
    c = base.new_core(device)
    state = c.initial_state(1,device=device,dtype=torch.float64); e = base.zero_e(c)
    rng = random.Random(7001+seed)
    for _ in range(PROTOCOL['conditioning_steps']):
        x = [rng.uniform(.5,1.5),rng.uniform(.5,1.5)]
        state,e = base.advance(c,state,e,x)
        action(c,e,state[0],[q[i]*base.target_family(x)[i] for i in range(2)])
    return c,e


@torch.no_grad()
def calibrate(device):
    inputs = []; labels = []
    for seed in PROTOCOL['calibration_seeds']:
        for angle in PROTOCOL['calibration_angles_pi']:
            q = direction(angle); _,e = condition(seed,q,device)
            inputs.append(features(e)); labels.append(q)
    x = torch.stack(inputs)
    y = torch.tensor(labels,device=device,dtype=torch.float64)
    ridge = PROTOCOL['decoder_ridge']*torch.eye(x.shape[1],device=device,dtype=x.dtype)
    weights = torch.linalg.solve(x.T@x+ridge,x.T@y)
    error = float((x@weights-y).square().mean())
    return weights,{'histories':len(inputs),'training_direction_mse':error,
                    'feature_count':x.shape[1],'weights':weights.tolist()}


@torch.no_grad()
def decode(e,weights):
    value = features(e)@weights; norm = float(value.norm())
    if not math.isfinite(norm):
        raise FloatingPointError('Nonfinite objective readout')
    return torch.zeros_like(value) if norm<1e-12 else value/norm


@torch.no_grad()
def continuation(seed,c,e,weights,control):
    # The goal is deliberately absent from this interface.
    c.W.weight.zero_(); state = c.initial_state(1,device=e.device,dtype=e.dtype)
    fixed_readout = decode(e,weights)
    initial_scale = column_scale(c,e)
    rng = random.Random(90001+seed); events = []
    for _ in range(PROTOCOL['continuation_steps']):
        x = [rng.uniform(.25,1.75),rng.uniform(.25,1.75)]
        decoded = fixed_readout if control=='frozen_reader' else decode(e,weights)
        state,e = base.advance(c,state,e,x)
        current_scale = column_scale(c,e)
        gain = max(.1,min(10.,current_scale/max(initial_scale,1e-12))) if control=='adaptive_amplitude' else 1.
        prediction = state[0][0].tolist()
        desired = [float(decoded[i])*base.target_family(x)[i] for i in range(2)]
        report = action(c,e,state[0],desired,gain)
        events.append({'inputs':x,'prediction':prediction,'decoded_direction':decoded.tolist(),
            'mean_E_column_RMS':current_scale,'action_gain':gain,**report})
    return events,initial_scale


def test_seed(seed,device,weights):
    conditioned = {angle:condition(seed,direction(angle),device)
        for angle in PROTOCOL['test_angles_pi']}
    rows = []
    for angle in PROTOCOL['test_angles_pi']:
        q = direction(angle)
        opposite = min(PROTOCOL['test_angles_pi'],key=lambda other:
            sum((direction(other)[i]+q[i])**2 for i in range(2)))
        for control in PROTOCOL['controls']:
            original,e = conditioned[angle]; c = copy.deepcopy(original)
            e = base.edited(c,e,control,conditioned[opposite][1])
            initial_readout = decode(e,weights).tolist()
            events,initial_scale = continuation(seed,c,e,weights,control)
            errors = [sum((event['prediction'][i]-q[i]*base.target_family(event['inputs'])[i])**2
                for i in range(2))/2 for event in events]
            if not all(math.isfinite(value) for value in errors):
                raise FloatingPointError('Nonfinite held-out continuation error')
            rows.append({'seed':seed,'angle_pi':angle,'control':control,
                'initial_decoded_direction':initial_readout,
                'initial_direction_squared_error':sum((initial_readout[i]-q[i])**2 for i in range(2)),
                'handoff_mean_E_column_RMS':initial_scale,
                'pre_action_mse_last_half':sum(errors[48:])/48,
                'pre_action_mse_by_event':errors,'events':events})
    return rows

def jobs(protocol):
    # Calibration is shared by every test seed, so the study is one job.
    return [{"calibration_seeds":protocol["calibration_seeds"],"test_seeds":protocol["test_seeds"]}]


def identifier(job):
    return "all"


def run(job, ctx):
    global PROTOCOL
    PROTOCOL=ctx.protocol
    base.PROTOCOL=ctx.matrix["continuation"]
    weights,calibration=calibrate(ctx.device)
    rows=[]
    for seed in PROTOCOL["test_seeds"]:
        rows.extend(test_seed(seed,ctx.device,weights))
    return {"calibration":calibration,"rows":rows}
