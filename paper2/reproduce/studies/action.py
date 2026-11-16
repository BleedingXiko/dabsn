"""Parameter action: teacher matching by direct readouts of E, with full-Jacobian controls.

Teacher matching by direct readouts of E, with full-Jacobian controls.

Runs the same FP64 reference equations on explicitly selected CPU or CUDA.
This isolates action geometry; it does not benchmark registered kernels.
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


def state_vector(core, inputs, initial):
    _, state = core._reference_forward_from_state(inputs, initial_state=initial,
                                                 return_final_state=True)
    return torch.stack(state, dim=-1).reshape(-1)


def prefix_operator(core, inputs, initial):
    eligibility = reference.tc.initial_core_eligibility(core, inputs.shape[0],
        device=inputs.device, dtype=inputs.dtype)
    with torch.no_grad():
        forward, tape = reference.tc.core_online_forward_reference(core, inputs,
                                                                  initial, eligibility)
        eligibility, _, _ = reference.tc.core_online_replay_reference(core, tape,
            reference.tc.CoreSignalGrads(), eligibility)
        K, _, _ = embedded_operator(core, eligibility)
        state = torch.stack(forward.final_state, dim=-1).reshape(-1)
    return state, K


def full_jacobian(core, inputs, initial):
    state = state_vector(core, inputs, initial)
    parameters = tuple(core.parameters())
    rows = []
    for coordinate in range(state.numel()):
        derivatives = torch.autograd.grad(state[coordinate], parameters,
            retain_graph=coordinate+1 < state.numel(), allow_unused=True)
        rows.append(torch.cat([(value if value is not None else
            torch.zeros_like(parameter)).reshape(-1)
            for parameter, value in zip(parameters, derivatives)]))
    return torch.stack(rows).detach()


def inverse_action(operator, error):
    U, singular, Vh = torch.linalg.svd(operator, full_matrices=False)
    threshold = PROTOCOL['svd_relative_cutoff'] * singular.max()
    keep = (singular > threshold) & (singular > 0)
    reciprocal = torch.zeros_like(singular)
    reciprocal[keep] = 1/singular[keep]
    displacement = -(Vh.T @ (reciprocal * (U.T @ error)))
    return displacement, int(keep.sum()), float(threshold)


def apply_action(core, displacement):
    cursor = 0
    with torch.no_grad():
        for parameter in core.parameters():
            width = parameter.numel()
            parameter.add_(displacement[cursor:cursor+width].reshape_as(parameter))
            cursor += width
    if cursor != displacement.numel():
        raise AssertionError('Parameter ordering or action length changed')


def one(seed, relative_perturbation, device):
    cfg = PROTOCOL
    core = reference.core(17000+seed, inputs=cfg['input_dim'], state=cfg['state_dim'],
                          dtype=torch.float64).to(device)
    initial = tuple(value.detach() for value in core.initial_state(cfg['batch'],
                      device=device, dtype=torch.float64))
    rng = torch.Generator(device='cpu').manual_seed(71001+seed)
    inputs = torch.randn(cfg['batch'],cfg['length'],cfg['input_dim'],
                         generator=rng,dtype=torch.float64).to(device)*cfg['input_std']
    parameters = torch.cat([p.detach().reshape(-1).cpu() for p in core.parameters()])
    direction = torch.randn(parameters.numel(),generator=rng,dtype=torch.float64)
    perturbation = direction/direction.norm() * parameters.norm() * relative_perturbation
    teacher = copy.deepcopy(core)
    apply_action(teacher, perturbation.to(device))
    with torch.no_grad():
        target = state_vector(teacher, inputs, initial).detach()
        initial_error = float((state_vector(core, inputs, initial)-target).square().sum())
    if not math.isfinite(initial_error) or initial_error <= 0:
        raise FloatingPointError('Teacher perturbation produced no finite discrepancy')
    students = {method:copy.deepcopy(core) for method in cfg['methods']}
    rows = []
    for step in range(1,cfg['repeated_actions']+1):
        # Match the adjoint norm to its own inverse-E action at its own state.
        # No action borrows a later result or the teacher's parameter vector.
        for method, student in students.items():
            before, K = prefix_operator(student, inputs, initial)
            error = before-target
            operator = full_jacobian(student,inputs,initial) if method=='H_inverse' else K
            if method in ('E_inverse','H_inverse'):
                displacement, retained_rank, cutoff = inverse_action(operator,error)
            elif method=='E_adjoint_matched_norm':
                inverse, retained_rank, cutoff = inverse_action(K,error)
                adjoint = -(K.T @ error)
                norm = float(adjoint.norm())
                displacement = (adjoint * (inverse.norm()/norm) if norm > 0
                                else torch.zeros_like(adjoint))
            else:
                displacement = torch.zeros(K.shape[1],device=device,dtype=K.dtype)
                retained_rank = None; cutoff = None
            if not bool(torch.isfinite(displacement).all()):
                raise FloatingPointError('Nonfinite direct parameter action')
            predicted_change = operator @ displacement
            apply_action(student,displacement)
            with torch.no_grad():
                after = state_vector(student,inputs,initial).detach()
            squared_error = float((after-target).square().sum())
            if not math.isfinite(squared_error):
                raise FloatingPointError('Nonfinite nonlinear replay error')
            rows.append({'method':method,'action':step,
                'error_before_squared':float(error.square().sum()),
                'error_after_squared':squared_error,
                'error_ratio_to_initial':squared_error/initial_error,
                'l2_error_reduction_from_initial':1-math.sqrt(squared_error/initial_error),
                'parameter_displacement_l2':float(displacement.norm()),
                'retained_singular_modes':retained_rank,'svd_cutoff':cutoff,
                'linear_prediction_error_l2':float((after-before-predicted_change).norm()),
                'reported_error_convention':'squared error and L2 reduction both explicit'})
    return {'seed':seed,'teacher_relative_perturbation':relative_perturbation,
        'teacher_parameter_displacement_l2':float(perturbation.norm()),
        'initial_squared_error':initial_error,'rows':rows}

def jobs(protocol):
    return [{"seed":seed,"teacher_relative_perturbation":relative}
            for seed in protocol["seeds"] for relative in protocol["teacher_relative_perturbations"]]


def identifier(job):
    return f"s{job['seed']}-p{job['teacher_relative_perturbation']:g}"


def run(job, ctx):
    global PROTOCOL
    PROTOCOL=ctx.protocol
    return one(job["seed"],job["teacher_relative_perturbation"],ctx.device)
