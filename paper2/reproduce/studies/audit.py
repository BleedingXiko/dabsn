"""Native-versus-reference audit of online credit at trained A5 weights (CUDA).

No optimizer updates. At the final weights of each A5 cohort run, the native
online forward and replay are compared with the framework reference equations
on the same native tape (core and long memory), and the full model's online
forward is compared with its ordinary forward. The reference is the implemented
online rule; BPTT gradient equality is not the acceptance criterion.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import dabsn.temporal_credit as tc
from dabsn.reproductions import common as paper1

import harness
from studies import continuity

NEEDS_NATIVE = True


def jobs(protocol):
    return [{"seed":seed} for seed in [protocol["a5_development_seed"],*protocol["a5_heldout_seeds"]]]


def identifier(job):
    return f"a5-s{job['seed']}"


def precision_for(job):
    return "fp32"


def drift_report(actual,expected,torch):
    """Record finite rollout disagreement without claiming tensor equivalence."""
    if actual.shape!=expected.shape:raise ValueError('Forward diagnostic shapes differ')
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise FloatingPointError('Nonfinite forward diagnostic')
    row={'max_abs':float((actual-expected).abs().max())}
    try:
        torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
        row['within_tolerance']=True
    except AssertionError as error:
        row.update(within_tolerance=False,error=str(error))
    return row


def pointwise_reference(core,budget,energy,saturation,y,expression,gate,novelty,
                        next_saturation,plasticity,write,torch,F):
    """Independent original equations evaluated at identical native operands.

    Each intermediate is checked separately. Feeding the native incoming
    state prevents prior FP32 rounding drift from masquerading as a local
    transition or replay error.
    """
    decay=torch.sigmoid(core.logit_saturation_decay)
    effective=novelty*(1-torch.sigmoid(core.logit_saturation_suppress)*next_saturation)
    k_signal=(core.k_s*gate+core.k_y*y+core.k_b*torch.tanh(budget)+core.k_n*effective+
              core.k_bias+core.k_saturation*next_saturation)
    r_signal=(core.r_s*gate+core.r_y*y+core.r_b*torch.tanh(budget)+core.r_n*effective+
              core.r_bias+core.r_saturation*next_saturation)
    cost=F.softplus(core.log_kappa)*torch.exp(.5*torch.tanh(k_signal))
    recovery=torch.clamp(torch.sigmoid(core.logit_recover)*torch.exp(.5*torch.tanh(r_signal)),0,1)
    return {'novelty':torch.tanh((expression-budget).abs()),
        'plasticity':gate*energy,'write':plasticity*expression,
        'saturation':decay*saturation+(1-decay)*(novelty*(1-energy)),
        'next_budget':(1-torch.sigmoid(core.logit_alpha))*budget+core.beta+F.softplus(core.log_lambda)*write,
        'next_energy':torch.clamp(energy+recovery*(1-energy)-cost*plasticity,0,1)}


def check_state_bindings(result,tape,initial_state,compare):
    """Require native initial energy and separately returned final states to agree."""
    return max(compare(tape.energy[:,0],initial_state[1],'initial energy',exact=True),
        compare(result.final_state[0],result.trajectory[:,-1,tape.state_dim:],'final budget',exact=True),
        compare(result.final_state[2],result.saturation[:,-1],'final saturation',exact=True))


def bounded_matvec_check(actual,inputs,weight,label,torch,F,transform=None,slope=1.0,extra_scale=None):
    """Native FP32 matvec against FP64 truth under the deterministic dot-product bound.

    For an n-term FP32 dot product the forward error is at most
    gamma_n * sum_i |a_i y_i| with gamma_n = n u / (1 - n u), u = 2^-24 (Higham, Accuracy and
    Stability of Numerical Algorithms, Thm 3.1), plus one rounding of the stored result. A
    fixed absolute tolerance cannot hold at trained weights, where large terms cancel to a
    small result; this bound is derived, not tuned. ``transform`` maps the pre-activation to
    the stored quantity; ``slope`` is its Lipschitz constant; ``extra_scale`` adds the
    bound of a further FP32 term in the same pre-activation (already multiplied by its gamma)."""
    n=weight.shape[1];u=2.0**-24;gamma=n*u/(1-n*u)
    truth=F.linear(inputs.double(),weight.double())
    scale=inputs.double().abs()@weight.double().abs().T
    if transform is not None:truth=transform(truth)
    bound=slope*(gamma*scale+(0 if extra_scale is None else extra_scale))+u*truth.abs()+1e-30
    error=(actual.double()-truth).abs()
    ratio=float((error/bound).max())
    if ratio>1:
        raise AssertionError(f"{label}: FP32 error exceeds the deterministic dot-product bound "
                             f"(max error/bound {ratio:.4f})")
    return float(error.max()),ratio


def same_state_core_checks(core,x,result,tape,initial_state,compare,torch,F,bound_ratios=None):
    """Check projections, matvecs and pointwise equations on the native tape."""
    errors=[check_state_bindings(result,tape,initial_state,compare)];hidden=tape.state_dim
    with torch.no_grad():
        projected=core.W(x);gate_projected=core.Wg(x)
        errors.append(compare(tape.y,torch.tanh(projected+tape.budget),'same-state y'))
        # The recurrent matvecs are 256-term FP32 dot products whose terms cancel at trained
        # weights: they are checked against FP64 truth under the deterministic bound.
        for step in range(x.shape[1]):
            y=tape.y[:,step]
            error,ratio=bounded_matvec_check(tape.expression[:,step],y,core.A.weight,
                'same-state expression step '+str(step),torch,F)
            errors.append(error)
            if bound_ratios is not None:bound_ratios.append(ratio)
            # The gate pre-activation also holds the input projection Wg x + b, itself an
            # FP32 dot product: its truth and its own bound enter the same check.
            inputs_step=x[:,step].double()
            projected_gate=F.linear(inputs_step,core.Wg.weight.double(),core.Wg.bias.double())
            terms=core.Wg.weight.shape[1]+1;unit=2.0**-24
            projection_bound=(terms*unit/(1-terms*unit))*(inputs_step.abs()@core.Wg.weight.double().abs().T
                                                           +core.Wg.bias.double().abs())
            error,ratio=bounded_matvec_check(tape.gate[:,step],y,core.Ug.weight,
                'same-state gate step '+str(step),torch,F,
                transform=lambda pre:F.hardsigmoid(pre+projected_gate),slope=1/6,
                extra_scale=projection_bound)
            errors.append(error)
            if bound_ratios is not None:bound_ratios.append(ratio)
        wanted=pointwise_reference(core,tape.budget,tape.energy,tape.saturation,tape.y,
            tape.expression,tape.gate,tape.novelty,tape.next_saturation,tape.plasticity,
            result.write,torch,F)
        actual={name:getattr(result,name) for name in ('novelty','plasticity','write','saturation')}
        actual['next_budget']=result.trajectory[...,hidden:]
        actual['next_energy']=torch.cat([tape.energy[:,1:],result.final_state[1][:,None]],dim=1)
        for name,value in wanted.items():errors.append(compare(actual[name],value,'same-state '+name))
    return max(errors)


def run(job, ctx):
    if ctx.device!="cuda":
        raise RuntimeError("The trained-weight native audit requires CUDA")
    from dabsn.kernels.triton import (cuda_core_online_forward, cuda_core_online_replay,
                                      cuda_long_read_online_forward, cuda_long_read_online_replay)
    continuity.PROTOCOL=ctx.matrix["continuity"]
    source={"task":"a5","seed":job["seed"],"method":"online"}
    directory,raw=ctx.dependency("continuity",continuity.identifier(source))
    cfg=raw["result"]["configuration"]
    checkpoint=directory/f"step-{cfg['steps']:06d}.pt"
    if harness.sha256(checkpoint)!=raw["result"]["checkpoint_sha256"][checkpoint.name]:
        raise ValueError("Trained audit checkpoint checksum differs")
    attempt=sorted(directory.glob("attempt-*.json"))[-1]
    record={"status":"running","checkpoint_sha256":harness.sha256(checkpoint),
        "raw_record_sha256":harness.sha256(attempt),"cases":[],
        "tolerance":{"rtol":3e-5,"atol":3e-6,
            "recurrent_matvecs":"FP64 truth, deterministic FP32 dot-product bound gamma_n*sum|a_i y_i| "
                                "plus one rounding (Higham Thm 3.1); the gate through hardsigmoid slope 1/6"},
        "optimizer_updates":0,
        "scope":"Same-native-state forward transitions and same-tape online replay; "
            "full-rollout forward drift is measured separately and is not certified equivalent"}
    net=continuity.model_for('a5',cfg,torch.device('cuda'))
    saved=torch.load(checkpoint,map_location='cuda',weights_only=True)
    net.load_state_dict(saved['model']);net.eval()
    core=net.backbone.blocks[0].core;read=net.backbone.blocks[0].read
    table,identity=paper1.a5_table()
    def compare(actual,expected,label='state or input',*,exact=False):
        try:
            torch.testing.assert_close(actual,expected,rtol=0 if exact else 3e-5,atol=0 if exact else 3e-6,
                msg=lambda message:label+': '+message)
        except AssertionError:
            case['failed_check']=label
            raise
        return float((actual-expected).abs().max())
    core_names={p:name for name,p in core.named_parameters()}
    read_names={p:name for name,p in read.named_parameters()}
    for batch,length in ((1,2),(cfg['batch'],32),(cfg['batch'],cfg['train_length'])):
        case={'length':length,'batch':batch,'passed':False,'phase':'forward'}
        record['cases'].append(case);harness.write_record(ctx.directory/'audit-progress.json',record)
        paper1.seed_all(83100+length)
        x,_=paper1.a5_batch(batch,length,table,identity,torch.device('cuda'))
        recurrent=core.initial_state(batch,device='cuda')
        eligibility=tc.initial_core_eligibility(core,batch,device='cuda',dtype=torch.float32)
        expected,expected_tape=tc.core_online_forward_reference(core,x,recurrent,eligibility)
        actual,tape=cuda_core_online_forward(core,x,recurrent,eligibility)
        drift={}
        for field in ('trajectory','novelty','plasticity','expression','write','energy','saturation'):
            drift[field]=drift_report(getattr(actual,field),getattr(expected,field),torch)
        for index,(left,right) in enumerate(zip(actual.final_state,expected.final_state)):
            drift['final_state_'+str(index)]=drift_report(left,right,torch)
        case['core_forward_rollout']=drift;harness.write_record(ctx.directory/'audit-progress.json',record)
        ratios=[]
        local_error=same_state_core_checks(core,x,actual,tape,recurrent,compare,torch,F,ratios)
        case.update(core_same_state_max_abs=local_error,
                    recurrent_matvec_max_error_over_bound=max(ratios),phase='core replay');harness.write_record(ctx.directory/'audit-progress.json',record)
        signals=tc.CoreSignalGrads(**{name:torch.randn(batch,length,cfg['hidden'],device='cuda')/(batch*length)
                                     for name in tc.CoreSignalGrads.__dataclass_fields__})
        # Use one identical native tape to isolate replay from forward rounding.
        ref_state,ref_grad,ref_input=tc.core_online_replay_reference(core,tape,signals,eligibility)
        state,gradient,input_gradient=cuda_core_online_replay(core,tape,signals,eligibility)
        replay=[compare(left,right) for left,right in zip(state.tensors(),ref_state.tensors())]
        replay.extend(compare(gradient[p],ref_grad[p],'core replay '+core_names[p]) for p in core.parameters())
        replay.append(compare(input_gradient,ref_input))
        case.update(core_online_replay_max_abs=max(replay),phase='long forward and replay')
        harness.write_record(ctx.directory/'audit-progress.json',record)
        long_state=read.initial_long_scan_state(batch,device='cuda')
        long_e=tc.initial_long_read_eligibility(read,batch,device='cuda')
        expected_long,_,expected_final=tc.long_read_online_forward(read,actual.write,
            actual.plasticity,actual.novelty,long_state,long_e)
        long_output,long_tape,long_final=cuda_long_read_online_forward(read,actual.write,
            actual.plasticity,actual.novelty,long_state,long_e)
        long_forward=[compare(long_output,expected_long)]
        long_forward.extend(compare(left,right) for left,right in zip(long_final,expected_final))
        cotangent=torch.randn_like(long_output)/(batch*length)
        ref_e,ref_g,ref_x=tc.long_read_online_replay(read,long_tape,cotangent,long_e)
        next_e,gradients,inputs=cuda_long_read_online_replay(read,long_tape,cotangent,long_e)
        long_replay=[compare(next_e.traces,ref_e.traces)]
        long_replay.extend(compare(gradients[p],ref_g[p],'long replay '+read_names[p]) for p in ref_g)
        long_replay.extend(compare(left,right) for left,right in zip(inputs,ref_x))
        temporal=tc.initial_online_temporal_state(net,batch,stream_policy='reset_each_batch',
            stream_ids=tuple(range(batch)),device='cuda',dtype=torch.float32,eligibility_dtype=torch.float32)
        with torch.no_grad():
            ordinary=net.forward_sequence(x)
            online,_=net.forward_sequence_online(x,temporal)
        logits_drift=drift_report(online,ordinary,torch)
        logits_drift['prediction_agreement']=float((online.argmax(-1)==ordinary.argmax(-1)).float().mean())
        case.update(passed=True,phase='complete',
            local_transition_and_replay_passed=True,
            forward_equivalence_passed=all(value['within_tolerance'] for value in drift.values())
                and logits_drift['within_tolerance'],
            long_forward_max_abs=max(long_forward),long_replay_max_abs=max(long_replay),
            full_model_online_forward=logits_drift)
        harness.write_record(ctx.directory/'audit-progress.json',record)
    record['status']='local_transition_and_replay_passed'
    record['forward_equivalence_passed']=all(case['forward_equivalence_passed'] for case in record['cases'])
    return record
