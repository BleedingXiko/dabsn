"""Device preflight: native-versus-reference parity, recovery, and measured target updates.

Runs first on the target device, before any learning study:

- mechanism: the registered online kernels against the reference rule (all
  eight emitted signals, credit, traces, forward); native BPTT and TBPTT
  derivatives against the reference; one real update and exact optimizer
  recovery for every central-study method; episode purity of the standard task
  generators (an episode is identical alone and inside any batch); on bf16
  hardware, a forward and a
  same-tape replay comparison against FP32.
- delayed_targets: two real updates and one evaluation batch at the full central
  configuration for every task, length and method, with measured time and memory.
- stream: two real updates plus exact checkpoint recovery for every arm of both
  stream studies (memory capacity and NARMA-10).
- forward_paths: the online-credit forward and the ordinary (BPTT/TBPTT) forward
  give the same outputs on the same weights, state and inputs, at every shape the
  matched studies use. Matched comparisons require identical forward computation.
"""
from __future__ import annotations

import copy
import io
import math
import time

import torch

import harness
import models
from studies import delayed_tasks
import reference as oracle
from reference import tc
from dabsn.temporal_credit import OnlineCoreFrame, initial_core_eligibility
from studies import stream

NEEDS_NATIVE = True
RUNTIME = None
MATRIX = harness.MATRIX
DELAYED = MATRIX["delayed"]


def jobs(protocol):
    return ([{"check":"mechanism"},{"check":"forward_paths"},{"check":"delayed_targets"}]
            + [{"check":"stream","study":study,**job}
               for study in ("continuous","sustained") for job in stream.preflight_jobs(MATRIX[study])])


def identifier(job):
    if job["check"]=="stream":
        return job["study"]+"-"+stream.identifier({k:v for k,v in job.items() if k not in ("check","study")})
    return job["check"]


def bf16_online_replay_probe(device):
    """Tiny native BF16 forward and same-tape replay comparison on the target.

    This follows the framework's low-precision CUDA replay test. The reference
    uses the actual native FP32 tape so this isolates replay derivatives from
    BF16 forward rounding. It does not change checkpoint recovery tolerances.
    """
    from dabsn.kernels.triton import cuda_core_online_forward, cuda_core_online_replay
    from dabsn.temporal_credit import (
        CoreSignalGrads, core_online_forward_reference, core_online_replay_reference,
    )
    model=models.Model(5,7,4,921,device)
    core=model.core
    inputs=torch.randn(2,3,5,device=device,dtype=torch.float32)*.1
    state=core.initial_state(2,device=device,dtype=torch.float32)
    eligibility=initial_core_eligibility(core,2,device=device,dtype=torch.float32)
    projection=harness.assert_projection_precision(model,inputs)
    expected,_=core_online_forward_reference(core,inputs,state,eligibility)
    with harness.autocast(device):
        actual,tape=cuda_core_online_forward(core,inputs,state,eligibility)
    forward_backend=getattr(core,'_last_core_online_backend',None)
    torch.testing.assert_close(actual.trajectory.float(),expected.trajectory,rtol=2e-2,atol=2e-2)
    if tape.y.dtype!=torch.float32 or any(value.dtype!=torch.float32
            for value in (*actual.final_state,*eligibility.tensors())):
        raise AssertionError('BF16 native online forward narrowed tape, state or eligibility storage')
    signals=CoreSignalGrads(**{name:torch.randn_like(actual.novelty)
                              for name in CoreSignalGrads.__dataclass_fields__})
    expected_e,expected_g,expected_x=core_online_replay_reference(core,tape,signals,eligibility)
    actual_e,actual_g,actual_x=cuda_core_online_replay(core,tape,signals,eligibility)
    eligibility_pairs=list(zip(actual_e.tensors(),expected_e.tensors()))
    gradient_pairs=[(actual_g[parameter],expected_g[parameter]) for parameter in core.parameters()]
    for left,right in (*eligibility_pairs,*gradient_pairs,(actual_x,expected_x)):
        if left.dtype!=torch.float32:
            raise AssertionError('BF16 replay narrowed FP32 eligibility, gradients or input credit')
        torch.testing.assert_close(left,right,rtol=2e-2,atol=2e-2)
    def maximum_error(pairs):
        return max(float((left-right).abs().max()) for left,right in pairs)
    return {'rtol':2e-2,'atol':2e-2,'projection_precision':projection,
        'forward_backend':forward_backend,'replay_backend':getattr(core,'_last_core_online_backend',None),
        'forward_max_absolute_error':float((actual.trajectory.float()-expected.trajectory).abs().max()),
        'eligibility_max_absolute_error':maximum_error(eligibility_pairs),
        'parameter_credit_max_absolute_error':maximum_error(gradient_pairs),
        'input_credit_max_absolute_error':float((actual_x-expected_x).abs().max()),
        'scope':'one tiny native BF16 forward; all emitted-signal replay derivatives against reference using the same native tape; no recovery tolerance change'}


def mechanism(device):
    runtime=RUNTIME
    profile=harness.precision_record(device)
    bf16_online_replay_parity=bf16_online_replay_probe(device) if profile['compute']=='bf16' else None
    # Matched methods see identical episodes: an episode is a pure function of its
    # identity, the same alone and inside any batch, at every horizon.
    for task in delayed_tasks.TASKS:
        for length in (128,512):
            whole,yw,_=delayed_tasks.delayed_batch(task,71,3,length,5,64,"cpu")
            for lane in range(5):
                alone,ya,_=delayed_tasks.delayed_batch(task,71,3+lane,length,1,64,"cpu")
                assert torch.equal(whole[lane:lane+1],alone) and torch.equal(yw[lane:lane+1],ya)
            assert torch.count_nonzero(whole[...,delayed_tasks.channels(task):])==0
    # Independent CPU reference -> selected native rule parity, all emitted signals.
    c=oracle.core(919,3,4,torch.float32)
    x=torch.randn(2,8,3)*.1
    r=c.initial_state(2,dtype=x.dtype)
    e=tc.initial_core_eligibility(c,2,device="cpu",dtype=x.dtype)
    out,_=tc.core_online_forward_reference(c,x,r,e)
    v=oracle.signals(out,912)
    expected=oracle.credit(c,x,v,[4,4])
    gpu=copy.deepcopy(c).to(device)
    state=gpu.initial_state(2,device=device,dtype=x.dtype)
    elig=initial_core_eligibility(gpu,2,device=device,dtype=x.dtype)
    vals=[]
    for start in (0,4):
        frame=OnlineCoreFrame(elig)
        outputs,state=gpu.forward_from_state(x[:,start:start+4].to(device),initial_state=state,
            return_cocktail=True,return_final_state=True,temporal_credit="online",
            eligibility_state=elig,online_frame=frame)
        fields=(outputs[0][...,:4],outputs[0][...,4:],*outputs[1:])
        loss=sum((a*getattr(v,f.name)[:,start:start+4].to(device)).sum()
                 for a,f in zip(fields,oracle.dataclasses.fields(v)))
        loss.backward()
        elig=frame.resulting_eligibility
        assert elig is not None
        vals.append(outputs[0].detach().cpu()); state=tuple(z.detach() for z in state)
    got=torch.cat([p.grad.detach().cpu().flatten() for p in gpu.parameters()])
    exact=oracle.flat(c,expected[1])
    error=oracle.rel(got,exact)
    trace_error=oracle.rel(torch.cat([z.cpu().flatten() for z in elig.tensors()]),
                           torch.cat([z.flatten() for z in expected[0].tensors()]))
    forward_error=oracle.maxdiff(torch.cat(vals,1),expected[4])
    if not all(math.isfinite(z) for z in (error,trace_error,forward_error)):
        raise AssertionError("nonfinite native parity error")
    if max(error,trace_error,forward_error)>2e-4:
        raise AssertionError(f"native parity failed: {error}, {trace_error}, {forward_error}")
    bptt_errors={}
    for method in ("bptt","tbptt"):
        gradients=[]; input_gradients=[]
        for reference,dev in ((True,"cpu"),(False,device)):
            model=copy.deepcopy(c).to(dev)
            xx=x.to(dev).detach().requires_grad_()
            rr=model.initial_state(2,device=dev,dtype=x.dtype)
            loss=0.
            for start in (0,4):
                if reference:
                    outputs,rr=model._reference_forward_from_state(xx[:,start:start+4],
                        initial_state=rr,return_cocktail=True,return_writes=True,return_final_state=True)
                else:
                    outputs,rr=model.forward_from_state(xx[:,start:start+4],
                        initial_state=rr,return_cocktail=True,return_final_state=True)
                fields=(outputs[0][...,:4],outputs[0][...,4:],*outputs[1:])
                term=sum((aa*getattr(v,f.name)[:,start:start+4].to(dev)).sum()
                    for aa,f in zip(fields,oracle.dataclasses.fields(v)))
                if method=="tbptt":
                    term.backward(); rr=tuple(z.detach() for z in rr)
                else: loss=loss+term
            if method=="bptt": loss.backward()
            gradients.append(torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).cpu().flatten()
                                        for p in model.parameters()]))
            input_gradients.append(xx.grad.cpu())
        err=oracle.rel(gradients[1],gradients[0])
        xerr=oracle.rel(input_gradients[1],input_gradients[0])
        if not math.isfinite(err+xerr) or max(err,xerr)>2e-4:
            raise AssertionError(f"{method} native derivative parity failed: {err}, {xerr}")
        bptt_errors[method]={"parameter_relative_l2":err,"input_relative_l2":xerr}
    # Verify one real update for every method; checkpoint recovery at sequence boundary.
    changes={}; projection_checks={}; bf16_forward_reference_parity=None
    recovery_rtol=1e-5
    chunk=DELAYED["main"]["online_chunk"]
    for method in DELAYED["main"]["methods"]:
        # The central study's model at small width: the full block, with its frozen arm.
        model=models.Block(19,8,4,11,device)
        frozen=models.FrozenCore(model) if method=="frozen_core" else None
        opt=torch.optim.Adam(frozen.trainable(model) if frozen else model.parameters(),lr=3e-4)
        xx,yy,_=delayed_tasks.delayed_batch("temporal_order",7,0,32,4,19,device)
        projection_checks[method]=harness.assert_projection_precision(model,xx)
        if profile['compute']=='bf16' and bf16_forward_reference_parity is None:
            core_model=models.Model(19,8,4,11,device)
            # Compare precision rounding separately from same-policy recovery.
            # This block enters no optimizer or online-credit intervention.
            with torch.no_grad():
                reference_output,reference_state=core_model.core.forward_from_state(
                    xx,return_final_state=True)
                reference_logits=core_model.head(torch.tanh(reference_output[0][:,-1,8:]))
                with harness.autocast(device):
                    low_output,low_state=core_model.core.forward_from_state(xx,return_final_state=True)
                    low_logits=core_model.head(torch.tanh(low_output[0][:,-1,8:]))
            torch.testing.assert_close(low_output[0].float(),reference_output[0],rtol=2e-2,atol=2e-2)
            torch.testing.assert_close(low_logits.float(),reference_logits,rtol=2e-2,atol=2e-2)
            for actual,expected_state in zip(low_state,reference_state):
                torch.testing.assert_close(actual.float(),expected_state,rtol=2e-2,atol=2e-2)
            bf16_forward_reference_parity={'rtol':2e-2,'atol':2e-2,
                'trajectory_max_absolute_error':float((low_output[0].float()-reference_output[0]).abs().max()),
                'readout_max_absolute_error':float((low_logits.float()-reference_logits).abs().max()),
                'scope':'one microscopic registered forward, BF16 versus FP32; no recovery tolerance change'}
            del reference_output,reference_state,reference_logits,low_output,low_state,low_logits,actual,expected_state
        before_core,before_read=_snapshot(model)
        models.update(model,opt,xx,yy,method,chunk,1.,frozen=frozen)
        changes[method]=dict(zip(("core","read"),_target_change(model,before_core,before_read,frozen)))
        stream=io.BytesIO()
        torch.save({"model":model.state_dict(),"optimizer":opt.state_dict()},stream)
        stream.seek(0)
        saved=torch.load(stream,map_location=device,weights_only=True)
        restored=models.Block(19,8,4,19,device); restored.load_state_dict(saved["model"])
        frozen2=models.FrozenCore(restored) if frozen else None
        opt2=torch.optim.Adam(frozen2.trainable(restored) if frozen2 else restored.parameters(),lr=3e-4)
        harness.restore_optimizer_state(opt2,saved["optimizer"])
        models.update(model,opt,xx,yy,method,chunk,1.,frozen=frozen)
        models.update(restored,opt2,xx,yy,method,chunk,1.,frozen=frozen2)
        for a,b in zip(model.parameters(),restored.parameters()):
            torch.testing.assert_close(a,b,rtol=recovery_rtol,atol=1e-6)
    return {"credit_relative_l2":error,"trace_relative_l2":trace_error,
            "forward_max_abs":forward_error,"update_norms":changes,
            "bptt_derivative_parity":bptt_errors,
            "episode_purity":"passed",
            "precision":profile,"mechanism_parity_compute":"fp32",
            "projection_precision":projection_checks,"recovery_parameter_rtol":recovery_rtol,
            "bf16_forward_reference_parity":bf16_forward_reference_parity,
            "bf16_online_replay_parity":bf16_online_replay_parity,
            "recovery":"model and optimizer after a complete sequence; recurrent state resets between sequences"}


def _target_change(model, before_core, before_read, frozen):
    """Every arm must change the read; the frozen-core arm must leave the core bitwise unchanged
    and every other arm must change it."""
    core = float(sum((p.detach() - b).square().sum() for p, b in zip(model.core.parameters(), before_core)).sqrt())
    read = float(sum((p.detach() - b).square().sum() for p, b in zip(model.block.read.parameters(), before_read)).sqrt())
    if not (math.isfinite(core) and math.isfinite(read)) or read <= 0:
        raise AssertionError("Target-configuration update did not change the read")
    if frozen is not None:
        frozen.verify()
    elif core <= 0:
        raise AssertionError("Target-configuration update did not change the core")
    return core, read


def _snapshot(model):
    return ([p.detach().clone() for p in model.core.parameters()],
            [p.detach().clone() for p in model.block.read.parameters()])


def delayed_targets(device):
    runtime=RUNTIME
    central=DELAYED["main"]
    rows=[]
    for task in DELAYED["tasks"]:
        for length in central["train_lengths_round_robin"]:
            for method in central["methods"]:
                model=models.Block(central["input_dim"],central["state_dim"],
                                   delayed_tasks.output_dim(task),100,device)
                frozen=models.FrozenCore(model) if method=="frozen_core" else None
                optimizer=torch.optim.Adam(frozen.trainable(model) if frozen else model.parameters(),
                    lr=DELAYED["development"]["learning_rates"][0],
                    betas=tuple(central["betas"]),weight_decay=central["weight_decay"])
                x,y,_=delayed_tasks.delayed_batch(task,17103,0,length,central["batch"],
                            central["input_dim"],device)
                projection_precision=harness.assert_projection_precision(model,x[:,:central["chunk"]])
                before_core,before_read=_snapshot(model)
                runtime.synchronize(); runtime.reset_peak()
                started=time.monotonic()
                loss,norm,_,_,_=models.update(model,optimizer,x,y,method,
                    central["online_chunk"],central["clip_norm"],delayed_tasks.objective(task),frozen)
                runtime.synchronize()
                first_seconds=time.monotonic()-started
                change,read_change=_target_change(model,before_core,before_read,frozen)
                first_allocated,first_reserved=runtime.peaks()
                runtime.reset_peak(); runtime.synchronize()
                started=time.monotonic()
                x2,y2,_=delayed_tasks.delayed_batch(task,17103,central["batch"],length,central["batch"],
                              central["input_dim"],device)
                second_loss,second_norm,_,_,_=models.update(model,optimizer,x2,y2,
                    method,central["online_chunk"],central["clip_norm"],delayed_tasks.objective(task),frozen)
                runtime.synchronize()
                second_seconds=time.monotonic()-started
                second_allocated,second_reserved=runtime.peaks()
                runtime.reset_peak(); runtime.synchronize()
                evaluation_batch=DELAYED['execution']['evaluation_batch']
                # Different batching changes neither episode identity nor
                # recurrent history. Compare the same eight episodes as a
                # separate batch and as lanes of the larger batch on target.
                xe,ye,ids=delayed_tasks.delayed_batch(task,730021,0,length,evaluation_batch,
                               central['input_dim'],device)
                xs,ys,small_ids=delayed_tasks.delayed_batch(task,730021,0,length,central['batch'],
                                     central['input_dim'],device)
                if ids[:len(small_ids)]!=small_ids or not torch.equal(xe[:len(small_ids)],xs) or not torch.equal(ye[:len(small_ids)],ys):
                    raise AssertionError('Evaluation batching changed episodes')
                equivalence={}
                for policy in ('full','final_chunk_only'):
                    large=xe.clone(); small=xs.clone()
                    if policy=='final_chunk_only':
                        large[:,:-central['chunk']]=0; small[:,:-central['chunk']]=0
                    a=models.delayed_predict(model,small,central['chunk'])
                    b=models.delayed_predict(model,large,central['chunk'])[:len(small_ids)]
                    error=float((a.float()-b.float()).abs().max())/(1+float(a.float().abs().max()))
                    if not math.isfinite(error) or error>DELAYED['execution']['evaluation_batch_equivalence_scaled_tolerance']:
                        raise AssertionError(f'Evaluation batching changed predictions: {policy}, {error}')
                    equivalence[policy]=error
                del xe,ye,xs,ys,large,small,a,b
                runtime.reset_peak(); runtime.synchronize()
                started=time.monotonic()
                delayed_tasks.delayed_evaluate(model,task,730021,length,evaluation_batch,evaluation_batch,
                         central["input_dim"],central["chunk"],device)
                runtime.synchronize()
                evaluation_batch_seconds=time.monotonic()-started
                rows.append({"task":task,"length":length,"method":method,
                             "precision":harness.precision_record(device),"projection_precision":projection_precision,
                             "loss":loss,"pre_clip_norm":norm,"core_update_norm":change,"read_update_norm":read_change,
                             "first_update_seconds":first_seconds,
                             "second_loss":second_loss,"second_pre_clip_norm":second_norm,
                             "second_update_seconds":second_seconds,
                             "evaluation_batch_seconds":evaluation_batch_seconds,
                             "evaluation_batch":evaluation_batch,
                             "evaluation_batch_equivalence":equivalence,
                             "peak_allocated_bytes":max(v for v in
                                (first_allocated,second_allocated) if v is not None)
                                if device=="cuda" else None,
                             "peak_reserved_bytes":max(v for v in
                                (first_reserved,second_reserved) if v is not None)
                                if device=="cuda" else None,
                             "backend":getattr(model.core,"_last_core_backend",None)})
    return {"target_configuration_updates":rows,"precision":harness.precision_record(device),
            "coverage":"two tasks x three lengths x four methods; first and second real updates plus one evaluation batch at full batch/state/input/chunk"}


def forward_paths(device):
    """Online-path versus ordinary forward, chunk by chunk, carrying the online state."""
    tolerance=MATRIX["delayed"]["execution"]["evaluation_batch_equivalence_scaled_tolerance"]
    rows=[]
    shapes=[(64,256),(64,512),(128,256)]
    for task in DELAYED["tasks"]:
        for inputs,state_dim in shapes:
            model=models.Block(inputs,state_dim,delayed_tasks.output_dim(task),100,device)
            x,_,_=delayed_tasks.delayed_batch(task,17103,0,512,DELAYED["main"]["batch"],inputs,device)
            chunk=DELAYED["main"]["chunk"]
            state=model.core.initial_state(x.shape[0],device=device)
            e=initial_core_eligibility(model.core,x.shape[0],device=device,dtype=torch.float32)
            worst=0.
            for start in range(0,x.shape[1],chunk):
                part=x[:,start:start+chunk]
                frame=OnlineCoreFrame(e)
                with harness.autocast(device):
                    online,online_state=model.core.forward_from_state(part,initial_state=state,
                        return_final_state=True,return_cocktail=True,temporal_credit="online",
                        eligibility_state=e,online_frame=frame)
                online_backend=getattr(model.core,"_last_core_online_backend",None)
                (online[0].sum()*0).backward(); e=frame.resulting_eligibility
                with torch.no_grad(),harness.autocast(device):
                    ordinary,ordinary_state=model.core.forward_from_state(part,initial_state=state,
                        return_final_state=True,return_cocktail=True)
                ordinary_backend=getattr(model.core,"_last_core_backend",None)
                for a,b in zip((*online,*online_state),(*ordinary,*ordinary_state)):
                    a=a.detach().float(); b=b.float()
                    worst=max(worst,float((a-b).abs().max())/(1+float(b.abs().max())))
                state=tuple(v.detach().float() for v in online_state)
            if not math.isfinite(worst) or worst>tolerance:
                raise AssertionError(f"Online and ordinary forward differ ({worst:.3e} > {tolerance}) at "
                                     f"{task} I={inputs} S={state_dim}: methods would not share one forward")
            rows.append({"task":task,"input_dim":inputs,"state_dim":state_dim,"length":512,
                         "max_scaled_difference":worst,"online_backend":online_backend,
                         "ordinary_backend":ordinary_backend})
    # The full block: chunked online forward against the ordinary full-sequence forward.
    block_rows=[]
    for task in DELAYED["tasks"]:
        for inputs,state_dim in shapes:
            model=models.Block(inputs,state_dim,delayed_tasks.output_dim(task),100,device)
            x,_,_=delayed_tasks.delayed_batch(task,17103,0,512,DELAYED["main"]["batch"],inputs,device)
            chunk=DELAYED["main"]["chunk"]
            with torch.no_grad(),harness.autocast(device):
                ordinary=model.net.forward_sequence(x).float()
            state=models.block_online_state(model,x.shape[0],chunk,device)
            parts=[]
            for start in range(0,x.shape[1],chunk):
                with harness.autocast(device):
                    output,pending=model.net.forward_sequence_online(x[:,start:start+chunk],state)
                (output.float().sum()*0).backward()
                state=models.finalize_online_temporal_state(pending)
                parts.append(output.detach().float())
            model.zero_grad(set_to_none=True)
            online=torch.cat(parts,1)
            worst=float((online-ordinary).abs().max())/(1+float(ordinary.abs().max()))
            if not math.isfinite(worst) or worst>tolerance:
                raise AssertionError(f"Block online and ordinary forward differ ({worst:.3e} > {tolerance}) at "
                                     f"{task} I={inputs} S={state_dim}")
            block_rows.append({"task":task,"input_dim":inputs,"state_dim":state_dim,"length":512,
                               "max_scaled_difference":worst})
    return {"rows":rows,"block_rows":block_rows,"tolerance":tolerance,"precision":harness.precision_record(device),
            "scope":"core: all emitted outputs and final states; block: all outputs; every chunk of a 512-step episode"}


def run(job, ctx):
    global RUNTIME
    RUNTIME=ctx.runtime
    if job["check"]=="mechanism":
        return mechanism(ctx.device)
    if job["check"]=="delayed_targets":
        return delayed_targets(ctx.device)
    if job["check"]=="forward_paths":
        return forward_paths(ctx.device)
    inner={k:v for k,v in job.items() if k not in ("check","study")}
    view=harness.Context(job["study"],inner,MATRIX[job["study"]],ctx.device,ctx.directory,ctx.results,
                         False,None,ctx.smoke,ctx.identity["reproduction_source_sha256"])
    result,_=stream.train(inner,view)
    return result
