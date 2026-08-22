"""Native C++/OpenMP runtime for the canonical DABSN core and read."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

_EXT = None
_TRIED = False
_EAGER_THREE_WAY_READ = None
_CORE_OP_CPU_REGISTERED = False
_READ_OP_CPU_REGISTERED = False
_SOURCE = Path(__file__).with_name("cpu_runtime.cpp")


def _prepare_cpp_build_env() -> str:
    build_root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not build_root:
        build_root = os.path.join("/private/tmp", "dabsn_torch_extensions")
        os.environ["TORCH_EXTENSIONS_DIR"] = build_root
    build_dir = os.path.join(build_root, "dabsn_cpu_runtime")
    os.makedirs(build_dir, exist_ok=True)
    venv_bin = os.path.join(sys.prefix, "bin")
    ninja = os.path.join(venv_bin, "ninja")
    if os.path.exists(ninja):
        path = os.environ.get("PATH", "")
        if venv_bin not in path.split(os.pathsep):
            os.environ["PATH"] = venv_bin + (os.pathsep + path if path else "")
    return build_dir


def _libomp_prefix() -> str | None:
    for candidate in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
        if os.path.isdir(os.path.join(candidate, "include")):
            return candidate
    for brew in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew", "brew"):
        try:
            prefix = subprocess.check_output(
                [brew, "--prefix", "libomp"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if prefix and os.path.isdir(prefix):
                return prefix
        except Exception:
            continue
    return None


def _torch_omp_paths() -> tuple[str | None, str | None]:
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if not os.path.exists(os.path.join(torch_lib, "libomp.dylib")):
        return None, None
    include = os.path.join(os.path.dirname(torch.__file__), "include")
    if not os.path.exists(os.path.join(include, "omp.h")):
        omp = _libomp_prefix()
        include = f"{omp}/include" if omp else include
    return torch_lib, include


def _load_ext(*, required: bool = False):
    global _EXT, _TRIED
    if _TRIED:
        if required and _EXT is None:
            raise RuntimeError("native DABSN CPU extension is unavailable")
        return _EXT
    _TRIED = True
    try:
        from torch.utils.cpp_extension import load

        build_dir = _prepare_cpp_build_env()
        cflags = ["-O3", "-std=c++17", "-ffp-contract=off"]
        ldflags: list[str] = []
        if sys.platform == "darwin":
            torch_lib, torch_include = _torch_omp_paths()
            if torch_lib:
                cflags += ["-Xpreprocessor", "-fopenmp", f"-I{torch_include}"]
                ldflags += [
                    f"-L{torch_lib}",
                    "-lomp",
                    f"-Wl,-rpath,{torch_lib}",
                ]
            else:
                omp = _libomp_prefix()
                if omp:
                    cflags += ["-Xpreprocessor", "-fopenmp", f"-I{omp}/include"]
                    ldflags += [f"-L{omp}/lib", "-lomp"]
        else:
            cflags += ["-fopenmp"]
            ldflags += ["-fopenmp"]
        _EXT = load(
            name="dabsn_cpu_runtime",
            sources=[str(_SOURCE)],
            build_directory=build_dir,
            extra_cflags=cflags,
            extra_ldflags=ldflags,
            verbose=False,
        )
    except Exception as exc:
        _EXT = None
        if required:
            raise RuntimeError(
                f"native DABSN CPU extension failed: {type(exc).__name__}: {exc}"
            ) from exc
    return _EXT


def _scalars(core):
    import torch.nn.functional as F

    gate_alpha = torch.sigmoid(core.logit_alpha.reshape(())).item()
    lam = F.softplus(core.log_lambda.reshape(())).item()
    saturation_suppress = torch.sigmoid(core.logit_saturation_suppress.reshape(())).item()
    return gate_alpha, lam, saturation_suppress


def _initial_core_state(core, e_in, initial_state=None):
    batch = int(e_in.shape[0])
    if initial_state is None:
        return core.initial_state(batch, device=e_in.device, dtype=e_in.dtype)
    if len(initial_state) != 3:
        raise ValueError("initial_state must contain budget, energy, and saturation")
    state = tuple(value.to(device=e_in.device, dtype=e_in.dtype) for value in initial_state)
    expected = (batch, int(core.hidden_dim))
    if any(value.shape != expected for value in state):
        raise ValueError(f"initial_state tensors must have shape {expected}")
    return state


def native_core_forward(
    core,
    e_in,
    return_writes=False,
    return_cocktail=False,
    *,
    initial_state=None,
    return_final_state=False,
):
    """Run the native CPU core for training or inference when available.

    Gradient-enabled calls use the custom autograd scan; inference uses the
    fused forward-only scan. Optional native execution falls back to the
    PyTorch reference path for unsupported inputs.
    """
    ext = _load_ext()
    if ext is None or e_in.is_cuda:
        return core._reference_forward_from_state(
            e_in,
            initial_state=initial_state,
            return_writes=return_writes,
            return_cocktail=return_cocktail,
            return_final_state=return_final_state,
        )
    initial_state = _initial_core_state(core, e_in, initial_state)
    core._last_core_backend = "cpu_native_cpp"
    # Native C++ forward with a reverse-scan autograd implementation.
    if torch.is_grad_enabled() and (
        e_in.requires_grad or any(p.requires_grad for p in core.parameters())
    ):
        U, nov, p, ay, write, energy, saturation, final_b, final_e, final_c = (
            native_core_forward_train(core, e_in, initial_state=initial_state)
        )
        result: tuple[Any, ...]
        if return_cocktail:
            result = U, nov, p, ay, write, energy, saturation
        elif return_writes:
            result = U, nov, p, ay, write
        else:
            result = U, nov, p
        if return_final_state:
            return result, (final_b, final_e, final_c)
        return result
    return _native_core_forward_infer(
        core,
        e_in,
        return_writes,
        return_cocktail,
        initial_state=initial_state,
        return_final_state=return_final_state,
    )


@torch.no_grad()
def _native_core_forward_infer(
    core,
    e_in,
    return_writes=False,
    return_cocktail=False,
    *,
    initial_state=None,
    return_final_state=False,
):
    """Inference path: fused forward-only scan (no autograd graph)."""
    ext = _load_ext()
    H = int(core.hidden_dim)
    gate_alpha, lam, saturation_suppress = _scalars(core)
    if ext is None or e_in.is_cuda:
        return core._reference_forward_from_state(
            e_in,
            initial_state=initial_state,
            return_writes=return_writes,
            return_cocktail=return_cocktail,
            return_final_state=return_final_state,
        )
    initial_b, initial_e, initial_c = _initial_core_state(core, e_in, initial_state)
    Wx = core.W(e_in).float().contiguous()
    Wgx = core.Wg(e_in).float().contiguous()
    GA_w = torch.cat([core.Ug.weight, core.A.weight], dim=0).float().contiguous()
    lcd = core.logit_saturation_decay.expand(H).float().contiguous()
    kc = core.k_saturation.float().contiguous()
    rc = core.r_saturation.float().contiguous()
    U, nov, p, ay, write, energy, saturation, final_b, final_e, final_c = ext.dabsn_core_scan_cpu(
        Wx,
        Wgx,
        GA_w,
        core.beta.float().contiguous(),
        core.log_kappa.float().contiguous(),
        core.logit_recover.float().contiguous(),
        core.k_s.float().contiguous(),
        core.k_y.float().contiguous(),
        core.k_b.float().contiguous(),
        core.k_n.float().contiguous(),
        core.k_bias.float().contiguous(),
        core.r_s.float().contiguous(),
        core.r_y.float().contiguous(),
        core.r_b.float().contiguous(),
        core.r_n.float().contiguous(),
        core.r_bias.float().contiguous(),
        lcd,
        kc,
        rc,
        initial_b.float().contiguous(),
        initial_e.float().contiguous(),
        initial_c.float().contiguous(),
        gate_alpha,
        lam,
        saturation_suppress,
    )
    result: tuple[Any, ...]
    if return_cocktail:  # admission read needs the energy/saturation tape
        result = U, nov, p, ay, write, energy, saturation
    elif return_writes:
        result = U, nov, p, ay, write
    else:
        result = U, nov, p
    if return_final_state:
        return result, (final_b, final_e, final_c)
    return result


class _ThreeWayReadFn(torch.autograd.Function):
    """Native CPU admitted three-way read with C++ forward and backward."""

    @staticmethod
    def forward(
        ctx,
        q,
        kb,
        wb,
        wb_next,
        cocktail,
        cb,
        key_bias_g,
        adm_g,
        scale,
        cocktail_gain,
        short_gain,
        pad_gain,
        induct_gain,
        allow,
        induct_allow,
        has_elig,
        induct_elig,
    ):
        ext = _load_ext()
        scale_f = float(scale.detach().cpu())
        cocktail_gain_f = float(cocktail_gain.detach().cpu())
        short_gain_f = float(short_gain.detach().cpu())
        pad_gain_f = float(pad_gain.detach().cpu())
        induct_gain_f = float(induct_gain.detach().cpu())
        qf = q.float().contiguous()
        kbf = kb.float().contiguous()
        wbf = wb.float().contiguous()
        wnf = wb_next.float().contiguous()
        cf = cocktail.float().contiguous()
        cbf = cb.float().contiguous()
        kbgf = key_bias_g.float().contiguous()
        admf = adm_g.float().contiguous()
        allowc = allow.contiguous()
        iallowc = induct_allow.contiguous()
        eligc = has_elig.contiguous()
        ieligc = induct_elig.contiguous()
        out = ext.three_way_read_cpu(
            qf,
            kbf,
            wbf,
            wnf,
            cf,
            cbf,
            kbgf,
            admf,
            allowc,
            iallowc,
            eligc,
            ieligc,
            scale_f,
            cocktail_gain_f,
            short_gain_f,
            pad_gain_f,
            induct_gain_f,
        )
        ctx.save_for_backward(
            qf,
            kbf,
            wbf,
            wnf,
            cf,
            cbf,
            kbgf,
            admf,
            allowc,
            iallowc,
            eligc,
            ieligc,
            scale,
            cocktail_gain,
            short_gain,
            pad_gain,
            induct_gain,
        )
        ctx.scale = scale_f
        ctx.cocktail_gain = cocktail_gain_f
        ctx.short_gain = short_gain_f
        ctx.pad_gain = pad_gain_f
        ctx.induct_gain = induct_gain_f
        return out.to(dtype=q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (
            q,
            kb,
            wb,
            wb_next,
            cocktail,
            cb,
            key_bias_g,
            adm_g,
            allow,
            induct_allow,
            has_elig,
            induct_elig,
            scale,
            cocktail_gain,
            short_gain,
            pad_gain,
            induct_gain,
        ) = ctx.saved_tensors
        ext = _load_ext()
        (
            gq,
            gkb,
            gwb,
            gwb_next,
            gcocktail,
            gcb,
            gkey,
            gadm,
            gscale,
            gcocktail_gain,
            gshort,
            gpad,
            ginduct,
        ) = ext.three_way_read_bwd(
            grad_out.float().contiguous(),
            q,
            kb,
            wb,
            wb_next,
            cocktail,
            cb,
            key_bias_g,
            adm_g,
            allow,
            induct_allow,
            has_elig,
            induct_elig,
            ctx.scale,
            ctx.cocktail_gain,
            ctx.short_gain,
            ctx.pad_gain,
            ctx.induct_gain,
        )
        return (
            gq.to(dtype=grad_out.dtype),
            gkb.to(dtype=grad_out.dtype),
            gwb.to(dtype=grad_out.dtype),
            gwb_next.to(dtype=grad_out.dtype),
            gcocktail.to(dtype=grad_out.dtype),
            gcb.to(dtype=grad_out.dtype),
            gkey.to(dtype=grad_out.dtype),
            gadm.to(dtype=grad_out.dtype),
            torch.as_tensor(gscale, device=scale.device, dtype=scale.dtype).reshape_as(scale),
            torch.as_tensor(
                gcocktail_gain, device=cocktail_gain.device, dtype=cocktail_gain.dtype
            ).reshape_as(cocktail_gain),
            torch.as_tensor(gshort, device=short_gain.device, dtype=short_gain.dtype).reshape_as(
                short_gain
            ),
            torch.as_tensor(gpad, device=pad_gain.device, dtype=pad_gain.dtype).reshape_as(
                pad_gain
            ),
            torch.as_tensor(ginduct, device=induct_gain.device, dtype=induct_gain.dtype).reshape_as(
                induct_gain
            ),
            None,
            None,
            None,
            None,
        )


def native_three_way_read(
    self,
    q,
    kb,
    wb,
    wb_next,
    cocktail,
    cb,
    key_bias_g,
    adm_g,
    scale,
    allow,
    induct_allow,
    has_elig,
    induct_elig,
) -> Tensor:
    """Native CPU path for ``DABSNRead._three_way_read``.

    This is geometry-agnostic: seq, field, and hybrid pass different masks into
    the same admitted-memory read. Uses a C++ autograd Function on CPU when
    gradients are required.
    """
    ext = _load_ext()
    if ext is None or q.is_cuda:
        assert _EAGER_THREE_WAY_READ is not None
        return _EAGER_THREE_WAY_READ(
            self,
            q,
            kb,
            wb,
            wb_next,
            cocktail,
            cb,
            key_bias_g,
            adm_g,
            scale,
            allow,
            induct_allow,
            has_elig,
            induct_elig,
        )

    needs_grad = (
        q.requires_grad
        or kb.requires_grad
        or wb.requires_grad
        or wb_next.requires_grad
        or cocktail.requires_grad
        or cb.requires_grad
        or key_bias_g.requires_grad
        or adm_g.requires_grad
        or scale.requires_grad
        or self.cocktail_gain.requires_grad
        or self.short_gain.requires_grad
        or self.pad_gain.requires_grad
        or self.induct_gain.requires_grad
    )
    # Small fields of up to 64 cells are faster through PyTorch's batched
    # matmul/softmax autograd. Larger admitted banks use the C++ path to avoid
    # Python-side score materialization.
    required = bool(getattr(type(self), "_cpu_native_required", False))
    if not required and needs_grad and q.shape[1] <= 64 and kb.shape[1] <= 64:
        assert _EAGER_THREE_WAY_READ is not None
        return _EAGER_THREE_WAY_READ(
            self,
            q,
            kb,
            wb,
            wb_next,
            cocktail,
            cb,
            key_bias_g,
            adm_g,
            scale,
            allow,
            induct_allow,
            has_elig,
            induct_elig,
        )
    self._last_three_way_backend = "cpu_native_cpp"
    if needs_grad:
        return _ThreeWayReadFn.apply(
            q,
            kb,
            wb,
            wb_next,
            cocktail,
            cb,
            key_bias_g,
            adm_g,
            scale,
            self.cocktail_gain,
            self.short_gain,
            self.pad_gain,
            self.induct_gain,
            allow,
            induct_allow,
            has_elig,
            induct_elig,
        )

    return ext.three_way_read_cpu(
        q.float().contiguous(),
        kb.float().contiguous(),
        wb.float().contiguous(),
        wb_next.float().contiguous(),
        cocktail.float().contiguous(),
        cb.float().contiguous(),
        key_bias_g.float().contiguous(),
        adm_g.float().contiguous(),
        allow.contiguous(),
        induct_allow.contiguous(),
        has_elig.contiguous(),
        induct_elig.contiguous(),
        float(scale),
        float(self.cocktail_gain.detach().cpu()),
        float(self.short_gain.detach().cpu()),
        float(self.pad_gain.detach().cpu()),
        float(self.induct_gain.detach().cpu()),
    ).to(dtype=q.dtype)


class _DABSNCoreScanFn(torch.autograd.Function):
    """Native CPU core scan with a reverse-time analytic backward.

    Projected inputs, recurrent weights, vector parameters, and scalar controls
    enter as graph tensors so gradients propagate through every core parameter.
    """

    @staticmethod
    def forward(
        ctx,
        Wx,
        Wgx,
        GA_w,
        beta,
        log_kappa,
        logit_recover,
        k_s,
        k_y,
        k_b,
        k_n,
        k_bias,
        r_s,
        r_y,
        r_b,
        r_n,
        r_bias,
        logit_c_decay,
        k_c,
        r_c,
        initial_b,
        initial_e,
        initial_c,
        gate_alpha,
        lam,
        saturation_suppress,
    ):
        ext = _load_ext()
        out = ext.dabsn_core_scan_fwd_train(
            Wx.contiguous(),
            Wgx.contiguous(),
            GA_w.contiguous(),
            beta,
            log_kappa,
            logit_recover,
            k_s,
            k_y,
            k_b,
            k_n,
            k_bias,
            r_s,
            r_y,
            r_b,
            r_n,
            r_bias,
            logit_c_decay,
            k_c,
            r_c,
            initial_b,
            initial_e,
            initial_c,
            float(gate_alpha),
            float(lam),
            float(saturation_suppress),
        )
        U, nov, p, ay, write, energy, saturation, bpre, cpre, final_b, final_e, final_c = out
        ctx.save_for_backward(
            bpre,
            energy,
            cpre,
            Wx,
            Wgx,
            GA_w,
            beta,
            log_kappa,
            logit_recover,
            k_s,
            k_y,
            k_b,
            k_n,
            k_bias,
            r_s,
            r_y,
            r_b,
            r_n,
            r_bias,
            logit_c_decay,
            k_c,
            r_c,
        )
        ctx.gate_alpha = float(gate_alpha)
        ctx.lam = float(lam)
        ctx.saturation_suppress = float(saturation_suppress)
        return U, nov, p, ay, write, energy, saturation, final_b, final_e, final_c

    @staticmethod
    def backward(ctx, gU, gNov, gP, gAy, gWrite, gEnergy, gCort, gFinalB, gFinalE, gFinalC):
        (
            bpre,
            epre,
            cpre,
            Wx,
            Wgx,
            GA_w,
            beta,
            log_kappa,
            logit_recover,
            k_s,
            k_y,
            k_b,
            k_n,
            k_bias,
            r_s,
            r_y,
            r_b,
            r_n,
            r_bias,
            logit_c_decay,
            k_c,
            r_c,
        ) = ctx.saved_tensors
        ext = _load_ext()
        gFinalB = torch.zeros_like(bpre[:, 0]) if gFinalB is None else gFinalB
        gFinalE = torch.zeros_like(epre[:, 0]) if gFinalE is None else gFinalE
        gFinalC = torch.zeros_like(cpre[:, 0]) if gFinalC is None else gFinalC
        g = ext.dabsn_core_scan_bwd(
            gU.contiguous(),
            gNov.contiguous(),
            gP.contiguous(),
            gAy.contiguous(),
            gWrite.contiguous(),
            gEnergy.contiguous(),
            gCort.contiguous(),
            gFinalB.contiguous(),
            gFinalE.contiguous(),
            gFinalC.contiguous(),
            bpre,
            epre,
            cpre,
            Wx,
            Wgx,
            GA_w,
            beta,
            log_kappa,
            logit_recover,
            k_s,
            k_y,
            k_b,
            k_n,
            k_bias,
            r_s,
            r_y,
            r_b,
            r_n,
            r_bias,
            logit_c_decay,
            k_c,
            r_c,
            ctx.gate_alpha,
            ctx.lam,
            ctx.saturation_suppress,
        )
        return tuple(g)


def native_core_forward_train(core, e_in, *, initial_state=None):
    """Run the native training scan and return its full state tuple.

    Gradients flow to the input and every core parameter. Optional native
    execution falls back to the reference path when the extension is unavailable
    or the input is not a CPU tensor.
    """
    import torch.nn.functional as F

    ext = _load_ext()
    H = int(core.hidden_dim)
    if ext is None or e_in.is_cuda:
        result, final_state = core._reference_forward_from_state(
            e_in,
            initial_state=initial_state,
            return_writes=True,
            return_cocktail=True,
            return_final_state=True,
        )
        return (*result, *final_state)
    initial_b, initial_e, initial_c = _initial_core_state(core, e_in, initial_state)
    Wx = core.W(e_in).float()
    Wgx = core.Wg(e_in).float()
    GA_w = torch.cat([core.Ug.weight, core.A.weight], dim=0).float()
    gate_alpha = torch.sigmoid(core.logit_alpha.reshape(()))
    lam = F.softplus(core.log_lambda.reshape(()))
    saturation_suppress = torch.sigmoid(core.logit_saturation_suppress.reshape(()))
    # .contiguous() is REQUIRED: logit_saturation_decay is a 0-dim scalar, so .expand(H)
    # is a stride-0 view -- the C++ reads lcd[i] as raw floats and would get garbage
    # for i>0 without materializing it.
    lcd = core.logit_saturation_decay.expand(H).float().contiguous()
    kc = core.k_saturation.float().contiguous()
    rc = core.r_saturation.float().contiguous()
    return _DABSNCoreScanFn.apply(
        Wx,
        Wgx,
        GA_w,
        core.beta.float(),
        core.log_kappa.float(),
        core.logit_recover.float(),
        core.k_s.float(),
        core.k_y.float(),
        core.k_b.float(),
        core.k_n.float(),
        core.k_bias.float(),
        core.r_s.float(),
        core.r_y.float(),
        core.r_b.float(),
        core.r_n.float(),
        core.r_bias.float(),
        lcd,
        kc,
        rc,
        initial_b.float().contiguous(),
        initial_e.float().contiguous(),
        initial_c.float().contiguous(),
        gate_alpha,
        lam,
        saturation_suppress,
    )


def _registered_cpu_core_scan(*inputs: Tensor) -> tuple[Tensor, ...]:
    """C++ CPU implementation for the fixed registered core-scan schema."""

    from .batched_runtime import _batched_forward_tapes

    wx, wgx, ug, a = inputs[:4]
    if wx.dtype != torch.float32:
        from dabsn.runtime.dispatch import warn_routing_once

        warn_routing_once(
            "core_scan_batched",
            f"native CPU core scan supports float32, not {wx.dtype}",
            requested_path="cpu_native_cpp",
            selected_path="registered_reference",
            corrective_action=(
                "use float32 CPU activations or run a supported low-precision CUDA backend"
            ),
            device="cpu",
            dtype=str(wx.dtype),
        )
        return _batched_forward_tapes(*inputs)
    (
        beta,
        log_kappa,
        logit_recover,
        k_s,
        k_y,
        k_b,
        k_n,
        k_bias,
        r_s,
        r_y,
        r_b,
        r_n,
        r_bias,
        logit_c_decay,
        k_c,
        r_c,
        logit_alpha,
        log_lambda,
        logit_c_suppress,
        initial_b,
        initial_e,
        initial_c,
    ) = inputs[4:]
    extension = _load_ext(required=True)
    assert extension is not None
    outputs = extension.dabsn_core_scan_fwd_train(
        wx.float().contiguous(),
        wgx.float().contiguous(),
        torch.cat((ug, a), dim=0).float().contiguous(),
        beta.float().contiguous(),
        log_kappa.float().contiguous(),
        logit_recover.float().contiguous(),
        k_s.float().contiguous(),
        k_y.float().contiguous(),
        k_b.float().contiguous(),
        k_n.float().contiguous(),
        k_bias.float().contiguous(),
        r_s.float().contiguous(),
        r_y.float().contiguous(),
        r_b.float().contiguous(),
        r_n.float().contiguous(),
        r_bias.float().contiguous(),
        logit_c_decay.float().contiguous(),
        k_c.float().contiguous(),
        r_c.float().contiguous(),
        initial_b.float().contiguous(),
        initial_e.float().contiguous(),
        initial_c.float().contiguous(),
        float(torch.sigmoid(logit_alpha)),
        float(torch.nn.functional.softplus(log_lambda)),
        float(torch.sigmoid(logit_c_suppress)),
    )
    (
        trajectory,
        novelty,
        plasticity,
        expression,
        write,
        energy_tape,
        saturation_tape,
        _budget_tape,
        _saturation_pre_tape,
        final_b,
        final_e,
        final_c,
    ) = outputs
    hidden = wx.shape[-1]
    y = trajectory[..., :hidden]
    gate_tape = torch.nn.functional.hardsigmoid(
        wgx.float() + torch.nn.functional.linear(y, ug.float())
    )
    return (
        trajectory,
        novelty,
        plasticity,
        expression,
        write,
        energy_tape,
        saturation_tape,
        gate_tape,
        final_b,
        final_e,
        final_c,
    )


def _registered_cpu_admitted_read(*inputs: Tensor) -> Tensor:
    from .admitted import admitted_three_way_read

    query = inputs[0]
    if query.dtype != torch.float32:
        from dabsn.runtime.dispatch import warn_routing_once

        warn_routing_once(
            "admitted_three_way_read",
            f"native CPU admitted read supports float32, not {query.dtype}",
            requested_path="cpu_native_cpp",
            selected_path="registered_reference",
            corrective_action=(
                "use float32 CPU activations or run a supported low-precision CUDA backend"
            ),
            device="cpu",
            dtype=str(query.dtype),
        )
        return admitted_three_way_read(*inputs)
    (
        query,
        bank_keys,
        bank_writes,
        next_writes,
        cocktail,
        bank_cocktail,
        bank_key_bias,
        bank_admission,
        scale,
        allow,
        induct_allow,
        eligible,
        induct_eligible,
        short_gain,
        pad_gain,
        induct_gain,
        cocktail_gain,
    ) = inputs
    extension = _load_ext(required=True)
    assert extension is not None
    return extension.three_way_read_cpu(
        query.contiguous(),
        bank_keys.contiguous(),
        bank_writes.contiguous(),
        next_writes.contiguous(),
        cocktail.contiguous(),
        bank_cocktail.contiguous(),
        bank_key_bias.contiguous(),
        bank_admission.contiguous(),
        allow.contiguous(),
        induct_allow.contiguous(),
        eligible.contiguous(),
        induct_eligible.contiguous(),
        float(scale),
        float(cocktail_gain),
        float(short_gain),
        float(pad_gain),
        float(induct_gain),
    )


def _registered_cpu_admitted_read_backward(*inputs: Tensor) -> tuple[Tensor, ...]:
    from .admitted import _native_backward_reference

    grad_output, query = inputs[:2]
    if query.dtype != torch.float32:
        return _native_backward_reference(*inputs)
    (
        grad_output,
        query,
        bank_keys,
        bank_writes,
        next_writes,
        cocktail,
        bank_cocktail,
        bank_key_bias,
        bank_admission,
        scale,
        allow,
        induct_allow,
        eligible,
        induct_eligible,
        short_gain,
        pad_gain,
        induct_gain,
        cocktail_gain,
    ) = inputs
    extension = _load_ext(required=True)
    assert extension is not None
    gradients = extension.three_way_read_bwd(
        grad_output.float().contiguous(),
        query.float().contiguous(),
        bank_keys.float().contiguous(),
        bank_writes.float().contiguous(),
        next_writes.float().contiguous(),
        cocktail.float().contiguous(),
        bank_cocktail.float().contiguous(),
        bank_key_bias.float().contiguous(),
        bank_admission.float().contiguous(),
        allow.contiguous(),
        induct_allow.contiguous(),
        eligible.contiguous(),
        induct_eligible.contiguous(),
        float(scale),
        float(cocktail_gain),
        float(short_gain),
        float(pad_gain),
        float(induct_gain),
    )
    return (
        gradients[0].to(query.dtype),
        gradients[1].to(bank_keys.dtype),
        gradients[2].to(bank_writes.dtype),
        gradients[3].to(next_writes.dtype),
        gradients[4].to(cocktail.dtype),
        gradients[5].to(bank_cocktail.dtype),
        gradients[6].to(bank_key_bias.dtype),
        gradients[7].to(bank_admission.dtype),
        gradients[8].to(scale.dtype).reshape_as(scale),
        gradients[10].to(short_gain.dtype).reshape_as(short_gain),
        gradients[11].to(pad_gain.dtype).reshape_as(pad_gain),
        gradients[12].to(induct_gain.dtype).reshape_as(induct_gain),
        gradients[9].to(cocktail_gain.dtype).reshape_as(cocktail_gain),
    )


def enable_native_cpu_scan(*, required: bool = False) -> bool:
    global _CORE_OP_CPU_REGISTERED
    from dabsn.core import DABSNCore

    if _load_ext(required=required) is None:
        return False
    if not _CORE_OP_CPU_REGISTERED:
        from .batched_runtime import _core_scan_batched_op

        _core_scan_batched_op.register_kernel("cpu")(_registered_cpu_core_scan)
        _CORE_OP_CPU_REGISTERED = True
    setattr(DABSNCore, "_cpu_native_enabled", True)
    setattr(DABSNCore, "_cpu_native_required", bool(required))
    return True


def enable_native_cpu_read(*, required: bool = False) -> bool:
    global _READ_OP_CPU_REGISTERED
    from dabsn.read import DABSNRead

    if _load_ext(required=required) is None:
        return False
    if not _READ_OP_CPU_REGISTERED:
        from .admitted import (
            _admitted_three_way_read_native_backward_op,
            _admitted_three_way_read_native_op,
        )

        _admitted_three_way_read_native_op.register_kernel("cpu")(_registered_cpu_admitted_read)
        _admitted_three_way_read_native_backward_op.register_kernel("cpu")(
            _registered_cpu_admitted_read_backward
        )
        _READ_OP_CPU_REGISTERED = True
    setattr(DABSNRead, "_cpu_native_read_enabled", True)
    setattr(DABSNRead, "_cpu_native_required", bool(required))
    return True


def enable_native_cpu_kernels(*, required: bool = False) -> dict[str, bool]:
    return {
        "core_scan": enable_native_cpu_scan(required=required),
        "three_way_read": enable_native_cpu_read(required=required),
    }


def native_cpu_status() -> dict[str, object]:
    extension = _load_ext(required=False)
    return {
        "extension_available": extension is not None,
        "core_scan_enabled": bool(
            getattr(
                __import__("dabsn.core", fromlist=["DABSNCore"]).DABSNCore,
                "_cpu_native_enabled",
                False,
            )
        ),
        "three_way_read_enabled": bool(
            getattr(
                __import__("dabsn.read", fromlist=["DABSNRead"]).DABSNRead,
                "_cpu_native_read_enabled",
                False,
            )
        ),
        "source": str(_SOURCE),
    }
