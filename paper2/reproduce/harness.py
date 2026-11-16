"""Shared execution services for every paper-2 study.

Importing this module verifies the installed DABSN framework against
``paper2/framework.lock.json`` before any numerical library is imported. The
reproduction runs only on the locked framework tree.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lock  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
LOCK_PATH = PAPER / "framework.lock.json"
MATRIX_PATH = HERE / "matrix_protocol.json"
MATRIX = json.loads(MATRIX_PATH.read_text())

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# ---------------------------------------------------------------- hashing

def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_value(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


# ---------------------------------------------------------- framework lock

def framework_package() -> Path:
    spec = importlib.util.find_spec("dabsn")
    if spec is None or spec.origin is None:
        raise SystemExit("DABSN is not installed. Install the locked framework (see paper2/README.md).")
    return Path(spec.origin).parent


def verify_lock() -> str:
    if not LOCK_PATH.is_file():
        raise SystemExit(f"Missing framework lock: {LOCK_PATH}")
    expected = json.loads(LOCK_PATH.read_text())["source_sha256"]
    actual = lock.files(framework_package())
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name])
    if missing or extra or changed:
        raise SystemExit("Installed DABSN differs from paper2/framework.lock.json: "
                         f"missing={missing[:5]} extra={extra[:5]} changed={changed[:5]}")
    return lock.tree_digest(actual)


FRAMEWORK_SHA256 = verify_lock()

import torch  # noqa: E402  (imported only after the lock passed)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# --------------------------------------------------------------- records

def write_record(path, record) -> None:
    """Publish valid JSON atomically."""
    path = Path(path)
    content = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name("." + path.name + ".writing")
    temporary.write_text(content)
    os.replace(temporary, path)


def finite(value, label: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(label)


def parameter_digest(model) -> str:
    checksum = hashlib.sha256()
    for name, parameter in model.named_parameters():
        tensor = parameter.detach().cpu().contiguous()
        checksum.update(canonical({"name": name, "shape": list(tensor.shape),
                                   "dtype": str(tensor.dtype)}).encode())
        checksum.update(tensor.numpy().tobytes())
    return checksum.hexdigest()


def provenance(device: str) -> dict:
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    import dabsn
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "torch": torch.__version__,
            "dabsn": dabsn.__version__, "framework_tree_sha256": FRAMEWORK_SHA256,
            "dependencies": {name: version(name) for name in ("numpy", "torch", "triton", "ninja", "safetensors")},
            "device": device}


# ------------------------------------------------------------- precision

def selected_precision(device) -> str:
    """fp32 or bf16. bf16 needs native CUDA support (compute capability >= 8.0)."""
    requested = os.environ.get("DABSN_PAPER2_PRECISION", "fp32").lower()
    device = torch.device(device)
    native_bf16 = (device.type == "cuda" and torch.cuda.is_available()
                   and torch.cuda.get_device_capability(device)[0] >= 8
                   and torch.cuda.is_bf16_supported(including_emulation=False))
    if requested == "auto":
        return "bf16" if native_bf16 else "fp32"
    if requested == "fp32":
        return "fp32"
    if requested == "bf16":
        if not native_bf16:
            raise ValueError("bf16 requires CUDA compute capability >= 8.0 with native support")
        return "bf16"
    raise ValueError("DABSN_PAPER2_PRECISION must be auto, fp32 or bf16")


def precision_record(device) -> dict:
    compute = selected_precision(device)
    return {"compute": compute, "parameters": "float32", "optimizer": "float32",
            "state": "float32", "eligibility": "float32", "readout": "float32", "loss": "float32"}


def autocast(device):
    from dabsn.runtime import autocast_context
    return autocast_context(torch.device(device), selected_precision(device))


def assert_projection_precision(model, x) -> dict:
    """Prove on the device that projections run at the declared compute dtype."""
    from dabsn.temporal_credit import initial_core_eligibility
    profile = precision_record(x.device)
    expected = torch.bfloat16 if profile["compute"] == "bf16" else torch.float32
    if x.dtype != torch.float32 or any(p.dtype != torch.float32 for p in model.parameters()):
        raise TypeError("Inputs and master parameters must have FP32 storage")
    with torch.no_grad():
        state = model.core.initial_state(x.shape[0], device=x.device, dtype=torch.float32)
        eligibility = initial_core_eligibility(model.core, x.shape[0], device=x.device, dtype=torch.float32)
        with autocast(x.device):
            projected = model.core.W(x)
            gate_projected = model.core.Wg(x)
            head = getattr(model, "head", None)
            readout = (head(torch.zeros(x.shape[0], model.core.hidden_dim, device=x.device, dtype=torch.float32))
                       if head is not None else torch.zeros(1, device=x.device))
    if any(value.dtype != expected for value in (projected, gate_projected)):
        raise AssertionError("Core projection dtype differs from the compute policy")
    if readout.dtype != torch.float32:
        raise AssertionError("The readout must compute in FP32 under every policy")
    if any(value.dtype != torch.float32 for value in (*state, *eligibility.tensors())):
        raise AssertionError("Recurrent state or eligibility storage is not FP32")
    return {"precision": profile, "core_projection_dtype": str(projected.dtype),
            "gate_projection_dtype": str(gate_projected.dtype),
            "readout_projection_dtype": str(readout.dtype)}


# --------------------------------------------------------------- devices

class Runtime:
    def __init__(self, device: str):
        if device not in ("cpu", "cuda"):
            raise ValueError("Select cpu or cuda explicitly")
        self.device = device

    def configure(self) -> dict:
        if self.device == "cpu":
            return {"device": "cpu", "machine": platform.machine(), "processor": platform.processor()}
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        info = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        return {"device": "cuda", "name": info.name, "total_memory": info.total_memory,
                "free_memory_fraction_at_start": free / total, "cuda": torch.version.cuda,
                "capability": list(torch.cuda.get_device_capability())}

    def synchronize(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()

    def reset_peak(self) -> None:
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def peaks(self):
        if self.device == "cpu":
            return None, None
        return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


def restore_optimizer_state(optimizer, state_dict) -> None:
    """Load optimizer state, keeping ordinary Adam step counters on CPU."""
    optimizer.load_state_dict(state_dict)
    for group in optimizer.param_groups:
        if group.get("capturable", False) or group.get("fused", False):
            continue
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            step = state.get("step")
            if getattr(step, "device", None) is not None and step.device.type != "cpu":
                state["step"] = step.detach().cpu().clone()


def compare_recovery(expected, actual) -> dict:
    """Largest absolute and scaled difference over a nested tensor structure."""
    errors = {"tensor_count": 0, "max_absolute_error": 0.0, "max_scaled_error": 0.0}

    def visit(left, right):
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
                raise AssertionError("Recovery tensor shape or dtype changed")
            a, b = left.detach().cpu(), right.detach().cpu()
            if not (a.is_floating_point() or a.is_complex()):
                # Masks, positions, counts and ids recover exactly or not at all.
                if not torch.equal(a, b):
                    raise AssertionError("Recovery changed discrete state")
                errors["tensor_count"] += 1
                return
            if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
                raise AssertionError("Recovery contains nonfinite state")
            difference = float((a - b).abs().max()) if a.numel() else 0.0
            scale = 1 + float(a.abs().max()) if a.numel() else 1.0
            errors["tensor_count"] += 1
            errors["max_absolute_error"] = max(errors["max_absolute_error"], difference)
            errors["max_scaled_error"] = max(errors["max_scaled_error"], difference / scale)
        elif isinstance(left, dict):
            if not isinstance(right, dict) or set(left) != set(right):
                raise AssertionError("Recovery mapping changed")
            for key in left:
                visit(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            if not isinstance(right, (tuple, list)) or len(left) != len(right):
                raise AssertionError("Recovery sequence changed")
            for a, b in zip(left, right):
                visit(a, b)
        elif left != right:
            raise AssertionError("Recovery scalar changed")

    visit(expected, actual)
    return errors


# -------------------------------------------------------------- stopping

_STOP = {"requested": False}


def _request_stop(signum, frame):
    _STOP["requested"] = True


def install_stop_handlers() -> None:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)


class Stopped(Exception):
    """A job stopped at an exact saved boundary; rerun with --resume."""


# --------------------------------------------------------------- context

class Context:
    """What a study sees: its job, protocol, device, directory and checkpoints."""

    RECOVERY_SECONDS = 600.0

    def __init__(self, study: str, job: dict, protocol: dict, device: str, directory: Path,
                 results: Path, resume: bool, max_seconds: float | None, smoke: bool,
                 source_sha256: str):
        self.study = study
        self.job = job
        self.protocol = protocol
        self.matrix = MATRIX
        self.device = device
        self.directory = Path(directory)
        self.results = Path(results)
        self.resume = resume
        self.smoke = smoke
        self.started = time.monotonic()
        self.deadline = None if max_seconds is None else self.started + max_seconds
        self.last_recovery = self.started
        self.runtime = Runtime(device)
        self.identity = {"study": study, "job": job, "device": device, "smoke": smoke,
                         "precision": precision_record(device), "framework_tree_sha256": FRAMEWORK_SHA256,
                         "reproduction_source_sha256": source_sha256}

    # -- time
    def stop_due(self) -> bool:
        return _STOP["requested"] or (self.deadline is not None and time.monotonic() >= self.deadline)

    def recovery_due(self) -> bool:
        return time.monotonic() - self.last_recovery >= self.RECOVERY_SECONDS

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    # -- checkpoints
    def checkpoint(self, name: str, payload: dict, *, replace: bool = False) -> dict:
        """Save atomically. Named scientific checkpoints are never overwritten."""
        destination = self.directory / f"{name}.pt"
        if destination.exists() and not replace:
            raise FileExistsError(destination)
        temporary = self.directory / f".{name}.writing"
        torch.save({**payload, "identity": self.identity}, temporary)
        os.replace(temporary, destination)
        if replace:
            self.last_recovery = time.monotonic()
        return {"path": destination.name, "sha256": sha256(destination)}

    def load(self, name: str, *, check_identity: bool = True):
        path = self.directory / f"{name}.pt"
        if not path.exists():
            return None
        saved = torch.load(path, map_location=self.device, weights_only=True)
        if check_identity and saved.get("identity") != self.identity:
            raise ValueError(f"{path.name}: job, device, precision or source changed since it was saved")
        return saved

    def latest(self, prefix: str = "step-"):
        """Newest of the step checkpoints and the rolling recovery checkpoint."""
        candidates = []
        for path in self.directory.glob(f"{prefix}*.pt"):
            candidates.append((int(path.stem[len(prefix):]), path.stem))
        recovery = self.load("recovery") if (self.directory / "recovery.pt").exists() else None
        best = max(candidates) if candidates else None
        if recovery is not None and (best is None or recovery["completed"] > best[0]):
            return recovery
        return None if best is None else self.load(best[1])

    def step_name(self, step: int) -> str:
        return f"step-{step:06d}"

    def checkpoint_hashes(self) -> dict:
        return {path.name: sha256(path) for path in sorted(self.directory.glob("step-*.pt"))}

    # -- dependencies
    def dependency(self, study: str, identifier: str) -> tuple[Path, dict]:
        directory = self.results / study / identifier
        attempts = sorted(directory.glob("attempt-*.json"))
        if not attempts:
            raise FileNotFoundError(f"Required {study} job {identifier} has not run")
        record = json.loads(attempts[-1].read_text())
        if record["status"] != "passed":
            raise ValueError(f"Required {study} job {identifier} did not pass")
        return directory, record


def maybe_stop(ctx: Context, save) -> None:
    """Save a recovery checkpoint when due, and stop cleanly when asked."""
    if ctx.stop_due():
        save()
        raise Stopped(f"stopped at a saved boundary after {ctx.elapsed():.0f}s")
    if ctx.recovery_due():
        save()


def nonnegative_finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
