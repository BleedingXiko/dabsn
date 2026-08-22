import json

import pytest
import torch

from dabsn import check_component
from dabsn.cli import build_parser
from dabsn.cli import main as cli_main
from dabsn.components import component_registry


def _config():
    return {"dim": 6, "ratio": 2.0}


def test_builtin_component_conformance_runs_real_local_checks():
    report = check_component("dabsn:residual_mlp", _config())
    assert report.passed, report.to_dict()
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["eager-forward-backward"] == "pass"
    assert statuses["config-schema-migration"] == "pass"
    assert statuses["non-contiguous-input"] == "pass"
    assert statuses["dynamic-axis-contract"] == "pass"
    assert statuses["determinism-declaration"] == "pass"
    assert statuses["amp-bf16"] == "pass"
    assert statuses["amp-fp16"] == "skip"
    assert statuses["compile-fullgraph"] == "pass"
    assert statuses["fake-tensor"] == "pass"
    assert statuses["torch-export"] == "pass"
    assert statuses["save-load-reconstruction"] == "pass"
    assert statuses["cuda-graph-replay"] == "skip"
    assert statuses["fsdp-wrapping"] == "skip"


@pytest.mark.parametrize("read_geometry", ["seq", "field", "hybrid"])
def test_dabsn_component_conformance_includes_real_bf16_and_fake_tensor_checks(
    read_geometry,
):
    report = check_component(
        "dabsn:block",
        {
            "input_dim": 6,
            "hidden_dim": 6,
            "state_dim": 7,
            "read_geometry": read_geometry,
            "residual": False,
        },
    )
    assert report.passed, report.to_dict()
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["eager-forward-backward"] == "pass"
    assert statuses["amp-bf16"] == "pass"
    assert statuses["compile-fullgraph"] == "pass"
    assert statuses["fake-tensor"] == "pass"


def test_component_check_cli_emits_json_and_returns_success(tmp_path, capsys):
    config = tmp_path / "component.json"
    config.write_text(json.dumps(_config()), encoding="utf-8")
    assert (
        cli_main(
            [
                "component",
                "check",
                "dabsn:residual_mlp",
                "--config",
                str(config),
                "--dtype",
                "fp32",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["provider_key"] == "dabsn:residual_mlp"


def test_component_check_cli_accepts_provider_parallel_topology_axes():
    args = build_parser().parse_args(
        [
            "component",
            "check",
            "dabsn:sparse_moe",
            "--config",
            "component.json",
            "--distributed",
            "ddp",
            "--parallel-axis",
            "data=2",
            "--parallel-axis",
            "expert=4",
        ]
    )
    assert args.parallel_axis == ["data=2", "expert=4"]


def test_conformance_executes_requested_distributed_wrapper(monkeypatch):
    class State:
        enabled = True

    calls = []

    def fake_wrap(module, state, *, precision):
        calls.append((module, state, precision))
        return module

    monkeypatch.setattr("dabsn.distributed.wrap_distributed", fake_wrap)
    provider = component_registry.resolve("dabsn:residual_mlp").provider
    original = provider.capabilities
    provider.capabilities = type(original)(
        **{
            **original.__dict__,
            "distributed": True,
        }
    )
    try:
        report = check_component(
            "dabsn:residual_mlp",
            _config(),
            dtype=torch.float32,
            distributed_state=State(),
        )
    finally:
        provider.capabilities = original
    assert report.passed, report.to_dict()
    assert calls and calls[0][2] == "fp32"
    assert calls[0][0].bindings[0].component_id == "conformance.0"
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["fsdp-wrapping"] == "pass"


def test_discovered_provider_requires_explicit_trust(monkeypatch):
    class Point:
        name = "fixture.untrusted"
        dist = None

    class Points:
        def select(self, *, group):
            assert group == "dabsn.components.v2"
            return [Point()]

    monkeypatch.setattr("importlib.metadata.entry_points", lambda: Points())
    component_registry.discover()
    report = check_component("fixture.untrusted", {"width": 4})
    assert report.passed is False
    assert report.checks[0].status == "fail"
    assert "untrusted" in report.checks[0].detail
