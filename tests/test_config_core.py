import torch

from dabsn import DABSNConfig, DABSNCore, DABSNLayerSpec, parse_dabsn_layer_specs


def test_compact_layer_spec_roundtrip() -> None:
    layers = parse_dabsn_layer_specs("seq:7:6,field:9,hybrid:11:8")
    assert layers == [
        DABSNLayerSpec(7, 6, "seq"),
        DABSNLayerSpec(9, None, "field"),
        DABSNLayerSpec(11, 8, "hybrid"),
    ]


def test_config_has_no_dropout_or_substrate_surface() -> None:
    fields = DABSNConfig.__dataclass_fields__
    assert "dropout" not in fields
    assert "substrate" not in fields


def test_core_forward_state_and_backward() -> None:
    torch.manual_seed(7)
    core = DABSNCore(input_dim=5, hidden_dim=6)
    inputs = torch.randn(2, 4, 5, requires_grad=True)
    result, state = core.forward_from_state(
        inputs,
        return_writes=True,
        return_cocktail=True,
        return_final_state=True,
    )
    trajectory, novelty, plasticity, expression, write, energy, saturation = result
    assert trajectory.shape == (2, 4, 12)
    for tensor in (novelty, plasticity, expression, write, energy, saturation):
        assert tensor.shape == (2, 4, 6)
    assert all(tensor.shape == (2, 6) for tensor in state)
    trajectory.sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_core_carried_state_matches_one_full_sequence() -> None:
    torch.manual_seed(11)
    core = DABSNCore(input_dim=5, hidden_dim=6)
    inputs = torch.randn(2, 7, 5)
    full, full_state = core.forward_from_state(
        inputs,
        return_writes=True,
        return_cocktail=True,
        return_final_state=True,
    )
    left, state = core.forward_from_state(
        inputs[:, :3],
        return_writes=True,
        return_cocktail=True,
        return_final_state=True,
    )
    right, final_state = core.forward_from_state(
        inputs[:, 3:],
        initial_state=state,
        return_writes=True,
        return_cocktail=True,
        return_final_state=True,
    )
    for full_tensor, left_tensor, right_tensor in zip(full, left, right):
        assert torch.equal(full_tensor, torch.cat([left_tensor, right_tensor], dim=1))
    assert all(
        torch.equal(chunked_state, one_call_state)
        for chunked_state, one_call_state in zip(final_state, full_state)
    )
