import pytest
import torch

from dabsn import DABSNSequenceLM
from dabsn.memory import (forward_with_memory, ingest, load, load_prepared,
                          prepare_for_inference, save, save_prepared)


@pytest.mark.parametrize("mode", ["dense", "sparse"])
@pytest.mark.parametrize("prepared", [False, True])
def test_learned_capacity_memory_continuation(mode, prepared, tmp_path):
    torch.manual_seed(404)
    model = DABSNSequenceLM(vocab=19, hidden_dim=5, depth=2,
        capacity={"mode": mode, "size": 16, "active_units": 4}).eval()
    ids = torch.randint(19, (2, 9))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    for _ in range(2):
        optimizer.zero_grad()
        model.forward_sequence(ids).square().mean().backward()
        optimizer.step()
    assert all(torch.count_nonzero(b.capacity_module.value) for b in model.backbone.blocks)
    full = model.forward_sequence(ids).detach()
    memory = ingest(model, ids[:, :5], chunk_size=2)
    assert all(layer.capacity is not None for layer in memory.layers)
    if prepared:
        memory = prepare_for_inference(model, memory, page_size=3)
        assert memory.format == "dabsn_prepared_memory_v3"
        assert memory.source_format == "dabsn_memory_v2"
        save_prepared(memory, tmp_path / "memory.dmem")
        restored = load_prepared(tmp_path / "memory.dmem", model=model)
    else:
        save(memory, tmp_path / "memory.dmem")
        restored = load(tmp_path / "memory.dmem", model=model)
    before = [layer.capacity.clone() for layer in restored.layers]
    query, same = forward_with_memory(model, ids[:, 5:], restored, chunk_size=1)
    assert same is restored
    for old, layer in zip(before, restored.layers):
        assert torch.equal(old, layer.capacity)
    torch.testing.assert_close(query, full[:, 5:], atol=2e-5, rtol=2e-4)
    continued, extended = forward_with_memory(model, ids[:, 5:], restored, extend=True)
    torch.testing.assert_close(continued, query, atol=2e-5, rtol=2e-4)
    assert extended.position == 9


def test_capacity_memory_rejects_missing_features_at_load(tmp_path):
    model = DABSNSequenceLM(vocab=19, hidden_dim=5, depth=1,
        capacity={"mode": "sparse", "size": 16, "active_units": 4})
    memory = ingest(model, torch.randint(19, (2, 3)))
    memory.layers[0].capacity = None
    save(memory, tmp_path / "invalid.dmem")
    with pytest.raises(ValueError, match="Capacity memory features"):
        load(tmp_path / "invalid.dmem", model=model)
