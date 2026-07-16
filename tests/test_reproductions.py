import pytest

from dabsn.reproductions.common import parser, run


@pytest.mark.parametrize("task", ["copy", "mqar", "keyvalue", "a5"])
def test_reproduction_reduced(task):
    command = ["--steps", "1", "--hidden", "4", "--train-length", "4", "--eval-lengths", "4", "--batch-size", "1", "--eval-batch-size", "1", "--val-batches", "1", "--seeds", "0"]
    if task in {"mqar", "keyvalue"}:
        command += ["--n-keys", "2", "--n-values", "2", "--n-pairs", "2"]
    args = parser(task).parse_args(command)
    rows = run(args)
    assert len(rows) == 1 and rows[0]["task"] == task
    assert 0.0 <= rows[0]["eval_acc"] <= 1.0
