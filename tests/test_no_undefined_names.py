"""No undefined name may reach a shipped module.

The CPU test suite cannot execute CUDA-only branches, so a name that is only
referenced on a GPU path is invisible to every runtime test here -- it raises
NameError on the user's hardware instead. That is not a hypothetical: a capture
guard was once used in the batched runtime without being imported, and because
the surrounding `or` short-circuits on `not Wx.is_cuda`, no CPU test ever
evaluated the term. It failed on an A100 and nowhere else.

A static pass over the source has no such blind spot: it sees every branch
whether or not this machine can run it. This test is the cheap standing guard
against that entire class of bug.
"""

from __future__ import annotations

import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "dabsn"


def test_no_undefined_names_anywhere_in_the_package():
    pyflakes_api = pytest.importorskip(
        "pyflakes.api", reason="pyflakes provides the static undefined-name pass"
    )
    from pyflakes import reporter as pyflakes_reporter

    class _Collect:
        """Reporter that keeps only undefined-name diagnostics."""

        def __init__(self) -> None:
            self.problems: list[str] = []

        def unexpectedError(self, filename, msg):  # noqa: N802 - pyflakes API
            self.problems.append(f"{filename}: {msg}")

        def syntaxError(self, filename, msg, lineno, offset, text):  # noqa: N802
            self.problems.append(f"{filename}:{lineno}: syntax error: {msg}")

        def flake(self, message):
            if "undefined name" in str(message).lower():
                self.problems.append(str(message))

    assert pyflakes_reporter is not None
    collector = _Collect()
    files = sorted(str(p) for p in _SRC.rglob("*.py"))
    assert files, "no package sources found to check"
    pyflakes_api.checkRecursive(files, collector)

    assert not collector.problems, (
        "undefined names in shipped code (these raise NameError on the branch "
        "that reaches them, which may be a GPU-only branch this suite cannot "
        "execute):\n  " + "\n  ".join(collector.problems)
    )
