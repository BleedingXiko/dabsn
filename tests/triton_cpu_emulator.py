"""Execute selected Triton function bodies on CPU for equation/indexing tests.

This is a test-only subset of language operations, not a Triton interpreter or
compiler. Passing these checks says nothing about device compilation, launch
legality, register use, races, or performance.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


class Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor = tensor.reshape(-1)
        self.offset = offset

    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + offset)

    def __sub__(self, offset):
        return Pointer(self.tensor, self.offset - offset)


def _load(pointer, mask=None, other=0):
    if mask is None:
        return pointer.tensor[pointer.offset]
    offset, mask = torch.broadcast_tensors(torch.as_tensor(pointer.offset), mask)
    result = torch.full(mask.shape, other, dtype=pointer.tensor.dtype)
    result[mask] = pointer.tensor[offset[mask]]
    return result


def _store(pointer, value, mask=None):
    if mask is None:
        pointer.tensor[pointer.offset] = value
    else:
        offset, value, mask = torch.broadcast_tensors(
            torch.as_tensor(pointer.offset), torch.as_tensor(value), mask
        )
        pointer.tensor[offset[mask]] = value[mask]


def load_kernel_bodies(path, names):
    """Return (function namespace, mutable language, Pointer constructor).

    Set ``language.program`` to an integer or coordinate tuple before calling
    a function. Include helper function names in ``names``.
    """
    language = SimpleNamespace(
        program=0,
        program_id=lambda axis: (
            language.program[axis] if isinstance(language.program, tuple) else language.program
        ),
        arange=torch.arange,
        load=_load,
        store=_store,
        float32=torch.float32,
        full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
        zeros=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
        sum=lambda value, axis: value.sum(dim=axis),
        sigmoid=torch.sigmoid,
        exp=torch.exp,
        log=torch.log,
        abs=torch.abs,
        where=torch.where,
        minimum=lambda x, y: torch.minimum(torch.as_tensor(x), torch.as_tensor(y)),
        maximum=lambda x, y: torch.maximum(torch.as_tensor(x), torch.as_tensor(y)),
    )
    path = Path(path)
    tree = ast.parse(path.read_text())
    definitions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            node.returns = None
            for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                argument.annotation = None
            definitions.append(node)
    if {node.name for node in definitions} != set(names):
        raise ValueError("requested kernel functions were not all found")
    scope = {"tl": language}
    exec(  # noqa: S102 - execute only selected local kernel definitions for tests
        compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), scope
    )
    return scope, language, Pointer
