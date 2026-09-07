"""Fixed-schema inference operator for one exactly admitted memory page.

The CPU implementation is the page-equation oracle. CUDA lowers the same
score-once equations with Inductor. No training backward is registered: this
operator consumes detached prepared inference memory, not a training bank.
"""
from __future__ import annotations

import torch
from torch import Tensor


@torch.library.custom_op("dabsn::prepared_admitted_page", mutates_args=())
def prepared_admitted_page(
    query: Tensor, cocktail: Tensor, query_positions: Tensor,
    write: Tensor, successor_write: Tensor, position: Tensor,
    scalar_features: Tensor, inverse_write_norm: Tensor, admission: Tensor,
    key_bias: Tensor, valid: Tensor, scale: Tensor, cocktail_gain: Tensor,
    short_max: Tensor, short_denominator: Tensor, short_numerator: Tensor,
    permanent_max: Tensor, permanent_denominator: Tensor, permanent_numerator: Tensor,
    induction_max: Tensor, induction_denominator: Tensor, induction_numerator: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    from dabsn.prepared_memory import _prepared_page_merge

    return _prepared_page_merge(
        query, cocktail, query_positions, write, successor_write, position,
        scalar_features, inverse_write_norm, admission, key_bias, valid, scale, cocktail_gain,
        short_max, short_denominator, short_numerator, permanent_max, permanent_denominator,
        permanent_numerator, induction_max, induction_denominator, induction_numerator,
    )


@prepared_admitted_page.register_fake
def _fake_prepared_page(
    query, cocktail, query_positions, write, successor_write, position,
    scalar_features, inverse_write_norm, admission, key_bias, valid, scale, cocktail_gain,
    short_max, short_denominator, short_numerator, permanent_max, permanent_denominator,
    permanent_numerator, induction_max, induction_denominator, induction_numerator,
):
    return tuple(torch.empty_like(value, dtype=query.dtype) for value in (
        short_max, short_denominator, short_numerator, permanent_max, permanent_denominator,
        permanent_numerator, induction_max, induction_denominator, induction_numerator,
    ))


@prepared_admitted_page.register_kernel("cuda")
def _cuda_prepared_page(*args):
    from dabsn.prepared_memory import _prepared_page_merge
    from .online_compile import strict_inductor_backend

    compiled = getattr(_cuda_prepared_page, "_compiled", None)
    if compiled is None:
        compiled = torch.compile(_prepared_page_merge, backend=strict_inductor_backend, fullgraph=True, dynamic=True)
        _cuda_prepared_page._compiled = compiled
    return compiled(*args)


@torch.library.custom_op("dabsn::prepared_admitted_single_page", mutates_args=())
def prepared_admitted_single_page(
    query: Tensor, cocktail: Tensor, query_starts: Tensor,
    write: Tensor, successor_write: Tensor, position: Tensor,
    scalar_features: Tensor, inverse_write_norm: Tensor, admission: Tensor,
    key_bias: Tensor, valid: Tensor, scale: Tensor, cocktail_gain: Tensor,
    short_gain: Tensor, pad_gain: Tensor, induct_gain: Tensor,
) -> Tensor:
    from dabsn.prepared_memory import _prepared_single_page_read

    return _prepared_single_page_read(
        query, cocktail, query_starts, write, successor_write, position,
        scalar_features, inverse_write_norm, admission, key_bias, valid, scale,
        cocktail_gain, short_gain, pad_gain, induct_gain,
    )


@prepared_admitted_single_page.register_fake
def _fake_prepared_single_page(
    query, cocktail, query_starts, write, successor_write, position,
    scalar_features, inverse_write_norm, admission, key_bias, valid, scale,
    cocktail_gain, short_gain, pad_gain, induct_gain,
):
    return torch.empty_like(query)


@prepared_admitted_single_page.register_kernel("cuda")
def _cuda_prepared_single_page(*args):
    from dabsn.prepared_memory import _prepared_single_page_read
    from .online_compile import strict_inductor_backend

    compiled = getattr(_cuda_prepared_single_page, "_compiled", None)
    if compiled is None:
        compiled = torch.compile(_prepared_single_page_read, backend=strict_inductor_backend, fullgraph=True, dynamic=True)
        _cuda_prepared_single_page._compiled = compiled
    return compiled(*args)
