"""Synthetic tasks for the paper 3 matrix, built from the published generators.

* **MQAR**: the generator of Arora et al., "Zoology" (2023),
  ``zoology/data/multiquery_ar.py``: vocabulary 8192, keys from the lower half
  and values from the upper half (each unique within a sequence), the
  key-value block first, then the keys queried at power-law gaps
  (``power_a = 0.01``), every other position filled with uniform random
  tokens (``random_non_queries=True``). Supervision is the value at each query.
* **Copy** and **selective copy**: the generator of the S4 repository,
  ``src/dataloaders/datasets/copying.py`` (``torch_copying_data``), the copying
  task of Arjovsky et al. (2016) and, with ``variable=True``, the selective
  copying task of Gu & Dao (2023, Mamba). Noise token 0, data tokens
  ``1..A-2``, ``M`` marker tokens ``A-1`` at the end; the data tokens are
  predicted, in order, at the markers.
* **Passkey**: the prompt of Mohtashami & Jaggi (2023, landmark attention,
  ``llama/run_test.py``), tokenized by word: task description, repeated
  garbage ("The grass is green. ..."), the information line at a uniformly
  random depth, more garbage, the question, then the key, whose digits are
  predicted.
* **Key-value recall** (mechanism study only): single-query associative recall
  over the Zoology key and value vocabularies, query after the pairs. The
  trained models are also tested with the same query shown before the pairs
  (``KVRecall.query_first``).

A sequence is a pair (x, y) of int64 arrays of length T; ``y`` holds the
target at supervised positions and ``IGNORE`` elsewhere. Every sample is a pure
function of an explicit NumPy generator, so a training batch is a pure function
of (seed, step) and every model is evaluated on byte-identical sequences.

Two axes, beyond the training distribution:

* **distance**: the content is fixed at the largest training load and filler
  is inserted between the facts and their use. The filler is each task's own
  non-content token stream: uniform random tokens for MQAR and recall (minus
  the sequence's own keys, so every query keeps exactly one answer), the noise
  token for (selective) copying, the garbage text for passkey.
* **load**: more items (pairs, copied tokens, key digits) with the filler held
  at its training size. For MQAR this reproduces Zoology's own test
  configurations (T = 4 x pairs: 512/128, 1024/256, ...).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IGNORE = -100
SPLITS = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class Sample:
    x: np.ndarray
    y: np.ndarray
    meta: dict


def _ignore(n: int) -> np.ndarray:
    return np.full(n, IGNORE, dtype=np.int64)


def _pack(parts, targets) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([np.asarray(p, dtype=np.int64) for p in parts])
    y = np.concatenate([np.asarray(t, dtype=np.int64) for t in targets])
    assert x.shape == y.shape
    return x, y


class Task:
    name: str
    vocab: int
    loads: tuple[int, ...]

    def __init__(self, loads):
        self.loads = tuple(int(v) for v in loads)

    @property
    def max_load(self) -> int:
        return max(self.loads)

    def train_load(self, rng) -> int:
        return int(self.loads[int(rng.integers(0, len(self.loads)))])

    def content_len(self, load: int) -> int:
        """Tokens that are not filler (an upper bound where content varies)."""
        raise NotImplementedError

    def load_axis_length(self, train_len: int, load: int) -> int:
        """Length for the load axis: filler fixed at its training size."""
        return self.content_len(load) + (train_len - self.content_len(self.max_load))

    def sample(self, rng, length: int, load: int, placement: str) -> Sample:
        raise NotImplementedError


# ----------------------------------------------------------------------
# MQAR (Zoology)

class MQAR(Task):
    name = "mqar"
    vocab = 8192
    power_a = 0.01

    def content_len(self, load):
        return 4 * load

    def _kv(self, rng, n):
        half = self.vocab // 2
        keys = rng.choice(np.arange(1, half), size=n, replace=False)
        values = rng.choice(np.arange(half, self.vocab), size=n, replace=False)
        return keys.astype(np.int64), values.astype(np.int64)

    def _queries(self, rng, keys, values, region):
        """Zoology's query region: keys at power-law gaps on even offsets,
        uniform random tokens everywhere else."""
        space = region // 2
        a = self.power_a
        p = a * np.arange(1, space + 1) ** (a - 1)
        p = p / p.sum()
        gaps = rng.choice(space, size=len(keys), replace=False, p=p)
        q = rng.integers(0, self.vocab, size=region).astype(np.int64)
        q[2 * gaps] = keys
        yq = _ignore(region)
        yq[2 * gaps] = values
        return q, yq

    def sample(self, rng, length, load, placement):
        n = load
        if length < 4 * n or length % 2:
            raise ValueError("MQAR needs an even length >= 4 x pairs")
        keys, values = self._kv(rng, n)
        kv = np.stack([keys, values], 1).reshape(-1)
        if placement == "random":  # Zoology's sequence at (length, n)
            q, yq = self._queries(rng, keys, values, length - 2 * n)
            x, y = _pack([kv, q], [_ignore(2 * n), yq])
            return Sample(x, y, {"load": n, "gap": 0})
        gap = length - 4 * n  # distance: filler between the pairs and the queries
        pool = np.setdiff1d(np.arange(self.vocab), keys, assume_unique=True)
        filler = rng.choice(pool, size=gap).astype(np.int64)
        q, yq = self._queries(rng, keys, values, 2 * n)
        x, y = _pack([kv, filler, q], [_ignore(2 * n), _ignore(gap), yq])
        return Sample(x, y, {"load": n, "gap": gap})


# ----------------------------------------------------------------------
# Single-query key-value recall (mechanism study)

class KVRecall(Task):
    """Pairs, filler, ``QUERY key`` -> value.

    The vocabulary keeps one reserved id (8193) that never occurs in any
    sequence."""

    name = "kv_recall"
    vocab = 8192 + 2
    QUERY = 8192

    def content_len(self, load):
        return 2 * load + 2

    def sample(self, rng, length, load, placement):
        n = load
        free = length - self.content_len(n)
        if free < 0:
            raise ValueError("content does not fit in the sequence length")
        half = 8192 // 2
        keys = rng.choice(np.arange(1, half), size=n, replace=False).astype(np.int64)
        values = rng.choice(np.arange(half, 8192), size=n, replace=False).astype(np.int64)
        which = int(rng.integers(0, n))
        gap = int(rng.integers(0, free + 1)) if placement == "random" else free
        pool = np.setdiff1d(np.arange(8192), keys, assume_unique=True)
        filler = rng.choice(pool, size=free).astype(np.int64)
        pre, mid = filler[:free - gap], filler[free - gap:]
        kv = np.stack([keys, values], 1).reshape(-1)
        x, y = _pack([pre, kv, mid, [self.QUERY, keys[which]]],
                     [_ignore(free - gap), _ignore(2 * n), _ignore(gap),
                      [IGNORE, values[which]]])
        return Sample(x, y, {"load": n, "gap": gap})

    def query_first(self, s: Sample) -> Sample:
        """The same sequence with the query also shown before the pairs.

        From a ``max_gap`` sample (pairs, filler, ``QUERY key``): ``QUERY key``
        is placed at the start and the last two filler tokens are dropped, so the
        length, the pairs, the question at the end and its target are unchanged.
        Used to test a model trained with the query last."""
        x = s.x
        if x[-2] != self.QUERY or s.meta["gap"] < 2 or s.x[0] == self.QUERY:
            raise ValueError("query_first needs a max_gap sample with at least 2 filler tokens")
        n = s.meta["load"]
        if len(x) - s.meta["gap"] - 2 != 2 * n:
            raise ValueError("query_first needs the pairs at the start (max_gap placement)")
        early = np.concatenate([x[-2:], x[:-4], x[-2:]])
        assert early.shape == x.shape
        return Sample(early, s.y.copy(), {**s.meta, "query_first": True})


# ----------------------------------------------------------------------
# Copying and selective copying (S4 ``torch_copying_data``)

class Copy(Task):
    """Arjovsky copying: ``M`` tokens, ``L`` noise tokens, ``M`` markers."""

    name = "copy"
    vocab = 10  # 8 data tokens, noise 0, marker 9 (Arjovsky et al.)
    variable = False

    def content_len(self, load):
        return 2 * load

    def sample(self, rng, length, load, placement):
        del placement  # the layout is fixed by the generator
        M, A = load, self.vocab
        L = length - 2 * M
        if L < 0:
            raise ValueError("content does not fit in the sequence length")
        tokens = rng.integers(1, A - 1, size=M).astype(np.int64)
        region = np.zeros(L + M, dtype=np.int64)
        if self.variable:
            inds = np.sort(rng.permutation(L + M)[:M])
        else:
            inds = np.arange(M)
        region[inds] = tokens
        markers = np.full(M, A - 1, dtype=np.int64)
        x, y = _pack([region, markers], [_ignore(L + M), tokens])
        return Sample(x, y, {"load": M, "gap": L, "first_pos": int(inds[0])})


class SelectiveCopy(Copy):
    """Selective copying (Mamba): the data tokens sit at random positions."""

    name = "selective_copy"
    vocab = 16  # 14 data tokens, noise 0, marker 15 (Gu & Dao)
    variable = True


# ----------------------------------------------------------------------
# Passkey (Mohtashami & Jaggi), word-tokenized

_TASK = ("There is an important info hidden inside a lot of irrelevant text . "
         "Find it and memorize them . I will quiz you about the important information there .")
_GARBAGE = "The grass is green . The sky is blue . The sun is yellow . Here we go . There and back again ."
_INFO = "The pass key is {K} . Remember it . {K} is the pass key ."
_QUESTION = "What is the pass key ? The pass key is"
_NL = "\n"
_WORDS = list(dict.fromkeys((_TASK + " " + _GARBAGE + " " + _INFO + " " + _QUESTION)
                            .replace("{K}", "").split()))
PASSKEY_VOCAB = [_NL] + [w for w in _WORDS] + [str(d) for d in range(10)]
_ID = {w: i for i, w in enumerate(PASSKEY_VOCAB)}


def _ids(text: str) -> np.ndarray:
    return np.array([_ID[w] for w in text.split()], dtype=np.int64)


def passkey_key(rng, digits: int) -> str:
    """Uniform on [1, 5 * 10^(digits-1)]; for 5 digits exactly M&J's
    ``random.randint(1, 50000)``."""
    n = int("".join(str(d) for d in rng.integers(0, 10, size=digits)))
    return str(1 + n % (5 * 10 ** (digits - 1)))


class Passkey(Task):
    name = "passkey"
    vocab = len(PASSKEY_VOCAB)
    _task = _ids(_TASK)
    _garbage = _ids(_GARBAGE)
    _question = _ids(_QUESTION)
    _info_fixed = len(_INFO.replace("{K}", "").split())

    def content_len(self, load):
        # task, info line (key twice), question, answer, four newlines
        return len(self._task) + self._info_fixed + 2 * load + len(self._question) + load + 4

    def _garbage_run(self, n: int) -> np.ndarray:
        return np.resize(self._garbage, n) if n > 0 else np.zeros(0, dtype=np.int64)

    def sample(self, rng, length, load, placement):
        del placement  # depth is always uniform, as in M&J; recorded in meta
        key = passkey_key(rng, load)
        kd = np.array([_ID[c] for c in key], dtype=np.int64)
        info = np.concatenate([_ids("The pass key is"), kd, _ids(". Remember it ."), kd,
                               _ids("is the pass key .")])
        nl = np.array([_ID[_NL]], dtype=np.int64)
        fixed = len(self._task) + len(info) + len(self._question) + len(kd) + 4
        n_garbage = length - fixed
        if n_garbage < 0:
            raise ValueError("content does not fit in the sequence length")
        pre = int(rng.integers(0, n_garbage + 1))
        out_t = np.concatenate([kd, [IGNORE]])  # "is"->d1, d1->d2, ..., last->none
        head = [self._task, nl, self._garbage_run(pre), nl, info, nl,
                self._garbage_run(n_garbage - pre), nl, self._question[:-1]]
        n_head = sum(len(h) for h in head)
        x, y = _pack(head + [self._question[-1:], kd],
                     [_ignore(n_head), out_t[:1], out_t[1:]])
        return Sample(x, y, {"load": load, "digits": len(kd),
                             "depth": pre / max(n_garbage, 1)})


# ----------------------------------------------------------------------

def make_task(name: str, loads=None) -> Task:
    default = {"mqar": (16, 32, 64), "kv_recall": (16, 32, 64), "copy": (10,),
               "selective_copy": (16,), "passkey": (5,)}
    loads = tuple(loads) if loads else default[name]
    table = {"mqar": MQAR, "kv_recall": KVRecall, "copy": Copy,
             "selective_copy": SelectiveCopy, "passkey": Passkey}
    return table[name](loads)


# ----------------------------------------------------------------------
# Batching. Training batches derive from (seed, step); evaluation samples
# derive from (eval_seed, axis, value, index) and are shared by all models.

def train_batch(task: Task, seed: int, step: int, batch: int, length: int):
    rng = np.random.default_rng([seed, SPLITS["train"], step])
    xs, ys = [], []
    for _ in range(batch):
        s = task.sample(rng, length, task.train_load(rng), "random")
        xs.append(s.x)
        ys.append(s.y)
    return np.stack(xs), np.stack(ys)


AXIS_CODES = {"val": 0, "distance": 1, "load": 2, "mech": 3, "extreme": 4}


def eval_sample(task: Task, eval_seed: int, split: str, axis: str, value: int,
                index: int, length: int, load: int, placement: str) -> Sample:
    rng = np.random.default_rng(
        [eval_seed, SPLITS[split], AXIS_CODES[axis], int(value), int(index)]
    )
    return task.sample(rng, length, load, placement)
