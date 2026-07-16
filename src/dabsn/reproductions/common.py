"""Shared, source-faithful task generation and training for paper 1."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from dabsn import DABSNLayerSpec, DABSNModel


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model(input_dim: int, out_dim: int, hidden: int, geometry: str, device: torch.device):
    return DABSNModel(
        input_dim,
        out_dim,
        [DABSNLayerSpec(hidden_dim=hidden, state_dim=hidden, read_geometry=geometry)],
        output_adapter="token",
    ).to(device)


def copy_batch(batch: int, length: int, vocab: int, device: torch.device):
    content = torch.randint(0, vocab, (batch, length), device=device)
    symbols = torch.empty(batch, 2 * length, dtype=torch.long, device=device)
    symbols[:, :length] = content
    symbols[:, length] = vocab
    symbols[:, length + 1 :] = content[:, :-1]
    targets = torch.full_like(symbols, -100)
    targets[:, length:] = content
    return F.one_hot(symbols, vocab + 1).float(), targets


@torch.no_grad()
def mqar_batch(batch: int, length: int, keys_n: int, values_n: int, pairs: int, device):
    if length < 2 * pairs:
        raise ValueError("MQAR length must be at least twice n_pairs")
    x = torch.zeros(batch, length, 3 + keys_n + values_n, device=device)
    targets = torch.zeros(batch, length, dtype=torch.long, device=device)
    mask = torch.zeros(batch, length, dtype=torch.bool, device=device)
    for row in range(batch):
        keys = torch.randperm(keys_n, device=device)[:pairs]
        values = torch.randint(0, values_n, (pairs,), device=device)
        binding_slots = sorted(random.sample(range(length // 2), pairs))
        later = [slot for slot in range(length) if slot > binding_slots[-1]]
        query_slots = sorted(random.sample(later, pairs))
        order = torch.randperm(pairs, device=device)
        for index, slot in enumerate(binding_slots):
            key, value = int(keys[index]), int(values[index])
            x[row, slot, 0] = 1
            x[row, slot, 3 + key] = 1
            x[row, slot, 3 + keys_n + value] = 1
        for query_index, slot in enumerate(query_slots):
            index = int(order[query_index])
            key, value = int(keys[index]), int(values[index])
            x[row, slot, 1] = 1
            x[row, slot, 3 + key] = 1
            targets[row, slot] = value
            mask[row, slot] = True
        unused = set(range(length)) - set(binding_slots) - set(query_slots)
        for slot in unused:
            x[row, slot, 2] = 1
    return x, targets, mask


@torch.no_grad()
def keyvalue_batch(batch: int, length: int, keys_n: int, values_n: int, pairs: int,
                   filler_prob: float, ordered: bool, device):
    if length < pairs + 1:
        raise ValueError("key-value length must be at least n_pairs + 1")
    x = torch.zeros(batch, length, 3 + keys_n + values_n, device=device)
    targets = torch.zeros(batch, dtype=torch.long, device=device)
    for row in range(batch):
        keys = torch.arange(pairs, device=device) if ordered else torch.randperm(keys_n, device=device)[:pairs]
        values = torch.randint(0, values_n, (pairs,), device=device)
        query_index = int(torch.randint(0, pairs, (1,), device=device))
        targets[row] = values[query_index]
        slots = list(range(pairs)) if ordered else sorted(random.sample(range(length - 1), pairs))
        by_slot = dict(zip(slots, range(pairs)))
        for slot in range(length - 1):
            if slot in by_slot:
                index = by_slot[slot]
                key, value = int(keys[index]), int(values[index])
                x[row, slot, 0] = 1
                x[row, slot, 3 + key] = 1
                x[row, slot, 3 + keys_n + value] = 1
            elif random.random() < filler_prob:
                x[row, slot, 2] = 1
        x[row, -1, 1] = 1
        x[row, -1, 3 + int(keys[query_index])] = 1
    return x, targets


def _parity(permutation) -> int:
    return sum(permutation[i] > permutation[j] for i in range(5) for j in range(i + 1, 5)) % 2


def a5_table():
    elements = [p for p in itertools.permutations(range(5)) if _parity(p) == 0]
    index = {p: i for i, p in enumerate(elements)}
    table = torch.empty(60, 60, dtype=torch.long)
    for i, left in enumerate(elements):
        for j, right in enumerate(elements):
            table[i, j] = index[tuple(right[left[k]] for k in range(5))]
    return table, index[tuple(range(5))]


@torch.no_grad()
def a5_batch(batch: int, length: int, table: torch.Tensor, identity: int, device):
    symbols = torch.randint(0, 60, (batch, length))
    targets = torch.empty(batch, length, dtype=torch.long)
    current = torch.full((batch,), identity, dtype=torch.long)
    for step in range(length):
        current = table[current, symbols[:, step]]
        targets[:, step] = current
    return F.one_hot(symbols, 60).float().to(device), targets.to(device)


def _loss(logits, targets, mask=None):
    if mask is None:
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
    return F.cross_entropy(logits[mask], targets[mask])


def _accuracy(logits, targets, mask=None):
    valid = targets != -100 if mask is None else mask
    return float(((logits.argmax(-1) == targets) & valid).sum() / valid.sum().clamp_min(1))


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    device = torch.device(args.device)
    results = []
    for seed in args.seeds:
        seed_all(seed)
        if args.task == "copy":
            net = model(args.vocab + 1, args.vocab, args.hidden, args.geometry, device)
        elif args.task in {"mqar", "keyvalue"}:
            net = model(3 + args.n_keys + args.n_values, args.n_values, args.hidden, args.geometry, device)
        else:
            net = model(60, 60, args.hidden, args.geometry, device)
            table, identity = a5_table()
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
        started = time.time()
        train_loss = float("nan")
        net.train()
        lengths = [2, 4, 8, 16, 32, 64, 128, args.train_length]
        lengths = sorted(set(length for length in lengths if length <= args.train_length))
        for step in range(1, args.steps + 1):
            if args.task == "copy":
                x, target = copy_batch(args.batch_size, args.train_length, args.vocab, device); mask = None
            elif args.task == "mqar":
                x, target, mask = mqar_batch(args.batch_size, args.train_length, args.n_keys, args.n_values, args.n_pairs, device)
            elif args.task == "keyvalue":
                x, target = keyvalue_batch(args.batch_size, args.train_length, args.n_keys, args.n_values, args.n_pairs, args.filler_prob, args.ordered_pairs, device); mask = None
            else:
                unlocked = max(1, min(len(lengths), 1 + int((len(lengths) - 1) * min(1.0, step / args.curriculum_steps))))
                length = random.choice(lengths[:unlocked])
                x, target = a5_batch(args.batch_size, length, table, identity, device); mask = None
            logits = net.forward_sequence(x)
            if args.task == "keyvalue":
                logits = logits[:, -1]
            loss = _loss(logits, target, mask)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip); optimizer.step()
            train_loss = float(loss.detach())
        trained_seconds = time.time() - started
        net.eval()
        for length in args.eval_lengths:
            losses, accuracies = [], []
            with torch.no_grad():
                for _ in range(args.val_batches):
                    if args.task == "copy":
                        x, target = copy_batch(args.eval_batch_size, length, args.vocab, device); mask = None
                    elif args.task == "mqar":
                        x, target, mask = mqar_batch(args.eval_batch_size, length, args.n_keys, args.n_values, args.n_pairs, device)
                    elif args.task == "keyvalue":
                        x, target = keyvalue_batch(args.eval_batch_size, length, args.n_keys, args.n_values, args.n_pairs, args.filler_prob, args.ordered_pairs, device); mask = None
                    else:
                        x, target = a5_batch(args.eval_batch_size, length, table, identity, device); mask = None
                    logits = net.forward_sequence(x)
                    if args.task == "keyvalue": logits = logits[:, -1]
                    losses.append(float(_loss(logits, target, mask)))
                    accuracies.append(_accuracy(logits, target, mask))
            results.append({
                "task": args.task, "model": f"dabsn_{args.geometry}", "seed": seed,
                "hidden": args.hidden, "train_length": args.train_length, "eval_length": length,
                "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
                "params": sum(p.numel() for p in net.parameters()), "train_loss_last": train_loss,
                "eval_loss": sum(losses) / len(losses), "eval_acc": sum(accuracies) / len(accuracies),
                "seconds_train": trained_seconds,
            })
    return results


def parser(task: str) -> argparse.ArgumentParser:
    defaults = {
        "copy": (48, 64, [64,128,256,512,1024,2048,3200], 1000, 48, .01),
        "mqar": (48, 64, [64,128,256,512,1024,2048,3200], 1000, 64, .006),
        "keyvalue": (48, 64, [64,128,256,512,1024,2000,3200], 2000, 64, .006),
        "a5": (256, 256, [128,256,512,1024,2048,4096,8192,16384], 60000, 64, .003),
    }[task]
    p = argparse.ArgumentParser(description=f"Canonical DABSN paper-1 {task} reproduction")
    p.set_defaults(task=task)
    p.add_argument("--hidden", type=int, default=defaults[0]); p.add_argument("--train-length", type=int, default=defaults[1])
    p.add_argument("--eval-lengths", nargs="+", type=int, default=defaults[2]); p.add_argument("--steps", type=int, default=defaults[3])
    p.add_argument("--batch-size", type=int, default=defaults[4]); p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=defaults[5]); p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2] if task != "a5" else [0,1])
    p.add_argument("--geometry", choices=["seq","field","hybrid"], default="seq")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--val-batches", type=int, default=10 if task != "a5" else 3)
    p.add_argument("--vocab", type=int, default=64); p.add_argument("--n-keys", type=int, default=16); p.add_argument("--n-values", type=int, default=16); p.add_argument("--n-pairs", type=int, default=8)
    p.add_argument("--filler-prob", type=float, default=.25); p.add_argument("--ordered-pairs", action="store_true")
    p.add_argument("--curriculum-steps", type=int, default=4000); p.add_argument("--csv", type=Path)
    return p


def main(task: str) -> None:
    args = parser(task).parse_args()
    rows = run(args)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"config": vars(args) | {"csv": str(args.csv) if args.csv else None}, "results": rows}, default=str))
