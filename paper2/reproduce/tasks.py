"""Deterministic sparse episodes for the two delayed-learning tasks.

This module imports only the standard library. Event content, event timing,
noise, and queries have independent keyed random streams. A returned episode
can be rendered into dense tensors by any CPU or GPU caller without changing
its scientific definition. All labels are stored separately from input events.
"""
from __future__ import annotations

import hashlib
import math
import random


SCHEMA = 1
TASKS = ("selective_retrieval", "conditional_computation")
N_KEYS = 16
N_VALUES = 16
VECTOR_DIM = 8
N_FUNCTIONS = 4


def _rng(task: str, seed: int, index: int, component: str) -> random.Random:
    identity = f"paper2-task-v1/{task}/{seed}/{index}/{component}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(identity).digest()[:16], "big"))


def _position_rng(task: str, seed: int, index: int, length: int):
    return _rng(task, seed, index, f"positions/{length}")


def _event(time: int, kind: str, indices: list[int], values: list[float]):
    if len(indices) != len(values) or len(set(indices)) != len(indices):
        raise ValueError("Each event needs distinct coordinates with matching values")
    return {"time": time, "kind": kind, "indices": indices, "values": values}


def _validate(length: int, inputs: int):
    if not isinstance(length, int) or length < 128:
        raise ValueError("The confirmatory task requires length >= 128")
    if inputs not in (64, 128):
        raise ValueError("Confirmatory task input dimensions are 64 and 128")


def _noise_positions(task: str, seed: int, index: int, length: int,
                     occupied: set[int]):
    # Roughly one distracting event per eight elapsed steps. The independent
    # noise stream is unaffected by switching distractors on or off.
    rng = _rng(task, seed, index, f"noise-times/{length}")
    candidates = [t for t in range(1, length-1) if t not in occupied]
    count = min(len(candidates), max(1, length//8))
    return sorted(rng.sample(candidates, count))


def _retrieval(seed: int, index: int, length: int, inputs: int,
               distractors: bool):
    task = "selective_retrieval"
    content = _rng(task, seed, index, "content")
    query_rng = _rng(task, seed, index, "query")
    keys = content.sample(range(N_KEYS), 8)
    # Every selected key appears twice. Independently sampled replacement
    # values can coincide by chance; the target always follows the latest write.
    writes = [(key, content.randrange(N_VALUES), occurrence)
              for occurrence in (0, 1) for key in keys]
    content.shuffle(writes)
    positions = sorted(_position_rng(task, seed, index, length).sample(
        range(1, length//4), len(writes)))
    tagged = [(t, key, value, occurrence)
              for t,(key,value,occurrence) in zip(positions,writes)]
    queried_key = query_rng.choice(keys)
    latest = {key: value for _,key,value,_ in tagged}
    target = latest[queried_key]
    events = [_event(t, "write", [key, 16+value, 32], [1.,1.,1.])
              for t,key,value,_ in tagged]
    occupied = set(positions)
    if distractors:
        noise = _rng(task, seed, index, "noise-content")
        for t in _noise_positions(task, seed, index, length, occupied):
            key = noise.randrange(N_KEYS); value = noise.randrange(N_VALUES)
            events.append(_event(t, "noise", [key,16+value,33], [1.,1.,1.]))
    events.append(_event(length-1,"query",[queried_key,34],[1.,1.]))
    events.sort(key=lambda e:e["time"])
    return {"schema":SCHEMA,"task":task,"seed":seed,"index":index,
            "length":length,"inputs":inputs,"events":events,"target":target,
            "metadata":{"writes":[{"time":t,"key":key,"value":value,
                                   "occurrence":occurrence}
                                  for t,key,value,occurrence in tagged],
                        "queried_key":queried_key,"latest_value":target,
                        "distractors":distractors}}


def target_function(cue: int, first: list[float], second: list[float]):
    if len(first)!=VECTOR_DIM or len(second)!=VECTOR_DIM or cue not in range(N_FUNCTIONS):
        raise ValueError("Invalid conditional-computation target")
    if cue==0: return [(a+b)/math.sqrt(2) for a,b in zip(first,second)]
    if cue==1: return [(a-b)/math.sqrt(2) for a,b in zip(first,second)]
    if cue==2: return [a*b for a,b in zip(first,second)]
    normalizer=math.sqrt((1-math.exp(-4))/2)
    return [math.sin(a+b)/normalizer for a,b in zip(first,second)]


def _conditional(seed: int, index: int, length: int, inputs: int,
                 distractors: bool):
    task="conditional_computation"
    content=_rng(task,seed,index,"content")
    query_rng=_rng(task,seed,index,"query")
    first=[content.gauss(0,1) for _ in range(VECTOR_DIM)]
    second=[content.gauss(0,1) for _ in range(VECTOR_DIM)]
    cue=query_rng.randrange(N_FUNCTIONS)
    positions=sorted(_position_rng(task,seed,index,length).sample(range(1,length//4),2))
    events=[_event(positions[0],"first",list(range(8))+[8],first+[1.]),
            _event(positions[1],"second",list(range(8))+[9],second+[1.])]
    if distractors:
        noise=_rng(task,seed,index,"noise-content")
        for t in _noise_positions(task,seed,index,length,set(positions)):
            value=[noise.gauss(0,1) for _ in range(VECTOR_DIM)]
            events.append(_event(t,"noise",list(range(8))+[10],value+[1.]))
    events.append(_event(length-1,"query",[11+cue],[1.]))
    events.sort(key=lambda e:e["time"])
    return {"schema":SCHEMA,"task":task,"seed":seed,"index":index,
            "length":length,"inputs":inputs,"events":events,
            "target":target_function(cue,first,second),
            "metadata":{"first":first,"second":second,"positions":positions,
                        "cue":cue,"distractors":distractors}}


def episode(task: str, seed: int, index: int, length: int, inputs: int=64,
            distractors: bool=True):
    """Return a sparse complete episode; parameters never affect generated data."""
    _validate(length,inputs)
    if task not in TASKS: raise ValueError(f"Unknown task: {task}")
    if not isinstance(seed,int) or not isinstance(index,int) or index<0:
        raise ValueError("seed and nonnegative index must be integers")
    generate = _retrieval if task=="selective_retrieval" else _conditional
    return generate(seed,index,length,inputs,distractors)


def dense_inputs(record):
    """Standard-library reference rendering; numerical runners may use scatter."""
    x=[[0.]*record["inputs"] for _ in range(record["length"])]
    for event in record["events"]:
        row=x[event["time"]]
        for index,value in zip(event["indices"],event["values"]):
            if not 0<=index<record["inputs"]: raise ValueError("Event index out of range")
            row[index]=value
    return x


# ------------------------------------------------------------------------
# Uninterrupted delayed-feedback stream (sustained study).
# Targets are regenerated from a keyed past event identity by the evaluator;
# the learner receives current inputs only.

SUSTAINED_DELAYS = (4,16,64)
SUSTAINED_INPUTS = 64
SUSTAINED_LENGTH = 32


def sustained_rng(seed,index,lane,component):
    key = f'paper2-sustained-v1/{seed}/{index}/{lane}/{component}'.encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:16],'big'))


def sustained_pair(seed,index,lane):
    random_source = sustained_rng(seed,index,lane,'event-content')
    return ([random_source.gauss(0,1) for _ in range(8)],
            [random_source.gauss(0,1) for _ in range(8)])


def sustained_episode(seed,index,lane):
    if not all(isinstance(v,int) and v >= 0 for v in (seed,index,lane)):
        raise ValueError('Seed, absolute interval and lane must be nonnegative integers')
    first,second = sustained_pair(seed,index,lane)
    query = sustained_rng(seed,index,lane,'query')
    delay_index = query.randrange(len(SUSTAINED_DELAYS))
    delay = SUSTAINED_DELAYS[delay_index]
    function = query.randrange(4)
    events = [
        {'time':0,'kind':'first','indices':list(range(8))+[8],'values':first+[1.]},
        {'time':1,'kind':'second','indices':list(range(8))+[9],'values':second+[1.]}
    ]
    noise = sustained_rng(seed,index,lane,'distractors')
    for position in range(4,SUSTAINED_LENGTH-1,4):
        events.append({'time':position,'kind':'noise','indices':list(range(8))+[10],
                       'values':[noise.gauss(0,1) for _ in range(8)]+[1.]})
    events.append({'time':SUSTAINED_LENGTH-1,'kind':'query','indices':[11+function,15+delay_index],
                   'values':[1.,1.]})
    target = None
    source_identity = None
    if index >= max(SUSTAINED_DELAYS):
        old_first,old_second = sustained_pair(seed,index-delay,lane)
        target = target_function(function,old_first,old_second)
        source_identity = [seed,index-delay,lane]
    return {'schema':1,'task':'sustained_conditional_computation',
        'identity':[seed,index,lane],'length':SUSTAINED_LENGTH,'inputs':SUSTAINED_INPUTS,'events':events,
        'delay_chunks':delay,'delay_steps':delay*SUSTAINED_LENGTH,'function':function,
        'target':target,'target_event_identity':source_identity}


def sustained_batch(seed,index,batch):
    if not isinstance(batch,int) or batch <= 0:
        raise ValueError('Positive integer batch required')
    return [sustained_episode(seed,index,lane) for lane in range(batch)]
