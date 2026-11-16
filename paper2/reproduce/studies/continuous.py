"""Short-term memory capacity (Jaeger 2001) learned on uninterrupted streams.

Each lane reads u(t) ~ U[-1, 1] forever and outputs u(t - k) for every declared delay
k at every step; the score is the standard memory capacity MC_k (squared correlation)
and its sum. The model is the stock DABSN core with a linear readout and no read, so
whatever the model keeps from the past lives in its recurrent state and whatever
teaches it to keep it arrives through E. Delays longer than a chunk can only be
learned from credit carried across chunk boundaries. Arms and procedure: stream.py.
"""
from studies.stream import NEEDS_NATIVE, identifier, jobs, preflight_jobs, run, train  # noqa: F401
