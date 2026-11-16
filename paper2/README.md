# Paper 2: online credit as a persistent object

This folder holds the paper and everything needed to reproduce it.

```
paper2/
  main.tex                      the paper (bibliography inline; builds with pdflatex or tectonic)
  results_all.json.gz           every job's run record and the selections (reproduce/collect.py)
  results/generated.tex         measured tables, written only by reproduce/analysis.py --paper
  figs/                         figures, written only by generate_figures.py from results_all.json.gz
  Makefile                      make (paper), make proofs, make results, make figures
  framework.lock.json           SHA-256 of every source file of the DABSN tree the paper uses
  proofs/                       exact-arithmetic proofs (Python standard library only)
  reproduce/                    every measured study, one resumable runner
```

## The object

The credit state E is DABSN's B-channel online eligibility (`CoreEligibilityState`):

- **Matrices** (`W`, `A`, `Wg`, `Ug`): one budget-only trace per entry.
- **Sixteen per-channel controls:** a three-state (budget, energy, saturation) trace each.
- **Four shared controls:** a three-state trace for every state channel.

That is `B * (2*S*I + 2*S^2 + 60*S)` scalars for batch `B`, input width `I` and state width `S`, independent of sequence length.

## Exact proofs (seconds, any machine)

```bash
cd paper2/proofs && python3 run_all.py
```

The script checks:

- terminal-Jacobian separation in both directions;
- the omitted-influence residual;
- the storage formula;
- partition invariance;
- the response geometry and direct-action algebra.

Arithmetic uses exact rationals and polynomials. The record is written to a new file and never overwrites an old one.

## Measured studies

Install the framework tree named by `framework.lock.json`. Every job refuses to run if any framework file differs.

```bash
pip install ./dabsn-source          # the locked tree
cd paper2/reproduce
python run.py --device cuda --results ../results --smoke     # every code path in minutes
python run.py --device cuda --results ../results-matrix      # the full matrix
python run.py --results ../results-matrix --status           # progress
python collect.py ../results-matrix                          # -> ../results_all.json.gz
```

`run.py` is resumable. If it is interrupted at any point (a closed notebook, a lost machine, `--max-hours`), run the same command again:

- Passed jobs are never rerun.
- An interrupted job continues from its newest checkpoint.
- Every attempt is kept.

Everything computes in FP32 (TF32 off) on any GPU; the reference-equation studies use FP64. Matched comparisons need one forward computation for every method: in FP32 the online-credit forward and the ordinary forward are bitwise identical, which the preflight asserts on your GPU before any learning job runs. The device and precision are fixed when a results directory is first used.

Stages, in order:

| Stage | What it measures |
|---|---|
| preflight | Native kernels against the reference rule; real updates and exact recovery at full size |
| oracles | Storage, single-step derivatives, partition, scalar counterexample, residual identity, delayed transport and learning, terminal-J separation, the credit operator K |
| action | Parameter action: teacher matching by direct readouts of E, against the full Jacobian |
| continuation | Objective-conditioned continuation: E steers behavior after the objective is removed |
| directions | Held-out objective directions and trace-derived action amplitude (update cadence) |
| ontology | The seven readouts: decoding, event arithmetic, inverse readouts |
| scaling, resource | Measured time and memory |
| continuous | Memory capacity (Jaeger 2001) on uninterrupted streams, stock core, six arms |
| delayed (+ development) | The central matched learning study: online, frozen core, BPTT on the adding and temporal order problems (Hochreiter and Schmidhuber 1997) |
| interventions | Causal edits of E at frozen checkpoints of the central study |
| ablation | Shape: (I, S) = (64, 512) and (128, 256), online against BPTT |
| sustained | NARMA-10 (Atiya and Parlos 2000) on uninterrupted streams, stock core, six arms |
| alignment | Online credit against exact BPTT-64 gradients, per parameter family, at trained stream weights |
| stream-window | TBPTT with a one-chunk window on both streams |
| stream-normalized (+ development) | The paced native action on the core (studies/native_action.py; readout under the stream arms' Adam regime) on both streams; development at the full budget |
| conjunction (+ development) | The core driven by E (E credit, the normalized action, pace from E; read and readout under the shared AdamW regime) against AdamW online and AdamW BPTT on copy, MQAR and key-value |
| continuity, audit | Paper-1 tasks against BPTT; the A5 cohort; native audit at trained weights |

The central study's task generators are in `studies/delayed_tasks.py`. No stage calls `tasks.py` or the `delayed_*` helpers in `models.py`; they are part of the shared sources that every study's recorded digest covers.

Each stage that has a development phase selects its learning rates from development seeds only. Those development seeds are never used for evaluation.

## From records to the paper

`results_all.json.gz` holds every job's latest run record and the selections, collected from the results directory by `reproduce/collect.py`. The analysis, the tables and the figures read it (a live results directory works too):

```bash
python reproduce/analysis.py results_all.json.gz --paper
```

The analysis requires every declared job to have passed, with one precision policy and one reproduction source per study. It names any failure, missing job or out-of-memory outcome, and writes `results/generated.tex`. `main.tex` includes that file verbatim. No number in the Results section is typed by hand.

The figures are built from the same file:

```bash
python3 generate_figures.py results_all.json.gz                 # figs/
```

For spreadsheet work, `reproduce/export_results.py results_all.json.gz --out DIR` writes every
study's passed records, unedited, as CSV tables (one row per job, one table per list of
records inside a result).
