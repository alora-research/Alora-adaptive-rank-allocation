# A-LoRA: Learning Where to Allocate Adaptation Capacity

> Anonymous repository accompanying our ICLR 2027 submission,
> *"Beyond Uniform Rank: Learning Where to Allocate Adaptation Capacity."*
> This repository is provided for double-blind review. Do not distribute
> or link to it in a way that reveals author identity.

A-LoRA is a lightweight, budget-invariant rank-reallocation mechanism for
LoRA. Instead of assigning every adapted module the same rank, A-LoRA
starts from a uniform allocation and, during training, moves rank one unit
at a time from low-importance modules to high-importance ones — where
importance is estimated from an EMA-smoothed, first-order gradient signal
that is already available at essentially no extra cost during
backpropagation. The total rank budget $\sum_l r_l = R$ is preserved
**exactly**, at every checkpoint, by construction: A-LoRA changes *where*
adaptation capacity lives, never *how much* of it exists.

No SVD, no orthogonality regularizer, no second-order computation.

---

## Contents

- [Method summary](#method-summary)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Reproducing paper results](#reproducing-paper-results)
- [Ablations](#ablations)
- [Known limitations](#known-limitations)
- [Reproducibility notes](#reproducibility-notes)
- [Citation](#citation)
- [License](#license)

---

## Method summary

For a pretrained Transformer with $L$ layers, LoRA modules are inserted
into the query and value projections of every layer, giving $2L$ adapted
modules. Each module $l$ maintains an EMA-smoothed importance score from
the mean absolute gradient of its own $B$ matrix:

```
s_l <- gamma * s_l + (1 - gamma) * mean(|grad(B_l)|)
```

At each allocation checkpoint (after a warm-up period, then every `K`
epochs), scores are normalized model-wide and converted into
rank-normalized receiver/donor scores:

```
s_tilde_l = s_l / (mean(s) + eps)
S_recv_l  = s_tilde_l / sqrt(r_l / r_0)      # favors high importance, low rank
S_don_l   = s_tilde_l * sqrt(r_l / r_0)      # favors low importance, high rank
```

The top receiver and bottom donor are selected; if their relative gap
clears a threshold `tau`, exactly one unit of rank moves between them,
subject to `[r_min, r_max]` bounds. Full pseudocode is in
`docs/algorithm.pdf` (Algorithm 1 in the paper) and matches the reference
implementation in `src/` line-for-line.

## Repository structure

```
.
├── src/
│   ├── train_qnli.py          # A-LoRA on QNLI (DeBERTa-v3-base)
│   ├── train_mrpc.py          # A-LoRA on MRPC (DeBERTa-v3-base)
│   ├── train_sst2_bert.py     # BERT-base ablation harness (Section 7)
│   ├── alora.py                # AdaptiveLoRALinear module + controller
│   └── optimizer_reset.py      # global vs. selective optimizer-reset policies
├── configs/
│   ├── bert_base_R96.yaml      # ablation config, R_total = 96
│   ├── deberta_v3_qnli.yaml
│   └── deberta_v3_mrpc.yaml
├── docs/
│   ├── algorithm.pdf           # rendered Algorithm 1 (horizontal, 2-column)
│   └── figures/                # analysis figure sources (matplotlib scripts)
├── results/
│   └── (populated by training scripts; see below)
└── README.md
```

## Installation

```bash
git clone https://github.com/anonymous/A-LoRA.git
cd A-LoRA
pip install -r requirements.txt
```

`requirements.txt` pins `transformers`, `datasets`, `sentencepiece`,
`scikit-learn`, and `torch` (FP32 training only — no mixed precision is
used anywhere in this codebase, by design, for reproducibility).

## Quickstart

```bash
# BERT-base, single dataset, uniform LoRA vs. A-LoRA
python src/train_sst2_bert.py --config configs/bert_base_R96.yaml

# DeBERTa-v3-base, QNLI
python src/train_qnli.py

# DeBERTa-v3-base, MRPC
python src/train_mrpc.py
```

Each script is self-contained (single-file, no external launcher) and
prints its final rank allocation, trainable parameter count, and best
validation metric at the end of the run. Checkpoints are saved to the
working directory as `{task}_alora_best.pt` / `{task}_alora_final.pt`,
each containing `model_state_dict`, the final per-module rank allocation,
and the full per-epoch training history.

## Configuration

Key hyperparameters (see each script's `CONFIG` block for the exhaustive
list):

| Name | Meaning | Value used in paper |
|---|---|---|
| `INITIAL_RANK` (`r_0`) | starting rank of every module | 8 |
| `MIN_RANK`, `MAX_RANK` | allocation bounds | 2, 16 |
| `EMA_DECAY` (`gamma`) | importance smoothing factor | 0.85 |
| `IMPORTANCE_GAP` (`tau`) | transfer gating threshold | 0.25 |
| `CONTROLLER_START_EPOCH` (`E_s`) | warm-up length before reallocation begins | 2–3 (task-dependent) |
| `CONTROLLER_INTERVAL` (`K`) | epochs between allocation checkpoints | 2 |
| `OPTIMIZER_RESET_MODE` | `"global"` or `"selective"` — see [Ablations](#ablations) | `"global"` (main results) |

> **Note on hyperparameter values in the paper text.** Section 5 and the
> reproducibility statement in the paper currently state $\tau = 0.5$ and
> EMA decay $\alpha = 0.9$; the values actually used by the scripts in this
> repository are $\tau = 0.25$ and $\gamma = 0.85$. We are reconciling this
> discrepancy before camera-ready — the numbers in this README and in the
> `configs/` directory reflect what the code actually runs.

## Reproducing paper results

| Table | Script | Backbone | Status |
|---|---|---|---|
| Table 1 (six NLU benchmarks) | `train_sst2_bert.py` (per-task configs) | BERT-base | five-seed macro-average, seeds `{42, 123, 456, 789, 999}` |
| Table 2 (efficiency profiling) | internal profiling harness, not yet included in this snapshot | DeBERTa-v1-base | single run per method (see paper §5.1 note on statistical scope) |
| Table 3 (ultra-low-budget benchmark) | `train_qnli.py`, `train_mrpc.py` + GLUE variants | DeBERTa-v3-base | **A-LoRA row pending** — being finalized |
| Table 4 (allocation-direction ablation) | `train_sst2_bert.py --variant {fixed,inverted,proposed}` | BERT-base, $R_{\text{total}}=96$ | complete |
| Table 5 (optimizer-reset ablation) | `--reset_mode {global,selective}` | BERT-base, $R_{\text{total}}=96$ | **pending** — selective-reset run not yet completed |

We are not publishing partially-filled result tables as final numbers;
entries marked *pending* above are left incomplete in this repository
until the corresponding runs are finished, rather than backfilled with
placeholder or estimated values.

## Ablations

**Allocation direction** (`--variant`): `fixed` holds rank uniform
throughout; `inverted` moves rank from *high*-importance to
*low*-importance modules (a deliberately wrong-signed control); `proposed`
is A-LoRA as described above. All three share identical hyperparameters,
data splits, and seeds, differing only in transfer direction.

**Optimizer reset policy** (`--reset_mode`): `global` rebuilds the AdamW
optimizer (and, where applicable, the LR scheduler) from scratch after
every transfer, discarding Adam moment estimates for *every* module.
`selective` preserves Adam state for every module except the two directly
involved in the transfer, mutating the existing optimizer's parameter
groups in place rather than reconstructing it. See `src/optimizer_reset.py`
for the implementation and an inline discussion of why the two policies
are not expected to differ only in "how much state is wiped" — under
`global`, the LR schedule is also restarted, which `selective` avoids.

## Known limitations

We list these here in the interest of transparency, matching the
limitations already acknowledged in the paper:

- **Allocation granularity is not unified across experiments.** The
  BERT-base ablations (Table 4/5) allocate one shared rank per layer
  ($L=12$ variables); the DeBERTa-v3 experiments and the core method
  allocate query and value projections independently ($2L$ variables). Both
  are valid instantiations of the mechanism, but results across the two
  settings are not directly comparable rank-for-rank.
- **The importance signal is a simple, first-order heuristic.** We have
  not yet benchmarked it against alternative proxies (squared gradient,
  parameter-gradient product) or a randomized control; this is planned
  future work (see the paper's Future Work section).
- **Table 2's efficiency numbers are single-run**, not five-seed averages
  like Table 1 — they are intended as a controlled efficiency comparison
  under identical hardware, not a statistical claim; see the caveat in the
  paper text.

## Reproducibility notes

- All experiments use FP32 throughout; no mixed-precision training is used
  anywhere in this codebase.
- Hardware: single NVIDIA T4 GPU (16GB VRAM) for all reported runs.
- Random seeds `{42, 123, 456, 789, 999}` are used for every five-seed
  result; single-seed results use seed `42` unless otherwise noted in the
  corresponding config file.
- Each `resize_rank()` call preserves existing rank components exactly
  (copied, not reinitialized); newly added $A$ rows are drawn from
  $\mathcal{N}(0, 0.02^2)$ and newly added $B$ columns are zero-initialized,
  so added capacity begins as a no-op and is shaped entirely by subsequent
  gradients.

## Citation

Citation details withheld for double-blind review; a full BibTeX entry
will be added upon de-anonymization.

## License

Code released under the MIT License upon de-anonymization. A `LICENSE`
file will be added at that time; until then, this repository is provided
for reviewer access only.
