# Seesaw: Budget-Preserving Rank Reallocation for Low-Rank Adaptation

Official implementation of **"Seesaw: Budget-Preserving Rank Reallocation for Low-Rank Adaptation"** (under review at ICLR 2027).

Seesaw redistributes a fixed LoRA rank budget across Transformer layers during training, using a lightweight gradient-derived importance signal — without ever changing the total rank sum. It separates *how much* adaptation capacity a model gets from *where* that capacity is placed.

## Overview

Conventional LoRA assigns every layer the same rank, assuming uniform adaptation demand. Seesaw instead:

- Tracks an EMA-smoothed importance score per module from gradient magnitude
- Identifies the highest-scoring receiver and lowest-scoring donor module
- Transfers a single rank unit between them when their gap clears a threshold τ
- Preserves the global rank budget exactly at every step (Σ ranks = R, always)

Two instantiations are provided:
- **Seesaw-v1** — fixed-formula version, query-projection only, evaluated on BERT-base
- **Seesaw-v2** — generalized version with z-score normalization, Q/K/V/O projections, evaluated on DeBERTa-v3-base

## Results

| Backbone | Task(s) | Comparison | Key result |
|---|---|---|---|
| BERT-base | 6 GLUE tasks | LoRA, AdaLoRA, DyLoRA | Best or competitive on 5/6 tasks at matched budget (R=96) |
| DeBERTa-v1 | SST-2 | Full FT, LoRA-FA, (IA)³ | Matches LoRA accuracy (94.38%), lowest val loss |
| DeBERTa-v3 | SST-2, CoLA, STS-B, MRPC | BitFit, Adapters, LoRA, AdaLoRA | Best on STS-B and MRPC; within 0.22 pts of AdaLoRA on average |

Full tables are in the paper (`paper/seesaw_iclr2027.tex`) and reproduced under `results/`.

## Repository structure

seesaw-lora/
├── src/
│ ├── models/adaptive_lora.py # AdaptiveLoRA module, rank resizing
│ ├── controller/
│ │ ├── seesaw_controller.py # importance EMA, threshold-gated transfer
│ │ └── normalization.py # raw / mean-divided / z-score scoring
│ ├── data/glue_loader.py
│ ├── train.py
│ └── evaluate.py
├── configs/ # per-backbone/task YAML configs
├── scripts/ # shell entrypoints for each experiment
├── ablations/ # fixed-static, inverted-dynamic, optimizer-reset, normalization
├── results/ # tables (CSV) and transfer logs
├── figures/ # paper figures (SVG/PDF)
└── paper/ # LaTeX source



## Installation

```bash
git clone https://github.com/<username>/seesaw-lora.git
cd seesaw-lora
pip install -r requirements.txt
```

Requires Python ≥3.9. Key dependencies: `torch`, `transformers`, `datasets`, `scikit-learn`, `accelerate`, `sentencepiece`.

## Usage

**Run the main BERT-base GLUE benchmark:**
```bash
bash scripts/run_bert_base_glue.sh
```

**Run the DeBERTa-v3 CoLA experiment (Seesaw-v2):**
```bash
python src/train.py --config configs/deberta_v3_cola.yaml
```

**Run the normalization ablation (raw / mean / z-score):**
```bash
bash scripts/run_ablations.sh --ablation normalization
```
```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu118
```
Each run saves the best and final checkpoints, plus a transfer log recording every rank reallocation event (`donor: r → r-1 | receiver: r → r+1`) to `results/logs/`.

## Key configuration

| Parameter | BERT-base (Seesaw-v1) | DeBERTa-v3 (Seesaw-v2) |
|---|---|---|
| Target modules | Query only | Query, Key, Value, Output |
| Total rank budget (R) | 96 | 827 |
| Rank bounds | [2, 16] | [2, 64] |
| Importance aggregation | Epoch-averaged | EMA (γ = 0.85) |
| Score normalization | None | Z-score |
| Transfer threshold (τ) | 0.10 (relative gap) | 0.25 (z-scored gap) |

## Ablations

- **Fixed Static** — uniform rank, no reallocation (baseline)
- **Inverted Dynamic** — reallocation direction reversed (negative control)
- **Optimizer-Reset Policy** — global vs. selective Adam state reset after resizing
- **Importance Normalization** — raw vs. mean-divided vs. z-score scoring, at matched and mismatched controller checkpoint frequencies

See `ablations/` for standalone scripts reproducing each result and the corresponding section in the paper.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{anonymous2027seesaw,
  title     = {Seesaw: Budget-Preserving Rank Reallocation for Low-Rank Adaptation},
  author    = {Anonymous},
  booktitle = {International Conference on Learning Representations},
  year      = {2027},
  note      = {Under review}
}
```

## License

[MIT]

## Acknowledgments

This repository is submitted as part of an anonymous double-blind review process for ICLR 2027. Author names and identifying details will be added upon acceptance.
