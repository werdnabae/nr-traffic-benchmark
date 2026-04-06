# NR Benchmarking

**Traffic Speed Forecasting Under Nonrecurrent Conditions:
A Causally-Stratified Large-Scale Benchmark**

*Andrew J. Bae — Carnegie Mellon University*

<!-- [![Paper](https://img.shields.io/badge/paper-TRC%20under%20review-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]() -->

---

## Overview

This repository contains code and results for a large-scale empirical
benchmark evaluating 17 traffic forecasting models under nonrecurrent (NR)
conditions — incidents, work zones, and weather events — on two independent
freeway networks.

The central contribution is a **causal observability framework** that
partitions evaluation into three informationally distinct regimes:

| Regime | Context window | Target |
|--------|----------------|--------|
| **Recurrent** | No confirmed NR | No NR |
| **Unobserved onset** | No confirmed NR | NR active (model is blind) |
| **Confirmed NR** | NR confirmed in context | NR active |

Key findings:
- NR MAE is **2–5× higher** than recurrent MAE across all 17 models and both networks
- **Onset is the dominant failure mode**: onset MAE exceeds confirmed NR MAE by 60–70%
- Model rankings **shift** under NR evaluation relative to aggregate evaluation
- NR-aware training strategies and richer inputs improve NR MAE by **at most 6%**

---

## Repository Structure

```
nr-benchmarking/
├── src/
│   ├── data/           # Data loading, NR labels, feature builder, dataset
│   ├── evaluation/     # Regime-stratified metrics, transition analysis
│   ├── models/         # Statistical, ML, DL, and spatial model wrappers
│   └── training/       # Unified trainer (4 strategies)
│
├── experiments/
│   ├── run_benchmark.py     # Single model × network × config × strategy
│   ├── run_sweep.py         # Full sweep with skip-if-exists
│   ├── run_baselines.py     # Non-DL baselines (CPU, no torch required)
│   └── tune_strategies.py   # Hyperparameter tuning for NR-aware strategies
│
├── scripts/
│   ├── sync_results.py          # Aggregate all result JSONs → summary table
│   ├── generate_results_table.py # Generate LaTeX table from results
│   ├── generate_nr_labels.py    # Regenerate NR labels from raw data (Mac)
│   ├── test_dl_pipeline.py      # Smoke test all DL models
│   └── setup_jupyterhub.sh      # JupyterHub first-time setup
│
├── results/            # Benchmark results (JSON per run, CSV summary)
├── paper/              # LaTeX source for the TRC submission
├── data/               # NOT included — see data/README.md
├── external/LargeST/   # NOT included — see external/LargeST/README.md
│
├── config.yaml         # All paths and hyperparameters
├── requirements.txt    # Python dependencies
└── AGENTS.md           # Full technical documentation
```

---

## Quick Start

### Non-DL baselines (CPU, no GPU required)

```bash
pip install -r requirements.txt
python experiments/run_baselines.py
```

### Full benchmark (GPU required)

```bash
# Phase 1: speed × standard × all 17 models
python experiments/run_sweep.py \
    --feature_configs speed \
    --strategies standard

# Phase 2: all 10 feature configs × standard
python experiments/run_sweep.py --strategies standard

# Phase 3: speed × all 4 strategies × DL models
python experiments/run_sweep.py \
    --feature_configs speed \
    --models lstm transformer hl lstm_st dcrnn agcrn stgcn gwnet \
              astgcn sttn stgode dgcrn d2stgnn
```

### Check progress

```bash
python scripts/sync_results.py          # Summary table to terminal
python scripts/sync_results.py --csv    # Also writes results/all_results.csv
```

---

## Networks

| Network | Location | Links | Period | NR Rate |
|---------|----------|-------|--------|---------|
| **TSMO** | Howard County, MD | 228 | Feb 2022–Feb 2023 | 3.97% |
| **Cranberry** | Pittsburgh, PA | 78 | Feb 2022–Jan 2024 | 2.42% |

Speed data is from INRIX commercial GPS trajectories via RITIS.
See `data/README.md` for access instructions.

---

## Models

17 models across four families:

| Family | Models |
|--------|--------|
| Statistical | Last Observation, Historical Average, Linear AR |
| ML | XGBoost |
| Per-link DL | LSTM, Transformer |
| Spatiotemporal | HL, LSTM-ST, DCRNN, AGCRN, STGCN, GWNet, ASTGCN, STTN, STGODE, DGCRN, D2STGNN |

Spatiotemporal models are implemented via the
[LargeST framework](https://github.com/liuxu77/LargeST)
with a unified `LargeSTRunner` adapter.

---

## Results

Pre-computed results for Phase 1 (speed × standard, all 17 models)
are in `results/`. Run `scripts/sync_results.py --csv` to regenerate
`results/all_results.csv`.

<!-- ---

## Citation

```bibtex
@article{bae2026nrbenchmark,
  author  = {Bae, Andrew J.},
  title   = {Traffic Speed Forecasting Under Nonrecurrent Conditions:
             A Causally-Stratified Large-Scale Benchmark},
  journal = {Transportation Research Part C: Emerging Technologies},
  year    = {2026},
  note    = {Under review}
}
``` -->

---

## License

MIT License. See `LICENSE` for details.

The LargeST model implementations in `external/LargeST/` are subject
to their own license terms.
