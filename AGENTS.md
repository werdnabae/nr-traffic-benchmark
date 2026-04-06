# AGENTS.md — Everything an Agent Needs to Know

This document is the single source of truth for any agent working on this project.
Read it fully before making any changes.

---

## What This Project Is

A benchmarking study for **Transportation Research Part C**. The paper asks:
**How do traffic forecasting models actually perform under nonrecurrent (NR)
conditions — incidents, weather events, work zones — when evaluated precisely
rather than in aggregate?**

This is NOT a new model paper. It is a rigorous empirical evaluation that uses
fine-grained spatiotemporal NR labels to expose where and why current models fail.

The prior paper (TRC, already published) produced ensemble labels that identify
NR disturbances at the individual (link, 5-min timestep) level. This project
uses those labels as the evaluation backbone.

---

## Repository Layout

```
nr-benchmarking/
├── AGENTS.md                        ← you are here
├── config.yaml                      ← single source of all paths and hyperparameters
├── requirements.txt
│
├── data/
│   ├── tsmo/                        ← TSMO network (Howard County MD, 228 links, 1 yr)
│   │   ├── tsmo_speed_data.parquet  ← 5-min speed by link (mph)
│   │   ├── tsmo_incident_reports.parquet  ← binary Waze incidents by link
│   │   ├── tsmo_network.geojson     ← road network geometry
│   │   ├── tsmo_upstream_mapping.json ← {tmc: [upstream_tmc, ...]}
│   │   ├── weather.pkl              ← scaled weather features (T × W DataFrame)
│   │   ├── nr_labels_full.parquet   ← binary NR labels (T × N), full time series
│   │   ├── v_rec_full.parquet       ← recurrent lower-bound speed (T × N, mph)
│   │   └── adj_mx.npy               ← (228, 228) binary adjacency matrix
│   └── cranberry/                   ← Cranberry network (Pittsburgh PA, 78 links, 2 yr)
│       └── (same structure)
│
├── src/
│   ├── data/
│   │   ├── loader.py                ← NetworkData class: loads everything, computes
│   │   │                               causal labels, 70/15/15 split, normalisation
│   │   ├── adjacency.py             ← builds adj matrix from GeoJSON + upstream mapping
│   │   ├── features.py              ← FeatureBuilder: assembles (T, N, C) tensors
│   │   ├── dataset.py               ← PyTorch Dataset: session-aware sliding windows
│   │   └── numpy_iterator.py        ← numpy equivalent (no torch dependency)
│   │
│   ├── evaluation/
│   │   ├── metrics.py               ← regime-stratified MAE/RMSE/MAPE
│   │   └── transitions.py           ← onset vs persistence analysis
│   │
│   ├── models/
│   │   ├── statistical/             ← last_observation, historical_avg, linear_ar
│   │   ├── ml/                      ← xgboost_model
│   │   ├── dl/                      ← lstm, transformer (per-link, shared weights)
│   │   └── spatial/
│   │       └── runner.py            ← LargeSTRunner: wraps all 12 LargeST models
│   │
│   └── training/
│       └── trainer.py               ← unified training loop, all 4 strategies
│
├── experiments/
│   ├── run_benchmark.py             ← single model × network × config × strategy
│   ├── run_sweep.py                 ← full sweep with skip-if-exists
│   ├── run_baselines.py             ← non-DL baselines only (no torch required)
│   └── tune_strategies.py           ← HP tuning for nr_loss_weight & finetune_lr_mult
│
├── scripts/
│   ├── generate_nr_labels.py        ← (Mac only) regenerate NR labels from raw data
│   ├── setup_jupyterhub.sh          ← JupyterHub first-time setup
│   ├── sync_results.py              ← aggregate all result JSONs into a summary table
│   ├── test_dl_pipeline.py          ← smoke test all 14 DL models
│   └── transfer_to_jupyterhub.sh    ← rsync to JupyterHub (requires cloudflared auth)
│
├── external/
│   └── LargeST/                     ← LargeST repo (downloaded, not cloned via git)
│       └── src/models/              ← 12 spatial-temporal model implementations
│
└── results/
    ├── tsmo/                        ← one JSON per (model, feature_config, strategy)
    ├── cranberry/
    ├── logs/                        ← per-job log files from run_sweep.py
    └── tuning_results.json          ← HP tuning grid search results
```

---

## The Forecasting Task

- **Input**: last 9 timesteps (45 min) × N links × C features
- **Target**: next 6 timesteps (30 min) × N links (speed in mph)
- **Resolution**: 5-minute intervals
- **Split**: chronological 70 / 15 / 15 on full time series per network
- **Session-aware**: windows never span overnight/weekend gaps

---

## The Two NR Label Variants

This is the most important concept in the project.

### full_nr  (T × N binary)
Raw ensemble labels from the prior paper. A cell is 1 when the ensemble
labeler identified a nonrecurrent disturbance at that link and timestep.
Used as ground-truth evaluation mask only.

### causal_nr  (context-dependent)
What the model can actually observe in real-time. The ensemble labeler's
persistence filter requires 3 consecutive flagged timesteps (15 min) before
confirming NR. The first 2 steps of any NR episode are back-labeled in
hindsight but were not observable.

For a sliding window ending at timestep T:
- `causal_nr[t, n] = 1`  iff  `full_nr[t, n] == 1`  AND  `observation_time[t, n] <= T`
- `observation_time[t, n]` = `episode_start + 2`  (earliest T at which the episode
  is confirmed; first 2 steps share the same observation_time = episode_start + 2)

**observation_time is precomputed in NetworkData and stored as a (T, N) float64 array.**

### causal_fixed  (T × N binary, precomputed)
`causal_fixed[t, n] = 1`  iff  `full_nr[t-2:t+1, n]` are all 1.
Used for:
- transition regime classification at the end of the context window
- loss weighting in the weighted_loss training strategy
- NR sample selection for nr_finetune strategy

---

## The Four Evaluation Regimes

Every prediction is classified at the individual **(timestep, link)** level:

| Regime | causal_fixed at end of context | full_nr in target |
|--------|-------------------------------|-------------------|
| Recurrent | 0 | 0 |
| Unobserved onset | 0 | 1 — hardest: model is informationally blind |
| Confirmed NR | 1 | 1 — model has NR signal in context |

Metrics: MAE, RMSE, MAPE for each regime, per-horizon (h1–h6) and averaged.

---

## Feature Configurations

The feature channel order is FIXED and matters for DGCRN and D2STGNN:

```
Channel 0:      speed (always first)
Channels 1,2:   TOD [0,1], DOW [0,1]  (if "time" in config — MUST come right after speed)
Channels 3+:    weather, incidents, nr_causal  (in that order)
```

| Config ID | Features | C |
|-----------|---------|---|
| speed | speed | 1 |
| speed_weather | speed + weather | 1+W |
| speed_incidents | speed + incidents | 2 |
| speed_nr | speed + causal NR label | 2 |
| speed_weather_incidents | speed + weather + incidents | 1+W+1 |
| speed_time | speed + TOD + DOW | 3 |
| speed_time_weather | speed + TOD + DOW + weather | 3+W |
| speed_time_incidents | speed + TOD + DOW + incidents | 4 |
| speed_time_nr | speed + TOD + DOW + causal NR | 4 |
| speed_time_weather_incidents | all | 3+W+1 |

W = 18 for TSMO, 12 for Cranberry (different weather station coverage).

---

## Models (18 total)

### Statistical (CPU, no training)
- `last_observation` — persist last observed speed
- `historical_average` — same DOW × TOD mean from training data
- `linear_ar` — global Ridge regression, lagged speed + time features + link OHE

### ML (CPU, training on full dataset)
- `xgboost` — global XGBoost, link ID as feature, direct multi-step (6 separate models)

### DL per-link (GPU, shared weights across links)
- `lstm` — Seq2Seq LSTM encoder-decoder
- `transformer` — Seq2Seq Transformer encoder-decoder

### LargeST spatial-temporal (GPU, full graph)
- `hl`, `lstm_st`, `dcrnn`, `agcrn`, `stgcn`, `gwnet`, `astgcn`, `sttn`,
  `stgode`, `dstagnn`, `dgcrn`, `d2stgnn`

**Special cases for DGCRN and D2STGNN:**
- These models internally read channels 1 and 2 as TOD/DOW position encodings
- They require C_in ≥ 3 (for DGCRN) or always append 2 time channels (D2STGNN)
- The runner.py pads with real TOD/DOW values from timestamps when time features
  are not already in the input
- Always use `speed_time` config or any `*_time_*` config when benchmarking these
  two for fair comparison with their positional encoding active

---

## Training Strategies (4)

| Strategy | Description | Novel HP |
|----------|------------|---------|
| standard | MSE loss, uniform weighting | — |
| weighted_loss | MSE with NR timesteps upweighted | `nr_loss_weight` |
| nr_finetune | Phase 1: standard on all data; Phase 2: standard on NR samples only, reduced LR | `finetune_lr_multiplier` |
| multi_objective | Speed MSE + auxiliary BCE on causal NR labels | — |

**Baselines (LO, HA, LinearAR, XGBoost) always run with `standard` strategy only.**

### Tuned Hyperparameters
Run `experiments/tune_strategies.py` BEFORE the main sweep. It tests:
- `nr_loss_weight` grid: [2, 5, 10, 20] — selected by validation NR MAE
- `finetune_lr_multiplier` grid: [0.05, 0.1, 0.2] — selected by validation NR MAE

Results saved to `results/tuning_results.json`. Best values written back to
`config.yaml` automatically. The sweep reads them from config.yaml.

---

## Result File Naming Convention

**Critical**: the skip-if-exists logic in `run_sweep.py` depends on this exact pattern.

```
results/{network}/{model}_{feature_config}_{strategy}.json
```

Examples:
```
results/tsmo/lstm_speed_standard.json
results/cranberry/gwnet_speed_time_weather_weighted_loss.json
results/tsmo/last_observation_speed_standard.json
```

If result files exist with any other naming pattern, the sweep will re-run them.

---

## Running the Full Sweep

### Recommended order

**Step 0** — One-time setup (JupyterHub):
```bash
cd ~/volume/nr-benchmarking
bash scripts/setup_jupyterhub.sh
```

**Step 1** — Hyperparameter tuning (~2.5 hrs, TSMO × LSTM × speed only):
```bash
python3 experiments/tune_strategies.py
```

**Step 2** — Phase 1: core benchmark (~12 hrs):
```bash
python3 experiments/run_sweep.py --feature_configs speed --strategies standard
```

**Step 3** — Phase 2: input config sweep (~5 days):
```bash
python3 experiments/run_sweep.py --strategies standard
```

**Step 4** — Phase 3: training strategies (~1.5 days):
```bash
python3 experiments/run_sweep.py \
  --feature_configs speed \
  --models lstm transformer hl lstm_st dcrnn agcrn stgcn gwnet astgcn sttn stgode dstagnn dgcrn d2stgnn
```

### Sweep behaviour
- **Skip-if-exists**: checks for result JSON before each job; safe to Ctrl+C and restart
- **Live output**: every epoch prints to terminal in real-time
- **Logs**: each job also writes to `results/logs/{network}_{model}_{config}_{strategy}.log`
- **Total jobs**: 1,192 across all phases (~20 days continuous on 3g.40gb GPU)

### Check progress
```bash
python3 scripts/sync_results.py          # summary table to terminal
python3 scripts/sync_results.py --csv    # also writes results/all_results.csv
python3 scripts/sync_results.py --watch 300  # refresh every 5 min
```

---

## LargeST Integration

LargeST lives at `external/LargeST/`. It was downloaded (not git-cloned) to allow
patching of upstream bugs.

### Patches applied to LargeST source (DO NOT REVERT)

| File | Bug | Fix |
|------|-----|-----|
| `src/models/lstm.py` | Used `t` (seq_len=9) instead of `self.horizon=6` in reshape | Use `self.horizon` |
| `src/models/dcrnn.py` | Same reshape bug | Use `self.horizon` |
| `src/models/stgode.py` | Hardcoded `temporal_dim=12` in STGCNBlock | Parametrised via `seq_len` |
| `src/models/dgcrn.py` | `temp2.repeat(self.horizon)` should use `tod.shape[-1]`; squeeze bug with F=1 | Fixed both |

### How the runner works
`src/models/spatial/runner.py` → `LargeSTRunner`:
1. Calls `_ensure_largeST_on_path()` which pre-registers `src.base.*` and `src.utils.*`
   in `sys.modules` to avoid conflict with our own `src/` package
2. Instantiates each model with correct kwargs (each model has different constructor signature)
3. Wraps `forward(batch)` to accept our `(B, L_in, N, C_in)` format
4. Normalises all outputs to `(B, L_out, N)`

---

## The JupyterHub

- **URL**: `https://jupyterhub.igneus.cfdata.org/user/abae/server2/lab`
- **Pod name**: `jovyan@jupyterhub-abae-server2`
- **Project path**: `~/volume/nr-benchmarking/`
- **Recommended server**: `BI Data Science - GPU nvidia.com/mig-3g.40gb`
  (48 CPU, 128GB RAM, 40GB GPU)
- **SSH**: Port 22 is closed. Cannot SSH in directly.

### How to transfer files TO the JupyterHub

SSH is closed. The working method is a reverse tunnel from the Mac:

```bash
# On Mac — authenticate (opens browser, one-time per session)
cloudflared access login jupyterhub.igneus.cfdata.org

# On Mac — serve files and create a public tunnel
cd /tmp && python3 -m http.server 8765 &
cloudflared tunnel --url http://localhost:8765 --protocol http2 --no-autoupdate &
# Wait ~10s, note the URL: https://random-words.trycloudflare.com

# On JupyterHub terminal — download
wget https://random-words.trycloudflare.com/yourfile
```

**Important notes:**
- Must use `--protocol http2` — default QUIC is blocked on the Cloudflare office network
- Some Cloudflare edge IPs (`198.41.200.*`) are unreachable; retry until you get `198.41.192.*`
- Tunnel URL is temporary and changes every session
- The update script `apply_update.py` (base64-encoded source files) is the standard
  way to push code changes without re-transferring the full 261MB dataset

### If packages are lost after server restart
```bash
cd ~/volume/nr-benchmarking && bash scripts/setup_jupyterhub.sh
```

---

## Key Design Decisions (Do Not Change Without Discussion)

1. **Metric stratification is per (timestep, link)**, not per window. A single window
   contributes independently to recurrent AND NR metrics.

2. **NR sample selection for training strategies** uses `full_nr.any()` in the target
   window (window-level binary) — not per-timestep causal label.

3. **Loss weighting** uses `causal_fixed` (not `full_nr`) so only confirmed NR steps
   are upweighted — maintains real-time consistency.

4. **Time features (TOD, DOW) must appear at channels 1 and 2**, immediately after
   speed, because DGCRN and D2STGNN hardcode reads of channels 1 and 2 as
   position encodings. The `FeatureBuilder._build()` method enforces this order.

5. **D2STGNN outputs `seq_len` steps** (not `horizon`). The runner truncates to
   `self.L_out` steps. This is intentional — D2STGNN was designed for
   `seq_len == horizon`, we adapt it.

6. **DCRNN and DGCRN require a `target`/`label` tensor** at forward time due to
   curriculum learning. The runner passes a dummy zeros tensor during inference/eval.
   DGCRN also requires `task_level=L_out` and `batches_seen=0`.

7. **Result filenames are canonical**: `{model}_{feature_config}_{strategy}.json`.
   Any deviation breaks the skip-if-exists logic in `run_sweep.py`.

---

## Config.yaml Key Parameters

```yaml
data:
  input_len: 9          # 45-min lookback
  output_len: 6         # 30-min forecast
  train_frac: 0.70
  val_frac: 0.15
  min_dur_steps: 3      # causal confirmation window (3 × 5min = 15min)
  batch_size: 64

training:
  lr: 0.001
  epochs: 100
  patience: 15          # early stopping on val MAE
  nr_loss_weight: ???   # set by tune_strategies.py
  finetune_lr_multiplier: ???  # set by tune_strategies.py
  device: "cuda"        # change to "mps" for Apple Silicon local testing
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: src.base.model` | LargeST __init__.py missing | `touch external/LargeST/src/base/__init__.py` etc., or run `setup_jupyterhub.sh` |
| `N=1431 instead of 228` | Column alignment bug | Fixed in `loader.py:_align_common_index` — do not re-introduce old reindex line |
| `websocket: bad handshake` on cloudflared SSH | SSH not enabled for this JupyterHub | Use reverse tunnel method instead |
| `results not skipped on restart` | Wrong filename saved | Must match `{model}_{feature_config}_{strategy}.json` exactly |
| DGCRN `IndexError: index 6 out of bounds` | `task_level` defaults to 12 | Runner passes `task_level=L_out` |
| D2STGNN `KeyError: seq_len` | model_args missing seq_len | Runner includes `"seq_len": L_in` in model_args |
| STGODE einsum mismatch | Hardcoded `temporal_dim=12` | Fixed in stgode.py — STGCNBlock now takes `seq_len` param |
| `Address already in use` on port 8765 | Old HTTP server still running | `pkill -f "http.server 8765"` |
| Sweep re-runs everything | result files have wrong names | Check files match canonical pattern; delete wrong-named ones |
```

---

## Local Development (Mac)

Non-DL baselines run locally without GPU:
```bash
cd /Users/abae/Documents/nr-benchmarking
python3 experiments/run_baselines.py  # LO, HA, LinearAR, XGBoost
```

DL + LargeST pipeline smoke test (requires torch):
```bash
python3 scripts/test_dl_pipeline.py --device mps --B 4
```
Expected: 14/14 PASS, all output shape `(4, 6, 228)`.

---

## Paper Context

**Venue**: Transportation Research Part C: Emerging Technologies  
**Type**: Empirical benchmarking / robustness analysis  
**Not**: A new model paper. Do not propose or implement new architectures.

**Central finding** (expected):
- NR MAE is 2–5× higher than recurrent MAE across all models
- Model rankings reverse under NR conditions (simpler ≥ complex)
- Unobserved onset MAE >> confirmed NR MAE — onset is the bottleneck
- Richer inputs and training strategies help modestly at best
- Findings replicate across both networks

**Prior paper**: The ensemble labeling TRC paper produced the NR labels used here.
The ITS paper made the conceptual argument that aggregate benchmarks hide NR failure.
This paper is the empirical proof.
