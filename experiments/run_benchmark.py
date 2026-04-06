#!/usr/bin/env python
"""
Core benchmark runner.

Trains and evaluates all models on one network + one feature configuration.
Results (metrics dict) are saved as JSON to results/{network}/{run_id}.json.

Usage
─────
# Single model, single config
python experiments/run_benchmark.py \\
    --network tsmo \\
    --feature_config speed \\
    --model lstm \\
    --strategy standard

# All baseline models, speed-only
python experiments/run_benchmark.py \\
    --network tsmo \\
    --feature_config speed \\
    --model all_baselines

# All spatial models
python experiments/run_benchmark.py \\
    --network cranberry \\
    --feature_config speed \\
    --model all_spatial

# Full sweep (all models × all feature configs) — submit as separate jobs
python experiments/run_benchmark.py --network tsmo --feature_config speed --model all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.data.loader import load_network
from src.data.adjacency import load_or_build_adjacency
from src.data.dataset import make_datasets
from src.evaluation.metrics import compute_metrics
from src.evaluation.transitions import full_transition_report

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

STATISTICAL_MODELS = ["last_observation", "historical_average", "linear_ar"]
ML_MODELS = ["xgboost"]
DL_MODELS = ["lstm", "transformer"]
SPATIAL_MODELS = [
    "hl",
    "lstm_st",
    "dcrnn",
    "agcrn",
    "stgcn",
    "gwnet",
    "astgcn",
    "sttn",
    "stgode",
    "dstagnn",
    "dgcrn",
    "d2stgnn",
]
ALL_MODELS = STATISTICAL_MODELS + ML_MODELS + DL_MODELS + SPATIAL_MODELS

MODEL_GROUPS = {
    "all_baselines": STATISTICAL_MODELS + ML_MODELS,
    "all_dl": DL_MODELS,
    "all_spatial": SPATIAL_MODELS,
    "all": ALL_MODELS,
}


# ─────────────────────────────────────────────────────────────────────────────
# Builder functions for each model category
# ─────────────────────────────────────────────────────────────────────────────


def build_model(model_name: str, nd, fb, config: dict, adj_mx: np.ndarray):
    """Instantiate a model by name."""
    data_cfg = config.get("data", {})
    L_in = data_cfg.get("input_len", 9)
    L_out = data_cfg.get("output_len", 6)

    # ── Statistical ──────────────────────────────────────────────────────────
    if model_name == "last_observation":
        from src.models.statistical.last_observation import LastObservation

        return LastObservation(nd.speed_mean, nd.speed_std, L_out)

    if model_name == "historical_average":
        from src.models.statistical.historical_avg import HistoricalAverage

        return HistoricalAverage(
            nd.speed_mean, nd.speed_std, nd.timestamps, input_len=L_in, output_len=L_out
        )

    if model_name == "linear_ar":
        from src.models.statistical.linear_ar import LinearAR

        ar_cfg = config.get("models", {}).get("linear_ar", {})
        return LinearAR(
            N=nd.N,
            speed_mean=nd.speed_mean,
            speed_std=nd.speed_std,
            input_len=L_in,
            output_len=L_out,
            alpha=ar_cfg.get("alpha", 1.0),
        )

    # ── ML ───────────────────────────────────────────────────────────────────
    if model_name == "xgboost":
        from src.models.ml.xgboost_model import XGBoostModel

        xgb_cfg = config.get("models", {}).get("xgboost", {})
        return XGBoostModel(
            N=nd.N,
            C_in=fb.C,
            speed_mean=nd.speed_mean,
            speed_std=nd.speed_std,
            input_len=L_in,
            output_len=L_out,
            **xgb_cfg,
        )

    # ── DL (per-link, shared weights) ─────────────────────────────────────────
    if model_name == "lstm":
        from src.models.dl.lstm import Seq2SeqLSTM

        lstm_cfg = config.get("models", {}).get("lstm", {})
        return Seq2SeqLSTM(
            C_in=fb.C,
            output_len=L_out,
            hidden_dim=lstm_cfg.get("hidden_dim", 128),
            num_layers=lstm_cfg.get("num_layers", 2),
            dropout=lstm_cfg.get("dropout", 0.1),
        )

    if model_name == "transformer":
        from src.models.dl.transformer import Seq2SeqTransformer

        tr_cfg = config.get("models", {}).get("transformer", {})
        return Seq2SeqTransformer(
            C_in=fb.C,
            output_len=L_out,
            d_model=tr_cfg.get("d_model", 128),
            nhead=tr_cfg.get("nhead", 8),
            num_encoder_layers=tr_cfg.get("num_encoder_layers", 3),
            num_decoder_layers=tr_cfg.get("num_decoder_layers", 3),
            dim_feedforward=tr_cfg.get("dim_feedforward", 512),
            dropout=tr_cfg.get("dropout", 0.1),
        )

    # ── Spatial (LargeST) ─────────────────────────────────────────────────────
    if model_name in SPATIAL_MODELS:
        from src.models.spatial.runner import build_largeST_model

        return build_largeST_model(model_name, nd, fb, config, adj_mx)

    raise ValueError(f"Unknown model '{model_name}'")


# ─────────────────────────────────────────────────────────────────────────────
# Training + evaluation
# ─────────────────────────────────────────────────────────────────────────────


def run_model(
    model_name: str,
    nd,
    fb,
    train_ds,
    val_ds,
    test_ds,
    config: dict,
    adj_mx: np.ndarray,
    strategy: str = "standard",
    out_dir: Path = Path("results"),
    run_id: str = "",
    feature_config: str = "",  # canonical key e.g. "speed_weather" (not the list)
) -> dict:
    """Train, evaluate, and return result dict for one model."""
    print(f"\n{'─' * 65}")
    print(f"  Model: {model_name.upper()}  |  Strategy: {strategy}")
    print(f"{'─' * 65}")

    data_cfg = config.get("data", {})
    L_out = data_cfg.get("output_len", 6)
    bs = data_cfg.get("batch_size", 128)
    nw = min(4, data_cfg.get("num_workers", 4))

    # Start from global training config; allow model-specific lr override
    train_cfg = dict(config.get("training", {}))
    model_lr = config.get("models", {}).get(model_name, {}).get("lr")
    if model_lr is not None:
        train_cfg["lr"] = model_lr
        print(f"  Using model-specific lr={model_lr} for {model_name}", flush=True)

    # pin_memory only valid with num_workers > 0
    pm = nw > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=pm,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm
    )

    t0 = time.time()
    model = build_model(model_name, nd, fb, config, adj_mx)

    # Use the canonical config key (e.g. "speed_weather") not str(list)
    cfg_key = feature_config if feature_config else "_".join(fb.config)
    result = {
        "model": model_name,
        "network": nd.network,
        "feature_config": cfg_key,
        "strategy": strategy,
        "run_id": run_id,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }

    # ── Statistical / ML models ───────────────────────────────────────────────
    if model_name in STATISTICAL_MODELS + ML_MODELS:
        if hasattr(model, "fit"):
            model.fit(train_loader, nd)

        pred, target, full_nr, regime = model.predict_dataset(test_loader)

    # ── DL / Spatial models ───────────────────────────────────────────────────
    else:
        from src.training.trainer import Trainer

        # HL has only 1 dummy parameter and no trainable forward path —
        # skip fit() entirely and just run prediction directly.
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        is_trivial = n_trainable < 10

        ckpt_dir = out_dir / nd.network / f"{model_name}_{strategy}_{run_id}"
        trainer = Trainer(
            model, nd, train_cfg, strategy=strategy, checkpoint_dir=ckpt_dir
        )
        if not is_trivial:
            trainer.fit(train_loader, val_loader)
        else:
            print(
                f"  Skipping training — model has {n_trainable} trainable params.",
                flush=True,
            )

        pred, target, full_nr, regime = trainer.predict(test_loader)
        result["train_history"] = {
            k: [round(v, 4) for v in vs if not np.isnan(v)]
            for k, vs in trainer.history.items()
        }

    # ── Metrics ───────────────────────────────────────────────────────────────
    core_metrics = compute_metrics(pred, target, full_nr)
    result.update(core_metrics)

    # Transition / anomaly prediction analysis
    # v_rec: use median across time for each link (T, N) → (N,)
    v_rec_median = np.nanmedian(nd.v_rec, axis=0)
    trans_metrics = full_transition_report(pred, target, full_nr, regime, v_rec_median)
    result.update(trans_metrics)

    elapsed = round(time.time() - t0, 1)
    result["elapsed_s"] = elapsed

    print(
        f"  overall MAE={core_metrics.get('overall_mae_avg', 'nan'):.4f}  "
        f"NR MAE={core_metrics.get('nr_mae_avg', 'nan'):.4f}  "
        f"rec MAE={core_metrics.get('recurrent_mae_avg', 'nan'):.4f}  "
        f"({elapsed}s)"
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    # cfg_key is already the canonical string (e.g. "speed_weather")
    fname = f"{model_name}_{cfg_key}_{strategy}.json"
    with open(out_dir / nd.network / fname, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NR Benchmarking — core experiment runner"
    )
    parser.add_argument("--network", required=True, choices=["tsmo", "cranberry"])
    parser.add_argument(
        "--feature_config",
        required=True,
        help="Feature config key from config.yaml (e.g. speed, speed_weather)",
    )
    parser.add_argument(
        "--model",
        default="all",
        help="Model name or group (e.g. lstm, all, all_spatial)",
    )
    parser.add_argument(
        "--strategy",
        default="standard",
        choices=["standard", "weighted_loss", "nr_finetune", "multi_objective"],
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_id", default="")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed or config.get("training", {}).get("seed", 2024)
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_cfg = config["data"]
    L_in = data_cfg.get("input_len", 9)
    L_out = data_cfg.get("output_len", 6)
    results_dir = PROJECT_ROOT / config["paths"].get("results", "results")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(
        f"\n[Benchmark] Network={args.network}  Feature={args.feature_config}"
        f"  Model={args.model}  Strategy={args.strategy}"
    )

    nd = load_network(args.network, args.config)
    print(nd.summary())

    # ── Feature config ────────────────────────────────────────────────────────
    feat_map = config.get("feature_configs", {})
    if args.feature_config not in feat_map:
        raise ValueError(
            f"Unknown feature config '{args.feature_config}'. "
            f"Available: {list(feat_map.keys())}"
        )
    feat_keys = feat_map[args.feature_config]
    train_ds, val_ds, test_ds, fb = make_datasets(nd, feat_keys, L_in, L_out)

    # ── Adjacency matrix ─────────────────────────────────────────────────────
    pcfg = config["paths"][args.network]
    adj_cache = PROJECT_ROOT / pcfg.get("adjacency", f"data/{args.network}/adj_mx.npy")
    A, A_fw, A_bw = load_or_build_adjacency(
        geojson_path=PROJECT_ROOT / pcfg["geojson"],
        upstream_path=PROJECT_ROOT / pcfg["upstream"],
        links=nd.links,
        cache_path=adj_cache,
        weight="binary",
    )

    # ── Model list ────────────────────────────────────────────────────────────
    model_arg = args.model.lower()
    if model_arg in MODEL_GROUPS:
        models_to_run = MODEL_GROUPS[model_arg]
    elif model_arg in ALL_MODELS:
        models_to_run = [model_arg]
    else:
        raise ValueError(
            f"Unknown model '{model_arg}'. Options: {ALL_MODELS + list(MODEL_GROUPS)}"
        )

    # ── Run ───────────────────────────────────────────────────────────────────
    all_results = []
    for mname in models_to_run:
        try:
            res = run_model(
                model_name=mname,
                nd=nd,
                fb=fb,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                config=config,
                adj_mx=A,
                strategy=args.strategy,
                out_dir=results_dir,
                run_id=args.run_id,
                feature_config=args.feature_config,  # canonical key
            )
            all_results.append(res)
        except Exception as e:
            print(f"  ERROR running {mname}: {e}")
            import traceback

            traceback.print_exc()

    # Summary table
    print(f"\n{'═' * 65}")
    print(f"  SUMMARY  {args.network.upper()} / {args.feature_config}")
    print(f"{'═' * 65}")
    print(
        f"  {'Model':<20} {'overall':>8} {'recurrent':>10} {'NR':>8}  "
        f"{'onset':>8}  {'confirmed':>10}"
    )
    print(f"  {'─' * 20} {'─' * 8} {'─' * 10} {'─' * 8}  {'─' * 8}  {'─' * 10}")
    for r in all_results:
        print(
            f"  {r['model']:<20} "
            f"{r.get('overall_mae_avg', float('nan')):8.4f} "
            f"{r.get('recurrent_mae_avg', float('nan')):10.4f} "
            f"{r.get('nr_mae_avg', float('nan')):8.4f}  "
            f"{r.get('unobserved_onset_mae', float('nan')):8.4f}  "
            f"{r.get('confirmed_nr_mae', float('nan')):10.4f}"
        )

    # Save combined results
    combined_path = (
        results_dir
        / args.network
        / f"summary_{args.feature_config}_{args.strategy}_{args.run_id}.json"
    ).resolve()
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved combined results → {combined_path}")


if __name__ == "__main__":
    main()
