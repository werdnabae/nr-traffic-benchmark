#!/usr/bin/env python3
"""
Standalone runner for non-DL baselines (no PyTorch required).

Models: Last Observation, Historical Average, Linear AR, XGBoost
Runs on both TSMO and Cranberry with the 'speed' feature config.

Usage
─────
# From nr-benchmarking/
python3 experiments/run_baselines.py
python3 experiments/run_baselines.py --network tsmo
python3 experiments/run_baselines.py --network tsmo --feature_config speed_weather
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_network
from src.data.numpy_iterator import make_numpy_iterators
from src.evaluation.metrics import compute_metrics
from src.evaluation.transitions import full_transition_report


# ─────────────────────────────────────────────────────────────────────────────


def run_model(model_name, model, nd, train_it, test_it, config, results_dir):
    """Fit and evaluate one model. Returns results dict."""
    data_cfg = config["data"]
    L_in = data_cfg["input_len"]
    L_out = data_cfg["output_len"]

    print(f"\n{'─' * 60}")
    print(f"  {model_name.upper()} on {nd.network.upper()}")
    print(f"{'─' * 60}")
    t0 = time.time()

    # ── Fit ──────────────────────────────────────────────────────────────────
    if hasattr(model, "fit"):
        model.fit(train_it, nd)

    # ── Predict on test set ───────────────────────────────────────────────────
    preds, targets, full_nrs, regimes = [], [], [], []
    for batch in test_it:
        if model_name in ("last_observation",):
            p = model.predict(batch)
        elif model_name == "historical_average":
            p = model.predict(batch)
        else:
            p = model.predict(batch, timestamps=nd.timestamps)
        preds.append(p)
        targets.append(batch["y_orig"])
        full_nrs.append(batch["full_nr"])
        regimes.append(batch["regime"])

    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    full_nr = np.concatenate(full_nrs, axis=0)
    regime = np.concatenate(regimes, axis=0)

    # ── Metrics ───────────────────────────────────────────────────────────────
    core = compute_metrics(pred, target, full_nr)
    v_rec_med = np.nanmedian(nd.v_rec, axis=0)  # (N,)
    trans = full_transition_report(pred, target, full_nr, regime, v_rec_med)

    elapsed = round(time.time() - t0, 1)

    result = {
        "model": model_name,
        "network": nd.network,
        "n_test": len(pred),
        "elapsed_s": elapsed,
    }
    result.update(core)
    result.update(trans)

    print(
        f"  overall_mae={core['overall_mae_avg']:.4f}  "
        f"recurrent_mae={core['recurrent_mae_avg']:.4f}  "
        f"nr_mae={core['nr_mae_avg']:.4f}  "
        f"({elapsed}s)"
    )
    print(
        f"  onset_mae={trans.get('unobserved_onset_mae', float('nan')):.4f}  "
        f"confirmed_mae={trans.get('confirmed_nr_mae', float('nan')):.4f}"
    )

    # Save
    out_dir = results_dir / nd.network
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{model_name}_speed_standard.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run non-DL baselines")
    parser.add_argument(
        "--network", choices=["tsmo", "cranberry", "both"], default="both"
    )
    parser.add_argument("--feature_config", default="speed")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    np.random.seed(config.get("training", {}).get("seed", 2024))

    data_cfg = config["data"]
    L_in = data_cfg["input_len"]
    L_out = data_cfg["output_len"]
    results_dir = PROJECT_ROOT / config["paths"].get("results", "results")

    feat_map = config["feature_configs"]
    feat_keys = feat_map[args.feature_config]

    networks = ["tsmo", "cranberry"] if args.network == "both" else [args.network]

    all_results = []

    for network in networks:
        print(f"\n{'=' * 60}")
        print(f"  Loading {network.upper()}...")
        print(f"{'=' * 60}")

        nd = load_network(network, args.config)
        print(nd.summary())

        train_it, val_it, test_it, fb = make_numpy_iterators(
            nd, feat_keys, L_in, L_out, batch_size=args.batch_size
        )

        # ── Build models ─────────────────────────────────────────────────────
        models_cfg = config.get("models", {})

        from src.models.statistical.last_observation import LastObservation

        lo = LastObservation(nd.speed_mean, nd.speed_std, L_out)

        from src.models.statistical.historical_avg import HistoricalAverage

        ha = HistoricalAverage(
            nd.speed_mean, nd.speed_std, nd.timestamps, input_len=L_in, output_len=L_out
        )

        from src.models.statistical.linear_ar import LinearAR

        ar_cfg = models_cfg.get("linear_ar", {})
        lar = LinearAR(
            N=nd.N,
            speed_mean=nd.speed_mean,
            speed_std=nd.speed_std,
            input_len=L_in,
            output_len=L_out,
            alpha=ar_cfg.get("alpha", 1.0),
        )

        from src.models.ml.xgboost_model import XGBoostModel

        xgb_cfg = models_cfg.get("xgboost", {})
        xgb = XGBoostModel(
            N=nd.N,
            C_in=fb.C,
            speed_mean=nd.speed_mean,
            speed_std=nd.speed_std,
            input_len=L_in,
            output_len=L_out,
            n_estimators=xgb_cfg.get("n_estimators", 500),
            max_depth=xgb_cfg.get("max_depth", 6),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            n_jobs=xgb_cfg.get("n_jobs", -1),
            random_state=xgb_cfg.get("random_state", 2024),
        )

        model_list = [
            ("last_observation", lo),
            ("historical_average", ha),
            ("linear_ar", lar),
            ("xgboost", xgb),
        ]

        for mname, model in model_list:
            try:
                r = run_model(mname, model, nd, train_it, test_it, config, results_dir)
                r["feature_config"] = args.feature_config
                all_results.append(r)
            except Exception as e:
                print(f"  ERROR in {mname}: {e}")
                import traceback

                traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  BASELINES SUMMARY  ({args.feature_config})")
    print(f"{'=' * 65}")
    hdr = f"  {'Model':<22} {'Net':<10} {'Overall':>8} {'Recurrent':>10} {'NR':>8} {'Onset':>8} {'Confirmed':>10}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for r in all_results:
        print(
            f"  {r['model']:<22} {r['network']:<10} "
            f"{r.get('overall_mae_avg', float('nan')):8.4f} "
            f"{r.get('recurrent_mae_avg', float('nan')):10.4f} "
            f"{r.get('nr_mae_avg', float('nan')):8.4f} "
            f"{r.get('unobserved_onset_mae', float('nan')):8.4f} "
            f"{r.get('confirmed_nr_mae', float('nan')):10.4f}"
        )

    combined_path = results_dir / f"baselines_{args.feature_config}.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved → {combined_path}")


if __name__ == "__main__":
    main()
