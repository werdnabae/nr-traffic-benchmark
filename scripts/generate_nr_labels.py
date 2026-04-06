#!/usr/bin/env python
"""
Generate full NR labels and v_rec arrays for both networks.

This is a one-time preprocessing step that must be run before any model
training.  It runs the ensemble labeler from the Anomaly Labeling project on
the *full* time series (train + val + test) using thresholds calibrated on the
first `calib_frac` of each network — the same protocol as the prior paper.

Outputs written to:
  data/{network}/nr_labels_full.parquet   — binary NR labels  (T × N, float32)
  data/{network}/v_rec_full.parquet       — v_rec speed       (T × N, float32)

Usage
-----
  python scripts/generate_nr_labels.py
  python scripts/generate_nr_labels.py --network tsmo
  python scripts/generate_nr_labels.py --network cranberry --force
  python scripts/generate_nr_labels.py --config path/to/config.yaml
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve(base: Path, rel: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT."""
    p = Path(rel)
    return p if p.is_absolute() else (base / p).resolve()


def add_anomaly_labeling_to_path(config: dict) -> Path:
    al_root = resolve(PROJECT_ROOT, config["paths"]["anomaly_labeling"])
    if not al_root.exists():
        raise FileNotFoundError(
            f"Anomaly Labeling project not found at:\n  {al_root}\n"
            "Update the 'anomaly_labeling' path in config.yaml."
        )
    if str(al_root) not in sys.path:
        sys.path.insert(0, str(al_root))
    return al_root


# ─────────────────────────────────────────────────────────────────────────────
# Core generation function
# ─────────────────────────────────────────────────────────────────────────────


def generate_for_network(
    network: str,
    config: dict,
    al_root: Path,
    force: bool = False,
) -> None:
    """Generate NR labels and v_rec for a single network."""

    # Late import — requires Anomaly Labeling on sys.path
    from src.ensemble_labeling import (
        load_network_data,
        get_geojson_path,
        select_tmcs,
        compute_session_ids,
        temporal_split,
        compute_free_flow_speed,
        compute_slowdown_speed,
        build_upstream_neighbors,
        four_point_slopes,
        frozen_thresholds,
        frozen_confirmation_thresholds,
        run_ensemble_labeler,
        INTERVAL_MIN,
        NETWORK_PARAMS,
        ALPHA_VREC,
        RECOVERY_SOFT_C_SHORT,
        RECOVERY_SOFT_C_LONG,
        HARD_SPEED_FACTOR,
        HARD_RECOVERY_SHORT,
        HARD_RECOVERY_LONG,
        MAX_GAP_MIN,
    )

    print(f"\n{'=' * 65}")
    print(f"  Generating NR labels: {network.upper()}")
    print(f"{'=' * 65}")

    # ── output paths ─────────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / "data" / network
    out_dir.mkdir(parents=True, exist_ok=True)
    nr_out = out_dir / "nr_labels_full.parquet"
    vrec_out = out_dir / "v_rec_full.parquet"

    if nr_out.exists() and vrec_out.exists() and not force:
        print(f"  Already exists at {out_dir}  (use --force to regenerate)")
        return

    # ── load raw data ─────────────────────────────────────────────────────────
    data_dir = al_root / "data" / network
    geo_path = get_geojson_path(data_dir, network)
    speed_raw, incident_raw, upstream_raw = load_network_data(data_dir, network)

    links = select_tmcs(speed_raw, geo_path, network)
    speed_raw = speed_raw[links].sort_index()
    incident_raw = incident_raw.reindex(columns=links, index=speed_raw.index).fillna(0)

    T, N = speed_raw.shape
    print(f"  Links: {N}  |  Timesteps: {T}")
    print(f"  Period: {speed_raw.index[0]}  →  {speed_raw.index[-1]}")

    # ── calibration split ─────────────────────────────────────────────────────
    calib_frac = config["data"]["calib_frac"]
    c_mask, _, n_calib, n_sess = temporal_split(speed_raw, calib_frac)
    calib_speed = speed_raw[c_mask]
    print(
        f"  Sessions: {n_sess} total | calibration: {n_calib} ({calib_frac * 100:.0f}%)"
    )

    # ── calibrated thresholds (broadcast to full time series) ─────────────────
    cfg_net = NETWORK_PARAMS[network.lower()]
    nr_cfg = config["nr_labeling"]
    snd_win = cfg_net["snd_window_min"]

    ffs = compute_free_flow_speed(calib_speed)

    snd_nr_thr = frozen_thresholds(
        calib_speed, speed_raw, "dow_timebin", nr_cfg["snd_c"], "minus", snd_win
    )
    snd_r_thr = frozen_thresholds(
        calib_speed, speed_raw, "dow_timebin", cfg_net["snd_c_report"], "minus", snd_win
    )
    conf_all = frozen_confirmation_thresholds(
        calib_speed, [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.84]
    )
    rec_s_thr = frozen_thresholds(
        calib_speed, speed_raw, "dow_time", RECOVERY_SOFT_C_SHORT, "minus"
    )
    rec_l_thr = frozen_thresholds(
        calib_speed, speed_raw, "dow_time", RECOVERY_SOFT_C_LONG, "minus"
    )
    # v_rec = median − ALPHA_VREC × IQR  (recurrent lower-bound speed)
    vrec_thr = frozen_thresholds(
        calib_speed, speed_raw, "dow_time", ALPHA_VREC, "minus"
    )

    # ── real-time features on the FULL time series ────────────────────────────
    print("  Computing real-time features on full time series...")
    up03 = build_upstream_neighbors(speed_raw, geo_path, upstream_raw)

    slow_full = compute_slowdown_speed(speed_raw, up03)
    nwg_full = four_point_slopes(speed_raw, weighted=False)
    wg_full = four_point_slopes(speed_raw, weighted=True)

    slow_calib = compute_slowdown_speed(calib_speed, up03)
    nwg_calib = four_point_slopes(calib_speed, weighted=False)
    wg_calib = four_point_slopes(calib_speed, weighted=True)

    slow_thr = frozen_thresholds(
        slow_calib, speed_raw, "dow_timebin", cfg_net["slowdown_c"], "plus", snd_win
    )
    nwg_thr = frozen_thresholds(
        nwg_calib, speed_raw, "dow_time", nr_cfg["grad_c"], "minus"
    )
    wg_thr = frozen_thresholds(
        wg_calib, speed_raw, "dow_time", nr_cfg["grad_c"], "minus"
    )

    def _closest_conf(f: float) -> pd.Series:
        f_r = round(f, 2)
        if f_r in conf_all:
            return conf_all[f_r]
        closest = min(conf_all.keys(), key=lambda k: abs(k - f))
        return conf_all[closest]

    conf_nr = _closest_conf(nr_cfg["conf_f"])
    conf_r = _closest_conf(cfg_net["confirmation_factor_report"])

    # ── run ensemble labeler on FULL time series ──────────────────────────────
    print("  Running ensemble labeler on full time series...")
    nr_labels = run_ensemble_labeler(
        speed_df=speed_raw,
        incident_df=incident_raw,
        slowdown_speed_df=slow_full,
        snd_threshold_nonreport=snd_nr_thr,
        snd_threshold_report=snd_r_thr,
        slowdown_threshold=slow_thr,
        nw_gradient_df=nwg_full,
        w_gradient_df=wg_full,
        nw_gradient_threshold=nwg_thr,
        w_gradient_threshold=wg_thr,
        confirmation_threshold_nonreport=conf_nr,
        confirmation_threshold_report=conf_r,
        recovery_soft_short=rec_s_thr,
        recovery_soft_long=rec_l_thr,
        ffs=ffs,
        hard_speed_factor=HARD_SPEED_FACTOR,
        hard_rec_short=HARD_RECOVERY_SHORT,
        hard_rec_long=HARD_RECOVERY_LONG,
        min_nonreport_min=nr_cfg["min_dur"],
        max_gap_min=MAX_GAP_MIN,
        interval_min=INTERVAL_MIN,
        verbose=True,
    )

    # ── save ──────────────────────────────────────────────────────────────────
    nr_labels.astype("float32").to_parquet(nr_out)
    vrec_thr.astype("float32").to_parquet(vrec_out)

    nr_rate = float((nr_labels == 1).values.mean() * 100)
    print(f"\n  NR rate (full series): {nr_rate:.2f}%")
    print(f"  Saved NR labels : {nr_out}")
    print(f"  Saved v_rec     : {vrec_out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate full NR labels for the NR Benchmarking project."
    )
    parser.add_argument(
        "--network",
        choices=["tsmo", "cranberry", "both"],
        default="both",
        help="Which network to process (default: both)",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate labels even if output files already exist",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    al_root = add_anomaly_labeling_to_path(config)
    networks = ["tsmo", "cranberry"] if args.network == "both" else [args.network]

    for network in networks:
        generate_for_network(network, config, al_root, force=args.force)

    print("\nDone. NR labels ready.")


if __name__ == "__main__":
    main()
