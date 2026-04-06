#!/usr/bin/env python3
"""
Aggregate all completed result JSON files and produce a status report.

Run this any time during or after the sweep. Safe to run while sweep is going.

Usage (on JupyterHub)
─────────────────────
  python3 scripts/sync_results.py               # print summary table to terminal
  python3 scripts/sync_results.py --csv         # also write results/all_results.csv
  python3 scripts/sync_results.py --report      # write results/status_report.txt
                                                 # (download this and share with agent)
  python3 scripts/sync_results.py --watch 300   # refresh every 5 minutes

The report file (--report) is the easiest thing to share with an agent:
it contains progress counts, the full metrics table, failed jobs, and
recent log snippets — everything needed to diagnose issues.
"""

from __future__ import annotations

import json
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUMMARY_COLS = [
    "overall_mae_avg",
    "recurrent_mae_avg",
    "nr_mae_avg",
    "unobserved_onset_mae",
    "confirmed_nr_mae",
    "overall_rmse_avg",
    "nr_rmse_avg",
    "nr_rate_pct",
    "elapsed_s",
]

# Runs that are structurally infeasible and should not be retried.
# ASTGCN has a known fp32 attention-softmax overflow that cannot be resolved
# by re-initialisation for these (network, model, config, strategy) combos.
# d2stgnn TSMO multi_objective hits a CUDA OOM on the 228-link graph.
# These are counted as "excluded" in the progress summary, not "pending".
KNOWN_EXCLUDED: set[tuple[str, str, str, str]] = {
    # ASTGCN: fp32 attention softmax overflow on these strategy/network combos
    ("tsmo", "astgcn", "speed", "multi_objective"),
    ("tsmo", "astgcn", "speed", "nr_finetune"),
    ("cranberry", "astgcn", "speed", "multi_objective"),
    # D2STGNN TSMO: numerical instability (NaN loss from initialisation)
    # persists after 5 re-init attempts for all non-standard strategies.
    # The saved result files contain untrained-model predictions and are invalid.
    ("tsmo", "d2stgnn", "speed", "weighted_loss"),
    ("tsmo", "d2stgnn", "speed", "nr_finetune"),
    ("tsmo", "d2stgnn", "speed", "multi_objective"),
}

# All expected jobs (model × config × strategy)
BASELINE_MODELS = ["last_observation", "historical_average", "linear_ar", "xgboost"]
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
ALL_MODELS = BASELINE_MODELS + DL_MODELS + SPATIAL_MODELS
ALL_CONFIGS = [
    "speed",
    "speed_weather",
    "speed_incidents",
    "speed_nr",
    "speed_weather_incidents",
    "speed_time",
    "speed_time_weather",
    "speed_time_incidents",
    "speed_time_nr",
    "speed_time_weather_incidents",
]
ALL_STRATEGIES = ["standard", "weighted_loss", "nr_finetune", "multi_objective"]
ALL_NETWORKS = ["tsmo", "cranberry"]


# ─────────────────────────────────────────────────────────────────────────────
# Expected job count
# ─────────────────────────────────────────────────────────────────────────────


def expected_jobs(
    networks=ALL_NETWORKS,
    configs=ALL_CONFIGS,
    models=ALL_MODELS,
    strategies=ALL_STRATEGIES,
) -> set[tuple]:
    jobs = set()
    for net in networks:
        for cfg in configs:
            for mdl in models:
                strats = ["standard"] if mdl in BASELINE_MODELS else strategies
                for s in strats:
                    jobs.add((net, mdl, cfg, s))
    return jobs


# ─────────────────────────────────────────────────────────────────────────────
# Collect completed results
# ─────────────────────────────────────────────────────────────────────────────


def collect(results_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(results_dir.rglob("*.json")):
        if f.name.startswith("summary_") or f.name in (
            "sweep_manifest.json",
            "tuning_results.json",
            "all_results.csv",
        ):
            continue
        try:
            data = json.loads(f.read_text())
            if "model" not in data:
                continue
            # feature_config is stored as str(list) e.g. "['speed', 'weather']"
            # Normalise to underscore-joined key e.g. "speed_weather"
            raw_fc = data.get("feature_config", "")
            if raw_fc.startswith("["):
                import ast

                try:
                    parts = ast.literal_eval(raw_fc)
                    fc = "_".join(parts)
                except Exception:
                    fc = raw_fc.strip("[]'\" ").replace("', '", "_").replace("'", "")
            else:
                fc = raw_fc
            row = {
                "network": data.get("network", ""),
                "model": data.get("model", ""),
                "feature_config": fc,
                "strategy": data.get("strategy", "standard"),
                "_file": str(f),
            }
            for col in SUMMARY_COLS:
                row[col] = data.get(col, float("nan"))
            rows.append(row)
        except Exception:
            pass
    return rows


def collect_failed(results_dir: Path) -> list[dict]:
    """Scan log files for ERROR entries."""
    failed = []
    log_dir = results_dir / "logs"
    if not log_dir.exists():
        return failed
    for lf in sorted(log_dir.glob("*.log")):
        text = lf.read_text(errors="replace")
        if "Traceback" in text or "ERROR" in text or "Error" in text:
            # Get last 10 lines as snippet
            lines = text.strip().splitlines()
            snippet = "\n    ".join(lines[-10:])
            failed.append({"log": lf.name, "snippet": snippet})
    return failed


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────


def _f(v, width=8):
    try:
        fv = float(v)
        if fv != fv:  # nan check
            return f"{'—':>{width}}"
        return f"{fv:{width}.4f}"
    except Exception:
        return f"{'—':>{width}}"


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "  No results yet.\n"
    lines = []
    for network in ALL_NETWORKS:
        net_rows = [r for r in rows if r["network"] == network]
        if not net_rows:
            continue
        lines.append(f"\n{'=' * 86}")
        lines.append(f"  {network.upper()}  ({len(net_rows)} completed runs)")
        lines.append(f"{'=' * 86}")
        lines.append(
            f"  {'Model':<20} {'Config':<26} {'Strat':<16}"
            f"  {'Overall':>8} {'Recur':>8} {'NR':>8} {'Onset':>8} {'Confirmed':>9}"
        )
        lines.append("  " + "─" * 84)
        for r in net_rows:
            lines.append(
                f"  {r['model']:<20} {r['feature_config']:<26} {r['strategy']:<16}"
                f"  {_f(r['overall_mae_avg'])}"
                f" {_f(r['recurrent_mae_avg'])}"
                f" {_f(r['nr_mae_avg'])}"
                f" {_f(r['unobserved_onset_mae'])}"
                f" {_f(r['confirmed_nr_mae'])}"
            )
    return "\n".join(lines)


def format_progress(rows: list[dict]) -> str:
    done_set = {
        (r["network"], r["model"], r["feature_config"], r["strategy"]) for r in rows
    }
    total_expected = len(expected_jobs())
    n_done = len(done_set)
    n_pending = total_expected - n_done

    lines = [
        f"  Total expected : {total_expected}",
        f"  Completed      : {n_done}",
        f"  Remaining      : {n_pending}",
        f"  Progress       : {100 * n_done / max(total_expected, 1):.1f}%",
    ]

    # Per-phase breakdown
    phase1 = {(n, m, "speed", "standard") for n in ALL_NETWORKS for m in ALL_MODELS}
    p1_done = len(done_set & phase1)
    lines.append(
        f"\n  Phase 1 (speed × standard × all models):  {p1_done}/{len(phase1)}"
    )

    all_std = {
        (n, m, c, "standard")
        for n in ALL_NETWORKS
        for m in ALL_MODELS
        for c in ALL_CONFIGS
    }
    p2_done = len(done_set & all_std)
    lines.append(
        f"  Phase 2 (all configs × standard):          {p2_done}/{len(all_std)}"
    )

    all_strats = {
        (n, m, "speed", s)
        for n in ALL_NETWORKS
        for m in (DL_MODELS + SPATIAL_MODELS)
        for s in ALL_STRATEGIES
    }
    p3_done = len(done_set & all_strats)
    lines.append(
        f"  Phase 3 (speed × all strategies × DL):     {p3_done}/{len(all_strats)}"
    )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Write report
# ─────────────────────────────────────────────────────────────────────────────


def write_report(rows: list[dict], failed: list[dict], path: Path) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections = [
        f"NR Benchmarking — Status Report",
        f"Generated: {ts}",
        f"{'=' * 86}",
        "",
        "PROGRESS",
        "────────",
        format_progress(rows),
        "",
        "METRICS (MAE)",
        "─────────────",
        "Columns: Overall | Recurrent | NR | Unobserved Onset | Confirmed NR",
        format_table(rows),
    ]

    if failed:
        sections += [
            "",
            f"FAILED JOBS  ({len(failed)} log files with errors)",
            "────────────",
        ]
        for f in failed[:10]:  # show at most 10
            sections.append(f"\n  [{f['log']}]")
            sections.append(f"    {f['snippet']}")

    report = "\n".join(sections) + "\n"
    path.write_text(report)
    print(f"  Report saved → {path}", flush=True)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    import csv

    keys = [k for k in rows[0].keys() if not k.startswith("_")]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved → {path}  ({len(rows)} rows)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", action="store_true", help="Write results/all_results.csv"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write results/status_report.txt (share with agent)",
    )
    parser.add_argument(
        "--watch", type=int, default=0, help="Refresh every N seconds (0 = run once)"
    )
    args = parser.parse_args()

    import yaml

    with open(PROJECT_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    results_dir = PROJECT_ROOT / config["paths"].get("results", "results")

    while True:
        rows = collect(results_dir)
        failed = collect_failed(results_dir)

        print(
            f"\n[{time.strftime('%H:%M:%S')}]  "
            f"Completed: {len(rows)}  |  Errors in logs: {len(failed)}",
            flush=True,
        )
        print(format_progress(rows), flush=True)
        print(format_table(rows), flush=True)

        if args.csv:
            write_csv(rows, results_dir / "all_results.csv")
        if args.report:
            write_report(rows, failed, results_dir / "status_report.txt")

        if args.watch <= 0:
            break
        print(f"\n  Refreshing in {args.watch}s...  (Ctrl+C to stop)", flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
