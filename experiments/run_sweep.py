#!/usr/bin/env python3
"""
Full benchmark sweep — runs every model × network × feature config × strategy.

Designed for long-running JupyterHub sessions:
  • Skips any run whose result file already exists (safe to re-start)
  • Logs each run to results/logs/
  • Prints a live progress table

Usage
─────
  # Full sweep (all combinations)
  python3 experiments/run_sweep.py

  # Only specific networks / configs / models
  python3 experiments/run_sweep.py --networks tsmo --feature_configs speed speed_time
  python3 experiments/run_sweep.py --models last_observation historical_average
  python3 experiments/run_sweep.py --strategies standard

  # Dry run — print what would run without executing
  python3 experiments/run_sweep.py --dry_run

Sweep matrix
────────────
  Networks       : tsmo, cranberry
  Feature configs: speed, speed_weather, speed_incidents, speed_nr,
                   speed_weather_incidents, speed_time, speed_time_weather,
                   speed_time_incidents, speed_time_nr, speed_time_weather_incidents
  Models         : last_observation, historical_average, linear_ar, xgboost,
                   lstm, transformer,
                   hl, lstm_st, dcrnn, agcrn, stgcn, gwnet, astgcn, sttn,
                   stgode, dstagnn, dgcrn, d2stgnn
  Strategies     : standard, weighted_loss, nr_finetune, multi_objective
                   (strategies only applied to DL models; baselines always standard)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Sweep definition
# ─────────────────────────────────────────────────────────────────────────────

ALL_NETWORKS = ["tsmo", "cranberry"]

ALL_FEATURE_CONFIGS = [
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

# Statistical and ML baselines — always run with 'standard' strategy only
BASELINE_MODELS = [
    "last_observation",
    "historical_average",
    "linear_ar",
    "xgboost",
]

# Deep learning models — run with all strategies
DL_MODELS = [
    "lstm",
    "transformer",
]

# LargeST spatial models — run with all strategies
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

ALL_STRATEGIES = ["standard", "weighted_loss", "nr_finetune", "multi_objective"]


def result_path(
    results_dir: Path, network: str, model: str, feature_config: str, strategy: str
) -> Path:
    return results_dir / network / f"{model}_{feature_config}_{strategy}.json"


def build_run_list(
    networks: list[str],
    feature_configs: list[str],
    models: list[str],
    strategies: list[str],
    results_dir: Path,
    skip_existing: bool = True,
) -> list[dict]:
    """Build the full list of (network, feature_config, model, strategy) jobs."""
    runs = []
    for network in networks:
        for feature_config in feature_configs:
            for model in models:
                # Baselines only run standard; DL/spatial run all strategies
                if model in BASELINE_MODELS:
                    applicable = ["standard"]
                else:
                    applicable = strategies

                for strategy in applicable:
                    rpath = result_path(
                        results_dir, network, model, feature_config, strategy
                    )
                    exists = rpath.exists()
                    runs.append(
                        {
                            "network": network,
                            "feature_config": feature_config,
                            "model": model,
                            "strategy": strategy,
                            "result_path": rpath,
                            "done": exists,
                        }
                    )
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────────


def run_one(job: dict, log_dir: Path) -> tuple[bool, float]:
    """
    Execute a single (network, feature_config, model, strategy) job.
    Returns True on success, False on failure.
    """
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_benchmark.py"),
        "--network",
        job["network"],
        "--feature_config",
        job["feature_config"],
        "--model",
        job["model"],
        "--strategy",
        job["strategy"],
        "--run_id",
        "",
    ]

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (
        log_dir / f"{job['network']}_{job['model']}"
        f"_{job['feature_config']}_{job['strategy']}.log"
    )

    t0 = time.time()
    try:
        with open(log_file, "w") as lf:
            # Stream output to both terminal (stdout) and log file in real-time
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                print(line, end="", flush=True)  # live terminal output
                lf.write(line)  # also save to log
            proc.wait(timeout=7200)

        elapsed = round(time.time() - t0, 1)
        return proc.returncode == 0, elapsed
    except subprocess.TimeoutExpired:
        proc.kill()
        return False, 7200.0
    except Exception:
        return False, round(time.time() - t0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Full benchmark sweep")
    parser.add_argument("--networks", nargs="+", default=ALL_NETWORKS)
    parser.add_argument("--feature_configs", nargs="+", default=ALL_FEATURE_CONFIGS)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES)
    parser.add_argument(
        "--dry_run", action="store_true", help="Print planned runs without executing"
    )
    parser.add_argument(
        "--no_skip", action="store_true", help="Re-run even if result file exists"
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)
    results_dir = PROJECT_ROOT / config["paths"].get("results", "results")
    log_dir = results_dir / "logs"

    runs = build_run_list(
        networks=args.networks,
        feature_configs=args.feature_configs,
        models=args.models,
        strategies=args.strategies,
        results_dir=results_dir,
        skip_existing=not args.no_skip,
    )

    total = len(runs)
    pending = [r for r in runs if not r["done"]]
    done = total - len(pending)

    print(f"\n{'=' * 65}")
    print(f"  NR Benchmarking — Full Sweep")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total   : {total}  |  Done: {done}  |  Pending: {len(pending)}")
    print(f"  Networks: {args.networks}")
    print(f"  Configs : {args.feature_configs}")
    print(f"  Models  : {len(args.models)} models")
    print(f"  Strategies: {args.strategies}")
    print(f"{'=' * 65}\n")

    if args.dry_run:
        print("DRY RUN — jobs that would execute:\n")
        for r in pending:
            print(
                f"  {r['network']:<12} {r['model']:<22} "
                f"{r['feature_config']:<30} {r['strategy']}"
            )
        print(f"\n  Total pending: {len(pending)}")
        return

    if not pending:
        print("All runs already complete.")
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    n_success = 0
    n_fail = 0
    elapsed_times = []
    sweep_start = time.time()

    for i, job in enumerate(pending, 1):
        tag = (
            f"{job['network']} / {job['model']} / "
            f"{job['feature_config']} / {job['strategy']}"
        )
        print(f"\n{'=' * 60}", flush=True)
        print(f"  Job {i}/{len(pending)}", flush=True)
        print(f"  Network  : {job['network'].upper()}", flush=True)
        print(f"  Model    : {job['model']}", flush=True)
        print(f"  Features : {job['feature_config']}", flush=True)
        print(f"  Strategy : {job['strategy']}", flush=True)
        print(f"  Started  : {datetime.now().strftime('%H:%M:%S')}", flush=True)
        print(f"{'=' * 60}", flush=True)

        result = run_one(job, log_dir)
        ok, elapsed = result
        elapsed_times.append(elapsed)

        status = "OK " if ok else "ERR"
        # Estimate remaining time
        avg_t = sum(elapsed_times) / len(elapsed_times)
        remain = int(avg_t * (len(pending) - i))
        eta_str = str(timedelta(seconds=remain))

        print(
            f"           {status}  {elapsed:.0f}s  |  avg={avg_t:.0f}s  "
            f"ETA={eta_str}  "
            f"({n_success + (1 if ok else 0)}/{i} ok)",
            flush=True,
        )

        if ok:
            n_success += 1
        else:
            n_fail += 1
            log_file = (
                log_dir / f"{job['network']}_{job['model']}"
                f"_{job['feature_config']}_{job['strategy']}.log"
            )
            print(f"           FAILED — see {log_file}", flush=True)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = round(time.time() - sweep_start)
    print(f"\n{'=' * 65}")
    print(f"  Sweep complete")
    print(f"  Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total    : {len(pending)}  |  Success: {n_success}  |  Failed: {n_fail}")
    print(f"  Wall time: {str(timedelta(seconds=total_elapsed))}")
    print(f"  Results  : {results_dir}")
    print(f"{'=' * 65}")

    # Save sweep manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "n_success": n_success,
        "n_fail": n_fail,
        "wall_time_s": total_elapsed,
        "networks": args.networks,
        "feature_configs": args.feature_configs,
        "models": args.models,
        "strategies": args.strategies,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "sweep_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
