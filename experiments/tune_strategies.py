#!/usr/bin/env python3
"""
Hyperparameter tuning for the two novel training strategy parameters.

Run ONCE before the main sweep. Tunes on TSMO × speed-only × validation set.
Writes the best values back to config.yaml so the main sweep uses them.

Parameters tuned
────────────────
nr_loss_weight       [2, 5, 10, 20]
  Upweight factor applied to NR timesteps in the weighted-loss strategy.
  Selected by: lowest NR MAE on the validation set.

finetune_lr_multiplier  [0.05, 0.1, 0.2]
  LR multiplier for Phase 2 of the nr_finetune strategy (base_lr × this).
  Selected by: lowest NR MAE on the validation set after fine-tuning.

Why LSTM × TSMO × speed?
  Fastest model that uses both strategies. TSMO is the larger network so
  the selection generalises. Speed-only keeps the comparison clean.

Why validation NR MAE?
  We are tuning to improve NR robustness, so overall MAE would give a
  biased selection toward recurrent-dominated performance.

Usage
─────
  python3 experiments/tune_strategies.py
  python3 experiments/tune_strategies.py --dry_run   # show grid, don't train
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_network
from src.data.dataset import make_datasets
from src.evaluation.metrics import compute_metrics, inverse_transform
from src.models.dl.lstm import Seq2SeqLSTM
from src.training.trainer import Trainer


# ─────────────────────────────────────────────────────────────────────────────
# Tuning grids
# ─────────────────────────────────────────────────────────────────────────────

NR_LOSS_WEIGHT_GRID = [2, 5, 10, 20]
FINETUNE_LR_MULT_GRID = [0.05, 0.1, 0.2]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_loaders(train_ds, val_ds, batch_size: int, num_workers: int = 4):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def _eval_nr_mae(trainer: Trainer, val_loader: DataLoader) -> float:
    """Return validation NR MAE after training."""
    trainer.model.eval()
    preds, targets, nrs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch_d = {
                k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            pred_norm = trainer.model(batch_d).cpu().numpy()
            pred_orig = inverse_transform(
                pred_norm, trainer.nd.speed_mean, trainer.nd.speed_std
            )
            preds.append(pred_orig)
            targets.append(batch["y_orig"].numpy())
            nrs.append(batch["full_nr"].numpy())

    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    nr = np.concatenate(nrs, axis=0)

    metrics = compute_metrics(pred, target, nr)
    return float(metrics.get("nr_mae_avg", float("nan")))


def _build_lstm(C_in: int, L_out: int, cfg: dict) -> Seq2SeqLSTM:
    lstm_cfg = cfg.get("models", {}).get("lstm", {})
    return Seq2SeqLSTM(
        C_in=C_in,
        output_len=L_out,
        hidden_dim=lstm_cfg.get("hidden_dim", 128),
        num_layers=lstm_cfg.get("num_layers", 2),
        dropout=lstm_cfg.get("dropout", 0.1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tune nr_loss_weight
# ─────────────────────────────────────────────────────────────────────────────


def tune_nr_loss_weight(
    nd, train_ds, val_ds, base_config: dict, L_out: int, C_in: int
) -> tuple[float, dict]:
    """
    Grid search over NR_LOSS_WEIGHT_GRID using weighted_loss strategy.
    Returns (best_weight, {weight: nr_mae}).
    """
    print("\n" + "─" * 60)
    print("  Tuning: nr_loss_weight  (weighted_loss strategy)")
    print("  Grid:", NR_LOSS_WEIGHT_GRID)
    print("─" * 60)

    batch_size = base_config["data"]["batch_size"]
    train_loader, val_loader = _make_loaders(train_ds, val_ds, batch_size)

    results = {}
    for w in NR_LOSS_WEIGHT_GRID:
        t0 = time.time()
        cfg = copy.deepcopy(base_config)
        cfg["training"]["nr_loss_weight"] = w

        torch.manual_seed(cfg["training"].get("seed", 2024))
        model = _build_lstm(C_in, L_out, cfg)
        trainer = Trainer(model, nd, cfg["training"], strategy="weighted_loss")
        trainer.fit(train_loader, val_loader)

        nr_mae = _eval_nr_mae(trainer, val_loader)
        elapsed = round(time.time() - t0, 1)
        results[w] = nr_mae
        print(f"  nr_loss_weight={w:<5}  val NR MAE={nr_mae:.4f}  ({elapsed}s)")

    best = min(results, key=results.get)
    print(f"\n  → Best: nr_loss_weight = {best}  (val NR MAE = {results[best]:.4f})")
    return float(best), results


# ─────────────────────────────────────────────────────────────────────────────
# Tune finetune_lr_multiplier
# ─────────────────────────────────────────────────────────────────────────────


def tune_finetune_lr(
    nd, train_ds, val_ds, base_config: dict, L_out: int, C_in: int
) -> tuple[float, dict]:
    """
    Grid search over FINETUNE_LR_MULT_GRID using nr_finetune strategy.
    Returns (best_multiplier, {mult: nr_mae}).
    """
    print("\n" + "─" * 60)
    print("  Tuning: finetune_lr_multiplier  (nr_finetune strategy)")
    print("  Grid:", FINETUNE_LR_MULT_GRID)
    print(
        f"  (base LR = {base_config['training']['lr']}, "
        f"finetune LR = base × multiplier)"
    )
    print("─" * 60)

    batch_size = base_config["data"]["batch_size"]
    train_loader, val_loader = _make_loaders(train_ds, val_ds, batch_size)

    results = {}
    for mult in FINETUNE_LR_MULT_GRID:
        t0 = time.time()
        cfg = copy.deepcopy(base_config)
        cfg["training"]["finetune_lr_multiplier"] = mult

        torch.manual_seed(cfg["training"].get("seed", 2024))
        model = _build_lstm(C_in, L_out, cfg)
        trainer = Trainer(model, nd, cfg["training"], strategy="nr_finetune")
        trainer.fit(train_loader, val_loader)

        nr_mae = _eval_nr_mae(trainer, val_loader)
        elapsed = round(time.time() - t0, 1)
        actual_lr = cfg["training"]["lr"] * mult
        results[mult] = nr_mae
        print(
            f"  multiplier={mult}  (ft LR={actual_lr:.2e})  "
            f"val NR MAE={nr_mae:.4f}  ({elapsed}s)"
        )

    best = min(results, key=results.get)
    print(
        f"\n  → Best: finetune_lr_multiplier = {best}  "
        f"(actual LR = {base_config['training']['lr'] * best:.2e}, "
        f"val NR MAE = {results[best]:.4f})"
    )
    return float(best), results


# ─────────────────────────────────────────────────────────────────────────────
# Write results to config.yaml
# ─────────────────────────────────────────────────────────────────────────────


def update_config(
    config_path: Path, best_nr_weight: float, best_ft_mult: float
) -> None:
    with open(config_path) as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if "nr_loss_weight:" in line and "upweight" in line:
            # Preserve the comment, update the value
            comment = line.split("#", 1)[1] if "#" in line else ""
            new_lines.append(
                f"  nr_loss_weight: {best_nr_weight}"
                f"{'  # ' + comment.strip() if comment else ''}\n"
            )
        elif "finetune_lr_multiplier:" in line:
            comment = line.split("#", 1)[1] if "#" in line else ""
            new_lines.append(
                f"  finetune_lr_multiplier: {best_ft_mult}"
                f"{'  # ' + comment.strip() if comment else ''}\n"
            )
        else:
            new_lines.append(line)

    with open(config_path, "w") as f:
        f.writelines(new_lines)

    print(f"\n  config.yaml updated:")
    print(f"    nr_loss_weight          = {best_nr_weight}")
    print(f"    finetune_lr_multiplier  = {best_ft_mult}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune nr_loss_weight and finetune_lr_multiplier"
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--dry_run", action="store_true", help="Print grid without training"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.dry_run:
        print("DRY RUN — would tune:")
        print(f"  nr_loss_weight grid       : {NR_LOSS_WEIGHT_GRID}")
        print(f"  finetune_lr_mult grid     : {FINETUNE_LR_MULT_GRID}")
        base_lr = config["training"]["lr"]
        print(
            f"  Actual finetune LRs tested: "
            f"{[round(base_lr * m, 6) for m in FINETUNE_LR_MULT_GRID]}"
        )
        n_runs = len(NR_LOSS_WEIGHT_GRID) + len(FINETUNE_LR_MULT_GRID)
        print(f"  Total LSTM training runs  : {n_runs}")
        print(f"  Estimated time            : ~{n_runs * 20} minutes")
        return

    print("=" * 60)
    print("  Strategy Hyperparameter Tuning")
    print("  Network: TSMO  |  Config: speed  |  Model: LSTM")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    data_cfg = config["data"]
    L_in = data_cfg["input_len"]
    L_out = data_cfg["output_len"]

    print("\nLoading TSMO data...")
    nd = load_network("tsmo", args.config)

    train_ds, val_ds, _, fb = make_datasets(nd, ["speed"], L_in, L_out)
    C_in = fb.C

    # ── Tune ──────────────────────────────────────────────────────────────────
    best_nr_weight, nr_weight_results = tune_nr_loss_weight(
        nd, train_ds, val_ds, config, L_out, C_in
    )
    best_ft_mult, ft_mult_results = tune_finetune_lr(
        nd, train_ds, val_ds, config, L_out, C_in
    )

    # ── Save results ──────────────────────────────────────────────────────────
    results_dir = PROJECT_ROOT / config["paths"].get("results", "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    tuning_results = {
        "nr_loss_weight": {
            "grid": NR_LOSS_WEIGHT_GRID,
            "results": {str(k): v for k, v in nr_weight_results.items()},
            "best": best_nr_weight,
        },
        "finetune_lr_multiplier": {
            "grid": FINETUNE_LR_MULT_GRID,
            "results": {str(k): v for k, v in ft_mult_results.items()},
            "best": best_ft_mult,
        },
    }
    out_path = results_dir / "tuning_results.json"
    with open(out_path, "w") as f:
        json.dump(tuning_results, f, indent=2)
    print(f"\n  Full tuning results saved → {out_path}")

    # ── Update config.yaml ────────────────────────────────────────────────────
    update_config(Path(args.config), best_nr_weight, best_ft_mult)

    print("\n" + "=" * 60)
    print("  Tuning complete. Run the sweep next:")
    print(
        "  python3 experiments/run_sweep.py --feature_configs speed "
        "--strategies standard"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
