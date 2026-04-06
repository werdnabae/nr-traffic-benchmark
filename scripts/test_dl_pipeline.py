#!/usr/bin/env python3
"""
Quick smoke-test of every DL model in the pipeline.

Tests:
  - Our own Seq2SeqLSTM and Seq2SeqTransformer
  - All 12 LargeST models (hl, lstm_st, dcrnn, agcrn, stgcn, gwnet,
    astgcn, sttn, stgode, dstagnn, dgcrn, d2stgnn)

For each model:
  1. Instantiate with real TSMO data (N=228, C=1, L_in=9, L_out=6)
  2. Run one forward pass with a tiny batch (B=4)
  3. Run one backward pass (loss.backward) to check gradients
  4. Report: PASS / FAIL + output shape + parameter count

Usage
─────
  python3 scripts/test_dl_pipeline.py
  python3 scripts/test_dl_pipeline.py --device cpu
  python3 scripts/test_dl_pipeline.py --skip dcrnn dgcrn   # skip slow/broken ones
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_network
from src.data.adjacency import load_or_build_adjacency
from src.data.numpy_iterator import make_numpy_iterators


# ─────────────────────────────────────────────────────────────────────────────


def make_fake_batch(B: int, L_in: int, N: int, C: int, device: torch.device) -> dict:
    """Tiny random batch on the target device."""
    return {
        "x": torch.randn(B, L_in, N, C, device=device, requires_grad=True),
        "y": torch.randn(B, 6, N, device=device),
        "y_orig": torch.randn(B, 6, N, device=device),
        "full_nr": torch.zeros(B, 6, N, device=device),
        "causal_fixed_end": torch.zeros(B, N, device=device),
        "nr_backfill": torch.zeros(B, L_in, N, device=device),
        "regime": torch.zeros(B, 6, N, dtype=torch.int8, device=device),
        "sample_idx": torch.zeros(B, dtype=torch.long),
    }


def test_model(name: str, model, batch: dict, device: torch.device) -> dict:
    """Run forward + backward, return result dict."""
    model = model.to(device)
    model.train()

    t0 = time.time()
    try:
        out = model(batch)

        # Verify output shape
        assert out.dim() == 3, f"Expected 3D output, got {out.shape}"
        B, L_out, N = out.shape
        assert L_out == 6, f"Expected horizon=6, got L_out={L_out}"

        # Backward pass (skip for HL — no trainable path through output)
        loss = out.mean()
        if loss.requires_grad:
            loss.backward()

        elapsed = round(time.time() - t0, 3)
        n_params = sum(p.numel() for p in model.parameters())
        return {
            "status": "PASS",
            "shape": tuple(out.shape),
            "params": n_params,
            "elapsed": elapsed,
            "error": None,
        }
    except Exception as e:
        return {
            "status": "FAIL",
            "shape": None,
            "params": None,
            "elapsed": round(time.time() - t0, 3),
            "error": f"{type(e).__name__}: {e}",
            "tb": traceback.format_exc(),
        }


# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps", help="cpu | mps | cuda[:N]")
    parser.add_argument("--skip", nargs="*", default=[], help="model names to skip")
    parser.add_argument(
        "--only", nargs="*", default=None, help="if set, only run these models"
    )
    parser.add_argument("--network", default="tsmo")
    parser.add_argument("--B", type=int, default=4, help="batch size")
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    device_str = args.device
    if device_str == "mps" and not torch.backends.mps.is_available():
        device_str = "cpu"
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    import yaml

    with open(PROJECT_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    print(f"Loading {args.network.upper()}...")
    nd = load_network(args.network)
    N, C, L_in, L_out = nd.N, 1, 9, 6

    # ── Adjacency ─────────────────────────────────────────────────────────────
    pcfg = config["paths"][args.network]
    adj_cache = PROJECT_ROOT / pcfg.get("adjacency", f"data/{args.network}/adj_mx.npy")
    A, _, _ = load_or_build_adjacency(
        geojson_path=PROJECT_ROOT / pcfg["geojson"],
        upstream_path=PROJECT_ROOT / pcfg["upstream"],
        links=nd.links,
        cache_path=adj_cache,
        weight="binary",
    )
    print(f"Adjacency: {A.shape}  density={A.mean():.3f}")

    # ── Fake batch ────────────────────────────────────────────────────────────
    batch = make_fake_batch(args.B, L_in, N, C, device)
    print(f"Batch: x={tuple(batch['x'].shape)}\n")

    results = {}

    # ── Our DL models ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("  OUR DL MODELS")
    print("=" * 60)

    from src.models.dl.lstm import Seq2SeqLSTM
    from src.models.dl.transformer import Seq2SeqTransformer

    our_models = {
        "lstm_ours": Seq2SeqLSTM(C_in=C, output_len=L_out),
        "transformer_ours": Seq2SeqTransformer(C_in=C, output_len=L_out),
    }
    for name, model in our_models.items():
        if name in args.skip:
            print(f"  {name:<20}  SKIPPED")
            continue
        if args.only and name not in args.only:
            continue
        r = test_model(name, model, batch, device)
        results[name] = r
        _print_result(name, r)

    # ── LargeST models ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  LARGST MODELS")
    print("=" * 60)

    from src.models.spatial.runner import LargeSTRunner, LARGST_MODEL_NAMES

    largst_cfg = config.get("models", {})

    for mname in LARGST_MODEL_NAMES:
        if mname in args.skip:
            print(f"  {mname:<20}  SKIPPED")
            continue
        if args.only and mname not in args.only:
            continue

        try:
            model = LargeSTRunner(
                model_name=mname,
                N=N,
                C_in=C,  # runner pads internally for dgcrn/d2stgnn
                L_in=L_in,
                L_out=L_out,
                adj_mx=A,
                model_cfg=largst_cfg.get(mname, {}),
                device=device_str,
            )
        except Exception as e:
            results[mname] = {
                "status": "FAIL (init)",
                "shape": None,
                "params": None,
                "elapsed": 0,
                "error": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc(),
            }
            _print_result(mname, results[mname])
            continue

        # DGCRN/D2STGNN need float TOD/DOW values in [0,1] — use a dedicated
        # batch with extra channels holding random time-like values.
        if mname in ("dgcrn", "d2stgnn"):
            # Build batch with C+extra channels containing [0,1] random values
            extra = 2 if mname == "d2stgnn" else max(0, 3 - C)
            if extra > 0:
                # Concatenate traffic channels (requires_grad) with [0,1]
                # time channels (no grad — just position info)
                x_traffic = torch.randn(
                    args.B, L_in, N, C, requires_grad=True, device=device
                )
                x_tod = torch.rand(args.B, L_in, N, extra, device=device)
                x_time = torch.cat([x_traffic, x_tod], dim=-1)
                batch_time = dict(batch)
                batch_time["x"] = x_time
                r = test_model(mname, model, batch_time, device)
            else:
                r = test_model(mname, model, batch, device)
        else:
            r = test_model(mname, model, batch, device)
        results[mname] = r
        _print_result(mname, r)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for r in results.values() if "PASS" in r["status"])
    n_fail = sum(1 for r in results.values() if "FAIL" in r["status"])
    print(f"  PASSED: {n_pass}   FAILED: {n_fail}   TOTAL: {len(results)}")

    if n_fail > 0:
        print("\n  FAILURES:")
        for name, r in results.items():
            if "FAIL" in r["status"]:
                print(f"\n  [{name}]  {r['error']}")
                if r.get("tb"):
                    # Print last 5 lines of traceback
                    tb_lines = r["tb"].strip().split("\n")
                    for line in tb_lines[-6:]:
                        print(f"    {line}")


def _print_result(name: str, r: dict) -> None:
    status = r["status"]
    shape = str(r["shape"]) if r["shape"] else "—"
    params = f"{r['params']:,}" if r["params"] else "—"
    elapsed = f"{r['elapsed']:.3f}s"
    err = f"  ERROR: {r['error']}" if r["error"] else ""
    print(f"  {name:<20}  {status:<12}  {shape:<20}  {params:<12}  {elapsed}{err}")


if __name__ == "__main__":
    main()
