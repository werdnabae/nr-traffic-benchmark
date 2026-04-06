#!/usr/bin/env python3
"""
Generate LaTeX table content for the NR benchmarking paper.

Outputs two LaTeX tabular bodies:
  - TSMO Phase 1 results (regime MAE + per-horizon NR MAE)
  - Cranberry Phase 1 results (regime MAE + per-horizon NR MAE)

Usage (on JupyterHub or Mac):
  python3 scripts/generate_results_table.py

Output: paper/table_phase1_body.tex  (paste into main.tex)
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
OUT = PROJECT_ROOT / "paper" / "table_phase1_body.tex"

MODELS_ORDERED = [
    (
        "Statistical / ML",
        [
            ("last_observation", "Last Obs."),
            ("historical_average", "Hist.\\ Avg."),
            ("linear_ar", "Linear AR"),
            ("xgboost", "XGBoost"),
        ],
    ),
    (
        "Per-link DL",
        [
            ("lstm", "LSTM"),
            ("transformer", "Transformer"),
        ],
    ),
    (
        "Spatial-temporal",
        [
            ("hl", "HL"),
            ("dgcrn", "DGCRN"),
            ("gwnet", "GWNet"),
            ("sttn", "STTN"),
            ("stgcn", "STGCN"),
            ("agcrn", "AGCRN"),
            ("dcrnn", "DCRNN"),
            ("dstagnn", "DSTAGNN"),
            ("stgode", "STGODE"),
            ("lstm_st", "LSTM-ST"),
            ("astgcn", "ASTGCN"),
            ("d2stgnn", "D2STGNN$^\\dagger$"),
        ],
    ),
]

# Models with anomalous NR behavior — add dagger in caption
ANOMALIES = {"d2stgnn", "dstagnn"}

# Known excluded strategy runs (not relevant to Phase 1 / standard config)


def load(network: str, model: str) -> dict | None:
    f = RESULTS / network / f"{model}_speed_standard.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def fmt(v, decimals=3) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return r"\text{---}"
    return f"{float(v):.{decimals}f}"


def row(network: str, model_id: str, label: str) -> str:
    d = load(network, model_id)
    if d is None:
        return f"  & {label} & " + r"\text{---} & " * 9 + r"\\" + "\n"

    rec = fmt(d.get("recurrent_mae_avg"))
    onset = fmt(d.get("unobserved_onset_mae"))
    conf = fmt(d.get("confirmed_nr_mae"))
    hr = [fmt(d.get(f"nr_mae_h{h}")) for h in range(1, 7)]

    return f"  & {label} & {rec} & {onset} & {conf} & " + " & ".join(hr) + r" \\" + "\n"


def build_network_table(network: str) -> str:
    lines = []
    for family, members in MODELS_ORDERED:
        n = len(members)
        first = True
        for model_id, label in members:
            if first:
                lines.append(f"\\multirow{{{n}}}{{*}}{{{family}}}\n")
                first = False
            lines.append(row(network, model_id, label))
        lines.append("\\midrule\n")
    # Remove last \midrule
    lines = lines[:-1]
    return "".join(lines)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_lines = []

    for network, label in [("tsmo", "TSMO"), ("cranberry", "Cranberry")]:
        out_lines.append(f"%% ──── {label} ────────────────────────────────────\n")
        out_lines.append(
            f"% Paste between \\midrule and \\bottomrule in tab:phase1_{network}\n"
        )
        out_lines.append(build_network_table(network))
        out_lines.append("\n")

    OUT.write_text("".join(out_lines))
    print(f"Written: {OUT}")
    print()

    # Also print summary to terminal
    for network in ["tsmo", "cranberry"]:
        print(f"\n{'=' * 70}")
        print(f"  {network.upper()} — speed/standard — NR MAE per horizon")
        print(f"{'=' * 70}")
        print(
            f"  {'Model':<20} {'Rec':>6} {'Onset':>7} {'Conf':>7}",
            "  h1    h2    h3    h4    h5    h6",
        )
        print("  " + "─" * 70)
        for _, members in MODELS_ORDERED:
            for model_id, label in members:
                d = load(network, model_id)
                if d is None:
                    print(f"  {label:<20} MISSING")
                    continue
                rec = d.get("recurrent_mae_avg", float("nan"))
                onset = d.get("unobserved_onset_mae", float("nan"))
                conf = d.get("confirmed_nr_mae", float("nan"))
                hrs = [d.get(f"nr_mae_h{h}", float("nan")) for h in range(1, 7)]
                h_str = "  ".join(f"{h:.2f}" for h in hrs)
                print(f"  {label:<20} {rec:6.3f} {onset:7.3f} {conf:7.3f}  {h_str}")


if __name__ == "__main__":
    main()
