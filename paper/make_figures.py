"""
Generate paper figures for the NR benchmarking paper.

Figure 1: Regime diagram — timeline showing the three evaluation regimes
Figure 2: Per-horizon MAE — how NR and recurrent MAE evolve across h1-h6
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS = Path(__file__).parent.parent / "results"
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

# ─── Shared style ────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    }
)

REGIME_COLORS = {
    "recurrent": "#cccccc",
    "onset": "#f4a460",  # sandy brown / orange
    "confirmed": "#c0392b",  # red
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Regime diagram
# ─────────────────────────────────────────────────────────────────────────────


def make_regime_diagram():
    """
    Show a stylised NR episode on a timeline.
    Context window (9 steps) ends at T_last.
    Target window (6 steps) is colour-coded by regime.
    """
    fig, (ax_speed, ax_regime) = plt.subplots(
        2,
        1,
        figsize=(6.5, 3.2),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        sharex=True,
    )

    # ── Notional speed trace ──────────────────────────────────────────────────
    # 16 total steps: 0..8 = context, 9..14 = target, 15 = padding
    steps = np.arange(16)
    # Freeway speeds: free-flow ~65 mph, episode starts at step 10
    speed = np.array(
        [
            65,
            64,
            65,
            63,
            64,
            65,
            64,
            63,
            65,  # context (steps 0-8)
            64,
            56,
            41,
            34,
            31,
            32,
            36,  # target (steps 9-15)
        ],
        dtype=float,
    )

    v_rec = 58.0  # recurrent lower bound (illustrative)
    T_last = 8  # last context step (0-indexed)
    t0_episode = 10  # episode starts at step 10
    t_confirm = t0_episode + 2  # confirmed at step 12

    # Shade context and target regions
    ax_speed.axvspan(-0.5, T_last + 0.5, alpha=0.08, color="steelblue", label="Context")
    ax_speed.axvspan(T_last + 0.5, 14.5, alpha=0.04, color="gray")

    # Plot speed
    ax_speed.plot(
        steps, speed, "o-", color="#2c3e50", linewidth=1.8, markersize=3.5, zorder=5
    )
    # v_rec reference line
    ax_speed.axhline(
        v_rec,
        linestyle="--",
        color="#7f8c8d",
        linewidth=1.1,
        label=r"$v^{\rm rec}_{n,s(t)}$",
    )

    # Shade below-v_rec region in target
    for s in range(9, 15):
        if speed[s] < v_rec:
            ax_speed.fill_between(
                [s - 0.5, s + 0.5],
                [speed[s], speed[s]],
                [v_rec, v_rec],
                alpha=0.25,
                color="#c0392b",
                zorder=3,
            )

    # ── Regime colouring in the target window ─────────────────────────────────
    def regime_color(s):
        if speed[s] >= v_rec:
            return REGIME_COLORS["recurrent"]
        if s < t_confirm:
            return REGIME_COLORS["onset"]
        return REGIME_COLORS["confirmed"]

    for s in range(9, 15):
        col = regime_color(s)
        for ax in (ax_speed, ax_regime):
            ax.axvspan(s - 0.5, s + 0.5, alpha=0.22, color=col, zorder=2)

    # ── Annotation — confirmation delay ──────────────────────────────────────
    ax_speed.annotate(
        "",
        xy=(t_confirm - 0.5, 52),
        xytext=(t0_episode - 0.5, 52),
        arrowprops=dict(arrowstyle="<->", color="#7f8c8d", lw=1.2),
    )
    ax_speed.text(
        (t0_episode + t_confirm) / 2 - 0.5,
        50,
        "2-step\ndelay",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555555",
    )

    # Vertical line at T_last
    for ax in (ax_speed, ax_regime):
        ax.axvline(
            T_last + 0.5, color="#2c3e50", linewidth=1.5, linestyle="-", zorder=6
        )

    # ── Axis decoration ───────────────────────────────────────────────────────
    ax_speed.set_ylabel("Speed (mph)", labelpad=4)
    ax_speed.set_ylim(22, 72)
    ax_speed.set_yticks([30, 45, 60])
    ax_speed.set_yticklabels(["30", "45", "60"])
    ax_speed.grid(True, alpha=0.3)

    # Labels
    ax_speed.text(
        4,
        69,
        "Context window\n(45 min)",
        ha="center",
        fontsize=8,
        color="steelblue",
        style="italic",
    )
    ax_speed.text(
        11.5,
        69,
        "Target window\n(30 min)",
        ha="center",
        fontsize=8,
        color="#555555",
        style="italic",
    )
    ax_speed.text(
        T_last + 0.55, 24, r"$T_{\rm last}$", fontsize=8, color="#2c3e50", va="bottom"
    )

    # ── Regime bar (bottom panel) ─────────────────────────────────────────────
    ax_regime.set_xlim(-0.5, 14.5)
    ax_regime.set_ylim(0, 1)
    ax_regime.set_yticks([])
    ax_regime.set_xticks(range(0, 15, 2))
    ax_regime.set_xticklabels([f"$t_{{{i}}}$" for i in range(0, 15, 2)], fontsize=7.5)
    ax_regime.set_xlabel("Timestep", labelpad=3)
    ax_regime.spines["left"].set_visible(False)

    # Regime blocks
    for s in range(9, 15):
        col = regime_color(s)
        ax_regime.fill_between([s - 0.5, s + 0.5], [0, 0], [1, 1], color=col, alpha=0.8)
        ax_regime.text(
            s,
            0.5,
            f"h{s - T_last}",
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="white" if col != REGIME_COLORS["recurrent"] else "#555",
        )

    ax_regime.axvline(T_last + 0.5, color="#2c3e50", linewidth=1.5, zorder=6)

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=REGIME_COLORS["recurrent"], alpha=0.7, label="Recurrent"),
        mpatches.Patch(
            color=REGIME_COLORS["onset"], alpha=0.7, label="Unobserved onset"
        ),
        mpatches.Patch(
            color=REGIME_COLORS["confirmed"], alpha=0.7, label="Confirmed NR"
        ),
    ]
    ax_speed.legend(
        handles=patches
        + [
            plt.Line2D(
                [0],
                [0],
                linestyle="--",
                color="#7f8c8d",
                lw=1.1,
                label=r"$v^{\rm rec}_{n,s(t)}$",
            )
        ],
        loc="lower left",
        fontsize=7.5,
        framealpha=0.9,
        ncol=2,
        columnspacing=0.8,
    )

    fig.savefig(FIGURES / "regime_diagram.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "regime_diagram.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  Saved: figures/regime_diagram.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Per-horizon MAE (appendix)
# ─────────────────────────────────────────────────────────────────────────────


def load_per_horizon(model, network="tsmo"):
    f = RESULTS / network / f"{model}_speed_standard.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return {
        "rec": [d.get(f"recurrent_mae_h{h}", float("nan")) for h in range(1, 7)],
        "nr": [d.get(f"nr_mae_h{h}", float("nan")) for h in range(1, 7)],
    }


def make_horizon_figure():
    """
    Per-horizon NR and recurrent MAE for representative models on TSMO.
    Shows that recurrent MAE grows steadily with horizon while NR MAE
    is consistently elevated and also grows, maintaining a large gap.
    """
    horizons = [5, 10, 15, 20, 25, 30]  # minutes

    models = [
        ("historical_average", "Hist. Avg.", "#9b59b6", "s"),
        ("linear_ar", "Linear AR", "#3498db", "^"),
        ("xgboost", "XGBoost", "#27ae60", "D"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=False)

    ax_rec, ax_nr = axes
    ax_rec.set_title("(a) Recurrent MAE", fontsize=9)
    ax_nr.set_title("(b) NR (combined) MAE", fontsize=9)

    for model_id, label, color, marker in models:
        data = load_per_horizon(model_id)
        if data is None:
            continue
        kw = dict(color=color, marker=marker, markersize=5, linewidth=1.6, label=label)
        ax_rec.plot(horizons, data["rec"], **kw)
        ax_nr.plot(horizons, data["nr"], **kw)

    for ax, ylabel in zip(axes, ["MAE (mph)", "MAE (mph)"]):
        ax.set_xlabel("Forecast horizon (min)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(horizons)
        ax.set_xticklabels([str(h) for h in horizons])

    ax_rec.set_ylim(3.0, 5.5)
    ax_nr.set_ylim(5.0, 16.0)
    ax_nr.yaxis.set_label_position("right")
    ax_nr.yaxis.tick_right()

    # Shared legend below both panels
    handles, labels = ax_rec.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=8,
        framealpha=0.9,
    )

    fig.subplots_adjust(bottom=0.22, wspace=0.05)
    fig.savefig(FIGURES / "horizon_mae.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "horizon_mae.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  Saved: figures/horizon_mae.pdf")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...")
    make_regime_diagram()
    make_horizon_figure()
    print("Done.")
