"""
Generate network diagrams for the two study networks (TSMO and Cranberry).
Saves paper/figures/networks.pdf and .png
"""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import defaultdict

DATA = Path(__file__).parent.parent / "data"
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "figure.dpi": 200,
    }
)

# Road-name to color mapping for the corridors we care about
TSMO_CORRIDORS = {
    "I 695": "#e74c3c",
    "I-695": "#e74c3c",
    "I 95": "#e67e22",
    "I-95": "#e67e22",
    "I 70": "#f1c40f",
    "I-70": "#f1c40f",
    "I 195": "#27ae60",
    "I-195": "#27ae60",
    "I 895": "#2980b9",
    "I-895": "#2980b9",
    "US 29": "#8e44ad",
    "US-29": "#8e44ad",
    "MD 100": "#16a085",
    "MD-100": "#16a085",
    "MD 32": "#c0392b",
    "MD-32": "#c0392b",
}

CRANBERRY_CORRIDORS = {
    "I 76": "#e74c3c",
    "I-76": "#e74c3c",
    "I 79": "#2980b9",
    "I-79": "#2980b9",
}


def load_geojson(path):
    with open(path) as f:
        return json.load(f)["features"]


def road_color(name, corridor_map, default="#aaaaaa"):
    if name is None:
        return default
    upper = name.upper().replace("-", " ").strip()
    for key, col in corridor_map.items():
        k = key.upper().replace("-", " ").strip()
        if k in upper:
            return col
    return default


def extract_coords(feature):
    """Return list of (lon, lat) pairs for a LineString or MultiLineString."""
    coords = feature["geometry"]["coordinates"]
    # If nested (MultiLineString or list-of-lists), flatten one level
    if coords and isinstance(coords[0][0], list):
        flat = []
        for ring in coords:
            flat.extend(ring)
        return flat
    return coords  # already flat [(lon, lat), ...]


def plot_network(ax, features, corridor_map, title, show_legend=True, nw_label=None):
    lons_all, lats_all = [], []

    # Group by road for color
    for feat in features:
        coords = extract_coords(feat)
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        lons_all += lons
        lats_all += lats
        rn = feat["properties"].get("roadname") or feat["properties"].get("roadnumber")
        col = road_color(rn, corridor_map)
        ax.plot(
            lons, lats, color=col, linewidth=0.7, solid_capstyle="round", alpha=0.85
        )

    # Axis cosmetics
    lon_pad = (max(lons_all) - min(lons_all)) * 0.04
    lat_pad = (max(lats_all) - min(lats_all)) * 0.04
    ax.set_xlim(min(lons_all) - lon_pad, max(lons_all) + lon_pad)
    ax.set_ylim(min(lats_all) - lat_pad, max(lats_all) + lat_pad)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.set_xlabel("Longitude", fontsize=7)
    ax.set_ylabel("Latitude", fontsize=7)
    ax.tick_params(labelsize=6.5)

    # North arrow
    x0, y0 = max(lons_all) - lon_pad * 0.5, max(lats_all) - lat_pad * 0.5
    ax.annotate(
        "N",
        xy=(x0, y0),
        xytext=(x0, y0 - lat_pad * 0.7),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="black"),
        ha="center",
        fontsize=7,
        fontweight="bold",
    )

    # Scale bar (rough: 1 deg lat ≈ 111 km)
    lat_mid = (min(lats_all) + max(lats_all)) / 2
    lon_per_km = 1.0 / (111.0 * np.cos(np.radians(lat_mid)))
    bar_km = 5
    bar_lon = bar_km * lon_per_km
    bx = min(lons_all) + lon_pad * 0.3
    by = min(lats_all) + lat_pad * 0.5
    ax.plot([bx, bx + bar_lon], [by, by], "k-", lw=2)
    ax.text(
        bx + bar_lon / 2, by + lat_pad * 0.15, f"{bar_km} km", ha="center", fontsize=6.5
    )

    if show_legend and corridor_map:
        # Build legend for corridors actually present
        present = set()
        for feat in features:
            rn = feat["properties"].get("roadname") or feat["properties"].get(
                "roadnumber"
            )
            if rn:
                upper = rn.upper().replace("-", " ").strip()
                for key in corridor_map:
                    k = key.upper().replace("-", " ").strip()
                    if k in upper:
                        present.add(key)
        handles = [
            Line2D([0], [0], color=corridor_map[k], lw=1.5, label=k)
            for k in sorted(present)
            if k in corridor_map
        ]
        if handles:
            ax.legend(
                handles=handles,
                fontsize=6,
                loc="lower right",
                framealpha=0.85,
                title="Corridor",
                title_fontsize=6.5,
            )

    # Network label box
    if nw_label:
        ax.text(
            0.02,
            0.98,
            nw_label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )


def main():
    tsmo_feats = load_geojson(DATA / "tsmo" / "tsmo_network.geojson")
    cran_feats = load_geojson(DATA / "cranberry" / "cranberry_network.geojson")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.8))

    plot_network(
        ax1,
        tsmo_feats,
        TSMO_CORRIDORS,
        title="(a) TSMO — Howard County, MD",
        show_legend=True,
        nw_label=f"N = 228 links\n1 year (Feb 2022–Feb 2023)\nNR rate: 3.97%",
    )

    plot_network(
        ax2,
        cran_feats,
        CRANBERRY_CORRIDORS,
        title="(b) Cranberry — Pittsburgh, PA",
        show_legend=True,
        nw_label=f"N = 78 links\n2 years (Feb 2022–Jan 2024)\nNR rate: 2.42%",
    )

    fig.suptitle("Study Networks", fontsize=9, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES / "networks.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "networks.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Saved: figures/networks.pdf")


if __name__ == "__main__":
    main()
