"""
Adjacency matrix construction for spatial-temporal models.

Builds a weighted directed adjacency from the upstream mapping JSON and the
network GeoJSON.  Returns both the raw matrix and the row-normalised diffusion
matrices used by DCRNN-family models.

Functions
─────────
build_adjacency_matrix   → (N, N) 0/1 sparse adj + weighted version
build_diffusion_matrices → (N, N) forward and backward transition matrices
                           for DCRNN bidirectional diffusion
save_adjacency           → save .npy files for LargeST compatibility
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import geopandas as gpd
import scipy.sparse as sp


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sym_norm_lap(A: np.ndarray) -> np.ndarray:
    """Symmetric normalised Laplacian: D^{-1/2} (D − A) D^{-1/2}."""
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt
    return L.astype(np.float32)


def _row_norm(A: np.ndarray) -> np.ndarray:
    """Row-normalise: D^{-1} A."""
    d = A.sum(axis=1, keepdims=True)
    return np.where(d > 0, A / d, 0.0).astype(np.float32)


def _scaled_lap(A: np.ndarray) -> np.ndarray:
    """Scaled Laplacian: 2L / λ_max − I  (used by STGCN ChebNet)."""
    L = _sym_norm_lap(A)
    eigs = np.linalg.eigvalsh(L)
    lmax = float(eigs.max())
    if lmax < 1e-6:
        return np.zeros_like(L)
    return ((2.0 / lmax) * L - np.eye(len(L))).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Distance-weighted adjacency
# ─────────────────────────────────────────────────────────────────────────────


def _load_link_lengths(geojson_path: str | Path) -> dict[str, float]:
    gdf = gpd.read_file(str(geojson_path))
    id_col = "tmc" if "tmc" in gdf.columns else "id_tmc"
    gdf[id_col] = gdf[id_col].astype(str)
    lengths: dict[str, float] = {}
    for _, row in gdf.iterrows():
        try:
            lengths[row[id_col]] = float(row["miles"])
        except (KeyError, ValueError, TypeError):
            pass
    return lengths


def build_adjacency_matrix(
    geojson_path: str | Path,
    upstream_path: str | Path,
    links: list[str],
    weight: str = "distance",  # "binary" | "distance" | "gaussian"
    sigma: float = 0.5,  # Gaussian kernel σ (miles)
    self_loops: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a directed adjacency matrix from the upstream mapping.

    Convention: A[i, j] = 1  (or weight > 0) means link j is an upstream
    neighbour of link i.  This reflects the direction of traffic propagation
    (upstream → downstream).

    Parameters
    ----------
    geojson_path  : path to *_network.geojson
    upstream_path : path to *_upstream_mapping.json
    links         : ordered list of N TMC strings (defines row/column order)
    weight        : "binary"   — unweighted 0/1
                    "distance" — inverse-distance weighted
                    "gaussian" — Gaussian kernel of edge length
    sigma         : bandwidth for Gaussian weighting (miles)
    self_loops    : if True add identity to A

    Returns
    -------
    A       : (N, N) float32 adjacency
    A_fw    : (N, N) row-normalised forward  transition matrix (D^{-1} A)
    A_bw    : (N, N) row-normalised backward transition matrix (D^{-1} A^T)
    """
    with open(upstream_path) as f:
        upstream: dict[str, list[str]] = json.load(f)

    lengths = _load_link_lengths(geojson_path)
    N = len(links)
    link_idx = {lnk: i for i, lnk in enumerate(links)}

    A = np.zeros((N, N), dtype=np.float32)

    for lnk, nbrs in upstream.items():
        i = link_idx.get(str(lnk))
        if i is None:
            continue
        for nbr in nbrs:
            j = link_idx.get(str(nbr))
            if j is None or i == j:
                continue
            # Edge length: average of the two link lengths (fallback: 0.1 mi)
            li = lengths.get(str(lnk), 0.1)
            lj = lengths.get(str(nbr), 0.1)
            d = (li + lj) / 2.0

            if weight == "binary":
                A[i, j] = 1.0
            elif weight == "distance":
                A[i, j] = 1.0 / max(d, 1e-3)
            else:  # gaussian
                A[i, j] = float(np.exp(-(d**2) / (2 * sigma**2)))

    if self_loops:
        A += np.eye(N, dtype=np.float32)

    # Row-normalised forward and backward (transpose) transition matrices
    A_fw = _row_norm(A)
    A_bw = _row_norm(A.T)

    return A, A_fw, A_bw


# ─────────────────────────────────────────────────────────────────────────────
# DCRNN diffusion matrices
# ─────────────────────────────────────────────────────────────────────────────


def build_diffusion_matrices(
    A_fw: np.ndarray,
    A_bw: np.ndarray,
    K: int = 2,
) -> list[np.ndarray]:
    """
    Build K-step bidirectional diffusion matrices for DCRNN.

    Returns a flat list:
      [I, A_fw, A_fw^2, ..., A_fw^K,
           A_bw, A_bw^2, ..., A_bw^K]
    (2K+1 matrices, each (N,N) float32).
    """
    N = A_fw.shape[0]
    I = np.eye(N, dtype=np.float32)
    mats = [I]
    Ak_fw, Ak_bw = A_fw.copy(), A_bw.copy()
    for _ in range(K):
        mats.append(Ak_fw.astype(np.float32))
        Ak_fw = Ak_fw @ A_fw
    for _ in range(K):
        mats.append(Ak_bw.astype(np.float32))
        Ak_bw = Ak_bw @ A_bw
    return mats


# ─────────────────────────────────────────────────────────────────────────────
# Chebychev polynomials (STGCN, ASTGCN)
# ─────────────────────────────────────────────────────────────────────────────


def build_cheb_polynomials(A: np.ndarray, K: int = 3) -> list[np.ndarray]:
    """
    Compute Chebyshev polynomials T_0, T_1, ..., T_{K-1} of the scaled
    Laplacian.  Each polynomial is (N, N) float32.
    """
    L_tilde = _scaled_lap(A)
    N = A.shape[0]
    polys = [np.eye(N, dtype=np.float32), L_tilde.copy()]
    for k in range(2, K):
        polys.append((2.0 * L_tilde @ polys[-1] - polys[-2]).astype(np.float32))
    return polys[:K]


# ─────────────────────────────────────────────────────────────────────────────
# Persistence / save
# ─────────────────────────────────────────────────────────────────────────────


def save_adjacency(
    A: np.ndarray,
    out_path: str | Path,
) -> None:
    """Save adjacency matrix as .npy (LargeST-compatible format)."""
    np.save(str(out_path), A)
    print(f"Saved adjacency matrix: {out_path}  shape={A.shape}")


def load_or_build_adjacency(
    geojson_path: str | Path,
    upstream_path: str | Path,
    links: list[str],
    cache_path: str | Path | None = None,
    weight: str = "binary",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load adjacency from cache (if it exists) or build and save.

    Returns (A, A_fw, A_bw).
    """
    if cache_path is not None:
        cp = Path(cache_path)
        if cp.exists():
            A = np.load(str(cp))
            A_fw = _row_norm(A)
            A_bw = _row_norm(A.T)
            print(f"Loaded cached adjacency: {cp}")
            return A, A_fw, A_bw

    A, A_fw, A_bw = build_adjacency_matrix(
        geojson_path, upstream_path, links, weight=weight, **kwargs
    )
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        save_adjacency(A, cache_path)

    return A, A_fw, A_bw
