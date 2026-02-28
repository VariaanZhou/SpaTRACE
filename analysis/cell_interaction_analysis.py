#!/usr/bin/env python3
"""
cell_interaction_analysis.py

Plotting utilities for SpaTRACE per-cell interaction/attention outputs.

IMPORTANT FILE-STRUCTURE NOTE
-----------------------------
Your pipeline produces TWO different kinds of artifacts:

A) Raw per-cell attention MATRICES (from per_cell_att_compute)
   <input_dir>/attentions/percell_attentions/
      meta_gene_orders.npz
      percell_000123.npz   # contains W_tf (TG,Gtf) and W_lr (TG,Glr)

B) Derived per-cell L->R "lr_score_tensor" (from percell_lr_inference)
   Typically written to out_dir (or sometimes the same attention folder):
      percell_lrscore_percell.npz
   This file contains:
      lr_score_tensor : (n_cells, n_lig, n_rec)
      cell_indices    : (n_cells,) mapping rows -> adata cell index
      ligand_list     : (n_lig,)
      receptor_list   : (n_rec,)

This module supports BOTH, but for LR spatial heatmaps it will prefer (B) if available,
because it directly stores L->R scores (no TG aggregation needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
import re

# matplotlib is imported inside plotting funcs


# ----------------------------- Small helpers -----------------------------

def _safe_np_load(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as Z:
        return {k: Z[k] for k in Z.files}


def _sanitize_filename(s: str, max_len: int = 160) -> str:
    s = str(s).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:max_len] if len(s) > max_len else s


def _get_spatial_xy(adata) -> np.ndarray:
    """Return (N,2) spatial coordinates. Prefers obsm['spatial']."""
    if hasattr(adata, "obsm") and "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"])
        if xy.ndim == 2 and xy.shape[1] >= 2:
            return xy[:, :2].astype(float, copy=False)
    # fallbacks
    if hasattr(adata, "obsm"):
        for key in ("X_spatial", "X_umap", "X_pca"):
            if key in adata.obsm:
                xy = np.asarray(adata.obsm[key])
                if xy.ndim == 2 and xy.shape[1] >= 2:
                    return xy[:, :2].astype(float, copy=False)
    raise KeyError("Could not find spatial coordinates in adata.obsm['spatial'] (or fallbacks).")


def _unparse_lr_name(lr_pair: str) -> Tuple[str, str]:
    """Parse '<ligand>_to_<receptor>' into (ligand, receptor)."""
    token = "_to_"
    if token not in lr_pair:
        raise ValueError(
            f"Bad lr_pair '{lr_pair}': expected '<ligand>_to_<receptor>'"
        )
    lig, rec = lr_pair.split(token, 1)
    return lig, rec


# ----------------------------- Resolution logic -----------------------------

def resolve_percell_attention_dir(percell_att_dir: Union[str, Path]) -> Path:
    """
    Resolve directory containing raw per-cell matrix files:
      meta_gene_orders.npz + percell_*.npz

    (This is used for TF->TG / LR->TG matrix-based plotting.)
    """
    root = Path(percell_att_dir)

    def ok(d: Path) -> bool:
        return (d / "meta_gene_orders.npz").is_file() and any(d.glob("percell_*.npz"))

    if ok(root):
        return root

    candidates = [root / "per_cell_attn", root / "percell_attentions"]
    for c in candidates:
        if ok(c):
            return c

    if root.is_dir():
        hit = [d for d in root.iterdir() if d.is_dir() and ok(d)]
        if len(hit) == 1:
            return hit[0]

    raise FileNotFoundError(
        "Could not locate raw per-cell attention artifacts. "
        "Expected meta_gene_orders.npz and percell_*.npz."
    )


def resolve_lrscore_npz(
    *,
    lrscore_npz: Optional[Union[str, Path]] = None,
    search_root: Optional[Union[str, Path]] = None,
    mode: str = "percell",
    stage: Optional[str] = None,
) -> Path:
    """
    Resolve the derived per-cell L->R score tensor file saved by percell_lr_inference.

    Default filename pattern:
      {mode}_lrscore_percell.npz
    or stage-aware:
      {mode}_lrscore_percell__{stage}.npz
    """
    if lrscore_npz is not None:
        p = Path(lrscore_npz)
        if not p.is_file():
            raise FileNotFoundError(f"lrscore_npz not found: {p}")
        return p

    if search_root is None:
        raise ValueError("Either lrscore_npz or search_root must be provided.")

    root = Path(search_root)
    name = f"{mode}_lrscore_percell__{stage}.npz" if stage else f"{mode}_lrscore_percell.npz"

    # deterministic search order
    candidates = [
        root / name,
        root / "attentions" / "percell_attentions" / name,
        root / "attentions" / "percell_attentions" / "results" / name,
        root.parent / name,
    ]
    for c in candidates:
        if c.is_file():
            return c

    # last resort: unique match
    patt = re.compile(re.escape(name) + r"$")
    hits = []
    for base in [root, root / "attentions", root / "attentions" / "percell_attentions", root.parent]:
        if base.is_dir():
            for p in base.rglob("*.npz"):
                if patt.search(p.name):
                    hits.append(p)
    hits = sorted(set(hits))
    if len(hits) == 1:
        return hits[0]

    msg = "\n".join([f"  - {p}" for p in candidates])
    raise FileNotFoundError(
        "Could not resolve per-cell LR score tensor (.npz).\n"
        f"Searched for: {name}\nCandidates:\n{msg}\n"
        "Tip: pass lrscore_npz explicitly (path to percell_lrscore_percell.npz)."
    )


# ----------------------------- Core loaders -----------------------------

def load_lrscore_tensor(
    *,
    lrscore_npz: Optional[Union[str, Path]] = None,
    search_root: Optional[Union[str, Path]] = None,
    mode: str = "percell",
    stage: Optional[str] = None,
) -> Dict[str, Any]:
    """Load {mode}_lrscore_percell(.npz) and return its dict."""
    p = resolve_lrscore_npz(lrscore_npz=lrscore_npz, search_root=search_root, mode=mode, stage=stage)
    Z = _safe_np_load(p)

    required = ("lr_score_tensor", "cell_indices", "ligand_list", "receptor_list")
    missing = [k for k in required if k not in Z]
    if missing:
        raise KeyError(
            f"Missing keys in lrscore file {p}: {missing}.\n"
            f"Available: {list(Z.keys())}"
        )
    return Z


# ----------------------------- Plotting -----------------------------

def plot_lr_attention_map(
    *,
    adata,
    lr_pair: str,
    lrscore_npz: Optional[Union[str, Path]] = None,
    search_root: Optional[Union[str, Path]] = None,
    mode: str = "percell",
    stage: Optional[str] = None,
    percell_att_dir_fallback: Optional[Union[str, Path]] = None,
    agg_over_tg_fallback: str = "mean",
    out_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    point_size: float = 8.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show: bool = False,
) -> Path:
    """
    Plot a spatial heatmap for a ligand-receptor pair.

    Preferred data source (your setup):
      percell_lrscore_percell.npz  (contains lr_score_tensor: C x n_lig x n_rec)

    Fallback (if lrscore file isn't available):
      raw percell_*.npz files containing W_lr (TG x LR) -> requires TG aggregation.

    Returns the saved PNG path.
    """
    import matplotlib.pyplot as plt

    lr_pair = str(lr_pair).strip()
    lig, rec = _unparse_lr_name(lr_pair)

    xy = _get_spatial_xy(adata)
    n_obs = int(adata.n_obs)

    vals = np.full((n_obs,), np.nan, dtype=np.float32)

    used = None

    # ---- Preferred: lrscore tensor ----
    if (lrscore_npz is not None) or (search_root is not None):
        Z = load_lrscore_tensor(lrscore_npz=lrscore_npz, search_root=search_root, mode=mode, stage=stage)

        lig_list = [str(x) for x in np.array(Z["ligand_list"]).tolist()]
        rec_list = [str(x) for x in np.array(Z["receptor_list"]).tolist()]

        if lig not in lig_list:
            raise KeyError(f"Ligand '{lig}' not found in ligand_list (len={len(lig_list)}).\nExample: {lig_list[:10]}")
        if rec not in rec_list:
            raise KeyError(f"Receptor '{rec}' not found in receptor_list (len={len(rec_list)}).\nExample: {rec_list[:10]}")
        li = lig_list.index(lig)
        ri = rec_list.index(rec)

        tensor = np.asarray(Z["lr_score_tensor"])
        cell_idx = np.asarray(Z["cell_indices"]).astype(int)

        if tensor.ndim != 3:
            raise ValueError(f"lr_score_tensor must be 3D (cells, lig, rec). Got shape: {tensor.shape}")
        if tensor.shape[0] != cell_idx.shape[0]:
            raise ValueError(f"lr_score_tensor cells {tensor.shape[0]} != cell_indices {cell_idx.shape[0]}")
        if li >= tensor.shape[1] or ri >= tensor.shape[2]:
            raise IndexError(f"Index out of bounds for tensor shape {tensor.shape}: li={li}, ri={ri}")

        percell_vals = tensor[:, li, ri].astype(np.float32, copy=False)

        # map into adata space via cell_indices
        good = (cell_idx >= 0) & (cell_idx < n_obs)
        vals[cell_idx[good]] = percell_vals[good]
        used = "lrscore_tensor"

    # ---- Fallback: raw per-cell W_lr matrices (TG x LR) ----
    if used is None:
        if percell_att_dir_fallback is None:
            raise FileNotFoundError(
                "Could not plot LR map because lrscore file was not provided/resolved, "
                "and percell_att_dir_fallback was not provided."
            )
        d = resolve_percell_attention_dir(percell_att_dir_fallback)
        meta = _safe_np_load(d / "meta_gene_orders.npz")
        lr_names = [str(x) for x in np.array(meta["lr_pair_names"]).tolist()]
        if lr_pair not in lr_names:
            # helpful nearest matches
            import difflib
            close = difflib.get_close_matches(lr_pair, lr_names, n=10, cutoff=0.6)
            msg = "\n".join([f"  - {c}" for c in close]) if close else "  (none)"
            raise KeyError(f"lr_pair '{lr_pair}' not found in meta lr_pair_names.\nClosest matches:\n{msg}")
        lr_idx = lr_names.index(lr_pair)

        files = sorted(d.glob("percell_*.npz"))
        for f in files:
            z = _safe_np_load(f)
            cell_i = int(np.array(z.get("cell_index", [-1])).ravel()[0])
            if cell_i < 0 or cell_i >= n_obs:
                continue
            if "W_lr" not in z:
                continue
            W = np.asarray(z["W_lr"], dtype=np.float32)  # (TG, LR)
            col = W[:, lr_idx]
            if agg_over_tg_fallback == "mean":
                vals[cell_i] = float(np.mean(col))
            elif agg_over_tg_fallback == "sum":
                vals[cell_i] = float(np.sum(col))
            elif agg_over_tg_fallback == "max":
                vals[cell_i] = float(np.max(col))
            elif agg_over_tg_fallback == "nonzero_mean":
                nz = col[col != 0]
                vals[cell_i] = float(np.mean(nz)) if nz.size else 0.0
            else:
                raise ValueError("agg_over_tg_fallback must be mean/sum/max/nonzero_mean")
        used = "W_lr_fallback"

    # ---- Plot ----
    if out_path is None:
        out_path = Path(f"attnmap_lr__{_sanitize_filename(lr_pair)}.png")
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=vals,
        s=float(point_size),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.0,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()  # common for spatial coordinates
    ax.set_xticks([])
    ax.set_yticks([])

    ttl = title if title is not None else f"{lr_pair} ({used})"
    ax.set_title(ttl)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("LR intensity")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_tf_attention_map(
    *,
    adata,
    tf_name: str,
    percell_att_dir: Union[str, Path],
    agg_over_tg: str = "mean",
    out_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    point_size: float = 8.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show: bool = False,
) -> Path:
    """
    Plot TF->TG attention map from raw per-cell W_tf matrices (TG x TF).

    NOTE: This uses the raw per-cell matrix outputs (percell_*.npz). If you later
    create a single-file TF score tensor similar to lr_score_tensor, we can add
    a preferred loader like plot_lr_attention_map().
    """
    import matplotlib.pyplot as plt
    import difflib

    tf_name = str(tf_name).strip()
    d = resolve_percell_attention_dir(percell_att_dir)
    meta = _safe_np_load(d / "meta_gene_orders.npz")
    tf_names = [str(x) for x in np.array(meta["tf_names"]).tolist()]

    if tf_name not in tf_names:
        close = difflib.get_close_matches(tf_name, tf_names, n=10, cutoff=0.6)
        msg = "\n".join([f"  - {c}" for c in close]) if close else "  (none)"
        raise KeyError(f"tf '{tf_name}' not found in meta tf_names.\nClosest matches:\n{msg}")
    tf_idx = tf_names.index(tf_name)

    xy = _get_spatial_xy(adata)
    n_obs = int(adata.n_obs)
    vals = np.full((n_obs,), np.nan, dtype=np.float32)

    files = sorted(d.glob("percell_*.npz"))
    for f in files:
        z = _safe_np_load(f)
        cell_i = int(np.array(z.get("cell_index", [-1])).ravel()[0])
        if cell_i < 0 or cell_i >= n_obs:
            continue
        if "W_tf" not in z:
            continue
        W = np.asarray(z["W_tf"], dtype=np.float32)  # (TG, TF)
        col = W[:, tf_idx]
        if agg_over_tg == "mean":
            vals[cell_i] = float(np.mean(col))
        elif agg_over_tg == "sum":
            vals[cell_i] = float(np.sum(col))
        elif agg_over_tg == "max":
            vals[cell_i] = float(np.max(col))
        elif agg_over_tg == "nonzero_mean":
            nz = col[col != 0]
            vals[cell_i] = float(np.mean(nz)) if nz.size else 0.0
        else:
            raise ValueError("agg_over_tg must be mean/sum/max/nonzero_mean")

    if out_path is None:
        out_path = Path(f"attnmap_tf__{_sanitize_filename(tf_name)}.png")
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=vals,
        s=float(point_size),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.0,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ttl = title if title is not None else f"{tf_name} (W_tf)"
    ax.set_title(ttl)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("TF intensity")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_lrtg_attention_map(*args, **kwargs):
    """Alias: historically LR->TG map; here routed to plot_lr_attention_map()."""
    return plot_lr_attention_map(*args, **kwargs)
