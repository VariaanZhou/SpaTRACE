#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from typing import Dict, Tuple, Sequence, List
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.spatial import cKDTree

# =========================
# Utilities & I/O helpers
# =========================

def _require(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return p

def load_names_from_npy(path: str | Path) -> List[str]:
    arr = np.load(_require(path), allow_pickle=True)
    return [str(x) for x in arr.tolist()]

def load_matrix(path: str | Path) -> np.ndarray:
    p = _require(path)
    ext = p.suffix.lower()
    if ext == ".npy":
        M = np.load(p, allow_pickle=False)
    elif ext in (".csv", ".tsv", ".txt"):
        sep = "," if ext == ".csv" else "\t"
        M = pd.read_csv(p, sep=sep, header=None).values
    else:
        raise ValueError(f"Unsupported matrix format: {ext}")
    return np.asarray(M, dtype=float)

def write_combined_csv(M: np.ndarray, ligands: Sequence[str], receptors: Sequence[str], stages: Sequence[str], out_path: Path) -> Path:
    cols = []
    for rname in receptors:
        cols.extend([f"{rname}|{st}" for st in stages])
    df = pd.DataFrame(np.asarray(M), index=ligands, columns=cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    return out_path

# =========================
# Expression filtering API
# =========================

def _to_dense(X):
    try:
        return X.toarray()
    except Exception:
        return np.asarray(X)

def compute_pct_expressed_by_celltype(
    adata,
    genes: Sequence[str],
    *,
    groupby: str,
    cell_types: Sequence[str] | None = None,
    expr_cutoff: float = 0.0,
) -> Dict[str, pd.DataFrame]:
    """For each cell type, % of cells with expression > expr_cutoff for each gene."""
    genes_present = [g for g in genes if g in adata.var_names]
    ct_series = adata.obs[groupby].astype(str)
    if cell_types is None:
        cell_types = ct_series.unique().tolist()

    out: Dict[str, pd.DataFrame] = {}
    for ct in cell_types:
        mask = (ct_series.values == ct)
        if mask.sum() == 0 or len(genes_present) == 0:
            out[ct] = pd.DataFrame(columns=["gene", "pct_expressed"])
            continue
        Xi = _to_dense(adata[mask, genes_present].X)
        pct = (Xi > expr_cutoff).sum(axis=0) / max(1, Xi.shape[0])
        out[ct] = pd.DataFrame({"gene": genes_present, "pct_expressed": pct})
    return out

def union_kept_genes(pct_by_ct: Dict[str, pd.DataFrame], *, pct_threshold: float) -> List[str]:
    keep: set[str] = set()
    for df in pct_by_ct.values():
        if df.empty:
            continue
        keep.update(df.loc[df["pct_expressed"] >= pct_threshold, "gene"].tolist())
    return sorted(keep)

# =========================
# Core compute helpers
# =========================

def present_names_and_indices(adata, wanted: Sequence[str], order_list: Sequence[str]) -> tuple[List[str], np.ndarray, np.ndarray]:
    present = [g for g in wanted if g in adata.var_names]
    if not present:
        return [], np.array([], int), np.array([], int)
    idx_adata = np.array([adata.var_names.get_loc(g) for g in present], dtype=int)
    pos = {name: i for i, name in enumerate(order_list)}
    idx_M = np.array([pos[g] for g in present], dtype=int)
    return present, idx_adata, idx_M

def per_gene_max(adata, gene_idx_adata: np.ndarray, *, row_block: int = 8192, dtype=np.float32) -> np.ndarray:
    n = adata.n_obs
    out = np.zeros(len(gene_idx_adata), dtype=dtype)
    for start in range(0, n, row_block):
        end = min(n, start + row_block)
        Xi = adata[start:end, gene_idx_adata].X
        try:
            Xi = Xi.toarray()
        except Exception:
            Xi = np.asarray(Xi)
        out = np.maximum(out, Xi.max(axis=0).astype(dtype, copy=False))
    out[out == 0] = 1.0
    return out

def extract_expr(adata, mask_rows: np.ndarray, gene_idx_adata: np.ndarray, *, norm_max: np.ndarray | None = None, row_block: int = 8192, dtype=np.float32) -> np.ndarray:
    rows = np.flatnonzero(mask_rows)
    out = np.zeros((len(rows), len(gene_idx_adata)), dtype=dtype)
    w = 0
    for start in range(0, len(rows), row_block):
        idx_rows = rows[start:start+row_block]
        Xi = adata[idx_rows, gene_idx_adata].X
        try:
            Xi = Xi.toarray()
        except Exception:
            Xi = np.asarray(Xi)
        Xi = Xi.astype(dtype, copy=False)
        if norm_max is not None:
            Xi /= norm_max
        out[w:w+len(idx_rows)] = Xi
        w += len(idx_rows)
    return out

def coords_from_adata(adata) -> np.ndarray:
    if 'spatial' in adata.obsm_keys():
        return adata.obsm['spatial']
    raise ValueError("adata.obsm['spatial'] missing.")

def get_stage_masks(adata, *, batch_key: str, stage_name: str, groupby: str, sender: str, receiver: str) -> tuple[np.ndarray, np.ndarray]:
    if batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key='{batch_key}' not found in adata.obs")
    if groupby not in adata.obs.columns:
        raise ValueError(f"groupby='{groupby}' not found in adata.obs")
    is_stage = (adata.obs[batch_key].astype(str).values == stage_name)
    ct = adata.obs[groupby].astype(str).values
    s_mask = (ct == sender) & is_stage
    r_mask = (ct == receiver) & is_stage
    return s_mask, r_mask

def sender_neighbors(sender_xy: np.ndarray, receiver_xy: np.ndarray, radius: float) -> list[list[int]]:
    tree = cKDTree(sender_xy)
    return tree.query_ball_point(receiver_xy, r=radius)

def compute_W_block(
    sender_expr: np.ndarray,          # (#sender × Lb)
    receiver_expr_block: np.ndarray,  # (#receiver × Rb)
    neigh_lists: Sequence[Sequence[int]],
    M_block: np.ndarray,              # (Lb × Rb)
    *,
    dtype=np.float32,
) -> np.ndarray:
    n_recv = receiver_expr_block.shape[0]
    Lb = sender_expr.shape[1]
    lig_near = np.zeros((n_recv, Lb), dtype=dtype)
    for u, neigh in enumerate(neigh_lists):
        if len(neigh):
            lig_near[u, :] = sender_expr[neigh, :].sum(axis=0)
    W = (receiver_expr_block.T @ lig_near).T  # (Lb × Rb) after transpose
    return (W * M_block).astype(dtype, copy=False)

# =========================
# New: within-stage batching & non-spatial fallback
# =========================

WITHIN_STAGE_KEY_CANDIDATES_DEFAULT = ('batch', 'Batch', 'sample', 'Sample', 'library_id', 'LibraryID')


def _find_within_stage_key(adata, candidates: Sequence[str]) -> str | None:
    for k in candidates:
        if k in adata.obs.columns:
            return k
    return None


def _reindex_W(W: np.ndarray, kept_ligs: Sequence[str], kept_recs: Sequence[str], ligands_all: Sequence[str], receptors_all: Sequence[str], fill=float('nan')) -> np.ndarray:
    Li = {g: i for i, g in enumerate(kept_ligs)}
    Rj = {g: j for j, g in enumerate(kept_recs)}
    L, R = len(ligands_all), len(receptors_all)
    out = np.full((L, R), fill, dtype=float)
    for li, lg in enumerate(ligands_all):
        si = Li.get(lg)
        if si is None:
            continue
        for rj, rc in enumerate(receptors_all):
            sj = Rj.get(rc)
            if sj is None:
                continue
            out[li, rj] = W[si, sj]
    return out


def _aggregate_within_stage(W_list: List[np.ndarray], weights: Sequence[float] | None, agg: str) -> np.ndarray:
    if not W_list:
        return None
    W_stack = np.stack(W_list, axis=2)  # (L, R, B)
    if agg == 'mean' or weights is None:
        return np.nanmean(W_stack, axis=2)
    w = np.asarray(weights, dtype=float)
    w = w / (np.nansum(w) + 1e-12)
    return np.nansum(W_stack * w.reshape((1, 1, -1)), axis=2)


# =========================
# Pair- and stage-level API
# =========================

def _compute_pair_W_for_stage(
    *,
    M_stage: np.ndarray,                # (L × R) aligned to ligand_order × receptor_order
    adata,
    ligand_order: Sequence[str],
    receptor_order: Sequence[str],
    batch_key: str,
    stage_name: str,
    groupby: str,
    sender: str,
    receiver: str,
    pct_threshold: float,
    expr_cutoff: float,
    keep_all_requested: bool,
    radius: float,
    receptor_block: int,
    row_block: int,
    dtype,
    use_local_intensity: bool,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Compute W (unnormalized) for ONE stage, optionally without spatial info.
       If keep_all_requested=True, bypass pct-threshold filtering.
    """
    # choose genes
    if keep_all_requested:
        ligs_used = [g for g in ligand_order if g in adata.var_names]
        recs_used = [g for g in receptor_order if g in adata.var_names]
    else:
        stage_mask = (adata.obs[batch_key].astype(str).values == stage_name)
        adata_stage = adata[stage_mask].copy()
        pct_sender = compute_pct_expressed_by_celltype(adata_stage, ligand_order, groupby=groupby, cell_types=[sender], expr_cutoff=expr_cutoff)
        pct_receiver = compute_pct_expressed_by_celltype(adata_stage, receptor_order, groupby=groupby, cell_types=[receiver], expr_cutoff=expr_cutoff)
        ligs_used = union_kept_genes(pct_sender, pct_threshold=pct_threshold)
        recs_used = union_kept_genes(pct_receiver, pct_threshold=pct_threshold)

    if not ligs_used or not recs_used:
        return np.zeros((0, 0), dtype=dtype), [], []

    lig_idx_in_order = np.array([ligand_order.index(g) for g in ligs_used], dtype=int)
    rec_idx_in_order = np.array([receptor_order.index(g) for g in recs_used], dtype=int)

    M_used = np.asarray(M_stage[np.ix_(lig_idx_in_order, rec_idx_in_order)], dtype=dtype)

    lig_present, lig_idx_adata, _ = present_names_and_indices(adata, ligs_used, ligand_order)
    rec_present, rec_idx_adata, _ = present_names_and_indices(adata, recs_used, receptor_order)
    Lp, Rp = len(lig_present), len(rec_present)
    if Lp == 0 or Rp == 0:
        return np.zeros((Lp, Rp), dtype=dtype), lig_present, rec_present

    if use_local_intensity:
        lig_max = per_gene_max(adata, lig_idx_adata, row_block=row_block, dtype=dtype)
        rec_max = per_gene_max(adata, rec_idx_adata, row_block=row_block, dtype=dtype)

        s_mask, r_mask = get_stage_masks(adata, batch_key=batch_key, stage_name=stage_name, groupby=groupby, sender=sender, receiver=receiver)
        nS, nR = int(s_mask.sum()), int(r_mask.sum())
        if nS == 0 or nR == 0:
            return np.zeros((Lp, Rp), dtype=dtype), lig_present, rec_present

        XY = coords_from_adata(adata)
        neigh_lists = sender_neighbors(XY[s_mask], XY[r_mask], radius)

        sender_expr = extract_expr(adata, s_mask, lig_idx_adata, norm_max=lig_max, row_block=row_block, dtype=dtype)
        receiver_expr = extract_expr(adata, r_mask, rec_idx_adata, norm_max=rec_max, row_block=row_block, dtype=dtype)

        posL = {g: i for i, g in enumerate(ligs_used)}
        posR = {g: i for i, g in enumerate(recs_used)}
        lig_pos = np.array([posL[g] for g in lig_present], dtype=int)
        rec_pos = np.array([posR[g] for g in rec_present], dtype=int)

        W_pair = np.zeros((Lp, Rp), dtype=dtype)
        for r0 in range(0, Rp, receptor_block):
            r1 = min(Rp, r0 + receptor_block)
            recv_blk = receiver_expr[:, r0:r1]
            M_blk = M_used[np.ix_(lig_pos, rec_pos[r0:r1])].astype(dtype, copy=False)
            W_blk = compute_W_block(sender_expr, recv_blk, neigh_lists, M_blk, dtype=dtype)
            W_pair[:, r0:r1] = W_blk
        return W_pair, lig_present, rec_present
    else:
        # Non-spatial fallback: pct_sender * pct_receiver * M
        stage_mask = (adata.obs[batch_key].astype(str).values == stage_name)
        adata_stage = adata[stage_mask].copy()
        pct_sender = compute_pct_expressed_by_celltype(adata_stage, lig_present, groupby=groupby, cell_types=[sender], expr_cutoff=expr_cutoff)
        pct_receiver = compute_pct_expressed_by_celltype(adata_stage, rec_present, groupby=groupby, cell_types=[receiver], expr_cutoff=expr_cutoff)
        s_map = dict(zip(pct_sender[sender]['gene'], pct_sender[sender]['pct_expressed'])) if lig_present else {}
        r_map = dict(zip(pct_receiver[receiver]['gene'], pct_receiver[receiver]['pct_expressed'])) if rec_present else {}
        W = np.zeros((Lp, Rp), dtype=dtype)
        for i, lg in enumerate(lig_present):
            lp = float(s_map.get(lg, 0.0))
            for j, rc in enumerate(rec_present):
                rp = float(r_map.get(rc, 0.0))
                W[i, j] = lp * rp * M_used[lig_idx_in_order[ligand_order.index(lg)], rec_idx_in_order[receptor_order.index(rc)]]
        return W, lig_present, rec_present


def combined_matrix_for_pair_across_stages(
    *,
    adata,
    M_by_stage: dict[str, np.ndarray],      # stage -> (L×R) weights aligned to orders
    ligand_order: Sequence[str],
    receptor_order: Sequence[str],
    stages: Sequence[str],
    batch_key: str,
    groupby: str,
    sender: str,
    receiver: str,
    pct_threshold: float = 0.1,
    expr_cutoff: float = 0.0,
    radius: float = 50.0,
    receptor_block: int = 128,
    row_block: int = 8192,
    dtype=np.float32,
    use_local_intensity: bool = True,
    within_stage_key_candidates: Sequence[str] = WITHIN_STAGE_KEY_CANDIDATES_DEFAULT,
    within_stage_agg: str = 'weighted',  # 'weighted' | 'mean' | 'none'
) -> tuple[np.ndarray, List[str], List[str]]:
    """
    Two-pass per (sender,receiver):
      1) discover union of active ligands/receptors across stages via pct-threshold;
      2) recompute each stage on the union (missing -> 0), per-stage max-normalize,
         then interleave columns (per receptor: stage1|stage2|...).
      Matches Script 2 behavior, with optional within-stage batch aggregation.
    """
    # pass 1: unions (respect within-stage batches but only for union discovery at stage level)
    lig_union: set[str] = set()
    rec_union: set[str] = set()
    per_stage_cache = {}  # stage -> list of (adata_sub, ligs_used, recs_used, W_raw)

    for st in stages:
        stage_mask = (adata.obs[batch_key].astype(str).values == st)
        adata_stage = adata[stage_mask].copy()
        if adata_stage.n_obs == 0:
            continue
        ws_key = _find_within_stage_key(adata_stage, within_stage_key_candidates) if within_stage_agg != 'none' else None

        if ws_key is None:
            W_raw, ligs_used, recs_used = _compute_pair_W_for_stage(
                M_stage=M_by_stage[st], adata=adata,
                ligand_order=ligand_order, receptor_order=receptor_order,
                batch_key=batch_key, stage_name=st, groupby=groupby,
                sender=sender, receiver=receiver,
                pct_threshold=pct_threshold, expr_cutoff=expr_cutoff,
                keep_all_requested=False, radius=radius,
                receptor_block=receptor_block, row_block=row_block,
                dtype=dtype, use_local_intensity=use_local_intensity
            )[0], _compute_pair_W_for_stage(
                M_stage=M_by_stage[st], adata=adata,
                ligand_order=ligand_order, receptor_order=receptor_order,
                batch_key=batch_key, stage_name=st, groupby=groupby,
                sender=sender, receiver=receiver,
                pct_threshold=pct_threshold, expr_cutoff=expr_cutoff,
                keep_all_requested=False, radius=radius,
                receptor_block=receptor_block, row_block=row_block,
                dtype=dtype, use_local_intensity=use_local_intensity
            )[1], _compute_pair_W_for_stage(
                M_stage=M_by_stage[st], adata=adata,
                ligand_order=ligand_order, receptor_order=receptor_order,
                batch_key=batch_key, stage_name=st, groupby=groupby,
                sender=sender, receiver=receiver,
                pct_threshold=pct_threshold, expr_cutoff=expr_cutoff,
                keep_all_requested=False, radius=radius,
                receptor_block=receptor_block, row_block=row_block,
                dtype=dtype, use_local_intensity=use_local_intensity
            )[2]
            lig_union.update(ligs_used); rec_union.update(recs_used)
            per_stage_cache[st] = [(None, ligs_used, recs_used, W_raw)]
        else:
            batch_vals = adata_stage.obs[ws_key].astype(str).unique().tolist()
            stage_ligs, stage_recs = set(), set()
            cache = []
            for b in batch_vals:
                ad_b = adata_stage[adata_stage.obs[ws_key].astype(str) == b].copy()
                # compute with filtering
                W_b, ligs_b, recs_b = _compute_pair_W_for_stage(
                    M_stage=M_by_stage[st], adata=ad_b,
                    ligand_order=ligand_order, receptor_order=receptor_order,
                    batch_key=batch_key, stage_name=st, groupby=groupby,
                    sender=sender, receiver=receiver,
                    pct_threshold=pct_threshold, expr_cutoff=expr_cutoff,
                    keep_all_requested=False, radius=radius,
                    receptor_block=receptor_block, row_block=row_block,
                    dtype=dtype, use_local_intensity=use_local_intensity
                )
                stage_ligs.update(ligs_b); stage_recs.update(recs_b)
                cache.append((ad_b, ligs_b, recs_b, W_b))
            lig_union.update(stage_ligs); rec_union.update(stage_recs)
            per_stage_cache[st] = cache

    lig_union = sorted(lig_union)
    rec_union = sorted(rec_union)
    L, R = len(lig_union), len(rec_union)
    if L == 0 or R == 0:
        return np.zeros((0, 0), dtype=dtype), lig_union, rec_union

    # pass 2: recompute on union with per-stage normalization & within-stage aggregation
    per_stage_normed: list[np.ndarray] = []
    for st in stages:
        stage_mask = (adata.obs[batch_key].astype(str).values == st)
        adata_stage = adata[stage_mask].copy()
        ws_key = _find_within_stage_key(adata_stage, within_stage_key_candidates) if within_stage_agg != 'none' else None

        if ws_key is None:
            W_raw, ligs_used, recs_used = _compute_pair_W_for_stage(
                M_stage=M_by_stage[st], adata=adata,
                ligand_order=ligand_order, receptor_order=receptor_order,
                batch_key=batch_key, stage_name=st, groupby=groupby,
                sender=sender, receiver=receiver,
                pct_threshold=0.0, expr_cutoff=expr_cutoff,  # union decided
                keep_all_requested=True, radius=radius,
                receptor_block=receptor_block, row_block=row_block,
                dtype=dtype, use_local_intensity=use_local_intensity
            )
            W_union = _reindex_W(W_raw, ligs_used, recs_used, lig_union, rec_union, fill=0.0)
        else:
            batch_vals = adata_stage.obs[ws_key].astype(str).unique().tolist()
            per_batch_W = []
            per_batch_weights = []
            for b in batch_vals:
                ad_b = adata_stage[adata_stage.obs[ws_key].astype(str) == b].copy()
                W_b, ligs_b, recs_b = _compute_pair_W_for_stage(
                    M_stage=M_by_stage[st], adata=ad_b,
                    ligand_order=ligand_order, receptor_order=receptor_order,
                    batch_key=batch_key, stage_name=st, groupby=groupby,
                    sender=sender, receiver=receiver,
                    pct_threshold=0.0, expr_cutoff=expr_cutoff,
                    keep_all_requested=True, radius=radius,
                    receptor_block=receptor_block, row_block=row_block,
                    dtype=dtype, use_local_intensity=use_local_intensity
                )
                Wb = _reindex_W(W_b, ligs_b, recs_b, lig_union, rec_union, fill=0.0)
                per_batch_W.append(Wb)
                recv_mask = (ad_b.obs[groupby].astype(str).values == receiver)
                per_batch_weights.append(int(np.sum(recv_mask)))

            if not per_batch_W:
                W_union = np.zeros((L, R), dtype=float)
            else:
                if within_stage_agg == 'weighted':
                    W_union = _aggregate_within_stage(per_batch_W, per_batch_weights, agg='weighted')
                elif within_stage_agg == 'mean':
                    W_union = _aggregate_within_stage(per_batch_W, None, agg='mean')
                else:  # none
                    # if 'none', default to mean across batches to produce a single canvas per stage
                    W_union = _aggregate_within_stage(per_batch_W, None, agg='mean')

        # per-stage max normalize
        m = float(np.nanmax(W_union)) if W_union.size else 0.0
        Wn = np.zeros_like(W_union) if m <= 0 else (W_union / (m + 1e-12)).astype(dtype, copy=False)
        per_stage_normed.append(Wn)

    # interleave: for each receptor, append stage columns
    blocks = []
    for r in range(R):
        for s_idx in range(len(stages)):
            blocks.append(per_stage_normed[s_idx][:, r])
    M_combined = np.stack(blocks, axis=1)  # (L × (R*len(stages)))
    return M_combined, lig_union, rec_union

# =========================
# Plotting (matplotlib)
# =========================

def plot_combined_matrix(
    M_combined: np.ndarray,
    ligands_all: Sequence[str],
    receptors_all: Sequence[str],
    stages: Sequence[str],
    *,
    sender: str,
    receiver: str,
    out_dir: Path,
    cmap: str = "Reds",
    figsize=(8, 6),
    dpi: int = 300,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    L, RC = M_combined.shape
    assert RC == len(receptors_all) * len(stages), "Combined matrix has unexpected column count."

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    with np.errstate(invalid='ignore'):
        masked = np.ma.masked_invalid(M_combined)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    im.cmap.set_bad(color="#f1f1f1")

    ax.set_yticks(np.arange(L))
    ax.set_yticklabels(ligands_all, fontsize=14)
    ax.set_ylabel(f"Ligands expressed by {sender}", fontsize=14)

    ax.set_xticks([r * len(stages) + (len(stages) // 2) for r in range(len(receptors_all))])
    ax.set_xticklabels(receptors_all, rotation=45, ha="right", fontsize=14)
    ax.set_xlabel("Per receptor: " + " | ".join(stages), fontsize=14)

    for rline in range(1, len(receptors_all)):
        ax.axvline(rline * len(stages) - 0.5, color="white", lw=0.8, alpha=0.9)

    ax.set_title(f"Receptors expressed by {receiver}", fontsize=15, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Normalized intensity (W / max W)", fontsize=12)
    fig.tight_layout()

    s_tag = sender.replace(" ", "_")
    r_tag = receiver.replace(" ", "_")
    base = f"{s_tag}__to__{r_tag}__combined_{'_'.join(stages)}"

    png = out_dir / (base + ".png")
    svg = out_dir / (base + ".svg")
    pdf = out_dir / (base + ".pdf")
    fig.savefig(png); fig.savefig(svg); fig.savefig(pdf)
    plt.close(fig)
    return {"png": png, "svg": svg, "pdf": pdf}

# =========================
# CLI
# =========================

# def main():
#     ap = argparse.ArgumentParser(description="Combined multi-stage LR heatmaps per (sender, receiver) — Script1 matched to Script2 features.")
#     ap.add_argument("--adata", required=True, help=".h5ad with spatial data and obs annotations.")
#     ap.add_argument("--lr_matrix", required=True, help="LR interaction intensity (L×R) .npy/.csv/.tsv")
#     ap.add_argument("--ligands", required=True, help="ligand_order.npy")
#     ap.add_argument("--receptors", required=True, help="receptor_order_lr.npy")
#     ap.add_argument("--batch_key", default="Time point")
#     ap.add_argument("--groupby", default="annotation")
#     ap.add_argument("--stages", nargs="+", default=("E12.5", "E14.5", "E16.5"))
#     ap.add_argument("--senders", nargs="+", required=True)
#     ap.add_argument("--receivers", nargs="+", required=True)
#     ap.add_argument("--pct_threshold", type=float, default=0.1)
#     ap.add_argument("--expr_cutoff", type=float, default=0.0)
#     ap.add_argument("--use_local_intensity", action="store_true", default=True, help="Use spatial neighbor-sum model; else percent-expression product.")
#     ap.add_argument("--radius", type=float, default=50.0)
#     ap.add_argument("--receptor_block", type=int, default=64)
#     ap.add_argument("--row_block", type=int, default=8192)
#     ap.add_argument("--figsize", nargs=2, type=float, default=(8, 6))
#     ap.add_argument("--dpi", type=int, default=300)
#     ap.add_argument("--out_dir", required=True)
#     ap.add_argument("--export_csv", action="store_true", default=True)
#     ap.add_argument("--prenorm_X", action="store_true", default=False, help="Pre-normalize adata.X per gene to max=1 (in-place), like Script 2.")
#     ap.add_argument("--within_stage_keys", nargs="*", default=list(WITHIN_STAGE_KEY_CANDIDATES_DEFAULT), help="Candidate obs keys for within-stage batches.")
#     ap.add_argument("--within_stage_agg", choices=["weighted", "mean", "none"], default="weighted", help="How to aggregate per-batch canvases into one per stage.")
#     args = ap.parse_args()
#
#     # Load inputs
#     adata = sc.read_h5ad(_require(args.adata))
#
#     # Optional in-place per-gene max normalization (like Script 2)
#     if args.prenorm_X:
#         try:
#             X = adata.X.toarray()
#         except AttributeError:
#             X = np.asarray(adata.X)
#         with np.errstate(invalid='ignore', divide='ignore'):
#             colmax = np.nanmax(X, axis=0)
#             colmax[colmax == 0] = 1.0
#             X = X / colmax
#         adata.X = X
#
#     if args.groupby not in adata.obs.columns:
#         # try to discover a cell-type column
#         candidates = ['cell_type', 'celltype', 'CellType', 'major_celltype', 'annotation', 'cluster', 'leiden']
#         found = next((c for c in candidates if c in adata.obs.columns), None)
#         if found is None:
#             raise ValueError(f"No cell-type column found for GROUPBY='{args.groupby}'.")
#         adata.obs[args.groupby] = adata.obs[found].astype(str)
#     else:
#         adata.obs[args.groupby] = adata.obs[args.groupby].astype(str)
#
#     if args.batch_key not in adata.obs.columns:
#         raise ValueError(f"batch_key='{args.batch_key}' not found in adata.obs")
#
#     LIGS = load_names_from_npy(args.ligands)
#     RECS = load_names_from_npy(args.receptors)
#     M = load_matrix(args.lr_matrix)
#     if M.shape != (len(LIGS), len(RECS)):
#         raise ValueError(f"LR matrix shape {M.shape} != ({len(LIGS)},{len(RECS)})")
#
#     # Build per-stage LR dict (same matrix reused unless you point to stage-specific files)
#     M_by_stage = {st: M for st in args.stages}
#
#     out_dir = Path(args.out_dir)
#     for s in args.senders:
#         for r in args.receivers:
#             M_comb, lig_union, rec_union = combined_matrix_for_pair_across_stages(
#                 adata=adata,
#                 M_by_stage=M_by_stage,
#                 ligand_order=LIGS,
#                 receptor_order=RECS,
#                 stages=tuple(args.stages),
#                 batch_key=args.batch_key,
#                 groupby=args.groupby,
#                 sender=s,
#                 receiver=r,
#                 pct_threshold=args.pct_threshold,
#                 expr_cutoff=args.expr_cutoff,
#                 radius=args.radius,
#                 receptor_block=args.receptor_block,
#                 row_block=args.row_block,
#                 dtype=np.float32,
#                 use_local_intensity=args.use_local_intensity,
#                 within_stage_key_candidates=tuple(args.within_stage_keys),
#                 within_stage_agg=args.within_stage_agg,
#             )
#
#             # Plot heatmaps (will be empty if size==0, but we still may want CSV)
#             if M_comb.size > 0:
#                 fig_paths = plot_combined_matrix(
#                     M_combined=M_comb,
#                     ligands_all=lig_union,
#                     receptors_all=rec_union,
#                     stages=tuple(args.stages),
#                     sender=s,
#                     receiver=r,
#                     out_dir=out_dir,
#                     figsize=tuple(args.figsize),
#                     dpi=args.dpi,
#                 )
#
#             # CSV
#             if args.export_csv:
#                 s_tag = s.replace(" ", "_")
#                 r_tag = r.replace(" ", "_")
#                 base = f"{s_tag}__to__{r_tag}__combined_{'_'.join(args.stages)}"
#                 write_combined_csv(M_comb, lig_union, rec_union, tuple(args.stages), out_dir / (base + ".csv"))
