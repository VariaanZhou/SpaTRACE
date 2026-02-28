from __future__ import annotations

import os
import glob
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from analysis.utils  import _unparse_lr_var_name
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

# per_cell_analysis.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt

# Optional: only needed if you want spatial/dpt visualizations
try:
    import scanpy as sc
except Exception:
    sc = None

import tensorflow as tf

# Your model helper (already defined in SpaTRACE_v1_0.py)
from model.SpaTRACE_v1_0 import infer_cpu


def _outer_abs(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """|A @ B^T| where A: (G_a, d), B: (G_b, d) -> (G_a, G_b)."""
    return np.abs(A @ B.T)

def _keep_top_k_per_col(M: np.ndarray, k: Optional[int]) -> np.ndarray:
    if k is None or k <= 0 or k >= M.shape[0]:
        return M
    idx = np.argpartition(-M, k-1, axis=0)[:k]  # top-k row indices (unordered)
    mask = np.zeros_like(M, dtype=bool)
    mask[idx, np.arange(M.shape[1])[None, :]] = True
    return np.where(mask, M, 0.0)

def _filter_matrix(M: np.ndarray, top_k: Optional[int], threshold: Optional[float], full_weights = True) -> Tuple[np.ndarray, np.ndarray]:
    X = M
    if not full_weights or top_k is not None:
        X = _keep_top_k_per_col(M, top_k)
    if threshold is None:
        mask = np.ones_like(X, dtype=bool)
        return X, mask
    mask = X > float(threshold)
    return np.where(mask, X, 0.0), mask

def _load_test_paths_from_npz(test_npz_path: str) -> np.ndarray:
    """Return object array of paths; each element is a list of metacell indices."""
    Z = np.load(test_npz_path, allow_pickle=True)
    if "paths" not in Z.files:
        raise ValueError(f"{test_npz_path} must contain 'paths'.")
    paths = Z["paths"]
    # normalize to list of python lists
    out = []
    for p in paths:
        if isinstance(p, (list, tuple, np.ndarray)):
            out.append(list(map(int, list(p))))
        else:
            raise ValueError("Unexpected path element; expected list/array of metacell indices.")
    return np.array(out, dtype=object)

def rowwise_nonzero_mean(mat2d: np.ndarray) -> np.ndarray:
    """
    For each LR row, average nonzero entries across TG columns.
    If a row is all zeros, returns 0 for that row.
    """
    nz = (mat2d != 0)
    sums = (mat2d * nz).sum(axis=1)
    counts = nz.sum(axis=1)
    counts = np.where(counts == 0, 1, counts)  # avoid /0
    return sums / counts

def build_lr_matrix(lr_scores, ligand_list, receptor_list, lr_names):
    """
    Build ligand×receptor matrix as a pandas DataFrame.

    lr_scores: 1D array aligned with lr_names
    lr_names: list like ["LigA_to_RecB", ...]
    """
    import numpy as np
    import pandas as pd

    lr_scores = np.asarray(lr_scores).reshape(-1)
    if lr_scores.shape[0] != len(lr_names):
        raise ValueError(
            f"lr_scores length {lr_scores.shape[0]} != len(lr_names) {len(lr_names)}"
        )

    lig_to_i = {l: i for i, l in enumerate(ligand_list)}
    rec_to_j = {r: j for j, r in enumerate(receptor_list)}

    M = np.zeros((len(ligand_list), len(receptor_list)), dtype=np.float32)

    for k, lr in enumerate(lr_names):
        ligand, receptor = _unparse_lr_var_name(lr)
        if ligand not in lig_to_i:
            raise ValueError(f"Ligand name {ligand} not found in ligand_list")
        if receptor not in rec_to_j:
            raise ValueError(f"Receptor name {receptor} not found in receptor_list")
        M[lig_to_i[ligand], rec_to_j[receptor]] = float(lr_scores[k])

    return pd.DataFrame(M, index=ligand_list, columns=receptor_list)

def normalize_global_max(df: pd.DataFrame) -> pd.DataFrame:
    maxv = np.max(df.to_numpy())
    if maxv == 0:
        return df.copy()
    return df / maxv

def tidy_long(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    out = df.stack().reset_index()
    out.columns = ["Ligand", "Receptor", "Score"]
    out["LR"] = out["Ligand"] + "_to_" + out["Receptor"]
    out["Stage"] = stage
    return out

def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.to_numpy()
    mu = arr.mean(axis=1, keepdims=True)
    sd = arr.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    z = (arr - mu) / sd
    return pd.DataFrame(z, index=df.index, columns=df.columns)

def make_clustermap(df: pd.DataFrame, stage: str, path_png: str):
    # z-score by row to highlight relative receptor preferences per ligand
    zdf = zscore_rows(df)
    g = sns.clustermap(
        zdf, cmap="vlag", center=0.0, xticklabels=False, yticklabels=False,
        linewidths=0, figsize=(10, 10)
    )
    plt.suptitle(f"LR Score (row z-score) — {stage}", y=1.02)
    g.cax.set_title("z-score", fontsize=9)
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close()

def make_top_pairs_barplot(df_long: pd.DataFrame, stage: str, path_png: str, top_n: int):
    topk = df_long.sort_values("Score", ascending=False).head(top_n).copy()
    plt.figure(figsize=(12, max(6, int(top_n * 0.3))))
    sns.barplot(data=topk, x="Score", y="LR", orient="h", dodge=False)
    plt.xlabel("LR Score")
    plt.ylabel("Ligand→Receptor")
    plt.title(f"Top {top_n} LR Pairs — {stage}")
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.close()



def make_heatmap(df: pd.DataFrame, stage: str, path_png: str):
    """
    A readable heatmap with **visible names** for ligands & receptors.
    """
    # pick subset to label clearly
    ligs, recs = list(df.index), list(df.columns)
    sub = df.loc[ligs, recs]

    plt.figure(figsize=(max(10, len(recs)*0.35), max(8, len(ligs)*0.35)))
    ax = sns.heatmap(sub, cmap="viridis", xticklabels=True, yticklabels=True, cbar_kws={"label": "Normalized LR score"})
    ax.set_xlabel("Receptor"); ax.set_ylabel("Ligand")
    ax.set_title(f"LR Score (highlighted names) — {stage}")

    # color & bold tick labels to emphasize selected items
    # (all shown are selected, but this makes them pop)
    for ticklab in ax.get_xticklabels():
        ticklab.set_color("#2b6cb0")   # blue
        ticklab.set_fontweight("bold")
        ticklab.set_rotation(90)
    for ticklab in ax.get_yticklabels():
        ticklab.set_color("#b83280")   # magenta
        ticklab.set_fontweight("bold")
        ticklab.set_rotation(0)

    # draw subtle grid lines to frame highlighted rows/cols
    ax.set_xticks(np.arange(len(recs)+1)-0.5, minor=True)
    ax.set_yticks(np.arange(len(ligs)+1)-0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.close()


from typing import Dict
import os
import glob
import numpy as np

def aggregate_LR_intensity(
    lr_tg_intensity_matrix,
    ligand_list,
    receptor_list,
    lr_names,
    out_dir,
    mode,
    stage=None,
    top_n_bar=20,
    top_k=None,
    # permutation test over gene columns (significance-based gene selection)
    n_permutations=1000,
    gene_top_k=None,
    gene_alpha=0.05 / 20,
    random_state=0,
):
    """
    Aggregate LR intensity with two optional filters:

    1) Gene (column) selection via permutation test:
       - observed column statistic = sum over rows
       - null by permuting columns within each row
       - one-sided p-values: P(null >= observed)
       - keep gene_top_k smallest p OR p<=gene_alpha (guard: keep best if none)

    2) Within each retained gene column, keep only its top-k entries (by value)
       before computing rowwise non-zero mean.

    Output:
      - Saves: {mode}_lrscore_mean(.npy/.npz) (+ optional pvals/mask .npy)
      - If mode=="global": also saves PNG plots (clustermap/bar/heatmap) using
        DataFrame-safe guards.
      - Returns: numpy array (ligand x receptor) normalized by global max.
    """
    import os
    import numpy as np

    # ------------------------
    # Sanity check: LR names
    # ------------------------
    for lr_name in lr_names:
        ligand, receptor = _unparse_lr_var_name(lr_name)
        if ligand not in ligand_list:
            raise ValueError(f"Ligand name {ligand} not found in the ligand list!")
        if receptor not in receptor_list:
            raise ValueError(f"Receptor name {receptor} not found in the receptor list!")

    X_np = np.asarray(lr_tg_intensity_matrix).copy()
    if X_np.ndim != 2:
        raise ValueError(f"lr_tg_intensity_matrix must be 2D, got shape {X_np.shape}")

    n_rows, n_cols = X_np.shape
    if n_cols == 0 or n_rows == 0:
        raise ValueError(f"lr_tg_intensity_matrix has invalid shape {X_np.shape}")

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------
    # 1) Permutation test to select gene columns
    # -----------------------------------------
    do_gene_selection = (gene_top_k is not None) or (gene_alpha is not None)

    pvals = None
    keep_cols_mask = np.ones(n_cols, dtype=bool)

    if do_gene_selection:
        if gene_top_k is not None:
            if not isinstance(gene_top_k, int) or gene_top_k <= 0:
                raise ValueError(f"gene_top_k must be a positive int or None, got {gene_top_k}")
        if gene_alpha is not None:
            if not (0.0 < float(gene_alpha) <= 1.0):
                raise ValueError(f"gene_alpha must be in (0, 1], got {gene_alpha}")
        if not isinstance(n_permutations, int) or n_permutations <= 0:
            raise ValueError(f"n_permutations must be a positive int, got {n_permutations}")

        rng = np.random.default_rng(random_state)

        # Observed stat per gene column
        obs = X_np.sum(axis=0)

        # Null distribution via shuffling columns within each row
        ge_count = np.zeros(n_cols, dtype=np.int64)

        for _ in range(n_permutations):
            perm_idx = rng.random((n_rows, n_cols)).argsort(axis=1)
            permuted = np.take_along_axis(X_np, perm_idx, axis=1)
            null = permuted.sum(axis=0)
            ge_count += (null >= obs)

        pvals = (ge_count + 1) / (n_permutations + 1)

        if gene_top_k is not None:
            k = min(gene_top_k, n_cols)
            keep_idx = np.argsort(pvals)[:k]
            keep_cols_mask = np.zeros(n_cols, dtype=bool)
            keep_cols_mask[keep_idx] = True
        else:
            keep_cols_mask = (pvals <= gene_alpha)
            if not np.any(keep_cols_mask):
                keep_cols_mask[np.argmin(pvals)] = True

        # Zero out non-selected columns
        X_np[:, ~keep_cols_mask] = 0

        # Save p-values and mask
        pval_name = f"{mode}_gene_perm_pvals__{stage}.npy" if stage else f"{mode}_gene_perm_pvals.npy"
        np.save(os.path.join(out_dir, pval_name), pvals, allow_pickle=True)

        mask_name = f"{mode}_gene_keep_mask__{stage}.npy" if stage else f"{mode}_gene_keep_mask.npy"
        np.save(os.path.join(out_dir, mask_name), keep_cols_mask, allow_pickle=True)

    # ------------------------------------------------
    # 2) Keep only top-k entries within each kept column
    # ------------------------------------------------
    if top_k is not None:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive int or None, got {top_k}")

        k = min(top_k, n_rows)
        cols_to_process = np.where(keep_cols_mask)[0] if do_gene_selection else np.arange(n_cols)

        for j in cols_to_process:
            col = X_np[:, j]
            if not np.any(col):
                continue
            top_idx = np.argpartition(col, -k)[-k:]
            mask = np.zeros(n_rows, dtype=bool)
            mask[top_idx] = True
            col[~mask] = 0
            X_np[:, j] = col

    # -----------------------------------------
    # Rowwise non-zero mean -> LR-pair scores
    # -----------------------------------------
    filt_scores = rowwise_nonzero_mean(X_np)

    # Build LR matrix (may be DataFrame or ndarray depending on your helpers)
    filt_df = build_lr_matrix(filt_scores, ligand_list, receptor_list, lr_names)

    # Normalize by global max (may return DataFrame or ndarray)
    filt_df_norm = normalize_global_max(filt_df)

    # Convert to numpy for saving/return
    if hasattr(filt_df_norm, "to_numpy"):
        arr = filt_df_norm.to_numpy()
    else:
        arr = np.asarray(filt_df_norm)
    arr = arr.astype(np.float32, copy=False)

    # ---- SAVE ONLY (always) ----
    file_name = f"{mode}_lrscore_mean__{stage}.npy" if stage else f"{mode}_lrscore_mean.npy"
    np.save(os.path.join(out_dir, file_name), arr, allow_pickle=True)

    np.savez_compressed(
        os.path.join(out_dir, file_name.replace(".npy", ".npz")),
        lr_score_matrix=arr,
        ligand_list=np.array(ligand_list, dtype=object),
        receptor_list=np.array(receptor_list, dtype=object),
        lr_names=np.array(lr_names, dtype=object),
        stage=np.array([stage], dtype=object),
        mode=np.array([mode], dtype=object),
        top_k=np.array([top_k], dtype=object),
        gene_top_k=np.array([gene_top_k], dtype=object),
        gene_alpha=np.array([gene_alpha], dtype=object),
        n_permutations=np.array([n_permutations], dtype=np.int32),
        random_state=np.array([random_state], dtype=np.int32),
    )

    # ---- OPTIONAL: PLOTS ONLY FOR GLOBAL ----
    if mode == "global":
        # Make sure we feed DataFrames into tidy_long / plotting helpers
        try:
            import pandas as pd

            if hasattr(filt_df_norm, "stack"):
                df_plot = filt_df_norm
            else:
                df_plot = pd.DataFrame(arr, index=ligand_list, columns=receptor_list)

            tidy_filt = tidy_long(df_plot, stage)

            cluster_name = "global_clustermap_lrscore.png"
            top_pair_name = "global_top_pairs_bar.png"
            heat_map_name = "global_heatmap_lrscore.png"

            make_clustermap(df_plot, stage, os.path.join(out_dir, cluster_name))
            make_top_pairs_barplot(tidy_filt, stage, os.path.join(out_dir, top_pair_name), top_n_bar)
            make_heatmap(df_plot, stage, os.path.join(out_dir, heat_map_name))
        except Exception as e:
            # Don't fail inference if plotting deps mismatch on cluster
            # (you still get npy/npz)
            print(f"[WARN] global plotting skipped due to error: {type(e).__name__}: {e}")

    return arr



# ----------------------------- IO helpers -----------------------------

def _auto_detect_project_npz(data_root: Path) -> Tuple[str, Path]:
    """
    Find a single '*_tensors_train.npz' under common locations and return (project, path).
    Mirrors run_experiment's auto-detect idea. :contentReference[oaicite:2]{index=2}
    """
    candidates: List[Path] = []

    # Common layouts:
    # (A) output_dir/data_triple/<proj>_tensors_train.npz  (from run_preprocess) :contentReference[oaicite:3]{index=3}
    candidates += list((data_root / "data_triple").glob("*_tensors_train.npz"))

    # (B) output_dir/<proj>/data_triple/<proj>_tensors_train.npz (some users keep project subdir)
    candidates += list(data_root.glob("*/data_triple/*_tensors_train.npz"))

    # (C) direct under root
    candidates += list(data_root.glob("*_tensors_train.npz"))

    # de-dup
    candidates = sorted(set(candidates))

    if len(candidates) == 0:
        raise FileNotFoundError(f"No '*_tensors_train.npz' found under {data_root}")

    if len(candidates) > 1:
        msg = "\n".join(str(p) for p in candidates[:20])
        raise RuntimeError(
            f"Multiple '*_tensors_train.npz' found under {data_root}. "
            f"Please keep only one, or modify _auto_detect_project_npz.\n{msg}"
        )

    train_npz = candidates[0]
    stem = train_npz.stem  # "<project>_tensors_train"
    project = stem[: -len("_tensors_train")]
    return project, train_npz


def _find_processed_adata(output_dir: Path, project: str):
    """
    Tries to load project_adata_processed.h5ad if scanpy is available.
    run_preprocess saves these exact filenames. :contentReference[oaicite:4]{index=4}
    """
    if sc is None:
        return None

    cand = [
        output_dir / f"{project}_adata_processed.h5ad",
        output_dir / project / f"{project}_adata_processed.h5ad",
    ]
    for p in cand:
        if p.is_file():
            return sc.read_h5ad(str(p))
    return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_np_load(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as Z:
        return {k: Z[k] for k in Z.files}

import tqdm

# ----------------------------- Core compute -----------------------------

def per_cell_att_compute(
    adata,
    model: tf.keras.Model,
    output_dir: str,
    *,
    train_npz_path: str,
    project_name: str,
    tlength: int = 5,
    n_paths_per_cell: int = 8,
    target_cell_indices: Optional[Iterable[int]] = None,
    seed: int = 0,
    attn_subdir: str = "per_cell_attn",
    save_per_path: bool = False,
) -> Path:
    """
    Correct per-cell extraction (cell-specific query row only).

    NOTE:
    - `output_dir` is ONLY where we write outputs.
    - `train_npz_path` explicitly points to {project}_tensors_train.npz for loading paths/features.
    """
    out_root = Path(output_dir)

    train_npz = Path(train_npz_path)
    if not train_npz.is_file():
        raise FileNotFoundError(f"train_npz_path not found: {train_npz}")

    project = str(project_name)
    blobs = _safe_np_load(train_npz)

    tf_all = blobs["tf"].astype(np.float32)[:, :tlength, :]           # (N, L, Gtf)
    lr_all = blobs["lr_pair"].astype(np.float32)[:, :tlength, :]      # (N, L, Glr)
    target_full = blobs["target"].astype(np.float32)                  # (N, L+1, Gtgt)
    tgt_in = target_full[:, 1:, :].astype(np.float32)[:, :tlength, :] # (N, L, Gtgt)

    paths = blobs["paths"]  # object array, each path length should be L+1 typically
    tf_names = np.array(blobs["tf_names"])
    tg_names = np.array(blobs["target_names"])
    lr_names = np.array(blobs["lr_pair_names"])

        # ---------------- ROBUST PATH PARSING + CHECKS ----------------
    n_cells = int(adata.n_obs)

    def _parse_path_entry(p):
        """
        Convert one path entry to a 1D np.int64 array.
        Supports:
          - list/tuple of ints
          - np.ndarray of ints/floats
          - object arrays containing lists/arrays
          - string forms like "[1 2 3]" or "1,2,3"
        """
        import numpy as np
        import re

        if p is None:
            return None

        # If already an ndarray, flatten it
        if isinstance(p, np.ndarray):
            arr = np.asarray(p).ravel()
            # object arrays often contain lists/arrays; handle below
            if arr.dtype != object:
                if np.issubdtype(arr.dtype, np.integer):
                    return arr.astype(np.int64, copy=False)
                if np.issubdtype(arr.dtype, np.floating):
                    if np.all(np.isfinite(arr)) and np.all(arr == np.floor(arr)):
                        return arr.astype(np.int64)
                    return None
            # fall through for object dtype

        # If list/tuple
        if isinstance(p, (list, tuple)):
            arr = np.asarray(p).ravel()
            if arr.dtype == object:
                # Could be nested lists; try flatten by concatenation
                flat = []
                for x in arr.tolist():
                    if isinstance(x, (list, tuple, np.ndarray)):
                        flat.extend(np.asarray(x).ravel().tolist())
                    else:
                        flat.append(x)
                arr = np.asarray(flat)
            if np.issubdtype(arr.dtype, np.integer):
                return arr.astype(np.int64, copy=False)
            if np.issubdtype(arr.dtype, np.floating):
                if np.all(np.isfinite(arr)) and np.all(arr == np.floor(arr)):
                    return arr.astype(np.int64)
                return None
            return None

        # If string
        if isinstance(p, str):
            # extract integers (handles spaces/commas/brackets)
            nums = re.findall(r"-?\d+", p)
            if not nums:
                return None
            return np.asarray([int(x) for x in nums], dtype=np.int64)

        # Otherwise: try numpy conversion
        try:
            arr = np.asarray(p).ravel()
        except Exception:
            return None

        if arr.dtype == object:
            # Try treating as list-like
            try:
                return _parse_path_entry(arr.tolist())
            except Exception:
                return None

        if np.issubdtype(arr.dtype, np.integer):
            return arr.astype(np.int64, copy=False)
        if np.issubdtype(arr.dtype, np.floating):
            if np.all(np.isfinite(arr)) and np.all(arr == np.floor(arr)):
                return arr.astype(np.int64)
            return None
        return None

    paths_i64 = []
    bad = []
    for pidx in range(len(paths)):
        arr = _parse_path_entry(paths[pidx])
        if arr is None or arr.size == 0:
            bad.append((pidx, "unparseable/empty"))
            if len(bad) >= 5:
                break
            continue

        lo = int(arr.min())
        hi = int(arr.max())
        if lo < 0 or hi >= n_cells:
            bad.append((pidx, f"index_range=[{lo},{hi}] not within [0,{n_cells-1}]"))
            if len(bad) >= 5:
                break
            continue

        paths_i64.append(arr)

    if bad:
        ex = "\n".join([f"  - path_idx={i}: {why}" for i, why in bad])
        raise ValueError(
            "Invalid `paths` detected after robust parsing.\n"
            f"adata.n_obs={n_cells}\nExamples:\n{ex}\n"
            "This usually means your paths are still referencing a different indexing space "
            "(e.g., metacells or a different adata ordering)."
        )

    # IMPORTANT: Replace original `paths` with parsed int paths
    paths = paths_i64
    # ---------------------------------------------------------------

    # Build inverted index: cell_id -> list[path_idx] where the path contains that cell
    inv: Dict[int, List[int]] = {}
    for pidx, p in enumerate(paths):
        for node in p.tolist():
            inv.setdefault(int(node), []).append(pidx)

    if target_cell_indices is None:
        target_cell_indices = range(n_cells)

    rng = np.random.default_rng(seed)

    save_dir = out_root if (attn_subdir in ("", None)) else (out_root / attn_subdir)
    _ensure_dir(save_dir)

    # Save metadata once
    meta_path = save_dir / "meta_gene_orders.npz"
    if not meta_path.exists():
        np.savez_compressed(
            meta_path,
            project=np.array([project], dtype=object),
            tf_names=tf_names,
            tg_names=tg_names,
            lr_pair_names=lr_names,
            tlength=np.array([tlength], dtype=np.int32),
            note=np.array(
                ["Per-cell TF->TG and LR->TG matrices extracted at the target cell's query position only."],
                dtype=object,
            ),
        )

    # Helper: find a usable query position for cell_i within a path
    # We only accept positions < tlength because model outputs net_* have Lq == tlength.
    def _pick_query_pos(path_1d: np.ndarray, cell_i: int) -> Optional[int]:
        hits = np.where(path_1d == cell_i)[0]
        if hits.size == 0:
            return None
        # Prefer earliest valid query position
        for h in hits.tolist():
            if int(h) < int(tlength):
                return int(h)
        return None  # cell only appears at positions not represented in query tokens

    for cell_i in tqdm.tqdm(target_cell_indices):
        cell_i = int(cell_i)
        cand = inv.get(cell_i, [])
        if len(cand) == 0:
            continue

        pick = cand if len(cand) <= n_paths_per_cell else rng.choice(cand, size=n_paths_per_cell, replace=False)
        pick = np.array(pick, dtype=np.int64)

        # Determine query position per picked path; drop paths where cell_i is not in query range
        qpos = []
        kept_path_indices = []
        for pidx in pick.tolist():
            p = paths[int(pidx)]
            pos = _pick_query_pos(p, cell_i)
            if pos is None:
                continue
            qpos.append(pos)
            kept_path_indices.append(int(pidx))

        if len(kept_path_indices) == 0:
            # No usable paths where cell_i occurs within query positions
            continue

        kept_path_indices = np.array(kept_path_indices, dtype=np.int64)
        qpos = np.array(qpos, dtype=np.int64)

        lr_in = lr_all[kept_path_indices]      # (B, L, Glr)
        tf_in = tf_all[kept_path_indices]      # (B, L, Gtf)
        tgt_in_b = tgt_in[kept_path_indices]   # (B, L, Gtgt)

        out = infer_cpu(model, lr_in, tf_in, tgt_in_b)

        net_tf = out["output_7"].numpy()  # (B, Lq, Lk, Gtgt, Gtf)
        net_lr = out["output_8"].numpy()  # (B, Lq, Lk, Gtgt, Glr)

        B = net_lr.shape[0]
        if B != qpos.shape[0]:
            raise RuntimeError("Internal error: batch size mismatch between outputs and qpos.")

        # Extract per-path matrices using ONLY the target cell's query row, average over keys only.
        W_tf_paths = np.zeros((B, net_tf.shape[3], net_tf.shape[4]), dtype=np.float32)  # (B, Gtgt, Gtf)
        W_lr_paths = np.zeros((B, net_lr.shape[3], net_lr.shape[4]), dtype=np.float32)  # (B, Gtgt, Glr)

        for b in range(B):
            q = int(qpos[b])
            # slice: (Lk, Gtgt, Gpred)
            tf_slice = net_tf[b, q, :, :, :]  # (Lk, Gtgt, Gtf)
            lr_slice = net_lr[b, q, :, :, :]  # (Lk, Gtgt, Glr)

            W_tf_paths[b] = np.mean(np.abs(tf_slice), axis=0).astype(np.float32)  # (Gtgt, Gtf)
            W_lr_paths[b] = np.mean(np.abs(lr_slice), axis=0).astype(np.float32)  # (Gtgt, Glr)

        # Aggregate across sampled paths (now truly cell-specific)
        W_tf_cell = np.mean(W_tf_paths, axis=0).astype(np.float32)
        W_lr_cell = np.mean(W_lr_paths, axis=0).astype(np.float32)

        out_cell = save_dir / f"percell_{cell_i:06d}.npz"
        np.savez_compressed(
            out_cell,
            cell_index=np.array([cell_i], dtype=np.int32),
            sampled_path_indices=kept_path_indices.astype(np.int32, copy=False),
            sampled_query_positions=qpos.astype(np.int32, copy=False),
            W_tf=W_tf_cell,
            W_lr=W_lr_cell,
        )

        if save_per_path:
            per_path_dir = save_dir / "per_path" / f"cell_{cell_i:06d}"
            _ensure_dir(per_path_dir)
            for b in range(B):
                np.savez_compressed(
                    per_path_dir / f"path_{int(kept_path_indices[b]):06d}.npz",
                    cell_index=np.array([cell_i], dtype=np.int32),
                    path_index=np.array([int(kept_path_indices[b])], dtype=np.int32),
                    query_pos=np.array([int(qpos[b])], dtype=np.int32),
                    W_tf=W_tf_paths[b],
                    W_lr=W_lr_paths[b],
                )

    return save_dir

# ----------------------------- Aggregation / inference -----------------------------

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np


def _safe_np_load(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as Z:
        return {k: Z[k] for k in Z.files}


def _unparse_lr_var_name(lr_name: str) -> Tuple[str, str]:
    token = "_to_"
    if token not in lr_name:
        raise ValueError(f"Bad LR var name '{lr_name}': expected '<ligand>_to_<receptor>'.")
    lig, rec = lr_name.split(token, 1)
    return lig, rec


def rowwise_nonzero_mean(X: np.ndarray) -> np.ndarray:
    """
    Mean across non-zero entries in each row. If a row is all zeros => 0.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"rowwise_nonzero_mean expects 2D, got {X.shape}")
    mask = (X != 0)
    sums = (X * mask).sum(axis=1)
    cnts = mask.sum(axis=1)
    out = np.zeros((X.shape[0],), dtype=np.float32)
    nz = cnts > 0
    out[nz] = (sums[nz] / cnts[nz]).astype(np.float32)
    return out


def build_lr_matrix(
    lr_scores: np.ndarray,
    ligand_list: List[str],
    receptor_list: List[str],
    lr_names: List[str],
) -> np.ndarray:
    """
    Build ligand×receptor matrix (numpy) from LR-pair score vector aligned with lr_names.
    """
    lig_to_i = {l: i for i, l in enumerate(ligand_list)}
    rec_to_j = {r: j for j, r in enumerate(receptor_list)}

    M = np.zeros((len(ligand_list), len(receptor_list)), dtype=np.float32)
    for k, lr in enumerate(lr_names):
        lig, rec = _unparse_lr_var_name(str(lr))
        M[lig_to_i[lig], rec_to_j[rec]] = float(lr_scores[k])
    return M


def normalize_global_max(df):
    """
    Normalize by global max.
    Accepts either:
      - pandas.DataFrame  -> returns pandas.DataFrame
      - numpy.ndarray     -> returns numpy.ndarray
    """
    import numpy as np
    try:
        import pandas as pd
    except Exception:
        pd = None

    # pandas DataFrame path
    if pd is not None and isinstance(df, pd.DataFrame):
        arr = df.to_numpy()
        maxv = float(np.max(arr)) if arr.size else 0.0
        if maxv <= 0:
            return df.copy()
        return df / maxv

    # numpy array path
    arr = np.asarray(df)
    maxv = float(np.max(arr)) if arr.size else 0.0
    if maxv <= 0:
        return arr.astype(np.float32, copy=False)
    return (arr / maxv).astype(np.float32, copy=False)


def percell_lr_inference(
    attn_save_dir: str,
    *,
    out_dir: Optional[str] = None,
    mode: str = "percell",
    stage: Optional[str] = None,
    # within-column top-k filtering (over LR rows)
    top_k: Optional[int] = None,
    # permutation test over TG columns (significance-based TG selection)
    n_permutations: int = 1000,
    gene_top_k: Optional[int] = None,
    gene_alpha: Optional[float] = 0.05 / 20,
    random_state: int = 0,
    # saving controls
    save_mean_lrtg: bool = False,
) -> np.ndarray:
    """
    TRUE per-cell LR inference (NO FIGURES).

    For each cell file percell_*.npz containing W_lr (TG, LR):
      X_cell = W_lr.T  -> (LR, TG)

    We compute TG-column selection ONCE (optional) using the mean LR×TG matrix,
    then apply the resulting TG mask to every cell for fast per-cell scoring.

    For each cell:
      - optional TG mask
      - optional within-column top-k over LR rows
      - rowwise_nonzero_mean => LR scores (len LR)
      - build ligand×receptor matrix (n_lig x n_rec)

    Saves:
      - {mode}_lrscore_percell(.npz): tensor (n_cells, n_lig, n_rec) + metadata
      - {mode}_lrscore_mean(.npy/.npz): mean across cells (compatibility)
      - optional: {mode}_mean_LR_by_TG(.npz): filtered mean LR×TG matrix (LR,TG)

    Returns:
      lr_score_tensor_norm: np.ndarray of shape (n_cells, n_lig, n_rec), float32
    """
    import os
    from pathlib import Path
    import numpy as np

    attn_dir = Path(attn_save_dir)
    outp = Path(out_dir) if out_dir is not None else attn_dir
    outp.mkdir(parents=True, exist_ok=True)

    meta_path = attn_dir / "meta_gene_orders.npz"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run per_cell_att_compute first.")
    meta = _safe_np_load(meta_path)

    lr_names = [str(x) for x in np.array(meta["lr_pair_names"]).tolist()]
    tg_names = [str(x) for x in np.array(meta["tg_names"]).tolist()]

    # Derive ligand/receptor lists from lr_names
    ligands = []
    receptors = []
    for lr in lr_names:
        lig, rec = _unparse_lr_var_name(lr)
        ligands.append(lig)
        receptors.append(rec)
    ligand_list = sorted(set(ligands))
    receptor_list = sorted(set(receptors))

    # Per-cell files
    percell_files = sorted(attn_dir.glob("percell_*.npz"))
    if len(percell_files) == 0:
        raise FileNotFoundError(f"No percell_*.npz found in {attn_dir}. Run per_cell_att_compute first.")

    # ------------------------
    # Load all W_lr and stack
    # ------------------------
    # W_lr per cell: (TG, LR)
    Wlr_cells = []
    cell_ids = []
    for p in percell_files:
        Z = _safe_np_load(p)
        W_lr = Z["W_lr"].astype(np.float32, copy=False)
        # optional stored cell_index
        if "cell_index" in Z:
            cell_ids.append(int(np.array(Z["cell_index"]).ravel()[0]))
        else:
            # fallback parse from filename
            cell_ids.append(int(p.stem.split("_")[-1]))
        Wlr_cells.append(W_lr)

    Wlr_stack = np.stack(Wlr_cells, axis=0)  # (C, TG, LR)
    C, TG, LR = Wlr_stack.shape

    if LR != len(lr_names):
        raise ValueError(f"Per-cell W_lr LR dim {LR} != len(lr_names) {len(lr_names)}")
    if TG != len(tg_names):
        raise ValueError(f"Per-cell W_lr TG dim {TG} != len(tg_names) {len(tg_names)}")

    # Mean LR×TG across cells for TG-selection
    mean_W_lr = np.mean(Wlr_stack, axis=0)           # (TG, LR)
    mean_LR_by_TG = mean_W_lr.T.copy()               # (LR, TG)

    X_sel = mean_LR_by_TG.copy()  # matrix used only for TG selection/topk preview
    n_rows, n_cols = X_sel.shape  # (LR, TG)

    # -----------------------------------------
    # 1) TG selection by permutation test (ONCE)
    # -----------------------------------------
    do_gene_selection = (gene_top_k is not None) or (gene_alpha is not None)
    pvals = None
    keep_cols_mask = np.ones(n_cols, dtype=bool)

    if do_gene_selection:
        if gene_top_k is not None and (not isinstance(gene_top_k, int) or gene_top_k <= 0):
            raise ValueError(f"gene_top_k must be a positive int or None, got {gene_top_k}")
        if gene_alpha is not None and (not (0.0 < float(gene_alpha) <= 1.0)):
            raise ValueError(f"gene_alpha must be in (0, 1], got {gene_alpha}")
        if not isinstance(n_permutations, int) or n_permutations <= 0:
            raise ValueError(f"n_permutations must be a positive int, got {n_permutations}")

        rng = np.random.default_rng(random_state)
        obs = X_sel.sum(axis=0)  # per TG

        ge_count = np.zeros(n_cols, dtype=np.int64)
        for _ in range(n_permutations):
            perm_idx = rng.random((n_rows, n_cols)).argsort(axis=1)
            permuted = np.take_along_axis(X_sel, perm_idx, axis=1)
            null = permuted.sum(axis=0)
            ge_count += (null >= obs)

        pvals = (ge_count + 1) / (n_permutations + 1)

        if gene_top_k is not None:
            k = min(int(gene_top_k), n_cols)
            keep_idx = np.argsort(pvals)[:k]
            keep_cols_mask = np.zeros(n_cols, dtype=bool)
            keep_cols_mask[keep_idx] = True
        else:
            keep_cols_mask = (pvals <= float(gene_alpha))
            if not np.any(keep_cols_mask):
                keep_cols_mask[np.argmin(pvals)] = True

        # Save pvals + mask (stage-aware)
        pval_name = f"{mode}_gene_perm_pvals__{stage}.npy" if stage else f"{mode}_gene_perm_pvals.npy"
        np.save(str(outp / pval_name), pvals, allow_pickle=True)

        mask_name = f"{mode}_gene_keep_mask__{stage}.npy" if stage else f"{mode}_gene_keep_mask.npy"
        np.save(str(outp / mask_name), keep_cols_mask, allow_pickle=True)

    # Optionally store filtered mean LR×TG for later visualization
    if save_mean_lrtg:
        X_mean = mean_LR_by_TG.copy()
        X_mean[:, ~keep_cols_mask] = 0.0
        lrtg_name = f"{mode}_mean_LR_by_TG__{stage}.npz" if stage else f"{mode}_mean_LR_by_TG.npz"
        np.savez_compressed(
            str(outp / lrtg_name),
            mean_LR_by_TG=X_mean.astype(np.float32),
            lr_names=np.array(lr_names, dtype=object),
            tg_names=np.array(tg_names, dtype=object),
            keep_tg_mask=keep_cols_mask,
        )

    # ------------------------------------------------
    # 2) Per-cell LR scoring
    # ------------------------------------------------
    if top_k is not None:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive int or None, got {top_k}")
        k_lr = min(int(top_k), LR)
    else:
        k_lr = None

    # Allocate output tensor: (C, n_lig, n_rec)
    n_lig = len(ligand_list)
    n_rec = len(receptor_list)
    lr_score_tensor = np.zeros((C, n_lig, n_rec), dtype=np.float32)

    # Precompute column indices to process for top-k
    cols_to_process = np.where(keep_cols_mask)[0] if do_gene_selection else np.arange(TG)

    for ci in range(C):
        # cell LR×TG
        X = Wlr_stack[ci].T.copy()  # (LR, TG)
        # apply TG mask
        X[:, ~keep_cols_mask] = 0.0

        # top-k over LR rows within each kept TG col
        if k_lr is not None:
            for j in cols_to_process:
                col = X[:, j]
                if not np.any(col):
                    continue
                top_idx = np.argpartition(col, -k_lr)[-k_lr:]
                mask = np.zeros(LR, dtype=bool)
                mask[top_idx] = True
                col[~mask] = 0.0
                X[:, j] = col

        # LR scores for this cell
        lr_scores = rowwise_nonzero_mean(X)  # (LR,)

        # ligand×receptor matrix for this cell
        lr_mat = build_lr_matrix(lr_scores, ligand_list, receptor_list, lr_names)

        # normalize later globally across all cells (for comparability)
        lr_score_tensor[ci] = np.asarray(lr_mat, dtype=np.float32)

    # ------------------------------------------------
    # 3) Global normalization across ALL CELLS
    # ------------------------------------------------
    maxv = float(np.max(lr_score_tensor)) if lr_score_tensor.size else 0.0
    if maxv > 0:
        lr_score_tensor_norm = lr_score_tensor / maxv
    else:
        lr_score_tensor_norm = lr_score_tensor

    # ------------------------------------------------
    # 4) Save per-cell tensor + mean matrix
    # ------------------------------------------------
    base_cell = f"{mode}_lrscore_percell__{stage}" if stage else f"{mode}_lrscore_percell"
    np.savez_compressed(
        str(outp / f"{base_cell}.npz"),
        lr_score_tensor=lr_score_tensor_norm.astype(np.float16),  # save space
        cell_indices=np.array(cell_ids, dtype=np.int32),
        ligand_list=np.array(ligand_list, dtype=object),
        receptor_list=np.array(receptor_list, dtype=object),
        lr_names=np.array(lr_names, dtype=object),
        tg_names=np.array(tg_names, dtype=object),
        keep_tg_mask=keep_cols_mask,
        gene_perm_pvals=(pvals if pvals is not None else np.full(TG, np.nan, dtype=np.float64)),
        params=np.array([{
            "mode": mode,
            "stage": stage,
            "top_k": top_k,
            "n_permutations": n_permutations,
            "gene_top_k": gene_top_k,
            "gene_alpha": gene_alpha,
            "random_state": random_state,
            "normalization": "global_max_over_all_cells",
        }], dtype=object),
    )

    # Also save the mean (compatibility + quick summaries)
    mean_lr_mat = np.mean(lr_score_tensor_norm, axis=0).astype(np.float32)  # (n_lig, n_rec)
    base_mean = f"{mode}_lrscore_mean__{stage}" if stage else f"{mode}_lrscore_mean"
    np.save(str(outp / f"{base_mean}.npy"), mean_lr_mat, allow_pickle=True)
    np.savez_compressed(
        str(outp / f"{base_mean}.npz"),
        lr_score_matrix=mean_lr_mat.astype(np.float32),
        ligand_list=np.array(ligand_list, dtype=object),
        receptor_list=np.array(receptor_list, dtype=object),
        lr_names=np.array(lr_names, dtype=object),
        tg_names=np.array(tg_names, dtype=object),
        keep_tg_mask=keep_cols_mask,
        gene_perm_pvals=(pvals if pvals is not None else np.full(TG, np.nan, dtype=np.float64)),
    )

    return lr_score_tensor_norm

# ----------------------------- Visualization helpers -----------------------------

def _load_meta(attn_save_dir: Path):
    meta_path = attn_save_dir / "meta_gene_orders.npz"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}.")
    return _safe_np_load(meta_path)


def _load_summary_or_build(attn_save_dir: Path) -> Dict[str, np.ndarray]:
    summ = attn_save_dir / "percell_summary.npz"
    if summ.exists():
        return _safe_np_load(summ)
    percell_lr_inference(str(attn_save_dir))
    return _safe_np_load(summ)


def _lr_name(ligand: str, receptor: str) -> str:
    return f"{ligand}_to_{receptor}"


def directional_map(
    ligand: str,
    receptor: str,
    attn_save_dir: str,
    *,
    output_png: Optional[str] = None,
    spatial_key: str = "spatial",
    pseudotime_key: str = "dpt_pseudotime",
) -> Dict[str, np.ndarray]:
    """
    directional_map(ligand, receptor, attn_save_dir)

    If LR pair not found in lr_pair_names, raises with a "not significant" message.
    Otherwise:
      - computes per-cell scalar LR signal = sum_TG W_lr[TG, LRpair]
      - if an adata_processed.h5ad is found alongside, makes a spatial scatter (if available)
      - also returns arrays for your downstream usage
    """
    d = Path(attn_save_dir)
    summ = _load_summary_or_build(d)
    lr_names = np.array(summ["lr_pair_names"])
    tg_names = np.array(summ["tg_names"])
    cell_ids = summ["cell_indices"].astype(np.int32)

    lr = _lr_name(ligand, receptor)
    hits = np.where(lr_names == lr)[0]
    if hits.size == 0:
        raise ValueError(
            f"LR pair '{lr}' is not in lr_pair_names (not significant / filtered out)."
        )
    lr_idx = int(hits[0])

    # Load per-cell matrices to get per-cell scalar (summary only has mean)
    percell_files = [d / f"percell_{int(i):06d}.npz" for i in cell_ids.tolist()]
    signals = np.zeros((len(percell_files),), dtype=np.float32)
    for k, p in enumerate(percell_files):
        Z = _safe_np_load(p)
        W_lr = Z["W_lr"].astype(np.float32)  # (Gtgt, Glr)
        signals[k] = float(np.sum(W_lr[:, lr_idx]))

    # Optional spatial plot if we can load adata
    meta = _safe_np_load(d / "meta_gene_orders.npz")
    project = str(meta["project"][0]) if "project" in meta else ""

    # Try to locate processed adata next to the analysis directory
    adata_obj = _find_processed_adata(d.parent, project) or _find_processed_adata(d, project)
    if output_png is None:
        output_png = str(d / f"directional_map__{lr}.png")

    if adata_obj is not None and spatial_key in adata_obj.obsm_keys():
        coords = np.asarray(adata_obj.obsm[spatial_key])
        # cell_ids are integer indices into original adata ordering
        xy = coords[cell_ids]
        plt.figure(figsize=(6, 5))
        plt.scatter(xy[:, 0], xy[:, 1], s=10, c=signals)
        plt.gca().invert_yaxis()
        plt.title(f"Directional LR signal: {lr}\n(sum over TG of W_lr[TG, {lr}])")
        plt.xlabel("x"); plt.ylabel("y")
        plt.colorbar(label="signal")
        plt.tight_layout()
        plt.savefig(output_png, dpi=300)
        plt.close()
    else:
        # fallback: histogram
        plt.figure(figsize=(6, 4))
        plt.hist(signals, bins=40)
        plt.title(f"Directional LR signal distribution: {lr}")
        plt.xlabel("signal"); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(output_png, dpi=300)
        plt.close()

    return {
        "cell_indices": cell_ids,
        "signal": signals,
        "lr_pair": np.array([lr], dtype=object),
        "lr_index": np.array([lr_idx], dtype=np.int32),
        "tg_names": tg_names,
    }


def lr_attn_map(
    ligand: str,
    receptor: str,
    attn_save_dir: str,
    *,
    top_tg: int = 50,
    output_png: Optional[str] = None,
) -> Path:
    """
    lr_attn_map(ligand, receptor, attn_save_dir)

    Builds a heatmap over top TGs for the specified LR pair using the *mean* LR->TG
    matrix across cells (from percell_summary.npz).
    """
    d = Path(attn_save_dir)
    summ = _load_summary_or_build(d)

    lr_names = np.array(summ["lr_pair_names"])
    tg_names = np.array(summ["tg_names"])
    mean_Wlr = summ["mean_W_lr"].astype(np.float32)  # (Gtgt, Glr)

    lr = _lr_name(ligand, receptor)
    hits = np.where(lr_names == lr)[0]
    if hits.size == 0:
        raise ValueError(f"LR pair '{lr}' is not in lr_pair_names (not significant / filtered out).")
    lr_idx = int(hits[0])

    v = mean_Wlr[:, lr_idx]  # (Gtgt,)
    order = np.argsort(-v)
    top = order[: int(top_tg)]
    mat = v[top][:, None]  # (top_tg, 1)

    if output_png is None:
        output_png = str(d / f"lr_attn_map__{lr}.png")

    plt.figure(figsize=(4.5, max(4.0, 0.18 * len(top))))
    plt.imshow(mat, aspect="auto")
    plt.yticks(np.arange(len(top)), tg_names[top].tolist(), fontsize=7)
    plt.xticks([0], [lr])
    plt.title(f"Mean LR→TG attention-like weight\n(top {len(top)} TGs)")
    plt.colorbar(label="mean weight")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    return Path(output_png)


def tf_tg_attn_map(
    tf_name: str,
    tg_name: str,
    attn_save_dir: str,
    *,
    output_png: Optional[str] = None,
) -> Path:
    """
    tf_tg_attn_map(tf, tg, attn_save_dir)

    Shows a single TF->TG scalar from the *mean* TF->TG matrix as a 1x1 heatmap.
    (Useful as a quick check; for broader views you can adapt to show top TGs for a TF.)
    """
    d = Path(attn_save_dir)
    summ = _load_summary_or_build(d)

    tf_names = np.array(summ["tf_names"])
    tg_names = np.array(summ["tg_names"])
    mean_Wtf = summ["mean_W_tf"].astype(np.float32)  # (Gtgt, Gtf)

    tf_hits = np.where(tf_names == tf_name)[0]
    tg_hits = np.where(tg_names == tg_name)[0]
    if tf_hits.size == 0:
        raise ValueError(f"TF '{tf_name}' not found in tf_names.")
    if tg_hits.size == 0:
        raise ValueError(f"TG '{tg_name}' not found in tg_names.")

    j = int(tf_hits[0])
    i = int(tg_hits[0])
    val = mean_Wtf[i, j]
    mat = np.array([[val]], dtype=np.float32)

    if output_png is None:
        output_png = str(d / f"tf_tg_attn_map__{tf_name}__to__{tg_name}.png")

    plt.figure(figsize=(4, 3))
    plt.imshow(mat, aspect="auto")
    plt.xticks([0], [tf_name], rotation=45, ha="right")
    plt.yticks([0], [tg_name])
    plt.title("Mean TF→TG weight")
    plt.colorbar(label="mean weight")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    return Path(output_png)


def stage_specific_overall_signals(
    attn_save_dir: str,
    *,
    n_bins: int = 5,
    pseudotime_key: str = "dpt_pseudotime",
    output_png: Optional[str] = None,
) -> Path:
    """
    Visualization: stage-specific overall signals.
    - Loads processed adata if available (needs scanpy).
    - Bins cells by pseudotime quantiles.
    - For each bin, averages per-cell total TF and LR signal:
        total_TF(cell) = sum_{TG,TF} W_tf[TG,TF]
        total_LR(cell) = sum_{TG,LR} W_lr[TG,LR]
    """
    d = Path(attn_save_dir)
    meta = _load_meta(d)
    project = str(meta["project"][0]) if "project" in meta else ""

    if sc is None:
        raise RuntimeError("scanpy is not available in this environment; can't load h5ad for stage signals.")

    adata_obj = _find_processed_adata(d.parent, project) or _find_processed_adata(d, project)
    if adata_obj is None:
        raise FileNotFoundError("Could not locate '<project>_adata_processed.h5ad' near the attention directory.")

    if pseudotime_key not in adata_obj.obs.columns:
        raise ValueError(f"'{pseudotime_key}' not found in adata.obs.")

    summ = _load_summary_or_build(d)
    cell_ids = summ["cell_indices"].astype(np.int32)

    # Load per-cell total signals
    total_tf = np.zeros((len(cell_ids),), dtype=np.float32)
    total_lr = np.zeros((len(cell_ids),), dtype=np.float32)

    for k, cid in enumerate(cell_ids.tolist()):
        Z = _safe_np_load(d / f"percell_{cid:06d}.npz")
        total_tf[k] = float(np.sum(Z["W_tf"]))
        total_lr[k] = float(np.sum(Z["W_lr"]))

    pt = np.asarray(adata_obj.obs[pseudotime_key].values, dtype=np.float32)
    pt_sel = pt[cell_ids]

    # quantile bins
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(pt_sel, qs)
    # ensure strict monotonic edges (rare if many ties)
    edges = np.unique(edges)
    if edges.size < 3:
        raise ValueError("Pseudotime has too many ties; can't form bins reliably.")

    bin_idx = np.digitize(pt_sel, edges[1:-1], right=True)
    nb = int(np.max(bin_idx)) + 1

    mean_tf = np.zeros((nb,), dtype=np.float32)
    mean_lr = np.zeros((nb,), dtype=np.float32)
    counts = np.zeros((nb,), dtype=np.int32)

    for b in range(nb):
        m = (bin_idx == b)
        counts[b] = int(np.sum(m))
        if counts[b] > 0:
            mean_tf[b] = float(np.mean(total_tf[m]))
            mean_lr[b] = float(np.mean(total_lr[m]))

    if output_png is None:
        output_png = str(d / "stage_specific_overall_signals.png")

    x = np.arange(nb)
    plt.figure(figsize=(7, 4))
    plt.plot(x, mean_tf, marker="o", label="mean total TF signal")
    plt.plot(x, mean_lr, marker="o", label="mean total LR signal")
    plt.xticks(x, [f"bin{i}\n(n={counts[i]})" for i in x])
    plt.title("Stage-specific overall signals (per-cell sums)")
    plt.xlabel("pseudotime bins (quantiles)")
    plt.ylabel("mean summed weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    return Path(output_png)
