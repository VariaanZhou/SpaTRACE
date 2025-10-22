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

def _load_metacell_to_stage(batchmap_csv: str, label_col: str, stage_col: str) -> Dict[int, str]:
    """CSV must have numeric 'metacell' IDs and a stage label in 'batch' (default col names)."""
    df = pd.read_csv(batchmap_csv)
    need = {label_col, stage_col}
    if not need.issubset(df.columns):
        raise ValueError(f"{batchmap_csv} must contain columns {need}")
    # ensure integers for metacell IDs
    return dict(zip(df[label_col].astype(int).values, df[stage_col].astype(str).values))


    # def _reduce_one(arr: np.ndarray, b: int) -> np.ndarray:
    #     """Return (G,d) for a single validation item b."""
    #     if arr.ndim == 4:      # (B, L, G, d)
    #         if reduce_time == "last":
    #             return arr[b, -1]           # (G,d) last time step
    #         elif reduce_time == "mean":
    #             return arr[b].mean(axis=0)  # (G,d)
    #         else:
    #             raise ValueError("reduce_time must be 'last' or 'mean'")
    #     elif arr.ndim == 3:    # (B, G, d)
    #         return arr[b]      # (G,d)
    #     else:
    #         raise ValueError(f"Unexpected embedding shape {arr.shape}")

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

def build_lr_matrix(values_per_lr: np.ndarray, ligands, receptors, lrpairs) -> pd.DataFrame:
    """
    Place LR scores back into a (ligand x receptor) table.
    values_per_lr must have length == len(lrpairs).
    """
    mat = pd.DataFrame(0.0, index=ligands, columns=receptors)
    if len(values_per_lr) != len(lrpairs):
        logging.warning(f"Scores({len(values_per_lr)}) != LR pairs({len(lrpairs)}); "
                        "we will fill where names match and leave others as 0.")
    for idx, pair in enumerate(lrpairs):
        lig, rec = _unparse_lr_var_name(pair)
        if lig in mat.index and rec in mat.columns:
            mat.at[lig, rec] = float(values_per_lr[idx])
    return mat

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

# def choose_highlights(df: pd.DataFrame) -> tuple[list, list]:
#     """Pick which ligands/receptors to show & highlight on the labeled heatmap."""
#     if HIGHLIGHT_LIGANDS:
#         ligs = [l for l in HIGHLIGHT_LIGANDS if l in df.index]
#     else:
#         ligs = df.sum(axis=1).sort_values(ascending=False).head(AUTO_TOP_LIGANDS).index.tolist()
#
#     if HIGHLIGHT_RECEPTORS:
#         recs = [r for r in HIGHLIGHT_RECEPTORS if r in df.columns]
#     else:
#         recs = df.sum(axis=0).sort_values(ascending=False).head(AUTO_TOP_RECEPTORS).index.tolist()
#
#     return ligs, recs

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

def load_percell_intensity_dict(save_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Reconstruct the per-stage dictionary produced by
    `aggregate_percell_intensity_from_embeddings(...)` by loading the
    per-stage NPZ files saved in `save_dir`.

    Expects files named: percell_intensity__{stage}.npz
    Returns: Dict[stage, {"tf_tg_sum": ..., "tf_tg_count": ..., "lr_tg_sum": ..., "lr_tg_count": ...}]
    """
    if not os.path.isdir(save_dir):
        raise NotADirectoryError(f"{save_dir} is not a directory")

    pattern = os.path.join(save_dir, "percell_intensity__*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No per-stage NPZs matching 'percell_intensity__*.npz' found in {save_dir}"
        )

    stage_sums: Dict[str, Dict[str, np.ndarray]] = {}
    required_keys = ("tf_tg_sum", "tf_tg_count", "lr_tg_sum", "lr_tg_count")

    for path in files:
        base = os.path.basename(path)
        # Extract stage name between prefix and ".npz" (no guessing beyond this exact pattern)
        prefix = "percell_intensity__"
        if not base.startswith(prefix) or not base.endswith(".npz"):
            raise ValueError(f"Unexpected filename format: {base}")
        stage = base[len(prefix):-4]

        Z = np.load(path, allow_pickle=True)
        for k in required_keys:
            if k not in Z.files:
                raise ValueError(f"{base} is missing required key '{k}'")

        stage_sums[stage] = {
            "tf_tg_sum":   Z["tf_tg_sum"],
            "tf_tg_count": Z["tf_tg_count"],
            "lr_tg_sum":   Z["lr_tg_sum"],
            "lr_tg_count": Z["lr_tg_count"],
        }

    return stage_sums


def aggregate_percell_intensity_from_embeddings(
    *,
    percell_emb_dir: str,             # folder with per-cell embeddings: embeddings_batch_*.npz
    test_npz_path: str,               # test npz containing 'paths' (object array of metacell-index lists)
    batchmap_csv: str,                # CSV with columns: metacell (int), batch (stage)
    label_col: str = "metacell",
    stage_col: str = "batch",
    reduce_time: str = "last",        # "last" (recommended) or "mean" across time steps
    top_k: Optional[int] = 200,
    threshold: Optional[float] = 0.01,
    save_dir: Optional[str] = None,   # if set, save per-stage npz and a CSV summary
    unknown_stage: str = "UNKNOWN",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Map **each validation item** to a **stage** (by the last metacell of its path),
    compute that item's TF->TG and LR->TG intensities, and aggregate directly into
    that stage (no majority voting across a batch file).

    Expected per-cell NPZ keys: "idx", "x_vq1", "tf_vq1", "recp_vq1"
      - idx: (B,) indices into the test set (these index 'paths')
      - x_vq1: (B,L,G_tg,d) or (B,G_tg,d)
      - tf_vq1: (B,L,G_tf,d) or (B,G_tf,d)
      - recp_vq1: (B,L,G_lr,d) or (B,G_lr,d)

    Returns per-stage dict with:
      - "tf_tg_sum", "tf_tg_count" of shape (G_tf, G_tg)
      - "lr_tg_sum", "lr_tg_count" of shape (G_lr, G_tg)
      (And with the ONLY modification: sums are max-normalized per stage.)
    """
    # 1) load mappings
    paths = _load_test_paths_from_npz(test_npz_path)  # object array of lists
    mc_to_stage = _load_metacell_to_stage(batchmap_csv, label_col, stage_col)

    # 2) list per-cell embedding files
    files = sorted(glob.glob(os.path.join(percell_emb_dir, "embeddings_batch_*.npz")))
    if not files:
        raise FileNotFoundError(f"No per-cell embedding NPZs found in {percell_emb_dir}")

    # 3) aggregation buffers
    stage_sums: Dict[str, Dict[str, np.ndarray]] = {}
    summary_rows = []

    def _reduce_one(arr: np.ndarray, b: int) -> np.ndarray:
        """Return (G,d) for a single validation item b."""
        if arr.ndim == 4:      # (B, L, G, d)
            if reduce_time == "last":
                return arr[b, -1]           # (G,d) last time step
            elif reduce_time == "mean":
                return arr[b].mean(axis=0)  # (G,d)
            else:
                raise ValueError("reduce_time must be 'last' or 'mean'")
        elif arr.ndim == 3:    # (B, G, d)
            return arr[b]      # (G,d)
        else:
            raise ValueError(f"Unexpected embedding shape {arr.shape}")

    for f in tqdm.tqdm(files):
        Z = np.load(f)
        for req in ("idx", "x_vq1", "tf_vq1", "recp_vq1"):
            if req not in Z.files:
                raise ValueError(f"{f} missing required key '{req}'")

        idx = Z["idx"]                    # (B,)
        Xtg = Z["x_vq1"]                  # (B,L,G_tg,d) or (B,G_tg,d)
        TF  = Z["tf_vq1"]                 # (B,L,G_tf,d) or (B,G_tf,d)
        LR  = Z["recp_vq1"]               # (B,L,G_lr,d) or (B,G_lr,d)

        B = idx.shape[0]
        # For each *cell* in this file, map to stage and aggregate
        for b in range(B):
            L = Xtg.shape[1]
            i = int(idx[b])
            if i < 0 or i >= len(paths):
                # skip invalid index
                continue
            path_i = list(paths[i])  # list of metacell IDs for this validation item

            for i in range(L):
                mc = int(path_i[i])
                stage = mc_to_stage.get(mc, unknown_stage)

                # Get embeddings for this single item
                X_tg = Xtg[b, i]
                TF_b = TF[b, i]
                LR_b = LR[b, i]

                # compute intensities and filter
                tf_tg = _outer_abs(TF_b, X_tg)   # (G_tf, G_tg)
                lr_tg = _outer_abs(LR_b, X_tg)   # (G_lr, G_tg)
                tf_f, tf_mask = _filter_matrix(tf_tg, top_k=top_k, threshold=threshold)
                lr_f, lr_mask = _filter_matrix(lr_tg, top_k=top_k, threshold=threshold)

                # init per-stage slots if first time
                if stage not in stage_sums:
                    stage_sums[stage] = {
                        "tf_tg_sum":   tf_f.copy(),
                        "tf_tg_count": tf_mask.astype(np.float64),
                        "lr_tg_sum":   lr_f.copy(),
                        "lr_tg_count": lr_mask.astype(np.float64),
                    }
                else:
                    stage_sums[stage]["tf_tg_sum"]   += tf_f
                    stage_sums[stage]["tf_tg_count"] += tf_mask.astype(np.float64)
                    stage_sums[stage]["lr_tg_sum"]   += lr_f
                    stage_sums[stage]["lr_tg_count"] += lr_mask.astype(np.float64)

        # optional per-file summary (how many cells went to each stage)
        file_counts = {}
        for b in range(B):
            i = int(idx[b])
            if i < 0 or i >= len(paths):
                continue
            path_i = list(paths[i])
            if len(path_i) == 0:
                st = unknown_stage
            else:
                last_mc = int(path_i[-1])
                st = mc_to_stage.get(last_mc, unknown_stage)
            file_counts[st] = file_counts.get(st, 0) + 1

        summary_rows.append({
            "file": os.path.basename(f),
            "n_items": int(B),
            "stage_counts_json": json.dumps(file_counts, ensure_ascii=False),
            "reduce_time": reduce_time,
            "top_k": top_k,
            "threshold": threshold,
        })

    # === NEW: max-normalize the output matrices per stage ===
    for stage, blobs in stage_sums.items():
        # TF->TG
        tf_sum = blobs["tf_tg_sum"]
        tf_max = float(np.max(tf_sum)) if tf_sum.size else 0.0
        if tf_max > 0:
            blobs["tf_tg_sum"] = tf_sum / tf_max
        # LR->TG
        lr_sum = blobs["lr_tg_sum"]
        lr_max = float(np.max(lr_sum)) if lr_sum.size else 0.0
        if lr_max > 0:
            blobs["lr_tg_sum"] = lr_sum / lr_max
        # counts remain unchanged

    # 4) optional saves
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # per-stage NPZs
        for stage, blobs in stage_sums.items():
            np.savez_compressed(
                os.path.join(save_dir, f"percell_intensity__{stage}.npz"),
                tf_tg_sum=blobs["tf_tg_sum"],
                tf_tg_count=blobs["tf_tg_count"],
                lr_tg_sum=blobs["lr_tg_sum"],
                lr_tg_count=blobs["lr_tg_count"],
                meta=dict(reduce_time=reduce_time, top_k=top_k, threshold=threshold, norm="max"),
            )
        # summary CSV
        pd.DataFrame(summary_rows).to_csv(os.path.join(save_dir, "percell_stage_summary.csv"), index=False)

    return stage_sums

'''
Infer the ligand->receptor interaction scores by aggregating the inferred LR->TG matrices.
The inputs can be either global or stage-specific percell LR->TG attention matrices; provide the ligand and receptor lists.
'''

def aggregate_LR_intensity(lr_tg_intensity_matrix, ligand_list, receptor_list, lr_names, out_dir, mode, stage=None, top_n_bar = 20):
    # Sanity check, making sure the name matches
    for lr_name in lr_names:
        ligand, receptor = _unparse_lr_var_name(lr_name)
        if ligand not in ligand_list:
            raise ValueError(f"Ligand name {ligand} not found in the ligand list!")
        if receptor not in receptor_list:
            raise ValueError(f"Receptor name {receptor} not found in the receptor list!")

    # Compute the mean across non-zero instances in each row.
    filt_scores = rowwise_nonzero_mean(lr_tg_intensity_matrix)
    filt_df = build_lr_matrix(filt_scores, ligand_list, receptor_list, lr_names)
    filt_df_norm = normalize_global_max(filt_df) # Normalize by the maximum intensity
    tidy_filt = tidy_long(filt_df_norm, stage)

    file_name = f"{mode}_lrscore_mean__{stage}.npy" if stage else f"{mode}_lrscore_mean.npy"
    np.save(os.path.join(out_dir, file_name), filt_df_norm, allow_pickle=True)

    cluster_name =  f"{mode}_clustermap_lrscore__{stage}.png" if stage else f"{mode}_clustermap_lrscore.png"
    top_pair_name = f"{mode}_top_pairs_bar__{stage}.png" if stage else f"{mode}_top_pairs_bar.png"
    heat_map_name = f"{mode}_heatmap_lrscore__{stage}.png" if stage else f"{mode}_heatmap_lrscore.png"
    # PLOTS
    make_clustermap(filt_df_norm, stage, os.path.join(out_dir, cluster_name))
    make_top_pairs_barplot(tidy_filt, stage, os.path.join(out_dir, top_pair_name), top_n_bar)
    make_heatmap(filt_df_norm, stage, os.path.join(out_dir, heat_map_name))

    return filt_df_norm.to_numpy()



