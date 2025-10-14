import os, glob, re, logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "aggregate_from_global_embeddings",
    "aggregate_lr_tg_by_bio_batch",
]

# module-scoped logger
logger = logging.getLogger(__name__)

# ============================== shared helpers ==============================

def _resolve_glob(data_dir: str, pattern: str) -> str:
    """Join `pattern` to `data_dir` unless pattern already includes a path."""
    return pattern if os.path.sep in pattern else os.path.join(data_dir, pattern)

_NUM_RE = re.compile(r"(\d+)")
def _numeric_key(p: str) -> int:
    """Sort by the last integer in filename (…_0009 before …_0100)."""
    m = list(_NUM_RE.finditer(os.path.basename(p)))
    return int(m[-1].group(1)) if m else 10**12

def _keep_top_k_per_column(arr: np.ndarray, k_: int) -> np.ndarray:
    """Keep only top-k entries per column (others set to 0)."""
    if arr.ndim != 2:
        raise ValueError(f"keep_top_k_per_column expects 2D, got {arr.shape}")
    n_rows = arr.shape[0]
    k_eff = min(max(k_, 0), n_rows)
    if k_eff == 0:
        return np.zeros_like(arr)
    sorted_idx = np.argsort(-arr, axis=0)
    mask = np.zeros_like(arr, dtype=bool)
    for c in range(arr.shape[1]):
        mask[sorted_idx[:k_eff, c], c] = True
    return np.where(mask, arr, 0.0)

def _mean_first_two_axes(a: np.ndarray) -> np.ndarray:
    """Mean over (B, L) if present; otherwise behave reasonably."""
    if a.ndim >= 3:
        return a.mean(axis=(0, 1))
    elif a.ndim == 2:
        return a.mean(axis=0)
    return a  # 1D or scalar

def _to_2d_row(v: np.ndarray) -> np.ndarray:
    """Ensure 2D for @ matmul: (1,D) if 1D; flatten if >2D."""
    v = np.asarray(v)
    if v.ndim == 1:
        return v[None, :]
    if v.ndim == 2:
        return v
    return v.reshape(1, -1)

def _save_heatmap(path: str, arr: np.ndarray, title: str, cbar_label: str,
                  xlabel: str = "Columns", ylabel: str = "Rows") -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(arr, aspect="auto", interpolation="nearest")
    cbar = plt.colorbar()
    cbar.set_label(cbar_label)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

# =================== processor 1: global-embeddings only ====================

def aggregate_from_global_embeddings(
    DATA_DIR: str = "embeddings/global_embeddings",
    PATTERN: str = "embeddings_batch_*.npz",  # glob pattern (not a file)
    OUT_DIR: str = "result",
    THRESHOLD: float = 50.0,
    TOPK_PER_COL: int = 100,
    FILTERED_OUT_PATH_LR: str = "attention_lr_tg_weights_all.npy",
    FILTERED_OUT_PATH_TF: str = "attention_tf_tg_weights_all.npy",
    COUNT_OUT_PATH_LR: str = "attention_weights_lr_tg_count.npy",
    COUNT_OUT_PATH_TF: str = "attention_weights_tf_tg_count.npy",
    FILTERED_HEATMAP_PATH_LR: str = "attention_lr_tg.png",
    FILTERED_HEATMAP_PATH_TF: str = "attention_tf_tg.png",
    COUNT_HEATMAP_PATH_LR: str = "attention_lr_tg_count.png",
    COUNT_HEATMAP_PATH_TF: str = "attention_tf_tg_count.png",
    MAKE_HEATMAPS: bool = True,
    LOGGER_LEVEL: Optional[int] = None,
) -> Dict[str, str]:
    """
    Rebuild LR→TG and TF→TG attentions from *global* embeddings batches and aggregate.
    Each input *.npz must contain: x_vq1 (targets), tf_vq1 (TFs), recp_vq1 (receptors).
    """
    if LOGGER_LEVEL is not None:
        logger.setLevel(LOGGER_LEVEL)
    os.makedirs(OUT_DIR, exist_ok=True)

    pattern_path = _resolve_glob(DATA_DIR, PATTERN)
    files = sorted(glob.glob(pattern_path), key=_numeric_key)
    if not files:
        raise RuntimeError(f"No files found matching pattern: {pattern_path}")
    logger.info("Found %d global-embedding batches.", len(files))

    sum_lr = sum_tf = None
    cnt_lr = cnt_tf = None
    dtype_lr = dtype_tf = None
    required = {"x_vq1", "tf_vq1", "recp_vq1"}

    for fn in files:
        try:
            with np.load(fn) as data:
                if not required.issubset(data.files):
                    missing = required - set(data.files)
                    raise KeyError(f"missing keys {missing}")
                x_vq1 = data["x_vq1"]       # (B,L,G_tgt,d)
                tf_vq1 = data["tf_vq1"]     # (B,L,G_tf,d)
                recp_vq1 = data["recp_vq1"] # (B,L,G_recp,d)

            # Means over (B,L) → (G, d)
            x_mean   = _mean_first_two_axes(x_vq1)
            tf_mean  = _mean_first_two_axes(tf_vq1)
            rec_mean = _mean_first_two_axes(recp_vq1)

            # 2D for matmul
            x2   = _to_2d_row(x_mean)    # (G_tg, d) or (1,d)
            tf2  = _to_2d_row(tf_mean)   # (G_tf, d) or (1,d)
            rec2 = _to_2d_row(rec_mean)  # (G_lr, d) or (1,d)

            # Reconstruct |A @ B^T|
            arr_tf = np.abs(tf2 @ x2.T)   # (G_tf, G_tg)
            arr_lr = np.abs(rec2 @ x2.T)  # (G_lr, G_tg)

            # Filter
            flr = _keep_top_k_per_column(arr_lr, TOPK_PER_COL)
            ftf = _keep_top_k_per_column(arr_tf, TOPK_PER_COL)
            mask_lr = flr > THRESHOLD
            mask_tf = ftf > THRESHOLD
            flr = np.where(mask_lr, flr, 0.0)
            ftf = np.where(mask_tf, ftf, 0.0)

        except Exception as e:
            logger.warning("Skipping '%s': %s", fn, e)
            continue

        if sum_lr is None:
            sum_lr = flr.astype(np.float64, copy=True)
            sum_tf = ftf.astype(np.float64, copy=True)
            cnt_lr = mask_lr.astype(np.float64, copy=True)
            cnt_tf = mask_tf.astype(np.float64, copy=True)
            dtype_lr, dtype_tf = arr_lr.dtype, arr_tf.dtype
            logger.info("Initialized with shapes LR=%s, TF=%s", sum_lr.shape, sum_tf.shape)
            continue

        # Shape alignment
        if flr.shape != sum_lr.shape or ftf.shape != sum_tf.shape:
            logger.warning("Shape mismatch in '%s' (LR %s, TF %s); expected LR %s, TF %s. Skipping.",
                           fn, flr.shape, ftf.shape, sum_lr.shape, sum_tf.shape)
            continue

        # Accumulate
        sum_lr += flr; sum_tf += ftf
        cnt_lr += mask_lr.astype(np.float64)
        cnt_tf += mask_tf.astype(np.float64)

    if sum_lr is None or sum_tf is None:
        raise RuntimeError("No valid arrays processed.")

    # Save artifacts
    paths = {
        "filtered_lr": os.path.join(OUT_DIR, FILTERED_OUT_PATH_LR),
        "filtered_tf": os.path.join(OUT_DIR, FILTERED_OUT_PATH_TF),
        "count_lr":    os.path.join(OUT_DIR, COUNT_OUT_PATH_LR),
        "count_tf":    os.path.join(OUT_DIR, COUNT_OUT_PATH_TF),
        "heat_lr":     os.path.join(OUT_DIR, FILTERED_HEATMAP_PATH_LR),
        "heat_tf":     os.path.join(OUT_DIR, FILTERED_HEATMAP_PATH_TF),
        "heat_cnt_lr": os.path.join(OUT_DIR, COUNT_HEATMAP_PATH_LR),
        "heat_cnt_tf": os.path.join(OUT_DIR, COUNT_HEATMAP_PATH_TF),
    }
    np.save(paths["filtered_lr"], sum_lr)
    np.save(paths["filtered_tf"], sum_tf)
    np.save(paths["count_lr"],    cnt_lr)
    np.save(paths["count_tf"],    cnt_tf)

    logger.info("Saved LR→TG sum to '%s'  (shape=%s, dtype=float64; src=%s)",
                paths['filtered_lr'], sum_lr.shape, dtype_lr)
    logger.info("Saved LR→TG cnt to '%s'   (shape=%s, dtype=float64)",
                paths['count_lr'], cnt_lr.shape)
    logger.info("Saved TF→TG sum to '%s' (shape=%s, dtype=float64; src=%s)",
                paths['filtered_tf'], sum_tf.shape, dtype_tf)
    logger.info("Saved TF→TG cnt to '%s'  (shape=%s, dtype=float64)",
                paths['count_tf'], cnt_tf.shape)

    if MAKE_HEATMAPS:
        _save_heatmap(paths["heat_lr"],     sum_lr, "LR→TG Attention (Summed)", "Attention weight (filtered & summed)")
        _save_heatmap(paths["heat_cnt_lr"], cnt_lr, "LR→TG Attention (Count)",  "Count (> threshold)")
        _save_heatmap(paths["heat_tf"],     sum_tf, "TF→TG Attention (Summed)", "Attention weight (filtered & summed)")
        _save_heatmap(paths["heat_cnt_tf"], cnt_tf, "TF→TG Attention (Count)",  "Count (> threshold)")

    logger.info("Done. Outputs in: %s", OUT_DIR)
    return paths

# ======== processor 2: bio-batch aggregation (topk/full/embeddings) ========

def aggregate_lr_tg_by_bio_batch(
    # -------- Inputs / discovery --------
    DATA_DIR: str,
    MODE: str,                              # "topk" | "full" | "embeddings"
    PATTERN: str,                           # glob for attention (topk/full) or embeddings (if EMB_PATTERN None)
    PATHS_FILE: str,                        # npy list of paths; each path = list[int metacell_idx]
    LABELS_NPY_CANDIDATES: List[str],       # possible label-order npy files
    LABEL2BATCH_CSV: str,
    MEMBERS_LONG_CSV: str,

    # -------- Behavior / thresholds --------
    THRESHOLD: float,
    TOPK_PER_COL: int,

    # -------- Output --------
    OUT_DIR: str,

    # -------- Misc --------
    LOGGER_LEVEL: Optional[int] = None,

    # -------- (Mode-specific knobs) --------
    DENSE_KEY: str = "weight_nt_lr",        # for MODE="full": key in npz holding dense LR→TG matrix
    EMB_PATTERN: Optional[str] = None,      # for MODE="embeddings": glob of embedding files (*.npz with x_vq1, recp_vq1)
    NUMERIC_REGEX: str = r"(\d+)",          # extract batch index for sorting
    MAKE_HEATMAPS: bool = True,
) -> None:
    """
    Aggregate LR→TG attention by biological batch.

    Modes:
      - "topk":  files have rows' TopK (lr_vals, lr_cols). Densify, then col-topk + threshold.
      - "full":  files have a dense LR→TG matrix (contains DENSE_KEY). Col-topk + threshold applied.
      - "embeddings": files have embeddings (x_vq1 target, recp_vq1 receptor).
                      Compute | recp_mean @ x_mean^T |, then col-topk + threshold.

    The PATHS_FILE order weights each file chunk by the majority biological batch of its paths.
    """
    if LOGGER_LEVEL is not None:
        logger.setLevel(LOGGER_LEVEL)
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- nested helpers (specific to this processor) ---
    num_re = re.compile(NUMERIC_REGEX)

    def numeric_key(path: str) -> int:
        m_all = list(num_re.finditer(os.path.basename(path)))
        return int(m_all[-1].group(1)) if m_all else 10**12

    def resolve_glob(data_dir: str, pattern: str) -> str:
        if any(sep in pattern for sep in (os.sep, "/")):
            return pattern
        return os.path.join(data_dir, pattern)

    def dense_from_rowwise_topk(vals: np.ndarray, cols: np.ndarray, n_cols: int, dtype=np.float32) -> np.ndarray:
        if vals.ndim != 2 or cols.ndim != 2 or vals.shape != cols.shape:
            raise ValueError(f"vals/cols must be 2D with same shape; got {vals.shape} and {cols.shape}")
        n_rows, K = vals.shape
        dense = np.zeros((n_rows, n_cols), dtype=dtype)
        row_idx = np.repeat(np.arange(n_rows, dtype=np.int64), K)
        col_idx = cols.reshape(-1).astype(np.int64, copy=False)
        data    = vals.reshape(-1).astype(dense.dtype, copy=False)
        valid = (col_idx >= 0) & (col_idx < n_cols)
        if not np.all(valid):
            row_idx = row_idx[valid]; col_idx = col_idx[valid]; data = data[valid]
        np.add.at(dense, (row_idx, col_idx), data)
        return dense

    def load_paths() -> List[List[int]]:
        if not os.path.exists(PATHS_FILE):
            raise FileNotFoundError(f"Missing test paths: {PATHS_FILE}")
        arr = np.load(PATHS_FILE, allow_pickle=True)
        return [list(map(int, p)) for p in arr]

    def load_metacell_labels_order() -> List[str]:
        for p in LABELS_NPY_CANDIDATES:
            if os.path.exists(p):
                lab = np.load(p, allow_pickle=True).tolist()
                return [str(x) for x in lab]
        raise FileNotFoundError(f"Could not find label order. Provide one of: {LABELS_NPY_CANDIDATES}")

    def load_label_to_batch_map() -> Dict[str, str]:
        import pandas as pd  # local import
        if os.path.exists(LABEL2BATCH_CSV):
            df = pd.read_csv(LABEL2BATCH_CSV)
            need = {"metacell_label", "batch"}
            if not need.issubset(df.columns):
                raise ValueError(f"{LABEL2BATCH_CSV} must contain columns {need}")
            sub = df[["metacell_label", "batch"]].dropna().drop_duplicates()
            return dict(zip(sub["metacell_label"].astype(str), sub["batch"].astype(str)))
        if os.path.exists(MEMBERS_LONG_CSV):
            df = pd.read_csv(MEMBERS_LONG_CSV)
            need = {"metacell_label", "batch"}
            if not need.issubset(df.columns):
                raise ValueError(f"{MEMBERS_LONG_CSV} must contain columns {need}")
            vc = df.groupby(["metacell_label", "batch"]).size().reset_index(name="n")
            vc = vc.sort_values(["metacell_label", "n"], ascending=[True, False])
            keep = vc.groupby("metacell_label").first().reset_index()
            return dict(zip(keep["metacell_label"].astype(str), keep["batch"].astype(str)))
        raise FileNotFoundError(f"Neither {LABEL2BATCH_CSV} nor {MEMBERS_LONG_CSV} found.")

    def classify_path_batch(path: List[int], idx_to_label: List[str], label_to_batch: Dict[str, str]) -> str:
        labs = [idx_to_label[mi] for mi in path if 0 <= mi < len(idx_to_label)]
        if not labs:
            return "UNKNOWN"
        batches = [label_to_batch.get(l, "UNKNOWN") for l in labs]
        import pandas as pd
        return pd.Series(batches).value_counts().index[0]  # majority vote

    # --- file list by mode ---
    if MODE not in {"topk", "full", "embeddings"}:
        raise ValueError("MODE must be one of {'topk','full','embeddings'}")
    pattern_to_use = resolve_glob(DATA_DIR, EMB_PATTERN or PATTERN) if MODE == "embeddings" else resolve_glob(DATA_DIR, PATTERN)

    files = sorted(glob.glob(pattern_to_use), key=numeric_key)
    if not files:
        raise RuntimeError(f"No files match {pattern_to_use}")
    logger.info("[%s] Found %d files.", MODE, len(files))

    # --- paths & chunking ---
    paths = load_paths()
    n_paths = len(paths); n_files = len(files)
    if n_paths == 0:
        raise RuntimeError("No test paths found.")
    if n_paths % n_files != 0:
        logger.warning("#paths (%d) not divisible by #files (%d). Using floor division; last chunk may be smaller.",
                       n_paths, n_files)
    chunk_size = n_paths // n_files
    if chunk_size == 0:
        raise RuntimeError("Too many files vs paths; chunk_size computed as 0.")

    idx_to_label   = load_metacell_labels_order()
    label_to_batch = load_label_to_batch_map()

    # --- infer (n_rows, n_cols) per mode ---
    def infer_shape_from_file(fn: str) -> Tuple[int, int]:
        if MODE == "topk":
            with np.load(fn) as z:
                lr_vals = z["lr_vals"]; lr_cols = z["lr_cols"]
            rows = int(lr_vals.shape[0]); max_col = int(np.max(lr_cols)) + 1 if lr_cols.size else 1
            return rows, max_col
        elif MODE == "full":
            with np.load(fn) as z:
                dense = z[DENSE_KEY]
            if dense.ndim != 2:
                raise ValueError(f"{fn}:{DENSE_KEY} must be 2D, got {dense.shape}")
            return int(dense.shape[0]), int(dense.shape[1])
        else:  # embeddings
            with np.load(fn) as z:
                x_vq1 = z["x_vq1"]; recp = z["recp_vq1"]
            return int(recp.shape[2]), int(x_vq1.shape[2])

    first_rows, max_cols = None, 0
    for fn in files:
        try:
            r, c = infer_shape_from_file(fn)
            if first_rows is None:
                first_rows = r
            max_cols = max(max_cols, c)
        except Exception as e:
            logger.warning("Skipping shape peek for '%s': %s", fn, e)
            continue
    if first_rows is None or max_cols == 0:
        raise RuntimeError("Could not infer matrix shape from files.")
    n_rows, n_cols = first_rows, max_cols
    logger.info("Inferred global shape: rows=%d, cols=%d", n_rows, n_cols)

    # --- aggregation ---
    bio_sum: Dict[str, np.ndarray] = {}
    bio_count: Dict[str, np.ndarray] = {}

    for i, fn in enumerate(files):
        start = i * chunk_size
        end   = n_paths if i == n_files - 1 else (i + 1) * chunk_size
        chunk = paths[start:end]
        if not chunk:
            logger.warning("Empty path chunk for file index %d; skipping '%s'.", i, fn)
            continue

        # batch weights for this chunk
        import pandas as pd
        path_batches = [classify_path_batch(p, idx_to_label, label_to_batch) for p in chunk]
        tot = len(path_batches)
        vc  = pd.Series(path_batches).value_counts()
        weights = {b: (cnt / tot) for b, cnt in vc.items()}

        try:
            # ---- build dense LR→TG for this file ----
            if MODE == "topk":
                with np.load(fn) as z:
                    if "lr_vals" not in z.files or "lr_cols" not in z.files:
                        raise KeyError("missing 'lr_vals' or 'lr_cols'")
                    lr_vals = z["lr_vals"]; lr_cols = z["lr_cols"]

                if lr_vals.shape[0] != n_rows:
                    use_rows = min(lr_vals.shape[0], n_rows)
                    if use_rows == 0:
                        logger.warning("%s: zero rows; skipping.", fn); continue
                    dense_file = dense_from_rowwise_topk(
                        lr_vals[:use_rows], lr_cols[:use_rows],
                        max(int(np.max(lr_cols[:use_rows])) + 1, 1), dtype=np.float32
                    )
                    dense = np.zeros((n_rows, n_cols), dtype=np.float32)
                    c_end = min(dense_file.shape[1], n_cols)
                    dense[:use_rows, :c_end] = dense_file[:, :c_end]
                else:
                    file_max_col = int(np.max(lr_cols)) if lr_cols.size else -1
                    file_cols    = max(file_max_col + 1, 1)
                    dense_file = dense_from_rowwise_topk(lr_vals, lr_cols, file_cols, dtype=np.float32)
                    dense = np.zeros((n_rows, n_cols), dtype=np.float32)
                    c_end = min(dense_file.shape[1], n_cols)
                    dense[:, :c_end] = dense_file[:, :c_end]

            elif MODE == "full":
                with np.load(fn) as z:
                    dense_file = z[DENSE_KEY]
                if dense_file.ndim != 2:
                    raise ValueError(f"{fn}:{DENSE_KEY} must be 2D, got {dense_file.shape}")
                dense = np.zeros((n_rows, n_cols), dtype=np.float32)
                r_end = min(dense_file.shape[0], n_rows)
                c_end = min(dense_file.shape[1], n_cols)
                dense[:r_end, :c_end] = dense_file[:r_end, :c_end].astype(np.float32, copy=False)

            else:  # "embeddings"
                with np.load(fn) as z:
                    x_vq1 = z["x_vq1"]       # (B,L,G_tgt,d)
                    recp  = z["recp_vq1"]    # (B,L,G_recp,d)
                x_mean   = _mean_first_two_axes(x_vq1)   # (G_tgt, d)
                rec_mean = _mean_first_two_axes(recp)    # (G_recp, d)
                dense_file = np.abs(_to_2d_row(rec_mean) @ _to_2d_row(x_mean).T)  # (G_recp, G_tgt)
                dense = np.zeros((n_rows, n_cols), dtype=np.float32)
                r_end = min(dense_file.shape[0], n_rows)
                c_end = min(dense_file.shape[1], n_cols)
                dense[:r_end, :c_end] = dense_file[:r_end, :c_end].astype(np.float32, copy=False)

            # post-filter
            filtered = _keep_top_k_per_column(dense, TOPK_PER_COL)
            mask = (filtered > THRESHOLD)
            filtered = np.where(mask, filtered, 0.0)

        except Exception as e:
            logger.warning("Skipping '%s': %s", fn, e)
            continue

        # weighted accumulation
        for bio, w in weights.items():
            if w <= 0:
                continue
            if bio not in bio_sum:
                bio_sum[bio]   = (filtered * w).astype(np.float64, copy=False)
                bio_count[bio] = (mask.astype(np.float64) * w)
            else:
                bio_sum[bio]   += filtered * w
                bio_count[bio] += mask.astype(np.float64) * w

    if not bio_sum:
        raise RuntimeError("No valid contributions aggregated.")

    # save per biological batch
    for bio, S in bio_sum.items():
        C = bio_count[bio]
        base_sum   = os.path.join(OUT_DIR, f"attention_lr_tg_weights_all__{bio}.npy")
        base_count = os.path.join(OUT_DIR, f"attention_weights_lr_tg_count__{bio}.npy")
        np.save(base_sum,   S.astype(np.float32, copy=False))
        np.save(base_count, C.astype(np.float32, copy=False))
        logger.info("[%s] Saved → %s, %s", bio, os.path.basename(base_sum), os.path.basename(base_count))
        if MAKE_HEATMAPS:
            _save_heatmap(os.path.join(OUT_DIR, f"attention_lr_tg__{bio}.png"),
                          S, f"LR→TG Attention (Summed) — {bio}",
                          f"Summed weight (top-{TOPK_PER_COL}, > {THRESHOLD})")
            _save_heatmap(os.path.join(OUT_DIR, f"attention_lr_tg_count__{bio}.png"),
                          C, f"LR→TG Attention (Weighted Count) — {bio}",
                          f"Weighted count kept (top-{TOPK_PER_COL}, > {THRESHOLD})")

    logger.info("Done. Per-biological-batch outputs in: %s", OUT_DIR)
