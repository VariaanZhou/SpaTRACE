#!/usr/bin/env python3
# Post-process GRAEST outputs with a specified input root directory.
# Uses known subpaths under --in_dir and writes into --out_dir/{global,bio_<mode>}

import argparse
import logging
import os
import sys
import glob
from pathlib import Path

from analysis.aggregate_intensity import aggregate_percell_intensity_from_embeddings
from analysis.utils import plot_lr_tg_heatmaps

REL_GLOBAL_EMB_DIR   = os.path.join("embeddings", "global_embeddings")
REL_PERCELL_EMB_DIR  = os.path.join("embeddings", "percell_embeddings")
REL_PERCELL_ATT_DIR  = os.path.join("embeddings", "percell_attentions")

REL_PATHS_FILE = os.path.join("data_triple", "all_paths_test.npy")
REL_LABELS_CANDIDATES = [
    os.path.join("data_triple", "meta_labels_by_batch.npy"),
    os.path.join("data_triple", "meta_labels.npy"),
]
REL_LABEL2BATCH_CSV  = os.path.join("data_triple", "label_to_batch.csv")
REL_MEMBERS_LONG_CSV = os.path.join("data_triple", "metacell_memberships_by_batch.csv")

# File patterns
EMB_BATCH_PATTERN = "embeddings_batch_*.npz"
BIO_TOPK_PATTERN  = "attn_percell_lr_topk_batch_*.npz"
BIO_FULL_PATTERN  = "attn_percell_lr_full_batch_*.npz"
NUMERIC_REGEX     = r"(\d+)"

def _log_level(val):
    try:
        return int(val)
    except Exception:
        name = str(val).upper()
        if not hasattr(logging, name):
            raise argparse.ArgumentTypeError(f"Invalid log level: {val}")
        return getattr(logging, name)

def _exists_with_files(dir_path: str, pattern: str) -> bool:
    return os.path.isdir(dir_path) and bool(glob.glob(os.path.join(dir_path, pattern)))

def main():
    ap = argparse.ArgumentParser(
        prog="graest-post",
        description="Aggregate attentions/embeddings from GRAEST outputs using a specified input root.",
    )
    ap.add_argument("-d", "--data_dir", required=True, help="Data directory containing the original .h5ad, as well as the preprocessed data folder; should be the same as the input to run_preprocess.py.")
    ap.add_argument("-i", "--input_dir", required=True,
                    help="Input directory containing embeddings/; should be the output of run_experiments.py.")
    ap.add_argument("-o", "--out_dir", required=True, help="Output directory to store analysis results.")
    ap.add_argument(
        "-n", "--project_name", type=str, default=None,
        help="Project prefix used by preprocess (e.g., 'MyProj'). If omitted, will auto-detect a single *_tensors_train.npz."
    )
    ap.add_argument("-b", "--batch_key", type=str, default='batch')
    ap.add_argument("-t", "--radius", type=float, default=50.0, help="Filter threshold (> keeps).")
    ap.add_argument("--topk_per_col", type=int, default=100, help="Top-K per column for filtering.")
    ap.add_argument("--log_level", type=_log_level, default=logging.INFO)
    ap.add_argument("--no_heatmaps", dest="make_heatmaps", action="store_false", help="Disable heatmap PNGs.")
    ap.set_defaults(make_heatmaps=True)
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")

    DATA_DIR = Path(args.data_dir)
    INPUT_DIR = Path(args.input_dir)
    OUTPUT_DIR = Path(args.out_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    PROJECT_NAME = args.project_name
    BATCH_KEY = args.batch_key
    RADIUS = args.radius
    TOPK_PER_COL = args.topk_per_col

    # Resolve inputs under in_root
    global_emb_dir  = os.path.join(INPUT_DIR, REL_GLOBAL_EMB_DIR)
    percell_emb_dir = os.path.join(INPUT_DIR, REL_PERCELL_EMB_DIR)
    percell_att_dir = os.path.join(INPUT_DIR, REL_PERCELL_ATT_DIR)

    test_npz_path = os.path.join(DATA_DIR, PROJECT_NAME, 'data_triple', f'{PROJECT_NAME}_tensors_test.npz')
    meta_labels_npy = os.path.join(DATA_DIR, PROJECT_NAME, f'{PROJECT_NAME}_metacell_membership.csv')
    batchmap_csv = os.path.join(DATA_DIR, PROJECT_NAME, f'{PROJECT_NAME}_metacell_batchmap.csv')

    paths_file         = os.path.join(INPUT_DIR, REL_PATHS_FILE)
    labels_candidates  = [os.path.join(INPUT_DIR, p) for p in REL_LABELS_CANDIDATES]
    label2batch_csv    = os.path.join(INPUT_DIR, REL_LABEL2BATCH_CSV)
    members_long_csv   = os.path.join(INPUT_DIR, REL_MEMBERS_LONG_CSV)


    # Aggregate percell intensity
    stage_sums = aggregate_percell_intensity_from_embeddings(
            percell_emb_dir = percell_emb_dir,  # directory with per-cell embeddings: embeddings_batch_*.npz
            test_npz_path = test_npz_path,  # <project>_tensors_test.npz (must contain 'paths')
            batchmap_csv = batchmap_csv,  # CSV with columns: metacell,batch (or override via args below)
            label_col = "metacell",
            stage_col = BATCH_KEY ,
            top_k = TOPK_PER_COL,  # filtering: keep top-k per TG column (None to disable)
            threshold = RADIUS,  # filtering: zero-out values <= threshold (None to disable)
            save_dir = args.out_dir,  # if provided, save per-stage .npz and a CSV summary
    )


    for stage, blobs in stage_sums.items():
        plot_lr_tg_heatmaps(
            sum_arr=blobs["lr_tg_sum"],
            count_arr=blobs["lr_tg_count"],
            stage=stage,
            out_dir=f"{args.out_dir}/figures/lr_heatmaps",
            title_prefix="Per-cell LR→TG",
            figsize=(10, 8),
            fontsize=14,
        )
        plot_lr_tg_heatmaps(
            sum_arr=blobs["tf_tg_sum"],
            count_arr=blobs["tg_tg_count"],
            stage=stage,
            out_dir=f"{args.out_dir}/figures/tf_heatmaps",
            title_prefix="Per-cell TF→TG",
            figsize=(10, 8),
            fontsize=14,
        )


if __name__ == "__main__":
    main()
