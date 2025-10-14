#!/usr/bin/env python3
# Post-process GRAEST outputs with a specified input root directory.
# Uses known subpaths under --in_dir and writes into --out_dir/{global,bio_<mode>}

import argparse
import logging
import os
import sys
import glob

from analysis.analysis_tools import aggregate_from_global_embeddings, aggregate_lr_tg_by_bio_batch


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
    ap.add_argument("--in_dir", required=True, help="Input directory containing embeddings/ and data_triple/ subfolders.")
    ap.add_argument("--out_dir", required=True, help="Output directory to store results.")
    ap.add_argument("--data_dir", required=True, help="Data directory containing the paths, ligand, receptors, TG identities, and metacell memberships.")
    ap.add_argument("--threshold", type=float, default=50.0, help="Filter threshold (> keeps).")
    ap.add_argument("--topk_per_col", type=int, default=100, help="Top-K per column for filtering.")
    ap.add_argument("--log_level", type=_log_level, default=logging.INFO)
    ap.add_argument("--no-heatmaps", dest="make_heatmaps", action="store_false", help="Disable heatmap PNGs.")
    ap.set_defaults(make_heatmaps=True)
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")

    in_root  = os.path.abspath(args.in_dir)
    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    # Resolve inputs under in_root
    global_emb_dir  = os.path.join(in_root, REL_GLOBAL_EMB_DIR)
    percell_emb_dir = os.path.join(in_root, REL_PERCELL_EMB_DIR)
    percell_att_dir = os.path.join(in_root, REL_PERCELL_ATT_DIR)

    paths_file         = os.path.join(in_root, REL_PATHS_FILE)
    labels_candidates  = [os.path.join(in_root, p) for p in REL_LABELS_CANDIDATES]
    label2batch_csv    = os.path.join(in_root, REL_LABEL2BATCH_CSV)
    members_long_csv   = os.path.join(in_root, REL_MEMBERS_LONG_CSV)

    # ---------------- GLOBAL aggregation (from global embeddings) ----------------
    if _exists_with_files(global_emb_dir, EMB_BATCH_PATTERN):
        out_global = os.path.join(out_root, "global")
        os.makedirs(out_global, exist_ok=True)
        logging.info("Global-embeddings aggregation → %s", out_global)
        try:
            aggregate_from_global_embeddings(
                DATA_DIR=global_emb_dir,
                PATTERN=EMB_BATCH_PATTERN,
                OUT_DIR=out_global,
                THRESHOLD=args.threshold,
                TOPK_PER_COL=args.topk_per_col,
                MAKE_HEATMAPS=args.make_heatmaps,
                LOGGER_LEVEL=args.log_level,
            )
        except RuntimeError as e:
            logging.error("Global aggregation failed: %s", e)
            sys.exit(1)
    else:
        logging.info("Global step skipped (no files like %s/%s).", global_emb_dir, EMB_BATCH_PATTERN)

    # ---------------- BIO aggregation (auto-detect mode) ----------------
    # Priority: precomputed TopK → dense Full → compute from percell Embeddings
    if _exists_with_files(percell_att_dir, BIO_TOPK_PATTERN):
        bio_mode     = "topk"
        bio_data_dir = percell_att_dir
        bio_pattern  = BIO_TOPK_PATTERN
        emb_pattern_override = None
    elif _exists_with_files(percell_att_dir, BIO_FULL_PATTERN):
        bio_mode     = "full"
        bio_data_dir = percell_att_dir
        bio_pattern  = BIO_FULL_PATTERN
        emb_pattern_override = None
    elif _exists_with_files(percell_emb_dir, EMB_BATCH_PATTERN):
        bio_mode     = "embeddings"
        bio_data_dir = percell_emb_dir
        bio_pattern  = EMB_BATCH_PATTERN
        emb_pattern_override = EMB_BATCH_PATTERN
    else:
        logging.info("Bio-batch step skipped (no TopK, no Full, no percell embeddings under %s).", in_root)
        return

    # Sanity check for metadata
    missing_meta = [p for p in [paths_file, label2batch_csv, members_long_csv] if not os.path.exists(p)]
    if missing_meta:
        logging.error("Missing required metadata files under --in_dir: %s", ", ".join(missing_meta))
        sys.exit(1)

    out_bio = os.path.join(out_root, f"bio_{bio_mode}")
    os.makedirs(out_bio, exist_ok=True)
    logging.info("Bio-batch aggregation (mode=%s) → %s", bio_mode, out_bio)

    try:
        aggregate_lr_tg_by_bio_batch(
            DATA_DIR=bio_data_dir,
            MODE=bio_mode,
            PATTERN=bio_pattern,
            PATHS_FILE=paths_file,
            LABELS_NPY_CANDIDATES=labels_candidates,
            LABEL2BATCH_CSV=label2batch_csv,
            MEMBERS_LONG_CSV=members_long_csv,
            THRESHOLD=args.threshold,
            TOPK_PER_COL=args.topk_per_col,
            OUT_DIR=out_bio,
            DENSE_KEY="weight_nt_lr",
            EMB_PATTERN=emb_pattern_override,   # only used if MODE="embeddings"
            NUMERIC_REGEX=NUMERIC_REGEX,
            MAKE_HEATMAPS=args.make_heatmaps,
            LOGGER_LEVEL=args.log_level,
        )
    except RuntimeError as e:
        logging.error("Bio aggregation failed: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
