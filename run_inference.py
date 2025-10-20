#!/usr/bin/env python3
# Post-process GRAEST outputs with a specified input root directory.
# Uses known subpaths under --in_dir and writes into --out_dir/{global,bio_<mode>}

import argparse
import logging
import os
import sys
import glob
from pathlib import Path

import numpy as np

from analysis.gene_interaction_inference import aggregate_percell_intensity_from_embeddings, aggregate_LR_intensity, load_percell_intensity_dict
from analysis.utils import plot_lr_tg_heatmaps, plot_lr_heatmap, read_list_txt

REL_GLOBAL_ATT_DIR   = os.path.join("attentions", "global_attentions")
REL_PERCELL_EMB_DIR  = os.path.join("embeddings", "percell_embeddings")
REL_PERCELL_ATT_DIR  = os.path.join("attentions", "percell_attentions")

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
    ap.add_argument("-t", "--threshold", type=float, default=0.01, help="Filter threshold (> keeps).")
    ap.add_argument('--skip_attentions', action='store_true', help="Set true to skip percell attention computation.")
    ap.add_argument("--topk_per_col", type=int, default=100, help="Top-K per column for filtering.")
    ap.add_argument("--top_n_bar", type = int, default=20, help="Top-N for the bar plots of LR interactions.")
    ap.add_argument("--log_level", type=_log_level, default=logging.INFO)
    ap.add_argument("--no_heatmaps", dest="make_heatmaps", action="store_false", help="Disable heatmap PNGs.")
    ap.add_argument("--full_weights", action="store_true", help="Save and use full attention matrices for gene-level interaction intensities.")
    ap.set_defaults(make_heatmaps=True)
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")

    DATA_DIR = Path(args.data_dir)
    INPUT_DIR = Path(args.input_dir)
    OUTPUT_DIR = Path(args.out_dir)
    GENE_OUT = OUTPUT_DIR / "gene_interactions"
    CELL_OUT = OUTPUT_DIR / "cell_interactions"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GENE_OUT.mkdir(parents=True, exist_ok=True)
    CELL_OUT.mkdir(parents=True, exist_ok=True)

    PROJECT_NAME = args.project_name
    BATCH_KEY = args.batch_key
    THRESHOLD = args.threshold
    TOPK_PER_COL = args.topk_per_col

    META_DIR = DATA_DIR / PROJECT_NAME
    LIGAND_DIR = META_DIR / f'{PROJECT_NAME}_ligands.txt'
    RECEPTOR_DIR = META_DIR / f'{PROJECT_NAME}_receptors.txt'
    TF_DIR = META_DIR/f'{PROJECT_NAME}_tfs.txt'
    TG_DIR = META_DIR/f'{PROJECT_NAME}_tgs.txt'
    LR_DIR = META_DIR / f'{PROJECT_NAME}_lr_pairs.txt'

    # Load gene names
    ligand_list = read_list_txt(LIGAND_DIR)
    receptor_list = read_list_txt(RECEPTOR_DIR)
    lr_list = read_list_txt(LR_DIR)
    tf_list = read_list_txt(TF_DIR)
    tg_list = read_list_txt(TG_DIR)

    # Resolve inputs under in_root
    global_att_dir  = os.path.join(INPUT_DIR, REL_GLOBAL_ATT_DIR)
    percell_emb_dir = os.path.join(INPUT_DIR, REL_PERCELL_EMB_DIR)
    percell_att_dir = os.path.join(INPUT_DIR, REL_PERCELL_ATT_DIR) # Save attention scores

    test_npz_path = os.path.join(META_DIR, 'data_triple', f'{PROJECT_NAME}_tensors_test.npz')
    meta_labels_npy = os.path.join(META_DIR, f'{PROJECT_NAME}_metacell_membership.csv')
    batchmap_csv = os.path.join(META_DIR, f'{PROJECT_NAME}_metacell_batchmap.csv')

    paths_file         = os.path.join(INPUT_DIR, REL_PATHS_FILE)
    labels_candidates  = [os.path.join(INPUT_DIR, p) for p in REL_LABELS_CANDIDATES]
    label2batch_csv    = os.path.join(INPUT_DIR, REL_LABEL2BATCH_CSV)
    members_long_csv   = os.path.join(INPUT_DIR, REL_MEMBERS_LONG_CSV)


    if not args.skip_attentions:
        # Aggregate percell intensity
        stage_sums = aggregate_percell_intensity_from_embeddings(
                percell_emb_dir = percell_emb_dir,  # directory with per-cell embeddings: embeddings_batch_*.npz
                test_npz_path = test_npz_path,  # <project>_tensors_test.npz (must contain 'paths')
                batchmap_csv = batchmap_csv,  # CSV with columns: metacell,batch (or override via args below)
                label_col = "metacell",
                full_weights=args.full_weights,
                stage_col = BATCH_KEY ,
                top_k = TOPK_PER_COL,  # filtering: keep top-k per TG column (None to disable)
                threshold = THRESHOLD,  # filtering: zero-out values <= threshold (None to disable)
                save_dir = GENE_OUT,  # save per-stage .npz and a CSV summary
        )
    else:
        stage_sums = load_percell_intensity_dict(GENE_OUT)

    # Aggregate_LR intensities
    # Start with global attentions
    global_attention_file = global_att_dir + 'attn_global_lr.npz'
    global_attention_mat = np.load(global_attention_file)
    # Compute L->R intensity first
    lr_tg_mat = aggregate_LR_intensity(global_attention_mat, ligand_list, receptor_list, lr_list, GENE_OUT, mode = 'global', top_n_bar = args.top_n_bar)
    if not args.no_heatmaps:
        plot_lr_heatmap(lr_intensity=lr_tg_mat, ligands=ligand_list, receptors=receptor_list, stage=None,
                        out_dir=GENE_OUT)

    # Compute LR intensity for percell intensity
    for stage in stage_sums:
        lr_tg_mat = stage_sums[stage]['lr_tg_sum']
        lr_tg_count = stage_sums[stage]['lr_tg_count']
        aggregate_LR_intensity(lr_tg_mat, ligand_list, receptor_list, lr_list, GENE_OUT, mode='percell',
                               top_n_bar=args.top_n_bar)
        aggregate_LR_intensity(lr_tg_count, ligand_list, receptor_list, lr_list, GENE_OUT, mode='percell',
                               top_n_bar=args.top_n_bar)
        if not args.no_heatmaps:
            plot_lr_heatmap(lr_intensity=lr_tg_mat, ligands=ligand_list, receptors=receptor_list, stage=stage, out_dir=GENE_OUT)
            plot_lr_heatmap(lr_intensity=lr_tg_count, ligands=ligand_list, receptors=receptor_list, stage=stage,
                            out_dir=GENE_OUT)

    if not args.no_heatmaps:
        # Plot the inferred percell LR->TG and TF->TG intensities as heatmaps.
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
