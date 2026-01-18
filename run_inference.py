#!/usr/bin/env python3
"""
Post-process GRAEST outputs for a given project/input root.

- Aggregates per-cell embeddings into per-stage TF→TG / LR→TG intensities.
- Aggregates global attentions into LR intensities.
- (Optionally) saves heatmaps and bar plots.

Inputs are discovered under --input_dir using known relative subpaths.
Outputs are written under --out_dir/{gene_interactions,cell_interactions,...}.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import glob
import scanpy as sc
import numpy as np

from analysis.gene_interaction_inference import (
    aggregate_percell_intensity_from_embeddings,
    aggregate_LR_intensity,
    load_percell_intensity_dict,
)

from analysis.cellular_interaction_inference import (
    combined_matrix_for_pair_across_stages,
    plot_combined_matrix,
    write_combined_csv,
)

from analysis.utils import (
    plot_lr_tg_heatmaps,
    # plot_lr_heatmap,  # keep import if you re-enable single heatmap plots
    read_list_txt,
)

# ----------------------------- Constants -----------------------------

REL_GLOBAL_ATT_DIR = Path("attentions") / "global_attentions"
REL_PERCELL_EMB_DIR = Path("embeddings") / "percell_embeddings"
REL_PERCELL_ATT_DIR = Path("attentions") / "percell_attentions"  # kept for completeness

REL_PATHS_FILE = Path("data_triple") / "all_paths_test.npy"
REL_LABELS_CANDIDATES = [
    Path("data_triple") / "meta_labels_by_batch.npy",
    Path("data_triple") / "meta_labels.npy",
]
REL_LABEL2BATCH_CSV = Path("data_triple") / "label_to_batch.csv"
REL_MEMBERS_LONG_CSV = Path("data_triple") / "metacell_memberships_by_batch.csv"

EMB_BATCH_PATTERN = "embeddings_batch_*.npz"
BIO_TOPK_PATTERN = "attn_percell_lr_topk_batch_*.npz"
BIO_FULL_PATTERN = "attn_percell_lr_full_batch_*.npz"

# ----------------------------- CLI Utils -----------------------------

def _log_level(val: str) -> int:
    try:
        return int(val)
    except Exception:
        name = str(val).upper()
        if not hasattr(logging, name):
            raise argparse.ArgumentTypeError(f"Invalid log level: {val}")
        return getattr(logging, name)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="graest-post",
        description="Aggregate attentions/embeddings from GRAEST outputs using a specified input root.",
    )
    ap.add_argument(
        "-d",
        "--data_dir",
        required=True,
        help=(
            "Data directory containing the original .h5ad and the preprocessed project folder "
            "(same as input to run_preprocess.py)."
        ),
    )
    ap.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help=(
            "Input directory that contains embeddings/ and attentions/ "
            "(output from run_experiments.py)."
        ),
    )
    ap.add_argument(
        "-o", "--out_dir", required=True, help="Output directory to store analysis results."
    )
    ap.add_argument(
        "-n",
        "--project_name",
        type=str,
        default=None,
        help="Project prefix used by preprocess (e.g., 'MyProj'). Required to locate gene lists.",
    )
    ap.add_argument(
        "-b",
        "--batch_key",
        type=str,
        default="batch",
        help="Column name for stage/batch in the batch map CSV.",
    )
    ap.add_argument(
        "--groupby",
        type=str,
        default="annotation",
        help="obs column for cell types in the spatial .h5ad (used in cellular inference).",
    )
    ap.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help=(
            "Stage names/order for cellular inference (e.g., E12.5 E14.5 E16.5). "
            "If omitted, stages are inferred from per-cell outputs."
        ),
    )
    ap.add_argument(
        "-f",
        "--filter_threshold",
        type=float,
        default=0.01,
        help="Filter threshold (> keeps).",
    )
    ap.add_argument(
        "-t",
        "--radius",
        type=float,
        default=50.0,
        help="Spatial neighbor radius (microns) for cellular LR intensity.",
    )
    ap.add_argument(
        "--topk_per_col", type=int, default=100, help="Top-K per column for filtering."
    )
    ap.add_argument(
        "--top_n_bar", type=int, default=20, help="Top-N for LR bar plots."
    )
    ap.add_argument(
        "--log_level",
        type=_log_level,
        default=logging.INFO,
        help="Numeric or named Python logging level (e.g., 20 or INFO).",
    )
    ap.add_argument(
        "--no_heatmaps", action="store_true", default=False, help="Disable heatmap PNG generation."
    )
    ap.add_argument(
        "--full_weights",
        action="store_true",
        default=False,
        help="(Reserved) Save/use full attention matrices for gene-level interactions.",
    )
    ap.add_argument(
        "--skip_percell",
        action="store_true",
        default=False,
        help="Skip per-cell aggregation & plots.",
    )
    ap.add_argument(
        "--skip_attentions",
        action="store_true",
        default=False,
        help="If set, do NOT recompute per-cell intensities; load from disk instead.",
    )
    ap.add_argument(
        "--pct_threshold",
        type=float,
        default=0.1,
        help=(
            "Percent-of-cells threshold for a gene to be considered expressed (used in cellular inference)."
        ),
    )
    ap.add_argument(
        "--expr_cutoff",
        type=float,
        default=0.0,
        help="Expression cutoff used for % expressed. Default is 0.0",
    )
    ap.add_argument(
        "--receptor_block",
        type=int,
        default=128,
        help="The receptor chunk size used during cellular CCC inference to avoid OOM.",
    )
    ap.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(8.0, 6.0),
        help="Figure size for cellular heatmaps, e.g., --figsize 8 6",
    )
    ap.add_argument(
        "--dpi", type=int, default=300, help="DPI for cellular heatmaps."
    )
    ap.add_argument(
        "--export_csv",
        action="store_true",
        default=True,
        help="Also export the combined cellular matrices as CSV.",
    )
    return ap.parse_args()


# ----------------------------- Path Builders -----------------------------

def build_project_paths(
    data_dir: Path, input_dir: Path, out_dir: Path, project_name: str
) -> dict:
    """
    Returns a dict of resolved paths used throughout the pipeline.
    """
    meta_dir = data_dir / project_name

    paths = {
        # I/O roots
        "DATA_DIR": data_dir,
        "INPUT_DIR": input_dir,
        "OUTPUT_DIR": out_dir,
        "GENE_OUT": out_dir / "gene_interactions",
        "CELL_OUT": out_dir / "cell_interactions",
        # Input (relative to INPUT_DIR)
        "global_att_dir": input_dir / REL_GLOBAL_ATT_DIR,
        "percell_emb_dir": input_dir / REL_PERCELL_EMB_DIR,
        "percell_att_dir": input_dir / REL_PERCELL_ATT_DIR,
        # Meta & tensors (relative to DATA_DIR/project_name)
        "test_npz_path": meta_dir / "data_triple" / f"{project_name}_tensors_test.npz",
        "batchmap_csv": meta_dir / f"{project_name}_metacell_batchmap.csv",
        "meta_labels_npy": meta_dir / f"{project_name}_metacell_membership.csv",  # kept if needed
        # Adata
        "adata_dp": data_dir / f"{project_name}_sc_adata.h5ad",
        "adata_all": data_dir / f"{project_name}_st_adata.h5ad",
        # Gene sets
        "ligands_txt": meta_dir / f"{project_name}_ligands.txt",
        "receptors_txt": meta_dir / f"{project_name}_receptors.txt",
        "lr_pairs_txt": meta_dir / f"{project_name}_lr_pairs.txt",
        "tfs_txt": meta_dir / f"{project_name}_tfs.txt",
        "tgs_txt": meta_dir / f"{project_name}_tgs.txt",
        "senders": meta_dir / f"{project_name}_senders.txt",
        "receivers": meta_dir / f"{project_name}_receivers.txt",
        # Misc (under INPUT_DIR; not all are used here)
        "paths_file": input_dir / REL_PATHS_FILE,
        "label2batch_csv": input_dir / REL_LABEL2BATCH_CSV,
        "members_long_csv": input_dir / REL_MEMBERS_LONG_CSV,
    }
    return paths


# ----------------------------- Validation -----------------------------

def _require_file(p: Path, desc: str) -> None:
    if not p.is_file():
        raise FileNotFoundError(f"Missing {desc}: {p}")


def _require_dir(p: Path, desc: str) -> None:
    if not p.is_dir():
        raise FileNotFoundError(f"Missing {desc}: {p}")


def validate_inputs(
    paths: dict, logger: logging.Logger, skip_percell: bool, skip_attentions: bool
) -> None:
    logger.debug("Validating input directories and key files...")

    _require_dir(paths["INPUT_DIR"], "input_dir")
    _require_dir(paths["DATA_DIR"], "data_dir")

    # Global attentions are required
    _require_dir(paths["global_att_dir"], "global attentions directory")
    ga_file = paths["global_att_dir"] / "attn_global_lr.npz"
    _require_file(ga_file, "global LR attention (attn_global_lr.npz)")
    logger.info(f"Found global attentions: {ga_file}")

    # Per-cell embeddings only required if we recompute per-cell intensities
    if not skip_percell and not skip_attentions:
        _require_dir(paths["percell_emb_dir"], "per-cell embeddings directory")
        pattern = str(paths["percell_emb_dir"] / EMB_BATCH_PATTERN)
        if not glob.glob(pattern):
            raise FileNotFoundError(f"No per-cell embedding NPZs match {pattern}")
        logger.info(f"Found per-cell embeddings under: {paths['percell_emb_dir']}")

    # Meta / tensors for per-cell aggregation
    if not skip_percell:
        _require_file(paths["test_npz_path"], "test tensors (contains 'paths')")
        _require_file(paths["batchmap_csv"], "metacell→batch map CSV")
        logger.info(f"Found test tensor: {paths['test_npz_path']}")
        logger.info(f"Found batch map CSV: {paths['batchmap_csv']}")

    # Gene lists (always needed)
    for k in ("ligands_txt", "receptors_txt", "lr_pairs_txt", "tfs_txt", "tgs_txt"):
        _require_file(paths[k], f"gene list: {k}")
        logger.info(f"Found {k}: {paths[k]}")


# ----------------------------- Loaders -----------------------------

def load_gene_lists(paths: dict, logger: logging.Logger) -> dict:
    logger.info("Loading gene lists...")
    genes = {
        "ligands": read_list_txt(paths["ligands_txt"]),
        "receptors": read_list_txt(paths["receptors_txt"]),
        "lr_pairs": read_list_txt(paths["lr_pairs_txt"]),
        "tfs": read_list_txt(paths["tfs_txt"]),
        "tgs": read_list_txt(paths["tgs_txt"]),
    }
    for name, lst in genes.items():
        logger.debug(f"Loaded {name}: {len(lst)} entries")
    logger.info("Gene lists loaded.")
    return genes


# ----------------------------- Workflows -----------------------------

def run_global_lr_intensity(paths: dict, genes: dict, top_n_bar: int, logger: logging.Logger) -> None:
    logger.info("Computing global LR intensities from global attentions...")
    ga_file = paths["global_att_dir"] / "attn_global_lr.npz"
    print(np.load(ga_file))
    vals = np.load(ga_file)["vals"]
    logger.debug(f"Global attention matrix shape: {vals.shape}")
    out_dir = paths["GENE_OUT"]
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_LR_intensity(
        vals,
        genes["ligands"],
        genes["receptors"],
        genes["lr_pairs"],
        out_dir,
        mode="global",
        top_n_bar=top_n_bar,
    )
    logger.info("Global LR intensities computed and saved.")


def run_percell_aggregation(
    paths: dict,
    batch_key: str,
    topk_per_col: int,
    threshold: float,
    recompute: bool,
    logger: logging.Logger,
) -> dict:
    """
    Returns stage_sums dict with lr_tg_sum, lr_tg_count, tf_tg_sum, tf_tg_count.
    """
    out_dir = paths["GENE_OUT"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if recompute:
        logger.info("Aggregating per-cell embeddings into per-stage intensities...")
        stage_sums = aggregate_percell_intensity_from_embeddings(
            percell_emb_dir=str(paths["percell_emb_dir"]),
            test_npz_path=str(paths["test_npz_path"]),
            batchmap_csv=str(paths["batchmap_csv"]),
            label_col="metacell",
            stage_col=batch_key,
            top_k=topk_per_col,
            threshold=threshold,
            save_dir=str(out_dir),
        )
        logger.info(f"Per-cell aggregation complete. Stages: {list(stage_sums.keys())}")
    else:
        logger.info("Loading previously computed per-cell intensities from disk...")
        stage_sums = load_percell_intensity_dict(str(out_dir))
        logger.info(f"Loaded per-cell intensities. Stages: {list(stage_sums.keys())}")

    return stage_sums


def run_percell_lr_intensity(
    stage_sums: dict, genes: dict, out_dir: Path, top_n_bar: int, logger: logging.Logger
) -> dict:
    logger.info("Computing LR intensities from per-cell LR→TG matrices (sum and count)...")
    per_cell_intensity = {}

    for stage, blobs in stage_sums.items():
        lr_tg_sum = blobs["lr_tg_sum"]
        lr_tg_count = blobs["lr_tg_count"]
        logger.debug(
            f"[{stage}] lr_tg_sum shape={lr_tg_sum.shape}, lr_tg_count shape={lr_tg_count.shape}"
        )

        lr_sum = aggregate_LR_intensity(
            lr_tg_sum,
            genes["ligands"],
            genes["receptors"],
            genes["lr_pairs"],
            out_dir,
            stage=stage,
            mode="percell",
            top_n_bar=top_n_bar,
        )
        lr_count = aggregate_LR_intensity(
            lr_tg_count,
            genes["ligands"],
            genes["receptors"],
            genes["lr_pairs"],
            out_dir,
            stage=stage,
            mode="percell",
            top_n_bar=top_n_bar,
        )
        per_cell_intensity[f"{stage}_lr_count"] = lr_count
        per_cell_intensity[f"{stage}_lr_sum"] = lr_sum
    logger.info("Per-cell LR intensities computed and saved for all stages.")
    return per_cell_intensity


def maybe_save_heatmaps(stage_sums: dict, out_root: Path, logger: logging.Logger) -> None:
    logger.info("Saving heatmaps for TF→TG and LR→TG (per-cell)...")
    for stage, blobs in stage_sums.items():
        plot_lr_tg_heatmaps(
            sum_arr=blobs["lr_tg_sum"],
            count_arr=blobs["lr_tg_count"],
            stage=stage,
            out_dir=str(out_root / "figures" / "lr_heatmaps"),
            title_prefix="Per-cell LR→TG",
            figsize=(10, 8),
            fontsize=14,
        )
        plot_lr_tg_heatmaps(
            sum_arr=blobs["tf_tg_sum"],
            count_arr=blobs["tf_tg_count"],
            stage=stage,
            out_dir=str(out_root / "figures" / "tf_heatmaps"),
            title_prefix="Per-cell TF→TG",
            figsize=(10, 8),
            fontsize=14,
        )
    logger.info("Heatmaps saved successfully.")


# ----------------------------- Main -----------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("graest-post")

    # Resolve core paths
    DATA_DIR = Path(args.data_dir).resolve()
    INPUT_DIR = Path(args.input_dir).resolve()
    OUTPUT_DIR = Path(args.out_dir).resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.project_name:
        raise ValueError("--project_name is required to locate gene lists and tensors.")

    paths = build_project_paths(DATA_DIR, INPUT_DIR, OUTPUT_DIR, args.project_name)

    # Ensure output subdirs exist
    paths["GENE_OUT"].mkdir(parents=True, exist_ok=True)
    paths["CELL_OUT"].mkdir(parents=True, exist_ok=True)

    # Validate presence of the key inputs
    validate_inputs(
        paths, logger, skip_percell=args.skip_percell, skip_attentions=args.skip_attentions
    )

    # Load gene, sender, and receptor lists
    genes = load_gene_lists(paths, logger)

    # Global LR intensity from global attentions
    run_global_lr_intensity(paths, genes, top_n_bar=args.top_n_bar, logger=logger)

    # Initialize holders so they exist even if we skip per-cell
    stage_sums = None
    per_cell_intensity = {}

    # Per-cell pipeline (optional)
    if not args.skip_percell:
        stage_sums = run_percell_aggregation(
            paths=paths,
            batch_key=args.batch_key,
            topk_per_col=args.topk_per_col,
            threshold=args.filter_threshold,
            recompute=(not args.skip_attentions),
            logger=logger,
        )

        per_cell_intensity = run_percell_lr_intensity(
            stage_sums=stage_sums,
            genes=genes,
            out_dir=paths["GENE_OUT"],
            top_n_bar=args.top_n_bar,
            logger=logger,
        )

        if not args.no_heatmaps:
            maybe_save_heatmaps(stage_sums, OUTPUT_DIR, logger)
            logger.info("Saving successful!")
        else:
            logger.info("Heatmap generation disabled (--no_heatmaps).")
    else:
        logger.info("Per-cell analysis skipped (--skip_percell).")

    logger.info("Gene Level Inference finished!")

    # Require per-cell intensities to be available for cellular level
    if stage_sums is None or not per_cell_intensity:
        logger.warning(
            "Skipping Cellular Level Inference because per-cell intensities are not available. "
            "Run without --skip_percell (and optionally without --skip_attentions) first."
        )
        return

    # Resolve stages for cellular inference
    inferred_stages = sorted(
        {k.split("_lr_")[0] for k in per_cell_intensity.keys() if k.endswith("_lr_sum")}
    )
    if args.stages is None:
        stages = inferred_stages
        logger.info(f"Using inferred stages for cellular inference: {stages}")
    else:
        stages = list(args.stages)
        missing = [st for st in stages if f"{st}_lr_sum" not in per_cell_intensity]
        if missing:
            raise ValueError(
                f"Requested stages {missing} not found in per-cell intensities. "
                f"Available: {inferred_stages}"
            )

    # Cellular Level Inference
    logger.info("Starting Cellular Level Inference")

    # Load adata, senders, receivers
    adata = sc.read_h5ad(paths["adata_all"])
    senders = read_list_txt(paths["senders"])
    receivers = read_list_txt(paths["receivers"])

    # Build stage→matrix dict once for all requested stages
    M_by_stage = {st: per_cell_intensity[f"{st}_lr_sum"] for st in stages}

    for s in senders:
        for r in receivers:
            M_comb, lig_union, rec_union = combined_matrix_for_pair_across_stages(
                adata=adata,
                M_by_stage=M_by_stage,
                ligand_order=genes["ligands"],
                receptor_order=genes["receptors"],
                stages=tuple(stages),
                batch_key=args.batch_key,
                groupby=args.groupby,
                sender=s,
                receiver=r,
                pct_threshold=args.pct_threshold,
                expr_cutoff=args.expr_cutoff,
                radius=args.radius,
                receptor_block=args.receptor_block,
                row_block=8192,
                dtype=np.float32,
            )

            if M_comb.size == 0:
                # Still write an empty CSV marker if requested
                if args.export_csv:
                    base = f"{s.replace(' ','_')}__to__{r.replace(' ','_')}__combined_{'_'.join(stages)}"
                    write_combined_csv(
                        M_comb,
                        lig_union,
                        rec_union,
                        tuple(stages),
                        paths["CELL_OUT"] / (base + ".csv"),
                    )
                continue

            # Plot heatmaps
            _ = plot_combined_matrix(
                M_combined=M_comb,
                ligands_all=lig_union,
                receptors_all=rec_union,
                stages=tuple(stages),
                sender=s,
                receiver=r,
                out_dir=paths["CELL_OUT"],
                figsize=tuple(args.figsize),
                dpi=args.dpi,
            )

            # CSV
            if args.export_csv:
                s_tag = s.replace(" ", "_")
                r_tag = r.replace(" ", "_")
                base = f"{s_tag}__to__{r_tag}__combined_{'_'.join(stages)}"
                write_combined_csv(
                    M_comb,
                    lig_union,
                    rec_union,
                    tuple(stages),
                    paths["CELL_OUT"] / (base + ".csv"),
                )


if __name__ == "__main__":
    main()
