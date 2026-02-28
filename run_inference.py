#!/usr/bin/env python3
"""
Gene-level post-processing for a given project/input root (NO METACELLS, NO FIGURES).

- Global LR inference from saved global gated LR matrix.
- Per-cell LR inference from saved per-cell LR→TG matrices (run-once artifact).
  Saves only .npy/.npz (no plots).

This script assumes:
  input_dir/
    attentions/global_attentions/gated_global_lr_full.npz
    attentions/percell_attentions/   (or your per-cell att dir)
      meta_gene_orders.npz
      percell_*.npz

and:
  data_dir/project_name/
    {project}_ligands.txt
    {project}_receptors.txt
    {project}_lr_pairs.txt
    {project}_tgs.txt   (optional, not strictly required if meta_gene_orders is used)

Per-cell analysis functions live in: analysis.gene_interaction_inference.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np

import scanpy as sc
import tensorflow as tf

from analysis.gene_interaction_inference_copy import (
    aggregate_LR_intensity,
    percell_lr_inference,
    per_cell_att_compute,   # <-- THIS must exist in your module
)
from analysis.utils import read_list_txt
from analysis.cell_interaction_analysis import plot_lr_attention_map, plot_tf_attention_map


# ----------------------------- Constants -----------------------------

REL_GLOBAL_ATT_DIR = Path("attentions") / "global_attentions"
# Update this constant to match where you store the "run-once" percell attention outputs.
# The original script used "attentions/percell_attentions".
REL_PERCELL_ATT_DIR = Path("attentions") / "percell_attentions"


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
        prog="spatrace-gene-infer",
        description="Gene-level inference from GRAEST outputs (no metacells, no figures).",
    )
    ap.add_argument(
        "-d",
        "--data_dir",
        required=True,
        help="Data directory that contains the preprocess project folder (same as input to run_preprocess.py).",
    )
    ap.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Input directory containing attentions/ (output from run_experiment).",
    )
    ap.add_argument(
        "-o",
        "--out_dir",
        required=True,
        help="Output directory to store gene interaction results.",
    )
    ap.add_argument(
        "-n",
        "--project_name",
        required=True,
        type=str,
        help="Project prefix used by preprocess (e.g. 'MyProj').",
    )
    ap.add_argument(
        "--log_level",
        type=_log_level,
        default=logging.INFO,
        help="Numeric or named Python logging level (e.g., 20 or INFO).",
    )

    # Global inference controls
    ap.add_argument(
        "--global_topk_per_col",
        type=int,
        default=None,
        help="Optional: keep only top-k entries per TG column before aggregation (global).",
    )
    ap.add_argument(
        "--global_top_n_bar",
        type=int,
        default=20,
        help="Kept only for API compatibility with aggregate_LR_intensity; no figures are saved here.",
    )

    # Per-cell inference controls (matches your no-figure percell_lr_inference)
    ap.add_argument(
        "--skip_percell",
        action="store_true",
        default=False,
        help="Skip per-cell LR inference.",
    )
    ap.add_argument(
        "--percell_top_k",
        type=int,
        default=None,
        help="Within each retained TG column, keep only its top-k LR entries before rowwise nonzero mean.",
    )
    ap.add_argument(
        "--percell_n_permutations",
        type=int,
        default=1000,
        help="Number of permutations for TG-column permutation test.",
    )
    ap.add_argument(
        "--percell_gene_top_k",
        type=int,
        default=None,
        help="Keep the gene_top_k TG columns with smallest p-values (permutation test).",
    )
    ap.add_argument(
        "--percell_gene_alpha",
        type=float,
        default=0.05 / 20,
        help="If gene_top_k is None, keep TG columns with p <= gene_alpha.",
    )
    ap.add_argument(
        "--percell_random_state",
        type=int,
        default=0,
        help="RNG seed for permutation test.",
    )
    ap.add_argument(
        "--percell_save_mean_lrtg",
        action="store_true",
        default=False,
        help="Also save the filtered mean LR×TG matrix as NPZ for later reuse (still no figures).",
    )

    # Path override if your percell dir name differs
    ap.add_argument(
        "--percell_att_relpath",
        type=str,
        default=str(REL_PERCELL_ATT_DIR),
        help="Relative path under input_dir to the per-cell attention directory.",
    )
    ap.add_argument(
        "--extract_percell_attn",
        action="store_true",
        default=False,
        help="If set, run per-cell attention extraction (per_cell_att_compute) before per-cell inference.",
    )
    ap.add_argument(
        "--percell_att_out_relpath",
        type=str,
        default=str(REL_PERCELL_ATT_DIR),
        help="Where to write extracted per-cell attentions under input_dir (relative path).",
    )
    ap.add_argument(
        "--sc_adata_path",
        type=str,
        default=None,
        help="Path to the single-cell .h5ad needed for per-cell attention extraction.",
    )
    ap.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the trained model checkpoint (format depends on your loader). Required if --extract_percell_attn.",
    )


    # ----------------------------- Optional plotting -----------------------------
    ap.add_argument(
        "--plot_adata_path",
        type=str,
        default=None,
        help="Path to a spatial .h5ad used only for plotting (must contain obsm['spatial']).",
    )
    ap.add_argument(
        "--plot_lr_pair",
        type=str,
        default=None,
        help="If set, plot an LR spatial heatmap for this '<ligand>_to_<receptor>' pair using percell_lrscore_percell.npz.",
    )
    ap.add_argument(
        "--plot_tf",
        type=str,
        default=None,
        help="If set, plot a TF spatial heatmap for this TF using raw per-cell W_tf matrices (requires percell_*.npz present).",
    )
    ap.add_argument(
        "--plot_stage",
        type=str,
        default=None,
        help="Optional stage suffix used when resolving '{mode}_lrscore_percell__{stage}.npz'.",
    )
    ap.add_argument(
        "--plot_point_size",
        type=float,
        default=1.0,
        help="Point size for scatter in attention maps.",
    )
    ap.add_argument(
        "--plot_cmap",
        type=str,
        default="Reds",
        help="Matplotlib colormap for attention maps.",
    )

    return ap.parse_args()


# ----------------------------- Validation -----------------------------

def _require_file(p: Path, desc: str) -> None:
    if not p.is_file():
        raise FileNotFoundError(f"Missing {desc}: {p}")


def _require_dir(p: Path, desc: str) -> None:
    if not p.is_dir():
        raise FileNotFoundError(f"Missing {desc}: {p}")


# ----------------------------- Loaders -----------------------------

def load_gene_lists(project_dir: Path, project_name: str, logger: logging.Logger) -> dict:
    """
    Load ligand/receptor/LR pair lists from preprocess outputs.
    """
    ligands_txt = project_dir / f"{project_name}_ligands.txt"
    receptors_txt = project_dir / f"{project_name}_receptors.txt"
    lr_pairs_txt = project_dir / f"{project_name}_lr_pairs.txt"

    _require_file(ligands_txt, "ligands list")
    _require_file(receptors_txt, "receptors list")
    _require_file(lr_pairs_txt, "lr pairs list")

    genes = {
        "ligands": read_list_txt(ligands_txt),
        "receptors": read_list_txt(receptors_txt),
        "lr_pairs": read_list_txt(lr_pairs_txt),
    }
    logger.info(
        "Loaded gene lists: ligands=%d receptors=%d lr_pairs=%d",
        len(genes["ligands"]), len(genes["receptors"]), len(genes["lr_pairs"])
    )
    return genes


# ----------------------------- Workflows -----------------------------

def run_global_lr_intensity(
    global_att_dir: Path,
    out_gene_dir: Path,
    genes: dict,
    *,
    top_n_bar: int,
    topk_per_col: int | None,
    logger: logging.Logger,
) -> None:
    """
    Global LR inference from gated_global_lr_full.npz.

    Note:
      - This function writes whatever aggregate_LR_intensity writes.
      - If your updated aggregate_LR_intensity no longer saves figures, great.
        This script itself does not create any figures.
    """
    ga_file = global_att_dir / "gated_global_lr_full.npz"
    _require_file(ga_file, "global LR attention (gated_global_lr_full.npz)")

    Z = np.load(ga_file, allow_pickle=True)
    if "weight" not in Z:
        raise KeyError(f"'weight' not found in {ga_file}. Keys: {list(Z.keys())}")

    # Your previous logic used transpose to get (TG, LR)
    W = Z["weight"].T
    logger.info("Global gated LR matrix loaded: shape(TG,LR)=%s", tuple(W.shape))

    out_gene_dir.mkdir(parents=True, exist_ok=True)

    aggregate_LR_intensity(
        W,
        genes["ligands"],
        genes["receptors"],
        genes["lr_pairs"],
        str(out_gene_dir),
        mode="global",
        stage=None,
        top_n_bar=top_n_bar,
        top_k=topk_per_col,
    )

    logger.info("Global LR inference complete (saved to %s).", out_gene_dir)

def ensure_percell_attentions(
    *,
    percell_att_dir: Path,
    extract_requested: bool,
    sc_adata_path: str | None,
    model_path: str | None,
    logger: logging.Logger,
    data_dir: Path,
    project_name: str,
    # model hparams (MUST match training)
    d_model: int = 128,
    dff: int = 128,
    num_heads: int = 5,
    dropout_rate: float = 0.0,
    num_layers: int = 1,
    # used to build model before load_weights
    build_seq_len: int = 1,
) -> None:
    """
    Ensure meta_gene_orders.npz + percell_*.npz exist.

    If missing and extract_requested=True, run per_cell_att_compute(adata, model, percell_att_dir).

    IMPORTANT:
    - model_path is weights-only (saved via model.save_weights()).
    - For subclassed models in Keras 3, you must BUILD the model (call once) before load_weights().
    """
    import numpy as np
    import scanpy as sc
    import tensorflow as tf

    meta = percell_att_dir / "meta_gene_orders.npz"
    has_percell = any(percell_att_dir.glob("percell_*.npz"))

    if meta.is_file() and has_percell:
        logger.info("Per-cell attention artifacts already exist: %s", percell_att_dir)
        return

    if not extract_requested:
        raise FileNotFoundError(
            f"Missing per-cell artifacts in {percell_att_dir}.\n"
            f"Expected: {meta} and percell_*.npz\n"
            "Either point --percell_att_relpath to the correct folder, or rerun with --extract_percell_attn."
        )

    if sc_adata_path is None or model_path is None:
        raise ValueError(
            "--extract_percell_attn was set, but --sc_adata_path and/or --model_path not provided."
        )

    percell_att_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading sc adata: %s", sc_adata_path)
    adata = sc.read_h5ad(sc_adata_path)

    # ---------------------------------------------------------
    # 1) Load training NPZ to recover input dimensions
    # ---------------------------------------------------------
    train_npz = Path(data_dir) / project_name / "data_triple" / f"{project_name}_tensors_train.npz"
    if not train_npz.is_file():
        raise FileNotFoundError(
            f"Training NPZ not found: {train_npz}\n"
            "Needed to recover LR/TF/TG dimensions to rebuild the model."
        )

    Z = np.load(train_npz, allow_pickle=True)
    for k in ("lr_pair", "tf", "target"):
        if k not in Z:
            raise KeyError(f"train NPZ missing key '{k}'. Found keys: {list(Z.keys())}")

    ligrecp_size = int(Z["lr_pair"].shape[-1])
    tf_gene_size = int(Z["tf"].shape[-1])
    target_gene_size = int(Z["target"].shape[-1])

    logger.info(
        "Recovered model dims from training NPZ: LR=%d, TF=%d, TG=%d",
        ligrecp_size, tf_gene_size, target_gene_size
    )

    # ---------------------------------------------------------
    # 2) Rebuild Transformer architecture (same as training)
    # ---------------------------------------------------------
    from model.SpaTRACE_v1_0 import Transformer  # matches your run_experiment.py

    model = Transformer(
        num_layers=int(num_layers),
        d_model=int(d_model),
        num_heads=int(num_heads),
        dff=int(dff),
        ligrecp_size=ligrecp_size,
        tf_gene_size=tf_gene_size,
        target_gene_size=target_gene_size,
        dropout_rate=float(dropout_rate),
    )

    # ---------------------------------------------------------
    # 3) BUILD the model (required before load_weights in Keras 3)
    # ---------------------------------------------------------
    L = int(build_seq_len)
    if L <= 0:
        raise ValueError(f"build_seq_len must be >= 1, got {build_seq_len}")

    dummy_lr = tf.zeros((1, L, ligrecp_size), dtype=tf.float32)
    dummy_tf = tf.zeros((1, L, tf_gene_size), dtype=tf.float32)
    dummy_tg = tf.zeros((1, L, target_gene_size), dtype=tf.float32)

    logger.info("Building model with dummy inputs: (B=1, L=%d)", L)
    _ = model([dummy_lr, dummy_tf, dummy_tg], training=False)

    # ---------------------------------------------------------
    # 4) Load weights
    # ---------------------------------------------------------
    logger.info("Loading weights: %s", model_path)
    model.load_weights(model_path)
    logger.info("Weights loaded successfully.")

    # ---------------------------------------------------------
    # 5) Run extractor
    # ---------------------------------------------------------
    logger.info("Extracting per-cell attentions into: %s", percell_att_dir)
    train_npz = Path(data_dir) / project_name / "data_triple" / f"{project_name}_tensors_train.npz"

    per_cell_att_compute(
        adata,
        model,
        str(percell_att_dir),
        train_npz_path=str(train_npz),
        project_name=project_name,
    )

    # Validate
    if not meta.is_file():
        raise RuntimeError(f"Extraction finished but {meta} not found.")
    if not any(percell_att_dir.glob("percell_*.npz")):
        raise RuntimeError(f"Extraction finished but no percell_*.npz found in {percell_att_dir}.")

    logger.info("Per-cell attention extraction complete.")

def _match_cell_identity(adata_1, adata_2):
    """
    Return a view/copy of adata_2 containing only the observations
    present in adata_1, in exactly the same order as adata_1.obs.
    
    Assumes:
        - adata_1.obs_names is a strict subset of adata_2.obs_names
    """
    # Optional safety check (remove if you want it leaner)
    if not adata_1.obs_names.isin(adata_2.obs_names).all():
        raise ValueError("adata_1.obs_names must be a subset of adata_2.obs_names")
    
    return adata_2[adata_1.obs_names].copy()

def run_percell_lr_inference(
    percell_att_dir: Path,
    out_gene_dir: Path,
    *,
    top_k: int | None,
    n_permutations: int,
    gene_top_k: int | None,
    gene_alpha: float | None,
    random_state: int,
    save_mean_lrtg: bool,
    logger: logging.Logger,
) -> None:
    """
    Per-cell LR inference using analysis.gene_interaction_inference.percell_lr_inference

    This should save ONLY .npy/.npz, no figures.
    """
    _require_dir(percell_att_dir, "per-cell attentions directory")

    # required run-once artifacts
    _require_file(percell_att_dir / "meta_gene_orders.npz", "meta_gene_orders.npz")
    # percell files exist?
    if not any(percell_att_dir.glob("percell_*.npz")):
        raise FileNotFoundError(f"No percell_*.npz found in {percell_att_dir}")

    out_gene_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running per-cell LR inference from %s", percell_att_dir)

    # stage is optional; keep None here unless you implement stage-specific splitting later
    _ = percell_lr_inference(
        attn_save_dir=str(percell_att_dir),
        out_dir=str(out_gene_dir),
        mode="percell",
        stage=None,
        top_k=top_k,
        n_permutations=n_permutations,
        gene_top_k=gene_top_k,
        gene_alpha=gene_alpha,
        random_state=random_state,
        save_mean_lrtg=save_mean_lrtg,
    )

    logger.info("Per-cell LR inference complete (saved to %s).", out_gene_dir)


# ----------------------------- Main -----------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("graest-gene-infer")

    DATA_DIR = Path(args.data_dir).resolve()
    INPUT_DIR = Path(args.input_dir).resolve()
    OUT_DIR = Path(args.out_dir).resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    project_dir = (DATA_DIR / args.project_name).resolve()
    _require_dir(project_dir, f"project dir '{args.project_name}'")

    global_att_dir = INPUT_DIR / REL_GLOBAL_ATT_DIR
    percell_att_dir = INPUT_DIR / Path(args.percell_att_relpath)

    _require_dir(INPUT_DIR, "input_dir")
    _require_dir(global_att_dir, "global attentions directory")

    # Output subdir
    gene_out = OUT_DIR / "gene_interactions"
    gene_out.mkdir(parents=True, exist_ok=True)

    # Gene lists from preprocess
    genes = load_gene_lists(project_dir, args.project_name, logger)

    # 1) Global LR inference
    run_global_lr_intensity(
        global_att_dir=global_att_dir,
        out_gene_dir=gene_out,
        genes=genes,
        top_n_bar=args.global_top_n_bar,
        topk_per_col=args.global_topk_per_col,
        logger=logger,
    )

    # 2) Per-cell LR inference (run-once artifact)
    if args.skip_percell:
        logger.info("Skipping per-cell inference (--skip_percell).")
    else:
        ensure_percell_attentions(
        percell_att_dir=percell_att_dir,
        extract_requested=args.extract_percell_attn,
        sc_adata_path=args.sc_adata_path,
        model_path=args.model_path,
        logger=logger,
        data_dir=DATA_DIR,
        project_name=args.project_name,
    )
        run_percell_lr_inference(
            percell_att_dir=percell_att_dir,
            out_gene_dir=gene_out,
            top_k=args.percell_top_k,
            n_permutations=args.percell_n_permutations,
            gene_top_k=args.percell_gene_top_k,
            gene_alpha=args.percell_gene_alpha,
            random_state=args.percell_random_state,
            save_mean_lrtg=args.percell_save_mean_lrtg,
            logger=logger,
        )

    # ----------------------------- Optional plots -----------------------------
    if args.plot_adata_path and (args.plot_lr_pair or args.plot_tf):
        ad_plot = sc.read_h5ad(args.plot_adata_path)
        sc_adata = sc.read_h5ad(args.sc_adata_path)
    
        plots_dir = OUT_DIR / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        ad_plot = _match_cell_identity(sc_adata, ad_plot)

    if args.plot_lr_pair is not None:
        lr_pair = str(args.plot_lr_pair).strip()

        # percell_lr_inference writes into gene_out (OUT_DIR/gene_interactions) by default
        lrscore_path = gene_out / (f"percell_lrscore_percell__{args.plot_stage}.npz" if args.plot_stage else "percell_lrscore_percell.npz")
        out_png = plots_dir / f"attnmap_lr__{lr_pair}.png"

        plot_lr_attention_map(
            adata=ad_plot,
            lr_pair=lr_pair,
            lrscore_npz=lrscore_path,
            out_path=out_png,
            cmap=args.plot_cmap,
            point_size=args.plot_point_size,
            stage=args.plot_stage,
        )

    if args.plot_tf is not None:
        tf_name = str(args.plot_tf).strip()
        out_png = plots_dir / f"attnmap_tf__{tf_name}.png"

        plot_tf_attention_map(
            adata=ad_plot,
            tf_name=tf_name,
            percell_att_dir=percell_att_dir,
            out_path=out_png,
            cmap=args.plot_cmap,
            point_size=args.plot_point_size,
        )

    logger.info("Gene-level inference finished successfully.")


if __name__ == "__main__":
    main()
