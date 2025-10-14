"""
Take a h5ad file and txt files as input. Sample the cell trajectories and save them to local.
(Logging added; fixes to args and IO paths.)
"""

import argparse
import time
import logging
import squidpy as sq
from scipy import sparse as sp

# Local utils (as you had)
from datasets.utils import gene_intersection, read_list_txt, verify_cell_types_exist, setup_logging, _write_one_list, _ensure_dir, _combine_lr_names
from datasets.preprocessing import *
# ---------------------- Helpers ----------------------
def _combine_name(dir, name, suffix):
    return os.path.join(dir, name + suffix)

# ---------------------- DE Genes ----------------------
def identify_de_genes_for_cell_types(
    adata,
    gene_set,
    cell_types=None,
    groupby='annotation',
    method='wilcoxon',
    pval_threshold=0.05,
    logfc_threshold=0.25,
    *,
    control_group=None,       # batch label to use as control (e.g., "Ctrl_1")
    batch_key='batch',        # obs column that stores batch labels
    logger=None,
):
    if logger is None:
        logger = logging.getLogger("graest_preprocess")

    if cell_types is None:
        if pd.api.types.is_categorical_dtype(adata.obs[groupby]):
            cell_types = list(adata.obs[groupby].cat.categories)
        else:
            cell_types = list(pd.unique(adata.obs[groupby]))

    # Restrict gene_set to genes present in adata
    gene_set = [g for g in gene_set if g in adata.var_names]
    if len(gene_set) == 0:
        raise ValueError("None of the genes in 'gene_set' are present in adata.var_names.")

    de_results = {}
    logger.info(
        "Starting DE for %d cell types (groupby=%s, method=%s, control_group=%s)",
        len(cell_types), groupby, method, control_group
    )
    t0 = time.perf_counter()

    if control_group is None:
        # vs rest
        for ct in cell_types:
            logger.debug("DE (vs rest): cell_type=%s", ct)
            sc.tl.rank_genes_groups(
                adata,
                groupby=groupby,
                groups=[ct],
                reference='rest',
                method=method
            )
            df = sc.get.rank_genes_groups_df(adata, group=ct)
            before = len(df)
            df = df[df['names'].isin(gene_set)]
            df = df[(df['pvals_adj'] < pval_threshold) &
                    (df['logfoldchanges'].abs() >= logfc_threshold)]
            de_results[ct] = df.reset_index(drop=True)
            logger.debug("  kept %d/%d DE genes (after thresholds)", len(df), before)
    else:
        # within-CT: case (non-control batches) vs ctrl (control_group)
        if batch_key not in adata.obs.columns:
            raise ValueError(f"'{batch_key}' not found in adata.obs but required for control_group mode.")
        if control_group not in set(map(str, adata.obs[batch_key].astype(str))):
            raise ValueError(f"Control group '{control_group}' not found in adata.obs['{batch_key}'].")

        for ct in cell_types:
            mask_ct = adata.obs[groupby] == ct
            if mask_ct.sum() == 0:
                logger.warning("No cells for cell_type=%s; skipping.", ct)
                de_results[ct] = pd.DataFrame(columns=['names','scores','logfoldchanges','pvals','pvals_adj'])
                continue

            ad_sub = adata[mask_ct].copy()
            batches_sub = ad_sub.obs[batch_key].astype(str)
            is_ctrl = batches_sub == str(control_group)
            if is_ctrl.sum() == 0:
                logger.warning("No control cells for cell_type=%s; skipping.", ct)
                de_results[ct] = pd.DataFrame(columns=['names','scores','logfoldchanges','pvals','pvals_adj'])
                continue
            if (~is_ctrl).sum() == 0:
                logger.warning("No case cells for cell_type=%s; skipping.", ct)
                de_results[ct] = pd.DataFrame(columns=['names','scores','logfoldchanges','pvals','pvals_adj'])
                continue

            tmp_col = "__de_tmp_group"
            ad_sub.obs[tmp_col] = np.where(is_ctrl, "ctrl", "case")
            ad_sub.obs[tmp_col] = ad_sub.obs[tmp_col].astype('category')

            logger.debug("DE (case vs ctrl): cell_type=%s | n_case=%d | n_ctrl=%d",
                         ct, int((~is_ctrl).sum()), int(is_ctrl.sum()))
            sc.tl.rank_genes_groups(
                ad_sub,
                groupby=tmp_col,
                groups=['case'],
                reference='ctrl',
                method=method
            )
            df = sc.get.rank_genes_groups_df(ad_sub, group='case')
            before = len(df)
            df = df[df['names'].isin(gene_set)]
            df = df[(df['pvals_adj'] < pval_threshold) &
                    (df['logfoldchanges'].abs() >= logfc_threshold)]
            ad_sub.obs.drop(columns=[tmp_col], inplace=True)
            de_results[ct] = df.reset_index(drop=True)
            logger.debug("  kept %d/%d DE genes (after thresholds)", len(df), before)

    # Union across cell types
    de_union = set()
    total_kept = 0
    for df in de_results.values():
        if not df.empty:
            de_union.update(df['names'])
            total_kept += len(df)

    logger.info("DE complete in %.2fs. Unique DE genes: %d (total rows kept: %d)",
                time.perf_counter() - t0, len(de_union), total_kept)
    return de_results, de_union


# ---------------------- Widely expressed ----------------------
def identify_widely_expressed_genes_for_cell_types(
    adata,
    gene_set,
    cell_types=None,
    groupby='cell_type',
    pct_threshold=0.1,
    expr_cutoff=0,
    *,
    batch_key='batch',
    return_batchwise=True,
    logger=None
):
    if logger is None:
        logger = logging.getLogger("graest_preprocess")

    # Restrict to genes present
    gene_set_present = [g for g in gene_set if g in adata.var_names]
    if not gene_set_present:
        logger.warning("No genes from gene_set are present in adata.var_names.")
        return ({} if return_batchwise else {}), set()

    # Resolve cell types
    if cell_types is None:
        if pd.api.types.is_categorical_dtype(adata.obs[groupby]):
            cell_types = list(adata.obs[groupby].cat.categories)
        else:
            cell_types = list(pd.unique(adata.obs[groupby]))

    logger.info("Computing 'widely expressed' genes (%d candidates). pct>=%.3f, cutoff>%g",
                len(gene_set_present), pct_threshold, expr_cutoff)
    t0 = time.perf_counter()

    def _pct_expr_for_subset(sub_ad):
        X = sub_ad.X
        n = sub_ad.n_obs
        if n == 0:
            return np.zeros(len(gene_set_present), dtype=float)

        idx = adata.var_names.isin(gene_set_present)
        if sp.issparse(X):
            expressed = (X[:, idx] > expr_cutoff)
            counts = np.asarray(expressed.sum(axis=0)).ravel()
        else:
            Xg = X[:, idx]
            counts = (Xg > expr_cutoff).sum(axis=0)
        return counts / float(n)

    union = set()

    # Per-batch if possible
    if batch_key in adata.obs.columns:
        batches = list(pd.unique(adata.obs[batch_key].astype(str)))
        results = {}
        for b in sorted(batches):
            results[b] = {}
            mask_b = adata.obs[batch_key].astype(str) == b
            ad_b = adata[mask_b, :]

            for ct in cell_types:
                mask_ct = ad_b.obs[groupby] == ct
                ct_adata = ad_b[mask_ct, :]
                if ct_adata.n_obs == 0:
                    results[b][ct] = pd.DataFrame(columns=['gene', 'pct_expressed'])
                    continue

                pcts = _pct_expr_for_subset(ct_adata)
                df = pd.DataFrame({'gene': gene_set_present, 'pct_expressed': pcts})
                df = df[df['pct_expressed'] >= pct_threshold].sort_values(
                    by='pct_expressed', ascending=False
                ).reset_index(drop=True)
                results[b][ct] = df
                union.update(df['gene'].tolist())
            logger.debug("Batch %s: union so far = %d genes", b, len(union))

        logger.info("Wide-expression complete in %.2fs. Union genes: %d",
                    time.perf_counter() - t0, len(union))
        return (results if return_batchwise else {}), union

    # Fallback: no batch column
    results = {}
    for ct in cell_types:
        mask_ct = adata.obs[groupby] == ct
        ct_adata = adata[mask_ct, :]
        if ct_adata.n_obs == 0:
            results[ct] = pd.DataFrame(columns=['gene', 'pct_expressed'])
            continue

        pcts = _pct_expr_for_subset(ct_adata)
        df = pd.DataFrame({'gene': gene_set_present, 'pct_expressed': pcts})
        df = df[df['pct_expressed'] >= pct_threshold].sort_values(
            by='pct_expressed', ascending=False
        ).reset_index(drop=True)
        results[ct] = df
        union.update(df['gene'].tolist())

    logger.info("Wide-expression complete in %.2fs. Union genes: %d",
                time.perf_counter() - t0, len(union))
    return results, union


# ---------------------- Filter constitutive ----------------------
def filter_non_constitutive_genes(adata, gene_list, max_threshold=0.3, percentage=0.8, expr_cutoff=0,
                                  groupby='annotation', logger=None):
    if logger is None:
        logger = logging.getLogger("graest_preprocess")

    logger.info("Filtering constitutive genes: max_threshold=%.3f, percentage=%.3f",
                max_threshold, percentage)

    gene_present = [g for g in gene_list if g in adata.var_names]

    all_cts = adata.obs[groupby].unique().tolist()

    max_expr_dict = {}
    for ct in all_cts:
        mask = adata.obs[groupby] == ct
        sub = adata[mask, gene_present]
        X = sub.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        # fraction of cells expressing each gene
        max_expr = (X > expr_cutoff).sum(axis=0) / X.shape[0]
        max_expr_dict[ct] = pd.Series(max_expr, index=gene_present)

    total_cts = len(all_cts)
    filtered_union = set()
    for gene in gene_present:
        ct_count = sum(max_expr_dict[ct].loc[gene] >= max_threshold for ct in all_cts)
        if (ct_count / total_cts) <= percentage:
            filtered_union.add(gene)

    logger.info("Constitutive filtering kept %d/%d genes", len(filtered_union), len(gene_present))
    return filtered_union


# ---------------------- Spatial enrichment ----------------------
def spatial_enrichment_senders(
    adata,
    input_dir,
    project_name,
    *,
    mode="all",          # "per_batch" or "all"
    cluster_key="annotation",
    batch_key="batch",
    interest=None,
    z_threshold=2.0,
    coord_type="generic",
    radius=50,
    n_perms=1000,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger("graest_preprocess")

    if "spatial" not in adata.obsm_keys():
        raise ValueError("adata.obsm['spatial'] not found (requires 2D spatial coordinates).")
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"'{cluster_key}' not found in adata.obs. Available: {list(adata.obs.columns)}")
    if mode not in {"per_batch", "all"}:
        raise ValueError("mode must be 'per_batch' or 'all'.")

    if not pd.api.types.is_categorical_dtype(adata.obs[cluster_key]):
        adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")

    out_dir = os.path.join(input_dir, project_name)
    os.makedirs(out_dir, exist_ok=True)

    def _zdf_for(adata_sub):
        sq.gr.spatial_neighbors(adata_sub, coord_type=coord_type, radius=radius)
        sq.gr.nhood_enrichment(adata_sub, cluster_key=cluster_key, n_perms=n_perms)

        key = f"{cluster_key}_nhood_enrichment"
        if key not in adata_sub.uns:
            raise RuntimeError(f"Expected adata.uns['{key}'] after enrichment.")
        enr = adata_sub.uns[key]
        if "z_score" in enr:
            z_mat = enr["z_score"]
        elif "zscore" in enr:
            z_mat = enr["zscore"]
        else:
            z_mat = next(v for v in enr.values() if isinstance(v, np.ndarray))

        if isinstance(enr.get("params"), dict) and "categories" in enr["params"]:
            clusters = list(enr["params"]["categories"])
        else:
            clusters = list(adata_sub.obs[cluster_key].cat.categories)

        return pd.DataFrame(z_mat, index=clusters, columns=clusters)

    def _senders_from_zdf(z_df, interest_list):
        if interest_list is None:
            recv = list(z_df.columns)
        else:
            recv = [r for r in interest_list if r in z_df.columns]
        if not recv:
            return set()

        senders = set()
        for s in z_df.index:
            vals = z_df.loc[s, recv]
            if s in vals.index:
                vals = vals.drop(s, errors="ignore")
            if np.any(vals.values > z_threshold):
                senders.add(s)
        return senders

    t0 = time.perf_counter()
    union_senders = set()

    if mode == "per_batch":
        if batch_key not in adata.obs.columns:
            raise ValueError(f"'{batch_key}' not found in adata.obs (required for mode='per_batch').")
        batches = sorted(map(str, pd.unique(adata.obs[batch_key].astype(str))))
        logger.info("Running spatial enrichment per batch (%d batches)...", len(batches))

        for b in batches:
            logger.debug("Batch %s: computing enrichment ...", b)
            ad_b = adata[adata.obs[batch_key].astype(str) == b].copy()
            z_df_b = _zdf_for(ad_b)
            batch_senders = _senders_from_zdf(z_df_b, interest)

            batch_file = os.path.join(out_dir, f"{b}_sender.txt")
            with open(batch_file, "w") as f:
                for s in sorted(batch_senders):
                    f.write(f"{s}\n")
            logger.info("Batch %s: %d sender(s) saved to %s", b, len(batch_senders), batch_file)

            union_senders.update(batch_senders)

        union_file = os.path.join(out_dir, f"{project_name}_sender.txt")
        with open(union_file, "w") as f:
            for s in sorted(union_senders):
                f.write(f"{s}\n")
        logger.info("Union senders (%d) saved to %s (elapsed %.2fs)",
                    len(union_senders), union_file, time.perf_counter() - t0)

    else:
        logger.info("Running spatial enrichment on full dataset ...")
        z_df = _zdf_for(adata.copy())
        union_senders = _senders_from_zdf(z_df, interest)

        union_file = os.path.join(out_dir, f"{project_name}_sender.txt")
        with open(union_file, "w") as f:
            for s in sorted(union_senders):
                f.write(f"{s}\n")
        logger.info("Union senders (%d) saved to %s (elapsed %.2fs)",
                    len(union_senders), union_file, time.perf_counter() - t0)

    return sorted(union_senders)

def save_acquired_gene_sets(out_dir, project_name, *, ligands, receptors, tfs, tgs, lrs, logger=None):
    """
    Save acquired gene sets to:
      INPUT_DIR/PROJECT_NAME/{PROJECT_NAME}_<kind>_acquired.txt
    Returns a dict of filepaths written.
    """
    _ensure_dir(out_dir)
    paths = {
        "ligands":  _combine_name(out_dir, project_name, '_ligands.txt'),
        "receptors": _combine_name(out_dir, project_name, '_receptors.txt'),
        "tfs":      _combine_name(out_dir, project_name, '_tfs.txt'),
        "tgs":      _combine_name(out_dir, project_name, '_tgs.txt'),
        "lrs": _combine_name(out_dir, project_name, '_lr_pairs.txt')
    }

    _write_one_list(paths["ligands"],   ligands)
    _write_one_list(paths["receptors"], receptors)
    _write_one_list(paths["tfs"],       tfs)
    _write_one_list(paths["tgs"],       tgs)
    _write_one_list(paths["lrs"], lrs)

    if logger:
        logger.info("Saved ligands to:   %s", paths["ligands"])
        logger.info("Saved receptors to: %s", paths["receptors"])
        logger.info("Saved TFs to:       %s", paths["tfs"])
        logger.info("Saved TGs to:       %s", paths["tgs"])
        logger.info("Saved LRs to:       %s", paths["lrs"])

    return paths

# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser(prog="GRAEST_Chat_Preprocess")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],
                        help="Logging verbosity (default: INFO)")

    parser.add_argument("-i", "--input_dir", required=True, type=str,
                        help="Directory containing the input h5ad and the ligand/receptor/TF txt files.")
    parser.add_argument("-n", "--project_name", required=True, type=str,
                        help="Project name (also used as the base filename).")
    parser.add_argument("-o", "--output_dir", required=False, type=str,
                        help="(Reserved) Directory to save sampled trajectories/expressions (unused in this script).")
    parser.add_argument("-b", "--batch_key", default='batch', type=str,
                        help="obs column name storing batch info (default: 'batch').")
    parser.add_argument("-c", "--control_name", default=None, type=str,
                        help="Batch label to use as control. If None, each batch is compared vs rest.")
    parser.add_argument("-d", "--radius", default=50.0, type=float,
                        help="Neighborhood radius for spatial neighbors (default: 50.0).")
    parser.add_argument("-r", "--receiver", required=True, type=str,
                        help="TXT filename (within INPUT_DIR/PROJECT_NAME*) that lists receiver cell types.")
    parser.add_argument("--n_neighbors", default=10, type=int,
                        help="Number of neighbors to aggregate metacells (default: 10).")
    parser.add_argument("-a", "--annotation_key", default="annotation", type=str,
                        help="obs column name for cell-type annotations (default: 'annotation').")
    parser.add_argument("--pt_key", default="dpt_pseudotime", type=str,
                        help="obs column name for pseudotime (default: 'dpt_pseudotime').")
    parser.add_argument("--sp_key", default="spatial", type=str,
                        help="obsm key for spatial coordinates (default: 'spatial').")

    # Spatial enrichment
    parser.add_argument("-s", "--sender", default=None, type=str,
                        help="TXT filename for sender cell types; if omitted, inferred via spatial enrichment.")
    parser.add_argument("--batch_wise", action="store_true", default=False,
                        help="If set, run spatial enrichment per-batch; otherwise use all data.")
    parser.add_argument("-z", "--z_threshold", default=3.0, type=float,
                        help="Z-score threshold to call sender->receiver enrichment (default: 3.0).")
    parser.add_argument("--n_perm", default=1000, type=int,
                        help="Number of permutations for spatial enrichment (default: 1000).")

    # Gene identification
    parser.add_argument("--logfc_threshold", default=0.25, type=float,
                        help="Abs log2 fold change threshold for DE (default: 0.25).")
    parser.add_argument("--p_val_threshold", default=0.05, type=float,
                        help="Adjusted p-value threshold for DE (default: 0.05).")
    parser.add_argument("--non_constitutive", action='store_true', default=False,
                        help="If set, remove constitutive genes from the identified set.")
    parser.add_argument("--pct_threshold", default=0.1, type=float,
                        help="Minimum fraction of cells expressing a gene to call it active (default: 0.1).")
    parser.add_argument("--max_threshold", default=0.3, type=float,
                        help="Threshold for constitutive filtering (default: 0.3).")
    parser.add_argument("--percentage", default=0.8, type=float,
                        help="Fraction of cell types allowed to be active before labeled constitutive (default: 0.8).")
    parser.add_argument("--expr_cutoff", default=0.0, type=float,
                        help="Expression cutoff for 'expressed' > cutoff (default: 0.0).")

    # Path Sampling
    parser.add_argument("-l", "--path_len", default=3, type=int, help="Length of the sampled cell paths (default: 3).")
    parser.add_argument("--num_repeats", default = 10, type=int, help="Number of times to repeat the sampling for each cell (default: 10).")
    parser.add_argument("-k", "--k_primary", default = 5, type=int, help="Number of temporal neighbors (default: 5).")
    # Parallel Computations
    parser.add_argument("--n_jobs", default=-1, type=int,
                        help="Number of jobs to run in parallel (default: -1, use all available CPUs).")

    args = parser.parse_args()
    logger = setup_logging(args.log_level)

    # I/O directories
    INPUT_DIR = args.input_dir
    PROJECT_NAME = args.project_name
    # Create a subfolder in the Input directory if output directory not provided
    OUTPUT_DIR = args.output_dir
    if not OUTPUT_DIR:
        OUTPUT_DIR = os.path.join(INPUT_DIR, PROJECT_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    # Parameters
    BATCH_KEY = args.batch_key
    ANNOTATION_KEY = args.annotation_key
    CONTROL_GROUP = args.control_name
    RADIUS = args.radius
    PT_KEY = args.pt_key
    SPATIAL_KEY = args.sp_key

    N_JOBS = args.n_jobs

    # Path sampling parameters
    PATH_LEN = args.path_len
    NUM_REPEATS = args.num_repeats
    K_PRIMARY = args.k_primary

    if BATCH_KEY is None:
        raise ValueError("Please provide a batch name via --batch_key.")
    if CONTROL_GROUP is None:
        logger.info("No control group provided; DE will use each cell type vs rest (or per-CT case vs ctrl only if provided).")

    # Paths
    INPUT_H5AD = _combine_name(INPUT_DIR, PROJECT_NAME, '_adata_all.h5ad')
    LIGAND_FILE = _combine_name(INPUT_DIR, PROJECT_NAME, '_ligand.txt')
    RECEPTOR_FILE = _combine_name(INPUT_DIR, PROJECT_NAME, '_receptor.txt')
    TF_FILE = _combine_name(INPUT_DIR, PROJECT_NAME, '_tf.txt')
    RECEIVER_FILE = _combine_name(INPUT_DIR, PROJECT_NAME, args.receiver)

    # Load AnnData
    logger.info("Reading h5ad: %s", INPUT_H5AD)
    adata = sc.read_h5ad(INPUT_H5AD)

    # Processed to ensure spatial enrichment
    if SPATIAL_KEY != "spatial":
        if SPATIAL_KEY not in adata.obsm_keys():
            raise ValueError(f"obsm key '{SPATIAL_KEY}' not found; cannot run spatial enrichment.")
        adata.obsm["spatial"] = adata.obsm[SPATIAL_KEY]
    # Determine genes present
    gene_names = list(adata.var_names)
    logger.info("AnnData loaded: n_cells=%d, n_genes=%d", adata.n_obs, adata.n_vars)

    # Load LR/TF lists intersected with var_names
    ligands = gene_intersection(gene_names, LIGAND_FILE)
    receptors = gene_intersection(gene_names, RECEPTOR_FILE)
    tfs = gene_intersection(gene_names, TF_FILE)
    logger.info("Loaded gene sets: ligands=%d, receptors=%d, TFs=%d", len(ligands), len(receptors), len(tfs))

    # Read receivers & optional senders
    receivers = read_list_txt(RECEIVER_FILE, str)
    verify_cell_types_exist(adata, receivers, ANNOTATION_KEY)
    logger.info("Receiver cell types: %s", receivers)

    if args.sender:
        SENDER_FILE = _combine_name(INPUT_DIR, PROJECT_NAME, args.sender)
        senders = read_list_txt(SENDER_FILE, str)
        logger.info("Sender cell types provided in %s", SENDER_FILE)
        verify_cell_types_exist(adata, senders, ANNOTATION_KEY)
        logger.info("Sender cell types: %s", senders)
    else:
        logger.info("No sender file provided; inferring via spatial enrichment (batch_wise=%s).", args.batch_wise)
        mode = 'per_batch' if args.batch_wise else 'all'
        senders = list(spatial_enrichment_senders(
            adata, INPUT_DIR, PROJECT_NAME,
            mode=mode,
            cluster_key=ANNOTATION_KEY,
            batch_key=BATCH_KEY,
            interest=receivers,
            z_threshold=args.z_threshold,
            radius=RADIUS,
            n_perms=args.n_perm,
            logger=logger
        ))
        logger.info("Inferred sender cell types: %s", senders)

    all_cell_types = list(set(senders + receivers))
    logger.debug("All involved cell types (senders ∪ receivers): %s", all_cell_types)

    # DE analysis
    logger.info("Running DE for ligands/receptors/TFs (control=%s) ...", CONTROL_GROUP)
    _, de_ligands = identify_de_genes_for_cell_types(
        adata, ligands, cell_types=senders, groupby=ANNOTATION_KEY,
        pval_threshold=args.p_val_threshold, logfc_threshold=args.logfc_threshold,
        control_group=CONTROL_GROUP, batch_key=BATCH_KEY, logger=logger
    )
    _, de_receptors = identify_de_genes_for_cell_types(
        adata, receptors, cell_types=receivers, groupby=ANNOTATION_KEY,
        pval_threshold=args.p_val_threshold, logfc_threshold=args.logfc_threshold,
        control_group=CONTROL_GROUP, batch_key=BATCH_KEY, logger=logger
    )
    _, de_tfs = identify_de_genes_for_cell_types(
        adata, tfs, cell_types=receivers, groupby=ANNOTATION_KEY,
        pval_threshold=args.p_val_threshold, logfc_threshold=args.logfc_threshold,
        control_group=CONTROL_GROUP, batch_key=BATCH_KEY, logger=logger
    )
    logger.info("DE unions: ligands=%d, receptors=%d, TFs=%d",
                len(de_ligands), len(de_receptors), len(de_tfs))

    # Widely expressed target genes (across batches)
    _, tgs = identify_widely_expressed_genes_for_cell_types(
        adata, gene_names, cell_types=receivers, groupby=ANNOTATION_KEY,
        pct_threshold=args.pct_threshold, expr_cutoff=args.expr_cutoff,
        batch_key=BATCH_KEY, logger=logger
    )
    logger.info("Widely expressed target genes: %d", len(tgs))
    combined = set().union(de_ligands, de_receptors, de_tfs, tgs)  # FIX: use set union, not +

    if args.non_constitutive:
        logger.info("Filtering non-constitutive genes...")
        filtered_genes = filter_non_constitutive_genes(
            adata, combined,
            max_threshold=args.max_threshold,
            percentage=args.percentage,
            expr_cutoff=args.expr_cutoff,
            groupby=ANNOTATION_KEY,
            logger=logger
        )
        filtered_ligands = filtered_genes.intersection(de_ligands)
        filtered_receptors = filtered_genes.intersection(de_receptors)
        filtered_tfs = filtered_genes.intersection(de_tfs)
        filtered_tgs = filtered_genes.intersection(tgs)
        lr_pairs = _combine_lr_names(filtered_ligands, filtered_receptors)
        logger.info(
            "After filtering: ligands=%d, receptors=%d, TFs=%d, TGs=%d, lr_pairs=%d",
            len(filtered_ligands), len(filtered_receptors), len(filtered_tfs), len(filtered_tgs), len(lr_pairs)
        )

        save_acquired_gene_sets(
            OUTPUT_DIR, PROJECT_NAME,
            ligands = filtered_ligands,
            receptors =filtered_receptors,
            tfs = filtered_tfs,
            tgs = filtered_tgs,
            lrs = lr_pairs,
            logger=logger
        )
        # Unify the names
        ligands = filtered_ligands
        receptors = filtered_receptors
        tfs = filtered_tfs
        tgs = filtered_tgs
    else:
        lr_pairs = _combine_lr_names(ligands, receptors)
        save_acquired_gene_sets(
            OUTPUT_DIR, PROJECT_NAME,
            ligands=ligands,
            receptors=receptors,
            tfs=tfs,
            tgs=tgs,
            lrs=lr_pairs,
            logger=logger
        )

    # Now, we conduct the path sampling procedure
    adata_all = adata[:, adata.var_names.isin(combined)].copy()
    adata_all = run_umap(adata_all)  # run umap on genes without dpt

    # Extract the adata_dp, receiver cells only
    adata_dp = adata_all[adata_all.obs[ANNOTATION_KEY].isin(receivers), :].copy()

    sc.tl.leiden(adata_all, key_added="clusters", resolution=30)
    logger.info("Leiden completed.")

    # Construct patial neighborhoods
    adata_neighbor, adata_lr, lr_var_names, present_lig, present_rec = build_neighbor_and_lr(adata_all,
                                                                                              adata_dp,
                                                                                              ligands,
                                                                                              receptors,
                                                                                              coords_key = SPATIAL_KEY,
                                                                                              radius = RADIUS,
                                                                                              n_jobs=N_JOBS
                                                                                              )
    adata_dp.obs['clusters'] = adata_all.obs.loc[adata_dp.obs_names, 'clusters']

    # Merge metacells
    adata_dp, merge_idx, membership, batch_map = merge_metacell_with_batch(adata_dp, batch_key=BATCH_KEY, pseudotime_key=PT_KEY, n_neighbors = args.n_neighbors)
    adata_neighbor, _, _, _ = merge_metacell_with_batch(adata_neighbor, batch_key=BATCH_KEY, pseudotime_key=PT_KEY, n_neighbors = args.n_neighbors)
    adata_lr, _, _, _ = merge_metacell_with_batch(adata_lr, batch_key=BATCH_KEY, pseudotime_key=PT_KEY, n_neighbors = args.n_neighbors)

    # Save the metacell memberships and batch information
    save_metacell_membership_and_batch(membership, batch_map, OUTPUT_DIR, PROJECT_NAME)

    # Re-compute pseudotime on the metacells with dpt
    iroot_idx = choose_iroot_safely(adata_dp, merge_idx, clusters_key="clusters", umap_key="X_umap")
    adata_dp.uns["iroot"] = iroot_idx
    sc.tl.dpt(adata_dp, key_added=PT_KEY)

    # Normalize gene_expressions
    normalize_genes(adata_dp)
    normalize_genes(adata_neighbor)
    normalize_genes(adata_lr)

    # slice down both adata and the neighbor‐averaged copy
    adata = adata_dp.copy()
    adata_neighbor = adata_neighbor.copy()

    idx_all = np.arange(len(adata))
    np.random.shuffle(idx_all)

    temporal_neighbors, _ = derive_temporal_neighborhood(
        adata, umap_key="X_umap", pseudotime_key=PT_KEY,
        k_primary=K_PRIMARY, k_fallback_scan=10, out_degree=2
    )

    paths_per_round = sample_paths(
        temporal_neighbors,
        len_path=PATH_LEN,
        n_rounds=2,  # train/test
        repeats_per_round=NUM_REPEATS,
        rng=np.random.default_rng(42),
        # draw_fn=draw_path, draw_example_n=10, draw_fn_kwargs={"adata": adata},
    )

    # 2) Collect expression tensors along those paths
    per_round = collect_expression_tensors_for_paths(
        paths_per_round,
        adata_neighbor=adata_neighbor,
        adata=adata,
        adata_lr=adata_lr,
        lig_genes=ligands,
        rec_genes=receptors,
        tf_genes=tfs,
        target_genes=tgs,
        lr_var_names=list(adata_lr.var_names),
        len_path=PATH_LEN,
    )

    # 3) Save to disk (NPZ bundles; optionally classic NPYs too)
    save_round_tensors_npz(
        per_round,
        out_dir=os.path.join(OUTPUT_DIR, 'data_triple'),
        split_names=["train", "test"],
        project_name=PROJECT_NAME,
        save_npz=True,
        also_save_npy=True,
    )

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
