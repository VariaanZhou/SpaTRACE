import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any
import os
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import radius_neighbors_graph

def merge_metacell_with_batch(
    adata,
    *,
    clusters_key: str = "clusters",
    batch_key: str = "batch",
    pseudotime_key: str = "pseudotime",   # set to "pseudo_time" if that’s your column
    n_neighbors: int = 10
):
    """
    Build a metacell AnnData aggregated by `clusters_key`, keep the majority
    batch membership for each metacell, and return the membership mapping.

    Returns
    -------
    meta : AnnData
        Metacell-level AnnData with:
          - meta.obs.index == metacell IDs (cluster labels)
          - meta.obs['batch'] == majority batch per metacell
          - meta.obsm['X_umap'] == mean UMAP per metacell
    merge_idx : str or int
        Cluster label of the cell with minimal pseudotime (if available), else -1.
    membership : dict[str, list[str]]
        Mapping: metacell ID -> list of composing cell obs_names.
    batch_map : dict[str, str]
        Mapping: metacell ID -> majority batch.
    """
    # --- Determine merge target via pseudotime, if provided
    if pseudotime_key in adata.obs.columns:
        min_cell = adata.obs[pseudotime_key].idxmin()
        merge_idx = adata.obs.loc[min_cell, clusters_key]
    else:
        merge_idx = -1

    # --- Pull original cluster codes/categories
    cat = adata.obs[clusters_key].astype("category")
    old_codes = cat.cat.codes.values
    cats_all = cat.cat.categories

    # --- Compact categories to those actually present
    unique_codes, inv = np.unique(old_codes, return_inverse=True)
    new_categories = cats_all[unique_codes]  # metacell IDs
    n_obs, n_vars = adata.X.shape
    n_clust = len(new_categories)

    # --- Build indicator matrix H (K × n_obs)
    rows = inv
    cols = np.arange(n_obs)
    data = np.ones(n_obs, dtype=float)
    H = csr_matrix((data, (rows, cols)), shape=(n_clust, n_obs))

    # --- Aggregate expression and UMAP
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    umap = adata.obsm["X_umap"]

    counts = np.asarray(H.sum(axis=1)).ravel()
    cluster_mean_X = H.dot(X) / counts[:, None]
    mean_umap = H.dot(umap) / counts[:, None]

    # --- Majority batch per metacell + membership map
    # indices of cells per metacell (list of arrays)
    # For each metacell k, member indices are np.where(inv == k)[0]
    membership = {}
    batch_map = {}

    # materialize obs_names, batch col
    obs_names = np.asarray(adata.obs_names)
    if batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key '{batch_key}' not found in adata.obs")

    cell_batches = adata.obs[batch_key].astype(str).values

    for k, mc in enumerate(new_categories):
        member_idx = np.where(inv == k)[0]
        membership[str(mc)] = obs_names[member_idx].tolist()

        # majority batch (deterministic tie-break: lexicographically smallest among max)
        if len(member_idx) == 0:
            batch_map[str(mc)] = "NA"
        else:
            vc = pd.Series(cell_batches[member_idx]).value_counts()
            max_count = vc.max()
            winners = sorted(vc[vc == max_count].index.tolist())
            batch_map[str(mc)] = winners[0]

    # --- Assemble meta AnnData
    meta = sc.AnnData(
        X=cluster_mean_X,
        obs=pd.DataFrame(index=new_categories),
        var=adata.var.copy()
    )
    meta.obsm["X_umap"] = mean_umap
    meta.obs[batch_key] = pd.Series([batch_map[str(mc)] for mc in new_categories],
                                    index=new_categories, dtype="category")

    # Optional: keep the compact cluster label as a column too
    meta.obs[clusters_key] = meta.obs.index.astype("category")

    # --- Rebuild neighbor graph on metacells
    sc.pp.neighbors(meta, n_neighbors=n_neighbors, use_rep="X_umap")

    return meta, merge_idx, membership, batch_map


def save_metacell_membership_and_batch(membership, batch_map, out_dir, project):
    """
    Save membership and batch_map dictionaries to CSV.

    Parameters
    ----------
    membership : dict
        Mapping: metacell ID -> list of composing cell obs_names.
    batch_map : dict
        Mapping: metacell ID -> majority batch.
    out_dir : str
        Directory to save the membership information
    """
    # membership: expand into rows (metacell, cell_id)
    membership_rows = []
    for mc, cells in membership.items():
        for c in cells:
            membership_rows.append({"metacell": mc, "cell_id": c})
    membership_df = pd.DataFrame(membership_rows)
    membership_file = os.path.join(out_dir, f"{project}_metacell_membership.csv")
    membership_df.to_csv(membership_file, index=False)

    # batch map: one row per metacell
    batch_df = pd.DataFrame([
        {"metacell": mc, "batch": batch}
        for mc, batch in batch_map.items()
    ])
    batch_map_file = os.path.join(out_dir, f"{project}_metacell_batchmap.csv")
    batch_df.to_csv(batch_map_file, index=False)

    print(f"Saved: {membership_file} and {batch_map_file}")

# Prepare the data for training
def draw_path(adata, path, cnt, t):
    plt.clf()
    spatial = adata.obsm['X_umap']
    plt.scatter(spatial[:, 0], spatial[:, 1], color='grey', s=1)
    plt.scatter(spatial[path[0], 0], spatial[path[0], 1], color='red', s=10)
    for i in range(len(path) - 1):
        plt.plot([spatial[path[i], 0], spatial[path[i + 1], 0]], [spatial[path[i], 1], spatial[path[i + 1], 1]],
                 color='red')
        plt.scatter(spatial[path[i + 1], 0], spatial[path[i + 1], 1], color='red', s=1)
    plt.savefig('fig/path_' + str(t) + '_' + str(cnt) + '_temporal.png')

def run_umap(adata, n_neighbors=100, min_dist=0.01):
    """
    Run PCA + UMAP on the AnnData object unless both 'X_pca' and 'X_umap'
    already exist in adata.obsm. Returns the updated AnnData.
    """
    # If PCA and UMAP already exist, skip computation
    if "X_umap" in adata.obsm:
        print("Skipping neighbors+UMAP: 'X_umap' already present.")
        return adata

    # Compute PCA
    sc.pp.pca(
        adata,
        n_comps=50,
        svd_solver='arpack'
    )

    # Build graph & run UMAP on PCA space
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep='X_pca'
    )
    sc.tl.umap(adata, min_dist=min_dist)
    return adata

# helper to extract and squeeze .X with sparse-check
def extract_array(adata_obj, obs_idx, var_idx):
    subX = adata_obj[obs_idx, var_idx].X
    if hasattr(subX, "toarray"):
        subX = subX.toarray()
    return np.squeeze(subX)



# def derive_spatial_neighbors(
#     adata,
#     *,
#     coords_key: str = "spatial",
#     radius: float = 50.0,
#     include_self: bool = True,
#     n_jobs: int = -1
# ):
#     """
#     Build a sparse radius-neighborhood graph from 2D coordinates in `adata.obsm[coords_key]`.
#
#     Returns
#     -------
#     A : scipy.sparse.csr_matrix
#         (n_cells x n_cells) binary connectivity (1 if within radius).
#     """
#     if coords_key not in adata.obsm_keys():
#         raise ValueError(f"'{coords_key}' not found in adata.obsm")
#
#     A = radius_neighbors_graph(
#         adata.obsm[coords_key],
#         radius=radius,
#         mode="connectivity",
#         include_self=False,  # add explicitly below to keep behavior consistent across sklearn
#         n_jobs=n_jobs,
#     ).tocsr()
#
#     if include_self:
#         A.setdiag(1)
#
#     return A

def build_neighbor_and_lr(
    adata_all,
    adata_dp,
    all_lig,
    all_recept,
    *,
    coords_key: str = "spatial",
    radius: float = 50.0,
    n_jobs: int = -1,
    dtype=np.float32,
):
    """
    Create:
      1) adata_neighbor: a copy of adata_all with neighbor-smoothed expression
         (sum over radius neighbors, on expm1-transformed counts).
      2) adata_lr: LR pair features built as log1p( expm1(lig_smooth) * expm1(receptor) ).

    Both are finally subset to adata_dp.obs_names for row alignment.

    Parameters
    ----------
    adata_all : AnnData
        Full dataset with adata_all.obsm[coords_key] present.
    adata_dp : AnnData
        Dataset whose obs_names define the final cell subset (e.g., pseudotime subset).
    all_lig : list[str]
        Ligand gene names (columns in adata_all.var_names).
    all_recept : list[str]
        Receptor gene names (columns in adata_all.var_names).
    coords_key : str
        Key in .obsm with 2D/ND coordinates for radius neighbors (default "spatial").
    radius : float
        Neighborhood radius for radius_neighbors_graph (default 50.0).
    n_jobs : int
        Parallelism for sklearn neighbor graph (-1 = all cores).
    dtype : numpy dtype
        Cast arrays to this dtype to save memory (default float32).

    Returns
    -------
    adata_neighbor : AnnData
        Copy of adata_all (same var), subset to adata_dp.obs_names after smoothing.
        NOTE: X is *not* overwritten; this object is returned to keep meta info aligned.
              If you want the smoothed matrix on it, uncomment the line indicated below.
    adata_lr : AnnData
        LR pair feature matrix, subset to adata_dp.obs_names.
    lr_var_names : list[str]
        Names of LR features (ligand_to_receptor order: ligands major).
    present_lig : list[str]
        The ligand genes actually found in adata_all.var_names.
    present_rec : list[str]
        The receptor genes actually found in adata_all.var_names.
    """
    if coords_key not in adata_all.obsm_keys():
        raise ValueError(f"'{coords_key}' not found in adata_all.obsm")

    # Determine which lig/receptors exist
    var_index = {g: i for i, g in enumerate(adata_all.var_names)}
    present_lig = [g for g in all_lig if g in var_index]
    present_rec = [g for g in all_recept if g in var_index]
    if not present_lig or not present_rec:
        raise ValueError("No overlap between provided ligands/receptors and adata_all.var_names.")

    # Radius neighbor graph (binary connectivity), add self-loops
    A = radius_neighbors_graph(
        adata_all.obsm[coords_key],
        radius=radius,
        mode="connectivity",
        include_self=False,  # add explicitly below
        n_jobs=n_jobs,
    ).tocsr()
    A.setdiag(1)

    # Make a neighbor-smoothed version of expm1(X)
    X_all = adata_all.X
    if hasattr(X_all, "toarray"):
        X_all = X_all.toarray()
    X_all = X_all.astype(dtype, copy=False)
    X_lin = np.expm1(X_all)                      # undo log1p if present
    X_smooth = A.dot(X_lin)                      # sum over neighbors

    # Build LR features: log1p( expm1(lig_smooth) * expm1(receptor) )
    lig_idx = [var_index[g] for g in present_lig]
    rec_idx = [var_index[g] for g in present_rec]

    X_lig = X_smooth[:, lig_idx].astype(dtype, copy=False)          # (n, L)
    X_rec = adata_all[:, present_rec].X
    if hasattr(X_rec, "toarray"):
        X_rec = X_rec.toarray()
    X_rec = X_rec.astype(dtype, copy=False)                         # (n, R)

    lr_prod = X_lig[:, :, None] * np.expm1(X_rec)[:, None, :]       # (n, L, R)
    X_lr = np.log1p(lr_prod).reshape(adata_all.n_obs, -1)           # (n, L*R)

    # Build AnnData for LR features
    lr_var_names = [f"{lig}_to_{rec}" for lig in present_lig for rec in present_rec]
    adata_lr = sc.AnnData(X=X_lr)
    adata_lr.var_names = lr_var_names
    adata_lr.obs = adata_all.obs.copy()
    adata_lr.uns = adata_all.uns
    adata_lr.obsm = adata_all.obsm
    adata_lr.obsp = adata_all.obsp

    # Start neighbor output as a full copy to preserve meta
    adata_neighbor = adata_all.copy()
    # If you want to store the smoothed matrix back onto adata_neighbor, uncomment:
    # adata_neighbor.X = X_smooth

    # Align both to adata_dp.obs_names (no deep copy of var)
    adata_neighbor = adata_neighbor[adata_dp.obs_names, :]
    adata_lr = adata_lr[adata_dp.obs_names, :]

    return adata_neighbor, adata_lr, lr_var_names, present_lig, present_rec



def derive_temporal_neighborhood(
    adata,
    *,
    umap_key: str = "X_umap",
    pseudotime_key: str = "dpt_pseudotime",
    k_primary: int = 5,
    k_fallback_scan: int = 10,
    out_degree: int = 2,
) -> Tuple[Dict[int, List[int]], np.ndarray]:
    """
    Reimplementation of the script segment building temporal neighbors:

      • For each cell i, get its k_primary nearest neighbors in UMAP (incl. self).
      • Keep only neighbors with strictly larger pseudotime.
      • Sort by pseudotime descending and keep up to `out_degree` (default 2).
      • If not enough, scan the first k_fallback_scan nearest (excluding self)
        and take the first with higher pseudotime until filled.

    Returns
    -------
    temporal_neighbors : dict[int, list[int]]
        i -> list of successor indices (len 0..out_degree).
    D : np.ndarray
        Pairwise UMAP distances (float32) used to construct neighbors.
    """
    if umap_key not in adata.obsm_keys():
        raise ValueError(f"'{umap_key}' not found in adata.obsm")
    if pseudotime_key not in adata.obs.columns:
        raise ValueError(f"'{pseudotime_key}' not found in adata.obs")

    X = adata.obsm[umap_key]
    pt = adata.obs[pseudotime_key].values
    n = X.shape[0]

    # pairwise distances in UMAP
    D = squareform(pdist(X)).astype(np.float32)

    temporal_neighbors: Dict[int, List[int]] = {}

    for i in range(n):
        nn = np.argsort(D[i])  # ascending; nn[0] is i itself

        # Primary set: first k_primary neighbors (may include self)
        primary = nn[:k_primary]

        # Filter to forward-in-time neighbors only, and sort by pt descending
        forward = [(j, pt[j]) for j in primary if j != i and pt[j] > pt[i]]
        forward.sort(key=lambda x: -x[1])  # higher pseudotime first
        chosen = [j for j, _ in forward[:out_degree]]

        # Fallback scan: first k_fallback_scan (skip self at nn[0])
        if len(chosen) < out_degree:
            for j in nn[1 : 1 + k_fallback_scan]:
                if pt[j] > pt[i] and j not in chosen:
                    chosen.append(j)
                    if len(chosen) >= out_degree:
                        break

        temporal_neighbors[i] = chosen

    return temporal_neighbors, D


def sample_paths(
    temporal_neighbors: Dict[int, List[int]],
    *,
    len_path: int = 3,
    n_rounds: int = 2,            # match the script's two outer loops (train/test)
    repeats_per_round: int = 10,  # match the script's inner "for repeat in range(10)"
    rng: Optional[np.random.Generator] = None,
    draw_fn=None,                 # optional: draw_fn(adata, path, cnt, t)
    draw_example_n: int = 0,
    draw_fn_kwargs: Optional[dict] = None,
) -> List[List[List[int]]]:
    """
    Reimplementation of the script’s path sampling:

      • For each round t in [0..n_rounds-1]:
          For each node i, attempt a forward random walk of length `len_path`
          using `temporal_neighbors`. Repeat `repeats_per_round` times.
          Keep only complete paths (len == len_path + 1).

      • Optionally draw the first `draw_example_n` paths of each round using draw_fn.

    Returns
    -------
    paths_per_round : list of rounds, where each round is a list of paths.
        paths_per_round[t] -> List[List[int]] with each inner list a node index path.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    nodes = list(temporal_neighbors.keys())
    draw_kwargs = draw_fn_kwargs or {}

    paths_per_round: List[List[List[int]]] = []

    for t in range(n_rounds):
        round_paths: List[List[int]] = []
        draw_count = 0

        for _ in range(repeats_per_round):
            for i in nodes:
                path = [i]
                cur = i
                for _step in range(len_path):
                    nxts = temporal_neighbors.get(cur, [])
                    if not nxts:
                        break
                    nxt = int(rng.choice(nxts))
                    path.append(nxt)
                    cur = nxt

                # keep only complete paths
                if len(path) == len_path + 1:
                    round_paths.append(path)
                    if draw_fn is not None and draw_count < draw_example_n:
                        # expected signature: draw_fn(adata, path, cnt, t)
                        draw_fn(path=path, cnt=draw_count, t=t, **draw_kwargs)
                        draw_count += 1

        paths_per_round.append(round_paths)

    return paths_per_round



def collect_expression_tensors_for_paths(
    paths_per_round: List[List[List[int]]],
    *,
    adata_neighbor,           # ligand source (smoothed)
    adata,                    # receptor/TF/target source
    adata_lr,                 # LR-pair source (all columns used)
    lig_genes: List[str],
    rec_genes: List[str],
    tf_genes: List[str],
    target_genes: List[str],
    lr_var_names: Optional[List[str]] = None,  # if None, will use adata_lr.var_names
    len_path: int = 3,
) -> List[Dict[str, np.ndarray]]:
    """
    For each round, materialize expression tensors along sampled paths:
      - ligand:  (n_paths, len_path+1, |lig_genes|)        from adata_neighbor
      - receptor:(n_paths, len_path+1, |rec_genes|)        from adata
      - tf:      (n_paths, len_path+1, |tf_genes|)         from adata
      - target:  (n_paths, len_path+1, |target_genes|)     from adata
      - lr_pair: (n_paths, len_path+1, |adata_lr.var|)     from adata_lr
      - label:   shifted targets = target[:, 1:, :]        (n_paths, len_path, |target_genes|)
      - target_trunc: target[:, :len_path, :]              (n_paths, len_path, |target_genes|)

    Returns
    -------
    per_round : list of dict
        Each dict has keys:
        ['paths','ligand','receptor','tf','target','lr_pair','label','target_trunc'].
    """

    # Build quick var_name -> idx maps (safe if some genes missing)
    var_idx_neighbor = {g: i for i, g in enumerate(adata_neighbor.var_names)}
    var_idx_main     = {g: i for i, g in enumerate(adata.var_names)}
    lr_names         = list(adata_lr.var_names) if lr_var_names is None else lr_var_names

    lig_idx = [var_idx_neighbor[g] for g in lig_genes if g in var_idx_neighbor]
    rec_idx = [var_idx_main[g]     for g in rec_genes   if g in var_idx_main]
    tf_idx  = [var_idx_main[g]     for g in tf_genes    if g in var_idx_main]
    tgt_idx = [var_idx_main[g]     for g in target_genes if g in var_idx_main]

    if lr_var_names is not None:
        assert list(adata_lr.var_names) == list(lr_var_names), "lr_var_names mismatch adata_lr.var_names"

    # Sparse-safe .X getter
    def _get_block(adata_obj, obs_idx: List[int], var_idx: List[int]) -> np.ndarray:
        X = adata_obj[obs_idx, var_idx].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X)

    per_round: List[Dict[str, Any]] = []

    for round_paths in paths_per_round:
        # Collect per-path matrices
        lig_list, rec_list, tf_list, tgt_list, lr_list = [], [], [], [], []

        for p in round_paths:
            if len(p) != (len_path + 1):
                # only keep complete paths
                continue

            # Extract along the path
            lig  = _get_block(adata_neighbor, p, lig_idx)             # (len_path+1, L)
            rec  = _get_block(adata,          p, rec_idx)             # (len_path+1, R)
            tf   = _get_block(adata,          p, tf_idx)              # (len_path+1, T)
            tgt  = _get_block(adata,          p, tgt_idx)             # (len_path+1, G)
            lrp  = _get_block(adata_lr,       p, list(range(adata_lr.n_vars)))  # all LR vars

            lig_list.append(lig)
            rec_list.append(rec)
            tf_list.append(tf)
            tgt_list.append(tgt)
            lr_list.append(lrp)

        if len(lig_list) == 0:
            # round with no complete paths
            per_round.append({
                "paths": [],
                "ligand": np.empty((0, len_path+1, len(lig_idx))),
                "receptor": np.empty((0, len_path+1, len(rec_idx))),
                "tf": np.empty((0, len_path+1, len(tf_idx))),
                "target": np.empty((0, len_path+1, len(tgt_idx))),
                "lr_pair": np.empty((0, len_path+1, adata_lr.n_vars)),
                "label": np.empty((0, len_path,   len(tgt_idx))),
                "target_trunc": np.empty((0, len_path, len(tgt_idx))),
            })
            continue

        ligand_array  = np.stack(lig_list, axis=0)    # (N, len_path+1, L)
        recep_array   = np.stack(rec_list, axis=0)    # (N, len_path+1, R)
        tf_array      = np.stack(tf_list,  axis=0)    # (N, len_path+1, T)
        target_array  = np.stack(tgt_list, axis=0)    # (N, len_path+1, G)
        lr_pair_array = np.stack(lr_list,  axis=0)    # (N, len_path+1, LR)

        # shift labels: next-step targets
        label_array   = target_array[:, 1:, :]        # (N, len_path, G)
        target_trunc  = target_array[:, :len_path, :] # (N, len_path, G)

        per_round.append({
            "paths": round_paths,
            "ligand": ligand_array,
            "receptor": recep_array,
            "tf": tf_array,
            "target": target_array,
            "lr_pair": lr_pair_array,
            "label": label_array,
            "target_trunc": target_trunc,
        })

    return per_round


def save_round_tensors_npz(
    per_round: List[Dict[str, np.ndarray]],
    *,
    out_dir: str = "data_triple",
    split_names: Optional[List[str]] = None,   # e.g., ["train","test"]
    project_name: str = "",
    save_npz: bool = True,
    also_save_npy: bool = False,
) -> None:
    """
    Save tensors from `collect_expression_tensors_for_paths` to files.

    - If `split_names` is None, uses ["round0", "round1", ...]
    - Saves a single .npz per round (recommended), optionally individual .npy files.

    Files (examples):
      {out_dir}/{prefix}ligand_train.npz
      {out_dir}/{prefix}receptor_test.npy
    """
    os.makedirs(out_dir, exist_ok=True)

    if split_names is None:
        split_names = [f"round{i}" for i in range(len(per_round))]

    for t, blobs in enumerate(per_round):
        tag = split_names[t] if t < len(split_names) else f"round{t}"

        if save_npz:
            np.savez_compressed(
                os.path.join(out_dir, f"{project_name}_tensors_{tag}.npz"),
                ligand=blobs["ligand"],
                receptor=blobs["receptor"],
                tf=blobs["tf"],
                target=blobs["target"],
                lr_pair=blobs["lr_pair"],
                label=blobs["label"],
                target_trunc=blobs["target_trunc"],
                paths=np.array(blobs["paths"], dtype=object),
            )

        if also_save_npy:
            np.save(os.path.join(out_dir, f"{project_name}_ligand_{tag}.npy"),       blobs["ligand"])
            np.save(os.path.join(out_dir, f"{project_name}_receptor_{tag}.npy"),     blobs["receptor"])
            np.save(os.path.join(out_dir, f"{project_name}_tf_{tag}.npy"),           blobs["tf"])
            np.save(os.path.join(out_dir, f"{project_name}_target_{tag}.npy"),       blobs["target"])
            np.save(os.path.join(out_dir, f"{project_name}_lr_pair_{tag}.npy"),      blobs["lr_pair"])
            np.save(os.path.join(out_dir, f"{project_name}_label_{tag}.npy"),        blobs["label"])
            np.save(os.path.join(out_dir, f"{project_name}_target_trunc_{tag}.npy"), blobs["target_trunc"])

def choose_iroot_safely(meta, merge_idx, clusters_key="clusters", umap_key="X_umap"):
    import numpy as np
    if merge_idx != -1 and clusters_key in meta.obs:
        hits = np.where(meta.obs[clusters_key].values == merge_idx)[0]
        if len(hits):
            return int(hits[0])
    if umap_key in meta.obsm:
        return int(np.argmin(meta.obsm[umap_key][:, 0]))
    return 0

def normalize_genes(adata):
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    maxs = X.max(axis=0)
    maxs[maxs == 0] = 1.0
    adata.X = X / maxs