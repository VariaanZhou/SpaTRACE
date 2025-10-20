import scanpy as sc
import difflib
import os
import logging
import numpy as np
def gene_intersection(gene_names, gene_list_file):
    '''
    This function takes a list of gene names, as well as a gene list .txt file as inputs.

    :param gene_names: list of gene names (strings)
    :param gene_list_file: path to a .txt file containing one gene name per line
    :return: set containing the intersection of gene_names and the gene list from the file
    '''
    # Read gene list from file
    file_genes = read_list_txt(gene_list_file)

    # Ensure input is also a set
    input_genes = set(gene_names)

    # Return intersection
    return list(input_genes.intersection(file_genes))


def read_list_txt(list_file, to_type=str):
    '''
    This function reads a .txt file where each row corresponds to one entry.
    Each entry is cast to the specified type.

    :param list_file: path to a .txt file
    :param to_type: desired Python type (default: str)
    :return: list of entries converted to the given type
    '''
    with open(list_file, "r") as f:
        entries = [to_type(line.strip()) for line in f if line.strip()]  # skip empty lines
    return entries

def verify_cell_types_exist(adata, cell_types, column):
    """
    Ensure every requested cell type is present in adata.obs[column].
    Raises ValueError with a helpful message if any are missing.

    :param adata: AnnData object
    :param cell_types: list of requested cell type names (strings)
    :param column: exact obs column name that stores cell type annotations (e.g., "cell_type")
    """
    if column not in adata.obs.columns:
        raise ValueError(
            f"Column '{column}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    # Compare against unique observed categories (as strings for safety)
    present = set(map(str, adata.obs[column].unique()))
    missing = [ct for ct in cell_types if ct not in present]

    if missing:
        # Close matches to reduce typos
        suggestions = {
            m: difflib.get_close_matches(m, present, n=3, cutoff=0.6) for m in missing
        }
        msg_lines = [
            f"The following cell types are missing from adata.obs['{column}']: {missing}",
            f"Observed unique cell types (n={len(present)}): {sorted(present)[:20]}{' ...' if len(present) > 20 else ''}",
            "Suggestions (per missing type):"
        ]
        for m, s in suggestions.items():
            msg_lines.append(f"  - {m}: {s if s else 'no close matches'}")
        raise ValueError("\n".join(msg_lines))

def _pt_exists(adata, pt_key: str = "dpt_pseudotime", cell_types=None, groupby: str = "cell_type"):
    """
    Check if a pseudotime key exists in adata.obs and is valid
    (non-null values available) for the given cell types.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    pt_key : str, default 'dpt_pseudotime'
        Column in adata.obs that stores pseudotime.
    cell_types : list[str] or None
        Subset of cell types to check pseudotime validity for.
        If None, check across all cells.
    groupby : str, default 'cell_type'
        The obs column that stores cell type annotations.

    Returns
    -------
    bool
        True if pseudotime exists and is valid for the given cell types,
        False otherwise.
    """

    # --- Check if key exists
    if pt_key not in adata.obs.columns:
        return False

    # --- Full check: no subset
    if cell_types is None:
        return adata.obs[pt_key].notna().any()

    # --- Check for subset of cell types
    if groupby not in adata.obs.columns:
        raise ValueError(f"groupby '{groupby}' not found in adata.obs")

    mask = adata.obs[groupby].isin(cell_types)
    if mask.sum() == 0:
        raise ValueError(f"None of the requested cell_types {cell_types} found in adata.obs['{groupby}'].")

    return adata.obs.loc[mask, pt_key].notna().any()

def _check_no_reserved_strings(adata):
    '''
    _to_ string is used to represent the LR names, all genes must not contain these as substrings.

    Parameters
    ----------
    adata

    Returns
    -------

    '''
    pass

def _sort_gene_by_values(list):
    try:
        # For simulation data only
        return sorted(list, key=lambda x: int(x), reverse=True)
    except ValueError:
        return list
def _combine_lr_names(ligands, receptors):
    return [f"{ligand}_to_{receptor}" for ligand in ligands for receptor in receptors]

def _write_one_list(path, genes):
    # Accept list/set/iterable of strings; write sorted, unique
    with open(path, "w") as f:
        for g in genes:
            f.write(f"{g}\n")

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ---------------------- Logging ----------------------
def setup_logging(level: str = "INFO"):
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("graest_preprocess")
    # Reduce noisy loggers if needed:
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("scanpy").setLevel(logging.WARNING)
    logging.getLogger("anndata").setLevel(logging.WARNING)
    logging.getLogger("squidpy").setLevel(logging.WARNING)
    return logger