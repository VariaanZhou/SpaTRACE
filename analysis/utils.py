import numpy as np


def combine_embedding_file_name(input_dir, file_name, mode = 'global'):
    return input_dir / 'embeddings' / f'{mode}_embeddings' / file_name

def attention_from_embeddings(driver_emb, tg_emb):
    """
    Compute |driver_emb @ tg_emb.T|.

    driver_emb: (G_driver, d)
    tg_emb:     (G_tg, d)
    returns:    (G_driver, G_tg)
    """
    return np.abs(driver_emb @ tg_emb.T)

def _unparse_lr_var_name(lr_name):
    lr = lr_name.split('_to_')
    return lr[0], lr[1]

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


def load_embedding_npz_batch(npz_filename):
    embeddings = np.load(npz_filename)
    idx = embeddings['idx']
    tg_emb = embeddings['x_vq1']
    tf_emb = embeddings['tf_vq1']
    ligrecp_emb = embeddings['recp_vq1']
    return idx, tg_emb, tf_emb, ligrecp_emb


import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional

def plot_lr_tg_heatmaps(
    *,
    sum_arr: np.ndarray,       # (G_lr, G_tg) summed intensities
    count_arr: np.ndarray,     # (G_lr, G_tg) counts (how many times survived filtering)
    stage: str,                # stage label (e.g., "E12.5")
    out_dir: str,              # directory to save the figures
    title_prefix: str = "LR→TG",   # prefix for plot titles
    figsize: Tuple[int, int] = (8, 6),
    fontsize: int = 12,
    dpi: int = 300,
):
    """
    Plot LR→TG intensity heatmaps (summed weights and counts).

    Parameters
    ----------
    sum_arr : np.ndarray
        Aggregated LR→TG weights (float array).
    count_arr : np.ndarray
        Aggregated LR→TG counts (int/float array).
    stage : str
        Biological stage or batch identifier.
    out_dir : str
        Directory to save output PNGs.
    title_prefix : str
        Prefix to prepend to titles (default: "LR→TG").
    figsize : (int, int)
        Figure size in inches.
    fontsize : int
        Font size for labels and titles.
    dpi : int
        Resolution for saving PNG.
    """
    os.makedirs(out_dir, exist_ok=True)

    # summed weights heatmap
    plt.figure(figsize=figsize)
    im = plt.imshow(sum_arr, aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im)
    cbar.set_label("Summed weight", fontsize=fontsize)
    plt.xlabel("Target genes (TG)", fontsize=fontsize)
    plt.ylabel("Ligand–Receptor (LR)", fontsize=fontsize)
    plt.title(f"{title_prefix} Intensity — {stage}", fontsize=fontsize+2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"lr_tg_sum__{stage}.png"), dpi=dpi)
    plt.close()

    # counts heatmap
    plt.figure(figsize=figsize)
    im = plt.imshow(count_arr, aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im)
    cbar.set_label("Counts (survived filtering)", fontsize=fontsize)
    plt.xlabel("Target genes (TG)", fontsize=fontsize)
    plt.ylabel("Ligand–Receptor (LR)", fontsize=fontsize)
    plt.title(f"{title_prefix} Count — {stage}", fontsize=fontsize+2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"lr_tg_count__{stage}.png"), dpi=dpi)
    plt.close()


def plot_lr_heatmap(
    *,
    lr_intensity: np.ndarray,  # (G_lr, G_tg) summed intensities
    ligands: list,             # ligand/receptor labels for rows
    receptors: list,           # TG labels for columns
    stage: str,                 # stage label (e.g., "E12.5")
    out_dir: str,               # directory to save the figures
    mode: str,
    title_prefix: str = "LR→TG",
    figsize: tuple = (8, 6),
    fontsize: int = 12,
    dpi: int = 300,
):
    """
    Plot a ligand/receptor → target gene intensity heatmap.

    Parameters
    ----------
    lr_intensity : np.ndarray
        Array of shape (G_lr, G_tg) with summed intensities.
    ligands : list
        Row labels (ligands/receptors).
    receptors : list
        Column labels (target genes).
    stage : str
        Stage label for the plot title and filename.
    out_dir : str
        Directory where the figure will be saved.
    """
    if lr_intensity.shape != (len(ligands), len(receptors)):
        raise ValueError(
            f"Shape mismatch: lr_intensity {lr_intensity.shape} "
            f"!= ({len(ligands)}, {len(receptors)})"
        )

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=figsize, dpi=dpi)
    im = plt.imshow(lr_intensity, aspect="auto", cmap="viridis")

    plt.colorbar(im, fraction=0.046, pad=0.04, label="Summed Intensity")
    if mode == 'percell':
        plt.title(f"{title_prefix} Heatmap — Stage {stage}", fontsize=fontsize+2)
    else:
        plt.title(f"{title_prefix} Heatmap", fontsize=fontsize+2)

    # axis ticks
    plt.xticks(range(len(receptors)), receptors, rotation=90, fontsize=fontsize-2)
    plt.yticks(range(len(ligands)), ligands, fontsize=fontsize-2)

    plt.tight_layout()
    fig_name = f"{mode}_lr_heatmap__{stage}.png" if stage else f"{mode}_lr_heatmap.png"
    out_path = os.path.join(out_dir, fig_name)
    plt.savefig(out_path)
    plt.close()

    return out_path
