import numpy as np


def combine_embedding_file_name(input_dir, file_name, mode = 'global'):
    return input_dir / 'embeddings' / f'{mode}_embeddings' / file_name

def attention_from_embeddings(driver_emb, tg_emb):
    # Compute attention from individual driver embedding (LR/TF) and Target gene embedding.
    return np.abs(np.matmul(driver_emb, tg_emb, transpose_b=True))


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
