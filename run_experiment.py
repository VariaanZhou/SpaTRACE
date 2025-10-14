import os
import gc
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from model.GRAEST_Chat_v1_0 import (
    Transformer, CustomSchedule, masked_mse, l1_reg_loss, infer_cpu
)

def _feat_dim(x):
    return x.shape[2] if x.ndim == 3 else x.shape[1]


def _auto_detect_project(base: Path) -> str:
    cand = list(base.glob("*_tensors_train.npz"))
    if len(cand) == 1:
        stem = cand[0].stem  # "<project>_tensors_train"
        return stem[: -len("_tensors_train")]
    if len(cand) == 0:
        raise FileNotFoundError(f"No '*_tensors_train.npz' found in {base}")
    raise RuntimeError(
        f"Multiple '*_tensors_train.npz' found in {base}; please pass --project explicitly."
    )

def _load_npz_split(path: Path, tlen: int):
    Z = np.load(path)
    # keys from preprocess: tf, lr_pair, target, label (plus others we ignore)
    tf_exp = Z["tf"].astype("float32")[:, :tlen, :]
    ligrecp_exp = Z["lr_pair"].astype("float32")[:, :tlen, :]
    target_exp = Z["target"].astype("float32")
    target_exp_y = Z["label"].astype("float32")
    return tf_exp, ligrecp_exp, target_exp, target_exp_y
def main():
    parser = argparse.ArgumentParser(prog="GREATEST_Chat")

    # IO + data
    parser.add_argument(
        "-i", "--input_dir", required=True, type=str,
        help="Directory containing <project>_tensors_train.npz / _test.npz"
    )
    parser.add_argument(
        "-p", "--project", type=str, default=None,
        help="Project prefix used by preprocess (e.g., 'MyProj'). If omitted, will auto-detect a single *_tensors_train.npz."
    )
    parser.add_argument(
        "-o", "--out_dir", required=True, type=str,
        help="Directory to save results (weights/embeddings/attentions)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--tlength", type=int, default=3, help="Length of sampled paths (time steps)")
    parser.add_argument("--mmap", action="store_true",
                        help="(Ignored for .npz loading; kept for CLI compatibility)")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension (d_model)")
    parser.add_argument("--dff", type=int, default=256, help="Feed-forward hidden size")
    parser.add_argument("--num_heads", type=int, default=5, help="Number of attention heads")
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate")

    # Export / inference controls
    parser.add_argument("--infer_batch_size", type=int, default=1,
                        help="Batch size for inference/exports")
    parser.add_argument("--save_attentions", dest="save_attentions", action="store_true", default=True,
                        help="Save attention matrices (default: True)")
    parser.add_argument("--no-save_attentions", dest="save_attentions", action="store_false",
                        help="Disable saving attention matrices")
    parser.add_argument("--save_visuals", dest="save_visuals", action="store_true", default=True,
                        help="Save PNG previews of attention matrices (default: True)")
    parser.add_argument("--no-save_visuals", dest="save_visuals", action="store_false",
                        help="Disable PNG previews")
    parser.add_argument("--save_full_weights", dest="save_full_weights", action="store_true", default=False,
                        help="Save full dense attention matrices (default: False = top-k sparse)")
    parser.add_argument("--no-save_full_weights", dest="save_full_weights", action="store_false",
                        help="Save sparse top-k per row (recommended)")
    parser.add_argument("--topk_per_row", type=int, default=50, help="Top-k per row when saving sparse")
    parser.add_argument("--dtype_on_disk", type=str, choices=["float16", "float32"], default="float16",
                        help="Disk dtype for exported arrays")

    args = parser.parse_args()

    print("------------- GRAEST_Chat started -------------")

    # ------------------------------- Paths & sanity -------------------------------
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    project = args.project or _auto_detect_project(in_dir)

    train_npz = in_dir / f"{project}_tensors_train.npz"
    test_npz = in_dir / f"{project}_tensors_test.npz"

    if not train_npz.is_file():
        raise FileNotFoundError(f"Required file not found: {train_npz}")
    if not test_npz.is_file():
        raise FileNotFoundError(f"Required file not found: {test_npz}")

    # ------------------------------- Data -------------------------------
    tlength = int(args.tlength)
    if tlength <= 0:
        raise ValueError(f"--tlength must be > 0, got {tlength}")

    print("------------- Loading data (NPZ) -------------")

    # Training
    tf_exp, ligrecp_exp, target_exp, target_exp_y = _load_npz_split(train_npz, tlength)

    # Validation
    tf_exp_val, ligrecp_exp_val, target_exp_val, target_exp_y_val = _load_npz_split(test_npz, tlength)

    # Basic alignment checks (fail fast)
    n_train = tf_exp.shape[0]
    n_val = tf_exp_val.shape[0]
    for name, arr in {
        "ligrecp_exp": ligrecp_exp, "target_exp": target_exp, "target_exp_y": target_exp_y
    }.items():
        if arr.shape[0] != n_train:
            raise ValueError(f"Train size mismatch: tf_exp={n_train} vs {name}={arr.shape[0]}")
    for name, arr in {
        "ligrecp_exp_val": ligrecp_exp_val, "target_exp_val": target_exp_val, "target_exp_y_val": target_exp_y_val
    }.items():
        if arr.shape[0] != n_val:
            raise ValueError(f"Val size mismatch: tf_exp_val={n_val} vs {name}={arr.shape[0]}")

    print("------------- Data loaded -------------")
    print(
        f"Train: N={n_train}, tlength={tf_exp.shape[1]}, "
        f"TF_dim={_feat_dim(tf_exp)}, Recp_dim={_feat_dim(ligrecp_exp)}, TG_dim={_feat_dim(target_exp)}"
    )
    print(
        f"Val:   N={n_val},   tlength={tf_exp_val.shape[1]}, "
        f"TF_dim={_feat_dim(tf_exp_val)}, Recp_dim={_feat_dim(ligrecp_exp_val)}, TG_dim={_feat_dim(target_exp_val)}"
    )

    # ------------------------------- Model -------------------------------
    print("------------- Training started -------------")
    ligrecp_gene_size = ligrecp_exp.shape[-1]
    tf_gene_size       = tf_exp.shape[-1]
    target_gene_size   = _feat_dim(target_exp)

    transformer = Transformer(
        num_layers=1,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        ligrecp_size=ligrecp_gene_size,
        tf_gene_size=tf_gene_size,
        target_gene_size=target_gene_size,
        dropout_rate=args.dropout_rate,
    )

    learning_rate = CustomSchedule(args.d_model)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    transformer.compile(
        optimizer=optimizer,
        loss={
            "output_1": masked_mse,
            "output_2": masked_mse,
            "output_3": masked_mse,
            "output_4": masked_mse,
            "output_5": l1_reg_loss,
            "output_6": l1_reg_loss,
            "output_7": l1_reg_loss,
            "output_8": l1_reg_loss,
        },
        loss_weights={
            "output_1": 1.0, "output_2": 1.0, "output_3": 1.0, "output_4": 1.0,
            "output_5": 0.5, "output_6": 0.5, "output_7": 0.5, "output_8": 0.5,
        },
        metrics={"output_1": tf.keras.metrics.MeanSquaredError()},
    )

    y_train = {f"output_{k}": target_exp_y for k in range(1, 9)}
    y_val   = {f"output_{k}": target_exp_y_val for k in range(1, 9)}

    history = transformer.fit(
        (ligrecp_exp, tf_exp, target_exp),
        y_train,
        batch_size=args.batch_size,
        epochs=int(args.epochs),
        validation_data=([ligrecp_exp_val, tf_exp_val, target_exp_val], y_val),
    )
    print("------------- Training finished -------------")

    # ------------------------------- Output dirs -------------------------------
    weights_dir        = Path(args.out_dir) / "weights"
    emb_root_dir       = Path(args.out_dir) / "embeddings"
    attn_root_dir      = Path(args.out_dir) / "attentions"

    global_emb_dir     = emb_root_dir / "global_embeddings"
    percell_emb_dir    = emb_root_dir / "percell_embeddings"
    global_att_dir     = attn_root_dir / "global_attentions"
    percell_att_dir    = attn_root_dir / "percell_attentions"

    for p in [weights_dir, global_emb_dir, percell_emb_dir, global_att_dir, percell_att_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # ------------------------------- Save weights -------------------------------
    try:
        transformer.save_weights(str(weights_dir / "weights.weights.h5"))
    except Exception as e:
        print(f"[WARN] Model save failed, continuing to export anyway: {e}")

    # ------------------------------- Inference & export -------------------------------
    print("------------- Saving embeddings and attentions -------------")
    # Export config (from CLI)
    BATCH_SIZE        = int(args.infer_batch_size)
    SAVE_ATTENTIONS   = bool(args.save_attentions)
    SAVE_VISUALS      = bool(args.save_visuals)
    SAVE_FULL_WEIGHTS = bool(args.save_full_weights)
    TOPK_PER_ROW      = int(args.topk_per_row)
    DTYPE_ON_DISK     = np.float16 if args.dtype_on_disk == "float16" else np.float32

    n_batches = int(np.ceil(len(ligrecp_exp_val) / BATCH_SIZE))

    def _to_np16(x):
        # honor requested disk dtype (name kept for backwards clarity)
        return tf.cast(x, tf.as_dtype(DTYPE_ON_DISK)).numpy().astype(DTYPE_ON_DISK, copy=False)

    def _save_npz(path, **arrays):
        np.savez_compressed(path, **arrays)

    def _save_heatmap_png(path, arr_2d):
        plt.imsave(path, arr_2d, format="png")

    def _topk_sparse_rows(mat, k):
        k = tf.minimum(k, tf.shape(mat)[1])
        vals, idxs = tf.math.top_k(mat, k=k, sorted=False)
        return vals, idxs

    def _summarize_and_save_attentions(kind, x_vq, tf_vk, recp_vk, out_dir, batch_idx_base):
        if not SAVE_ATTENTIONS:
            return
        # collapse (B,L)
        x_vq_mean    = tf.reduce_mean(tf.reduce_mean(x_vq, axis=0), axis=0)     # (G_tgt, d)
        tf_vk_mean   = tf.reduce_mean(tf.reduce_mean(tf_vk, axis=0), axis=0)    # (G_tf,  d)
        recp_vk_mean = tf.reduce_mean(tf.reduce_mean(recp_vk, axis=0), axis=0)  # (G_rec, d)

        weight_nt    = tf.abs(tf.matmul(tf_vk_mean,   x_vq_mean, transpose_b=True))
        weight_nt_lr = tf.abs(tf.matmul(recp_vk_mean, x_vq_mean, transpose_b=True))

        if not SAVE_FULL_WEIGHTS:
            nt_vals, nt_cols = _topk_sparse_rows(weight_nt, TOPK_PER_ROW)
            lr_vals, lr_cols = _topk_sparse_rows(weight_nt_lr, TOPK_PER_ROW)

            _save_npz(out_dir / f"attn_{kind}_tf_topk_batch_{batch_idx_base:04d}.npz",
                      vals=_to_np16(nt_vals), cols=nt_cols.numpy().astype(np.int32, copy=False))
            _save_npz(out_dir / f"attn_{kind}_lr_topk_batch_{batch_idx_base:04d}.npz",
                      vals=_to_np16(lr_vals), cols=lr_cols.numpy().astype(np.int32, copy=False))

            if SAVE_VISUALS:
                R_nt = int(weight_nt.shape[0]);  C = int(weight_nt.shape[1])
                R_lr = int(weight_nt_lr.shape[0]); preview_cap = 4096
                if C <= preview_cap:
                    nt_preview = np.zeros((R_nt, C), dtype=DTYPE_ON_DISK)
                    lr_preview = np.zeros((R_lr, C), dtype=DTYPE_ON_DISK)
                    rows_nt = np.arange(R_nt, dtype=np.int32)[:, None]
                    rows_lr = np.arange(R_lr, dtype=np.int32)[:, None]
                    nt_preview[rows_nt, nt_cols.numpy()] = _to_np16(nt_vals)
                    lr_preview[rows_lr, lr_cols.numpy()] = _to_np16(lr_vals)
                    _save_heatmap_png(out_dir / f"attn_{kind}_tf_preview_{batch_idx_base:04d}.png", nt_preview)
                    _save_heatmap_png(out_dir / f"attn_{kind}_lr_preview_{batch_idx_base:04d}.png", lr_preview)
        else:
            nt_np = _to_np16(weight_nt)
            lr_np = _to_np16(weight_nt_lr)
            _save_npz(out_dir / f"attn_{kind}_tf_full_batch_{batch_idx_base:04d}.npz", weight_nt=nt_np)
            _save_npz(out_dir / f"attn_{kind}_lr_full_batch_{batch_idx_base:04d}.npz", weight_nt_lr=lr_np)
            if SAVE_VISUALS:
                _save_heatmap_png(out_dir / f"attn_{kind}_tf_full_{batch_idx_base:04d}.png", nt_np)
                _save_heatmap_png(out_dir / f"attn_{kind}_lr_full_{batch_idx_base:04d}.png", lr_np)

    for b in range(n_batches):
        sl = slice(b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, len(ligrecp_exp_val)))
        try:
            _ = transformer([ligrecp_exp_val[sl], tf_exp_val[sl], target_exp_val[sl]], training=False)
        except Exception as e:
            print(f"[INFO] Batch {b}: GPU error ({e}); falling back to CPU.")
            _ = infer_cpu(transformer, ligrecp_exp_val[sl], tf_exp_val[sl], target_exp_val[sl])

        # per-cell
        x_vq1_percell    = transformer.decoder.x_vq1_percell
        tf_vk1_percell   = transformer.decoder.tf_vk1_percell
        recp_vk1_percell = transformer.decoder.recp_vk_percell
        # global
        x_vq1_global     = transformer.decoder.x_vq1_global
        tf_vk1_global    = transformer.decoder.tf_vk1_global
        recp_vk1_global  = transformer.decoder.recp_vk_global

        # embeddings
        batch_indices = np.arange(sl.start, sl.stop, dtype=np.int64)
        np.savez_compressed(
            global_emb_dir / f"embeddings_batch_{b:04d}.npz",
            idx=batch_indices,
            x_vq1=_to_np16(x_vq1_global),
            tf_vq1=_to_np16(tf_vk1_global),
            recp_vq1=_to_np16(recp_vk1_global),
        )
        np.savez_compressed(
            percell_emb_dir / f"embeddings_batch_{b:04d}.npz",
            idx=batch_indices,
            x_vq1=_to_np16(x_vq1_percell),
            tf_vq1=_to_np16(tf_vk1_percell),
            recp_vq1=_to_np16(recp_vk1_percell),
        )

        # attentions
        _summarize_and_save_attentions("global",  x_vq1_global,  tf_vk1_global,  recp_vk1_global,
                                       global_att_dir,  b * BATCH_SIZE)
        _summarize_and_save_attentions("percell", x_vq1_percell, tf_vk1_percell, recp_vk1_percell,
                                       percell_att_dir, b * BATCH_SIZE)

        del (x_vq1_percell, tf_vk1_percell, recp_vk1_percell,
             x_vq1_global, tf_vk1_global, recp_vk1_global)
        gc.collect()
        tf.keras.backend.clear_session()

    print("[DONE] Weights, embeddings, and (global + percell) attentions exported.")

if __name__ == "__main__":
    main()
