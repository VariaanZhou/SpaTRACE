#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""
Runnable with TensorFlow 2.18 / Keras 3.

Notes:
- The L1 losses (outputs 5–8) are regularizers on inferred network tensors.
  They **ignore y_true** in the loss function, so we simply pass any placeholder
  (we reuse `target_exp_y`) to satisfy Keras’ API.
- All prior TF/Keras-3 plumbing (mask handling, keyword-only args, build stubs)
  is retained.

"""

# ------------------------------- Layers -------------------------------

class CellEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model: int, gene_size: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.gene_size = int(gene_size)
        self.proj_kernel = None  # created in build()

    def build(self, input_shape):
        self.proj_kernel = self.add_weight(
            name="proj_kernel",
            shape=(self.gene_size, self.d_model),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x, training=None):
        # x: (B, L, M)  ->  (B, L, d_model)
        return tf.einsum('blm,mn->bln', x, self.proj_kernel)

    def get_config(self):
        return {"d_model": self.d_model, "gene_size": self.gene_size, **super().get_config()}


class BaseAttentionGene(tf.keras.layers.Layer):
    """Self-attention over gene tokens with residual + LayerNorm."""
    def __init__(self, num_heads=1, key_dim=128, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout
        )
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.add = tf.keras.layers.Add()

    def call(self, x, training=None, mask=None):
        attn = self.mha(query=x, value=x, key=x,
                        use_causal_mask=False,
                        attention_mask=mask,
                        training=training)
        # x = self.add([x, self.dropout(attn, training=training)])
        # x = self.layernorm(x)
        return attn

    def build(self, input_shape):
        super().build(input_shape)


class GeneEmbedding(tf.keras.layers.Layer):
    """
    Produces per-gene token embeddings with a small value pathway,
    then applies self-attention over the (gene) tokens.
    """
    def __init__(self, d_model: int, gene_size: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.total_gene_size = int(gene_size)
        self.proj_kernel_token = None
        self.proj_kernel_disc_value = None
        self.proj_kernel_value = None
        self.proj_kernel2_value = None
        self.scale_factor = None
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.self_attention = BaseAttentionGene(num_heads=1, key_dim=d_model, dropout=0)

    def build(self, input_shape):
        self.proj_kernel_token = self.add_weight(
            name="proj_kernel_token",
            shape=(self.total_gene_size, self.d_model),
            initializer='random_normal',
            trainable=True,
        )
        self.proj_kernel_disc_value = self.add_weight(
            name="proj_kernel_disc_value",
            shape=(self.d_model, 1),
            initializer='random_normal',
            trainable=True,
        )
        self.proj_kernel_value = self.add_weight(
            name="proj_kernel_value",
            shape=(self.d_model, 1),
            initializer='random_normal',
            trainable=True,
        )
        self.proj_kernel2_value = self.add_weight(
            name="proj_kernel2_value",
            shape=(self.d_model, self.d_model),
            initializer='random_normal',
            trainable=True,
        )
        self.scale_factor = self.add_weight(
            name="scale_factor",
            shape=(),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x, *, gene_size: int, training=None):
        """
        x: (B, L, M_total) where M_total <= self.total_gene_size
        gene_size: integer number of leading tokens to return
        Returns:
          x_token[:, :, :gene_size, :], x[:, :, :gene_size, :]
        """
        # token pathway (IDs → token vectors), then L2 norm
        x_token = tf.einsum('blm,mn->blmn', tf.ones_like(x, dtype=tf.float32), self.proj_kernel_token)
        x_token = tf.nn.l2_normalize(x_token, axis=-1)

        # value pathway
        x_value = tf.einsum('blm,nk->blmn', x, self.proj_kernel_value)
        x_value = self.leaky_relu(x_value)
        x_value = tf.einsum('blmn,nk->blmk', x_value, self.proj_kernel2_value) + self.scale_factor * x_value
        x_value = tf.nn.softmax(x_value, axis=-1)
        x_value = tf.einsum('blmn,nk->blm', x_value, self.proj_kernel_disc_value)

        # fuse token & value
        x = tf.einsum('blmn,blm->blmn', x_token, x_value)  # (B, L, M_total, d_model)

        # fold for self-attn over tokens
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        x = tf.reshape(x, tf.concat([[B * L], tf.shape(x)[2:]], axis=0))  # (B*L, M_total, d_model)

        x = self.self_attention(x, training=training)  # (B*L, M_total, d_model)

        # restore (B, L, M_total, d_model)
        x = tf.reshape(x, tf.concat([[B, L], tf.shape(x)[1:]], axis=0))

        g = int(gene_size)
        return x_token[:, :, :g, :], x[:, :, :g, :]

    def get_config(self):
        return {"d_model": self.d_model, "gene_size": self.total_gene_size, **super().get_config()}


class CustomAttentionCell(tf.keras.layers.Attention):
    """
    Cross-attention that computes values from (vq, vk, vexp) tensors.
    """

    def __init__(self, use_scale=False, score_mode='dot', dropout=0.0, **kwargs):
        super().__init__(use_scale=use_scale, score_mode=score_mode, **kwargs)
        self.dropout = float(dropout)

    def _calculate_values(self, vq, vk, vexp):
        # vq: (B,Lq,Mq,d), vk: (B,Lk,Mk,d), vexp: (B,Lk,Mk)
        weights = tf.einsum('blmn,bkpn->blkmp', vq, vk)   # (B,Lq,Lk,Mq,Mk)
        v = tf.einsum('blkmp,bkp->blkm', weights, vexp)   # (B,Lq,Lk,Mq)
        return v

    def _apply_scores(self, scores, value, scores_mask=None, training=False):
        # scores: (B,Lq,Lk), value: (B,Lq,Lk,Mq)
        if scores_mask is not None:
            padding_mask = tf.logical_not(scores_mask)
            max_value = 65504.0 if scores.dtype == "float16" else 1.0e9
            scores -= max_value * tf.cast(padding_mask, scores.dtype)

        weights = tf.nn.softmax(scores, axis=-1)
        if training and self.dropout > 0.0:
            weights = tf.nn.dropout(weights, rate=self.dropout)

        result = tf.einsum('blk,blkm->blm', weights, value)  # (B,Lq,Mq)
        return result, weights

    def call(self, inputs, mask=None, training=None,
             return_attention_scores=False, use_causal_mask=False):
        # inputs = [q, (vexp, vq, vk), k]
        q = inputs[0]                     # (B,Lq,d)
        vexp, vq, vk = inputs[1]          # (B,Lk,Mk), (B,Lq,Mq,d), (B,Lk,Mk,d)
        k = inputs[2]                     # (B,Lk,d)

        # --- unpack masks safely ---
        q_mask = None
        k_mask = None
        if isinstance(mask, (list, tuple)) and len(mask) == 3:
            q_mask = mask[0]
            k_mask = mask[2]  # ignore middle tuple (vexp, vq, vk)

        # --- scores & values ---
        scores = self._calculate_scores(query=q, key=k)          # (B,Lq,Lk)
        v = self._calculate_values(vq=vq, vk=vk, vexp=vexp)      # (B,Lq,Lk,Mq)

        # --- build scores mask from key mask and causal mask ---
        scores_mask = None
        if k_mask is not None:
            if k_mask.dtype != tf.bool:
                k_mask = tf.not_equal(tf.cast(k_mask, tf.int32), 0)
            scores_mask = tf.expand_dims(k_mask, axis=-2)        # (B,1,Lk)

        if use_causal_mask:
            s = tf.shape(scores)
            causal_shape = tf.concat([tf.ones_like(s[:-2]), s[-2:]], axis=0)
            row = tf.cumsum(tf.ones(causal_shape, tf.int32), axis=-2)
            col = tf.cumsum(tf.ones(causal_shape, tf.int32), axis=-1)
            causal_mask = tf.greater_equal(row, col)             # broadcastable
            scores_mask = causal_mask if scores_mask is None else tf.logical_and(scores_mask, causal_mask)

        result, attention_scores = self._apply_scores(scores, v, scores_mask, training=training)

        if q_mask is not None:
            q_mask = tf.expand_dims(tf.cast(q_mask, tf.bool), axis=-1)  # (B,Lq,1)
            result *= tf.cast(q_mask, result.dtype)

        if return_attention_scores:
            return result, attention_scores
        return result


class BaseAttentionCell(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = CustomAttentionCell(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        super().build(input_shape)


class CrossAttention(BaseAttentionCell):
    def call(self, x_query, context_key, context_vexp, x_vq, context_vk, training=None):
        attn_output, attn_scores = self.mha(
            inputs=[x_query, (context_vexp, x_vq, context_vk), context_key],
            use_causal_mask=True,
            return_attention_scores=True,
            training=training,
        )
        self.last_attn_scores = attn_scores
        return attn_output

    def build(self, input_shape):
        super().build(input_shape)


class DecoderCrossLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.cross_attention = CrossAttention(use_scale=True, dropout=dropout_rate)
        # self.ffn = FeedForward(d_model, dff)

    def call(self, x_query, context_key, context_vexp, x_vq, context_vk, training=None):
        x = self.cross_attention(x_query, context_key, context_vexp, x_vq, context_vk, training=training)
        self.last_attn_scores = self.cross_attention.last_attn_scores
        # x = self.ffn(x, training=training)
        return x

    def build(self, input_shape):
        super().build(input_shape)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, target_gene_size, ligrecp_size, tf_gene_size, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.target_gene_size = int(target_gene_size)
        self.lr_size = ligrecp_size
        self.tf_gene_size = int(tf_gene_size)

        # cell embeddings
        self.embed_cell_target = CellEmbedding(d_model=d_model, gene_size=target_gene_size)
        self.embed_cell_tf = CellEmbedding(d_model=d_model, gene_size=tf_gene_size)
        self.embed_cell_ligrecp = CellEmbedding(d_model=d_model, gene_size=self.lr_size)

        # gene embeddings (capacity set to sum of all for safety)
        cap = self.lr_size + tf_gene_size + target_gene_size
        self.embed_gene_target  = GeneEmbedding(d_model=d_model, gene_size=cap)
        self.embed_gene_tf      = GeneEmbedding(d_model=d_model, gene_size=cap)
        self.embed_gene_ligrecp = GeneEmbedding(d_model=d_model, gene_size=cap)

        self.dec_cross_layers_tf = DecoderCrossLayer(
            d_model=target_gene_size, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
        )
        self.dec_cross_layers_ligand = DecoderCrossLayer(
            d_model=target_gene_size, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
        )

        # exposed internals (filled on call)
        self.last_attn_scores = None
        self.x_vq1_global = None
        self.tf_vk1_global = None
        self.recp_vk_global = None
        self.x_vq1_percell = None
        self.tf_vk1_percell = None
        self.recp_vk_percell = None

    def call(self, target_exp, tf_exp, ligrecp_exp, training=None):
        # --- embeddings ---
        cell_embedding_ligrecp = self.embed_cell_ligrecp(ligrecp_exp, training=training)
        gglob, gcell = self.embed_gene_ligrecp(
            tf.concat([ligrecp_exp, tf_exp, target_exp], axis=-1),
            gene_size=self.lr_size,
            training=training,
        )
        ligrecp_gene_embedding_global = gglob[:, :, :self.lr_size, :]
        ligrecp_gene_embedding_percell= gcell[:, :, :self.lr_size, :]

        cell_embedding_tf = self.embed_cell_tf(tf_exp, training=training)
        tf_gglob, tf_gcell = self.embed_gene_tf(
            tf.concat([tf_exp, ligrecp_exp, target_exp], axis=-1),
            gene_size=self.tf_gene_size,
            training=training,
        )

        cell_embedding_target = self.embed_cell_target(target_exp, training=training)
        tgt_gglob, tgt_gcell = self.embed_gene_target(
            tf.concat([target_exp, ligrecp_exp, tf_exp], axis=-1),
            gene_size=self.target_gene_size,
            training=training,
        )

        # ---- predictions (4 supervised outputs) ----
        target_exp_y1_global   = self.dec_cross_layers_tf(
            cell_embedding_target, cell_embedding_tf, tf_exp, tgt_gglob, tf_gglob, training=training
        )
        target_exp_y2_global   = self.dec_cross_layers_ligand(
            cell_embedding_target, cell_embedding_ligrecp, ligrecp_exp, tgt_gglob, ligrecp_gene_embedding_global, training=training
        )
        target_exp_y1_percell  = self.dec_cross_layers_tf(
            cell_embedding_target, cell_embedding_tf, tf_exp, tgt_gcell, tf_gcell, training=training
        )
        target_exp_y2_percell  = self.dec_cross_layers_ligand(
            cell_embedding_target, cell_embedding_ligrecp, ligrecp_exp, tgt_gcell, ligrecp_gene_embedding_percell, training=training
        )

        # ---- network tensors (4 additional outputs; large tensors) ----
        # Shapes ~ (B, L, L, Gtgt, Gtf/Grec)
        network_tf_global    = tf.einsum('blmn,bkpn->blkmp', tgt_gglob, tf_gglob)
        network_recp_global  = tf.einsum('blmn,bkpn->blkmp', tgt_gglob, ligrecp_gene_embedding_global)
        network_tf_percell   = tf.einsum('blmn,bkpn->blkmp', tgt_gcell, tf_gcell)
        network_recp_percell = tf.einsum('blmn,bkpn->blkmp', tgt_gcell, ligrecp_gene_embedding_percell)

        # expose for downstream inspection
        self.x_vq1_global   = tgt_gglob
        self.tf_vk1_global  = tf_gglob
        self.recp_vk_global = ligrecp_gene_embedding_global
        self.x_vq1_percell  = tgt_gcell
        self.tf_vk1_percell = tf_gcell
        self.recp_vk_percell= ligrecp_gene_embedding_percell

        return (target_exp_y1_global, target_exp_y2_global,
                target_exp_y1_percell, target_exp_y2_percell,
                network_tf_global, network_recp_global,
                network_tf_percell, network_recp_percell)

    def build(self, input_shape):
        super().build(input_shape)


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 ligrecp_size, tf_gene_size, target_gene_size,
                 dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.decoder = Decoder(
            num_layers=num_layers, d_model=d_model, target_gene_size=target_gene_size,
            ligrecp_size=ligrecp_size, tf_gene_size=tf_gene_size, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
        )

    def call(self, inputs, training=None):
        ligrecp_exp, tf_exp, target_exp = inputs
        (y1g, y2g, y1c, y2c,
         net_tf_g, net_rec_g, net_tf_c, net_rec_c) = self.decoder(target_exp, tf_exp, ligrecp_exp, training=training)

        # Return a DICT with 8 outputs
        return {
            "output_1": y1g,
            "output_2": y2g,
            "output_3": y1c,
            "output_4": y2c,
            "output_5": net_tf_g,
            "output_6": net_rec_g,
            "output_7": net_tf_c,
            "output_8": net_rec_c,
        }

    def build(self, input_shape):
        super().build(input_shape)

# ------------------------------- Losses -------------------------------

def masked_mse(y_true, y_pred, sentinel=-99999.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    mask = tf.not_equal(y_true, tf.constant(sentinel, y_pred.dtype))
    diff = y_pred - y_true
    diff = tf.where(mask, diff, tf.zeros_like(diff))
    denom = tf.maximum(tf.reduce_sum(tf.cast(mask, y_pred.dtype)), 1.0)
    return tf.reduce_sum(tf.square(diff)) / denom

# def masked_mse(label, pred):
#   mask = label != -99999
#   loss_object = tf.keras.losses.MeanSquaredError(
#     name='mean_squared_error')
#   loss = loss_object(label, pred)
#
#   mask = tf.cast(mask, dtype=loss.dtype)
#   loss *= mask
#
#   loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
#   return loss

def l1_reg_loss(y_true, y_pred):
    # Regularization-style L1: ignores y_true on purpose
    del y_true
    return tf.reduce_mean(tf.abs(y_pred))

def infer_cpu(model, lr, tf_in, tgt):
    with tf.device('/CPU:0'):
        return model([lr, tf_in, tgt], training=False)


# Noam scheduler is used for optimization
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return (tf.math.rsqrt(self.d_model) *
                tf.math.minimum(tf.math.rsqrt(step), step * (self.warmup_steps ** -1.5)))