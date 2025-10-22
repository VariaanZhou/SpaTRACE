# GREATEST_Chat
![Pipeline Overview](./assets/Method_Overview.png)

## Introduction

**GREATEST_Chat** (Granger REcurrent AuToEncoder for SpatialTemporal transcriptomics) is a pathway-free tool for **cell–cell communication (CCC) inference**, designed specifically for **developmental spatial transcriptomics** (or spatio-temporal transcriptomics) datasets.  

It enables:  
- **Cell–cell communication analysis**  
- **Gene regulatory network reconstruction**  
- **Ligand–receptor pair prediction**  

### How it works
GREATEST_Chat is a recurrent autoencoder trained on sampled cell trajectories from pseudotime. By learning the temporal dynamics of each **ligand–receptor pair, transcription factor, and target gene**, it captures semantic representations of cellular interactions. These embeddings can then be used to:  
- Reconstruct ligand–receptor interaction networks  
- Infer gene regulatory networks via score matching between LR/TF and TG embeddings  

### What this repo provides
- A **user-friendly interface** to run GREATEST_Chat on your own datasets  
- **Documentation and examples** from our experiments on:
  - Simulation Datasets (and their generation code)  
  - Mouse midbrain development  
  - Axolotl brain regeneration  





The workflow is organized into a **three-step pipeline**:

1. **Data preparation**: Taken an .h5ad data and given lists of ligands, receptors, TFs as inputs, it automatically extract DE genes, perform pseudotime analysis, and prepare input data for the model training.
2. **Model training**: Train the transformer model **GREATEST_Chat** on the prepared features.  
3. **Downstream analysis**: Perform feature selection and reconstruct ligand–receptor interactions, gene regulatory networks, and cellular interactions.


---

## 📂 Pipeline Overview

The pipeline consists of two main scripts:

1. **`run_preprocess.py`**  
   - Takes annotated single-cell `AnnData` objects as input.  
   - Builds spatial/temporal neighborhoods, metacells, and sampled paths.  
   - Extracts ligand, receptor, TF, and target features along these paths.  
   - Saves compact `.npz` bundles for training/testing.

2. **`run_experiment.py`**  
   - Consumes the `.npz` bundles.  
   - Trains the GRAEST_Chat transformer model.  
   - Saves learned weights, embeddings, and attentions.

---

## 🛠️ Requirements

- Python 3.9+
- Packages:
  - `scanpy`
  - `numpy`
  - `scipy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `tensorflow` (2.x)
- Plus the included model code: `model/GRAEST_Chat_v1_0.py`

---

## 🚀 Step 1: Preprocessing

### Run
```bash
python run_preprocess.py \
  --input my_input.h5ad \
  --out_dir ./outputs_preprocess \
  --project MyProj \
  --n_neighbors 15 \
  --len_path 3
```
## Outputs
```bash
outputs_preprocess/
└── data_triple/
    ├── MyProj_tensors_train.npz
    ├── MyProj_tensors_test.npz
    ├── recep_array_train.npy
    ├── ligand_array_train.npy
    ├── tf_array_train.npy
    ├── target_array_train.npy
    ├── label_array_train.npy
    ├── lr_pair_array_train.npy
    ├── all_paths_train.npy
    ├── recep_array_test.npy
    ├── ligand_array_test.npy
    ├── tf_array_test.npy
    ├── target_array_test.npy
    ├── label_array_test.npy
    ├── lr_pair_array_test.npy
    ├── all_paths_test.npy
    └── fig/ (diagnostic plots)
```
## 🚀 Step 2: Training & Experiment
### Run
```bash
python run_experiment.py \
  --input_dir ./outputs_preprocess/data_triple \
  --project MyProj \
  --out_dir ./outputs_experiment \
  --epochs 50 \
  --tlength 3 \
  --batch_size 16
```

## Outputs
```bash
outputs_experiment/
├── weights/
│   └── weights.weights.h5
├── embeddings/
│   ├── global_embeddings/
│   │   ├── embeddings_batch_0000.npz
│   │   ├── embeddings_batch_0001.npz
│   │   └── ...
│   └── percell_embeddings/
│       ├── embeddings_batch_0000.npz
│       ├── embeddings_batch_0001.npz
│       └── ...
└── attentions/
    ├── global_attentions/
    │   ├── attn_global_tf_topk_batch_0000.npz
    │   ├── attn_global_lr_topk_batch_0000.npz
    │   └── ...
    └── percell_attentions/
        ├── attn_percell_tf_topk_batch_0000.npz
        ├── attn_percell_lr_topk_batch_0000.npz
        └── ...
```
