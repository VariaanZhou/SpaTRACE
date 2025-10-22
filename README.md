# GREATEST_Chat
## Introduction

**GREATEST_Chat** (Granger REcurrent AuToEncoder for SpatialTemporal transcriptomics) is a pathway-free tool for **cell–cell communication (CCC)** inference, specifically designed for developmental spatial transcriptomics (spatio-temporal transcriptomics) datasets. It captures cellular dynamics across different developmental stages as well as interactions within the surrounding microenvironment, enabling efficient CCC inference in undercharacterized tissues or species.

It enables:  
- **Cell–cell communication analysis**  
- **Gene regulatory network reconstruction**  
- **Ligand–receptor pair prediction**

![Pipeline Overview](./assets/Method_Overview.png)
### How it works

**GREATEST_Chat** is a recurrent autoencoder trained on sampled cell trajectories from pseudotime. By modeling the temporal dynamics of each **ligand–receptor pair, transcription factor, and target gene** under **L1 regularization**, the model learns embeddings that capture semantic representations of cellular interactions.  

These embeddings can then be used to:  
- Reconstruct **ligand–receptor → target gene** relationships  
- Infer **TF → target gene** regulatory links via score matching  
- Build integrated **cell–cell communication and gene regulatory networks**

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
## Pipeline Overview

The pipeline is organized into **three main scripts**:

1. **`run_preprocess.py`**  
   - Constructs spatial and temporal neighborhoods, metacells, and sampled trajectories.  
   - Extracts ligand, receptor, TF, and target features along these paths.  
   - Outputs compact `.npz` bundles for model training and testing.  

2. **`run_experiment.py`**  
   - Trains the **GREATEST_Chat** transformer model on the preprocessed data.  
   - Produces learned weights, embeddings, and attention maps.  

3. **`run_inference.py`**  
   - Performs gene-level inference of regulatory relationships (**LR → TG**, **TF → TG**, and **L → R**).  
   - Aggregates gene-level and spatial information to predict **cellular-level interactions**.  

---

## Requirements

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

## Step 0: Input Preparation

Before running the pipeline, you must prepare a list of input files under a working directory and follow our naming convention. You need to define a **project name** (`project_name`), abd all input files should be named using this prefix.

### Required Input Files
The pipeline accepts `.h5ad` files as input. Please prepare the following **two stRNA datasets**:

- **`{project_name}_sc_adata.h5ad`**  
  Contains all *receiver cells*, their expression profiles, and pseudotime. If available, also include the UMAP and PCA. 
- **`{project_name}_st_adata.h5ad`**  
  Contains *all cells* (both sender and receiver) with associated expression profiles and spatial information.  

### Gene Lists
Provide text files containing your genes of interest:  

- **`{project_name}_ligand.txt`** — list of ligands  
- **`{project_name}_receptor.txt`** — list of receptors  
- **`{project_name}_tf.txt`** — list of transcription factors  
- **(Optional)** **`{project_name}_tg.txt`** — list of target genes (if omitted, the model will automatically infer target genes)  

### Cell Type Lists
Specify the cell types relevant to your analysis:  

- **`{project_name}_receiver.txt`** — list of receiver cell types  
- **(Optional)** **`{project_name}_sender.txt`** — list of sender cell types (if omitted, the model will use spatially enriched genes surrounding the receiver cells)  

## Step 1: Preprocessing
With the input data provided as in Step 0, our model will preprocess the data and sample cell trajectories as the input of model training. Please provide the following parameters (Here we use simulation data 1 as an example):
### Run
```bash
python run_preprocess.py \
  --data_dir ./inputs \
  --project_name MyProj \
  --batch_key batch \
  --annotation_key 'Cell Types' \
  --pt_key dpt_pseudotime \
  --sp_key spatial \
  --n_neighbors 10 \
  --path_len 3 \
  --num_repeats 10 \
  --k_primary 5 \
  --skip_de \
  --radius 1 \
  --n_jobs -1
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
## Step 2: Training & Experiment
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
