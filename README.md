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
  --data_dir ./experiments/simulation \
  --project_name simulation \
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
## Arguments explained
- `--data_dir` – working directory where you saved the input files.  
- `--project_name` – project name (also used as the base filename).  
- `--batch_key` – column in `obs` storing batch info.  
- `--annotation_key` – column in `obs` with cell type annotations.  
- `--pt_key` – pseudotime key (e.g., `dpt_pseudotime`).  
- `--sp_key` – key in `.obsm` for spatial coordinates.  
- `--n_neighbors` – number of neighbors to aggregate metacells (default: 10).  
- `--path_len` – sampled path length (default: 3).  
- `--num_repeats` – number of sampling paths for each metacell (default: 10).  
- `--k_primary` – closest *k* temporal neighbors considered as potential descendants (default: 5).  
- `--skip_de` – skip the differential expression step; use this if you want to include all ligands, receptors, TFs, and TGs directly.  
- `--radius` – neighborhood radius (default: 50; here set to 1).  
- `--n_jobs` – number of parallel tasks (default: `-1` uses all available CPUs).  

## Outputs
## Output directory structure

```bash
outputs_preprocess/
├── data_triple/
│   ├── MyProj_tensors_train.npz         # bundled training tensors
│   ├── MyProj_tensors_test.npz          # bundled testing tensors
│   ├── recep_array_train.npy            # receptor expressions (train)
│   ├── ligand_array_train.npy           # ligand expressions (train)
│   ├── tf_array_train.npy               # transcription factor expressions (train)
│   ├── target_array_train.npy           # target gene expressions (train)
│   ├── label_array_train.npy            # labels (train)
│   ├── lr_pair_array_train.npy          # ligand–receptor pair info (train)
│   ├── all_paths_train.npy              # sampled paths (train)
│   ├── recep_array_test.npy             # receptor expressions (test)
│   ├── ligand_array_test.npy            # ligand expressions (test)
│   ├── tf_array_test.npy                # transcription factor expressions (test)
│   ├── target_array_test.npy            # target gene expressions (test)
│   ├── label_array_test.npy             # labels (test)
│   ├── lr_pair_array_test.npy           # ligand–receptor pair info (test)
│   ├── all_paths_test.npy               # sampled paths (test)
│   └── fig/                             # diagnostic plots
├── MyProj_ligands.txt                   # identified ligands
├── MyProj_receptors.txt                 # identified receptors
├── MyProj_tfs.txt                       # identified transcription factors
├── MyProj_tgs.txt                       # identified target genes
├── MyProj_receivers.txt                 # receiver cell types
└── MyProj_senders.txt                   # sender cell types

```
## Step 2: Training & Experiment

After extracting essential genes and sampling the cell trajectories, you can call `run_experiment.py` to train the recurrent autoencoder on the data.  
The model will extract embeddings, model weights, and global attention scores between drivers and TGs, saving them under the user-provided output directory.  
If specified, per-cell attentions and visualizations can also be saved.  

Here we use the simulation data as an example.

### Run

```bash
python run_experiment.py \
  --data_dir ./experiments/simulation \
  --project simulation \
  --out_dir ./experiments/simulation/results \
  --d_model 64 \
  --dff 64 \
  --num_heads 3 \
  --epochs 50 \
  --tlength 3 \
  --batch_size 16 \
  --save_visuals \
  --save_percell_attentions

## Arguments explained

- `--data_dir` – same working directory used in `run_preprocess.py`.  
- `--project` – project name (must match the preprocess step).  
- `--out_dir` – directory for experiment outputs.  
- `--d_model` – embedding size for the model.  
- `--dff` – hidden layer size in the feedforward block.  
- `--num_heads` – number of attention heads.  
- `--epochs` – number of training epochs.  
- `--tlength` – trajectory path length.  
- `--batch_size` – batch size for training.  
- `--save_visuals` – save driver → TG attention visualizations.  
- `--save_percell_attentions` – save per-cell attention matrices.  

## Outputs
```bash
outputs_experiment/
├── weights/                        # trained model weights
│   └── weights.weights.h5
├── embeddings/
│   ├── global_embeddings/           # global embeddings across cells
│   │   ├── embeddings_batch_0000.npz
│   │   ├── embeddings_batch_0001.npz
│   │   └── ...
│   └── percell_embeddings/          # per-cell embeddings
│       ├── embeddings_batch_0000.npz
│       ├── embeddings_batch_0001.npz
│       └── ...
└── attentions/
    ├── global_attentions/           # top-k global attention scores
    │   ├── attn_global_tf_topk_batch_0000.npz
    │   ├── attn_global_lr_topk_batch_0000.npz
    │   └── ...
    └── percell_attentions/          # per-cell attention scores
        ├── attn_percell_tf_topk_batch_0000.npz
        ├── attn_percell_lr_topk_batch_0000.npz
        └── ...
```

## Step 3: Inference & Post-Processing

After training, you can run `run_inference.py` to aggregate and interpret the results.  
This script performs several levels of analysis on GRAEST outputs:

- **Gene-level aggregation**:  
  Aggregates per-cell embeddings into per-stage TF→TG and LR→TG interaction intensities.  
  Summarizes global attentions into LR intensities.

- **Cellular-level aggregation**:  
  Combines stage-specific matrices into cell-type–to–cell-type communication intensities.  
  Generates heatmaps and CSVs for sender–receiver pairs.

- **Visualization (optional)**:  
  Saves heatmaps, bar plots, and other figures for global and per-cell interactions.

---

### Run

```bash
python run_inference.py \
  --data_dir ./experiments/simulation \
  --input_dir ./experiments/simulation/results \
  --out_dir ./experiments/simulation/analysis \
  --project_name simulation \
  --batch_key batch \
  --groupby 'Cell Types' \
  --filter_threshold 0.2 \
  --radius 1 \
  --topk_per_col 100 \
  --top_n_bar 20 \
  --dpi 300 \
  --export_csv
```
## Arguments explained

- `--data_dir` – directory containing the original `.h5ad` and preprocessed project folder (same as used in `run_preprocess.py`).  
- `--input_dir` – directory with `embeddings/` and `attentions/` (output of `run_experiment.py`).  
- `--out_dir` – directory to store inference results (`gene_interactions/`, `cell_interactions/`, figures, CSVs).  
- `--project_name` – project prefix (e.g., `MyProj`), required to locate gene lists.  
- `--batch_key` – column in metadata representing developmental stage or batch.  
- `--groupby` – `obs` column used for cell-type grouping in cellular inference.  
- `--stages` – ordered list of stages to analyze (e.g., `E12.5 E14.5 E16.5`).  
- `--filter_threshold` – minimum intensity filter threshold (default: 0.01).  
- `--radius` – spatial neighbor radius (in microns) for cellular inference.  
- `--topk_per_col` – top-K entries per column for filtering during per-cell aggregation.  
- `--top_n_bar` – top-N entries shown in LR bar plots.  
- `--no_heatmaps` – disable generation of PNG heatmaps.  
- `--skip_percell` – skip per-cell aggregation and plots.  
- `--skip_attentions` – do not recompute per-cell intensities; load from disk instead.  
- `--export_csv` – save combined cellular communication matrices as CSV.  
- `--figsize` / `--dpi` – control figure size and resolution.  

## Outputs
```bash
inference/
├── gene_interactions/
│   ├── global_LR_intensity.csv
│   ├── percell_LR_stage_E12.5.csv
│   ├── percell_LR_stage_E14.5.csv
│   └── ...
├── cell_interactions/
│   ├── SenderA__to__ReceiverB__combined_E12.5_E14.5.csv
│   ├── SenderB__to__ReceiverC__combined_E16.5.csv
│   └── ...
└── figures/
    ├── lr_heatmaps/
    ├── tf_heatmaps/
    └── combined_stage_plots/
```
