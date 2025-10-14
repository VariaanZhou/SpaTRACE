# GREATEST_Chat

This repository provides a **two-step pipeline** for preparing temporal ligand–receptor / transcription factor / target features from single-cell data and training a transformer model (**GRAEST_Chat**) on those features.

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
