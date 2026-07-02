# Data Directory

This directory contains all raw and processed data used in the analysis.

## Directory Structure

| Folder/File | Description | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| `raw/` | Anonymized raw field and cost data. | - | - |
| `raw/cost/` | Fungicide and pesticide price records. | - | - |
| `raw/economics/` | Economic data regarding yield and revenue. | - | - |
| `processed/` | Datasets cleaned and formatted for analysis. | `raw/` | `cost_data.csv`, `data_{year}.npz` |
| `processed/results/` | Output files from the MLE model. | `src/01_MLE.py` | `mle_results.csv`, `mle_preds.csv`, `mle_edges_long.csv` |
| `processed/results/edge_weights/` | Individual edge weight CSVs for different scenarios. | `src/01_MLE.py` | scenario-specific edge weight files |
