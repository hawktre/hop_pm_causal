# Data Directory

This directory contains all raw and processed data used in the analysis.

## Directory Structure

| Folder/File | Description | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| `raw/` | Anonymized raw field and cost data. | - | - |
| `raw/cost/` | Fungicide and pesticide price records. | - | - |
| `raw/economics/` | Economic data regarding yield and revenue. | - | - |
| `processed/` | Datasets cleaned and formatted for analysis. | `raw/` | `cost_data.csv`, `data_{year}.npz` |
