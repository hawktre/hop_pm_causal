# Source Code Directory

This directory contains the Python scripts and Stan models used for the analysis.

## Files and Scripts

| File | Description | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| `00a_CleanData.py` | Cleans raw fungicide and incidence data, handles anonymization. | `../data/raw/cost/*.csv`, `../data/raw/*.csv` | `../data/processed/cost_data.csv` |
| `00b_FormatData.py` | Calculates distances and wind vectors, prepares `.npz` files for Stan. | `../data/processed/cost_data.csv` | `../data/processed/data_{year}.npz` |
| `01_MLE.py` | Executes the MLE model using `cmdstanpy`. | `../data/processed/data_{year}.npz`, `dispersal_mod.stan` | `../data/processed/results/mle_*.csv` |
| `02_compare_MLE.py` | Analyzes model results and generates diagnostic plots. | `../data/processed/results/*.csv` | Diagnostic plots in `../output/figures/` |
| `03_spatial_preds.py` | Creates spatial visualizations of model predictions. | `../data/processed/results/mle_preds.csv` | Spatial visualizations |
| `dispersal_mod.stan` | Stan model definition for dispersal. | - | Compiled Stan binary |
