# Source Code Directory

This directory contains the Python scripts and Stan models used for the analysis.

## Files and Scripts

| File | Description | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| `00a_CleanData.py` | Cleans raw fungicide and incidence data, handles anonymization. | `../data/raw/cost/*.csv`, `../data/raw/*.csv` | `../data/processed/cost_data.csv` |
| `00b_FormatData.py` | Calculates distances and wind vectors, prepares `.npz` files for Stan. | `../data/processed/cost_data.csv` | `../data/processed/data_{year}.npz` |
| `01_MLE.py` | Executes the MLE model using `cmdstanpy`. | `../data/processed/data_{year}.npz`, `dispersal_mod_*.stan` | `../results/mle/*.csv` |
| `01a_compare_mle.py` | Analyzes MLE model results and generates diagnostic plots. | `../results/mle/*.csv` | Diagnostic plots in `../output/figures/` |
| `01b_spatial_preds_mle.py` | Creates spatial visualizations of MLE model predictions. | `../results/mle/mle_preds.csv` | Spatial visualizations |
| `02_bayes.py` | Executes the Bayesian models using `cmdstanpy`. | `../data/processed/data_{year}.npz`, `dispersal_mod_*.stan` | `../results/stan_fits/` |
| `02a_explore_results.py` | Generates diagnostic plots (trace, autocorrelation) for Bayesian fits. | `../results/stan_fits/` | Diagnostics in `../output/figures/hmc_diagnostics/` |
| `02b_spatial_network_plots.py` | Visualizes the spatial transmission network. | `../results/stan_fits/` | Network plots in `../output/figures/` |
| `dispersal_mod_*.stan` | Stan model definitions (binomial, zero-inflated binomial, zero-inflated beta-binomial). | - | Compiled Stan binaries |
