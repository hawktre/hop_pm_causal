# Causal inference for spillover effects in the presence of network interference
## A case study of Hop Powdery Mildew in Oregon (2014 - 2017)

This project investigates the causal spillover effects of fungicide applications on the spread of Hop Powdery Mildew across a network of hop yards in Oregon. It uses Maximum Likelihood Estimation (MLE) and Stan-based modeling to estimate dispersal and the impact of interference between yards.

## Reproduction Instructions

To reproduce the analysis, follow these steps:

### 1. Set up a Python Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies
Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Install CmdStan
This project uses `cmdstanpy`, which requires a working installation of CmdStan. You can install it by running:

```bash
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```
In addition, please check their [installation instructions](https://mc-stan.org/cmdstanpy/installation.html#function-install-cmdstan).

### 4. Run the Analysis Scripts
Execute the scripts in the `src/` directory in the following order:

1.  **Clean Data:** `python src/00a_CleanData.py`
2.  **Format Data:** `python src/00b_FormatData.py`
3.  **Run MLE Model:** `python src/01_MLE.py`
4.  **Compare & Diagnostics:** `python src/02_compare_MLE.py`
5.  **Spatial Visualization:** `python src/03_spatial_preds.py`

Data cleaning code originally come from Josh Pedro's [2025 Hops Project Github](https://github.com/joshfpedro/hops-project-2025/tree/master/notebooks)

## Project Structure

| Folder/File | Description | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| `data/raw/` | Anonymized raw field and cost data. | - | Raw CSV files |
| `data/processed/` | Datasets cleaned and formatted for analysis. | - | `cost_data.csv`, `data_{year}.npz` |
| `data/processed/results/` | Output files from the MLE model. | - | `mle_results.csv`, `mle_preds.csv`, `mle_edges_long.csv` |
| `src/` | Python and Stan source code. | - | - |
| `src/00a_CleanData.py` | Cleans raw fungicide and incidence data, handles anonymization. | `data/raw/cost/*.csv`, `data/raw/*.csv` | `data/processed/cost_data.csv` |
| `src/00b_FormatData.py` | Calculates distances and wind vectors, prepares `.npz` files for Stan. | `data/processed/cost_data.csv` | `data/processed/data_{year}.npz` |
| `src/01_MLE.py` | Executes the MLE model using `cmdstanpy`. | `data/processed/data_{year}.npz`, `src/dispersal_mod.stan` | `data/processed/results/mle_*.csv` |
| `src/02_compare_MLE.py` | Analyzes model results and generates diagnostic plots. | `data/processed/results/*.csv` | Diagnostic plots in `output/figures/` |
| `src/03_spatial_preds.py` | Creates spatial visualizations of model predictions. | `data/processed/results/mle_preds.csv` | Spatial visualizations |
| `src/dispersal_mod.stan` | Stan model definition for dispersal. | - | - |
| `output/` | Quarto documents and generated figures for reporting. | - | `.qmd` files, PNG figures |
| `requirements.txt` | Python dependencies. | - | - |
| `renv.lock` / `renv/` | R environment configuration for Quarto reporting. | - | - |
| `.venv/` | Python virtual environment (ignored by git). | - | - |
