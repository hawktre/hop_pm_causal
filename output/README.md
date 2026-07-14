# Output Directory

This directory contains Quarto documents for reporting and figures generated during the analysis.

## Directory Structure

| Folder/File | Description | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| `figures/` | Diagnostic and results plots. | `src/01a_compare_mle.py`, `src/02a_explore_results.py` | PNG images |
| `model_results.qmd` | Quarto document summarizing model findings. | `results/mle/*.csv`, `results/stan_fits/**/*.csv` | Rendered report (HTML/PDF) |
| `modeling_considerations.qmd` | Quarto document discussing modeling choices. | - | Rendered report (HTML/PDF) |
| `references.bib` | BibTeX references for the reports. | - | - |
