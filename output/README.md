# Output Directory

This directory contains Quarto documents for reporting and figures generated during the analysis.

## Directory Structure

| Folder/File | Description | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| `figures/` | Diagnostic and results plots. | `src/02_compare_MLE.py` | PNG images |
| `model_results.qmd` | Quarto document summarizing model findings. | `data/processed/results/*.csv` | Rendered report (HTML/PDF) |
| `modeling_considerations.qmd` | Quarto document discussing modeling choices. | - | Rendered report (HTML/PDF) |
| `references.bib` | BibTeX references for the reports. | - | - |
