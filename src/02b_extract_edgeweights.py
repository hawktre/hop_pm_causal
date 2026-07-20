import os
import cmdstanpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import glob

# Read in the raw data
stacked = np.load("data/processed/stacked_data.npz", allow_pickle=True)['arr_0'].item()

# Set up results direcrtory to read from
results_dir = "results/stan_fits"

if os.path.exists(results_dir):
    # Find all models (directories under results/stan_fits/)
    models = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    for model in models:
        model_path = os.path.join(results_dir, model)
        months = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        
        for month in months:
            fit_dir = os.path.join(model_path, month)
            csv_files = glob.glob(os.path.join(fit_dir, "*.csv"))
            
            # Skip if there are no CSV files (fit outputs)
            if not csv_files:
                continue
                
            print(f"Extracting Edgewieghts for Model: {model}, Month: {month}...")
            
            try:
                # Read in the Stan fit results
                fit = cmdstanpy.from_csv(fit_dir)
                
                #Extract edge weights from the fit (n_draws x n_yards x n_yards)
                edge_weights = fit.stan_variable("edge_weights")

                #Compute in-centrality draws (sum over the columns) and format as data frame to write to csv
                in_centrality = np.sum(edge_weights, axis=1).transpose()
                in_centrality_df = pd.DataFrame(in_centrality)
                in_centrality_df.columns = [f'draw_{d}' for d in range(1, in_centrality.shape[1] + 1)]
                in_centrality_df.insert(0, 'year', stacked['year_vec'])
                in_centrality_df.insert(0, 'field_id', stacked['field_id'])
                      
                # Setup output directory for diagnostics
                output_dir = f"results/degree_centrality/{model}/"
                os.makedirs(output_dir, exist_ok=True)
                
                output_name = f"in_centrality_{month}.csv"
                write_path = output_dir + output_name
                in_centrality_df.to_csv(write_path, index = False)
                del(fit)

            except Exception as e:
                print(f"Error processing {model}/{month}: {e}")
                plt.close('all')
else:
    print(f"Results directory not found at: {results_dir}")


