import numpy as np
import pandas as pd
import cmdstanpy
import os



#Function to read in the data and separate it into objects stacked (across years) and by year
def load_and_preprocess_hop_data(years, data_path_template='data/processed/data_{year}_test.npz'):
    data_by_year = {}
    vector_keys = [
        'field_id', 'year_vec','tI1', 'y_apr', 'y_may', 'y_jun', 'y_jul',
        'n_apr', 'n_may', 'n_jun', 'n_jul',
        'a_apr', 'a_may', 'a_jun', 'a_jul',
        'sI1_apr', 'sI1_may', 'sI1_jun', 'sI1_jul',
        's_apr', 's_may', 's_jun', 's_jul'
    ]
    matrix_keys = ['distance', 'wind_apr', 'wind_may', 'wind_jun', 'wind_jul']

    #For each year
    for year in years:
        path = data_path_template.format(year=year)
        try:
            #Read in the data
            raw_data = np.load(path)
            #Number of observations in current year
            N_yr = int(raw_data['N'])
            #Start a dictionary
            processed = {'N': N_yr}
            #Assign the relevant data to that year
            for key in vector_keys:
                if key in raw_data:
                    processed[key] = raw_data[key].reshape(N_yr, 1)
            for key in matrix_keys:
                if key in raw_data:
                    processed[key] = raw_data[key]
            #Assign the data for that year
            data_by_year[year] = processed
        except FileNotFoundError:
            continue
    
    #Stack the data so that we have all the years stacked for each month (for the analysis)
    stacked_data = {}
    for key in vector_keys:
        available_arrays = [data_by_year[y][key] for y in data_by_year if key in data_by_year[y]]
        if available_arrays:
            stacked_data[key] = np.vstack(available_arrays)

    return data_by_year, stacked_data

# Prepare the data for how stan expects it
def prepare_stan_inputs(analysis_month, data_by_year, stacked_data, years, prior_config):
    #Define the transitions
    month_map = {'may': 'apr', 'jun': 'may', 'jul': 'jun'}
    lag_month = month_map[analysis_month]
    #Get the size of the largest year so that we can pad the data with zeros
    N_max = int(max(data_by_year[y]['N'] for y in years))
    #Total number of years
    T = len(years)
    
    #Initialize lists to track the indicies in teh stacked data
    year_starts, year_ends, year_sizes = [], [], []
    current_idx = 1 
    
    #Get the indices and format distance and wind data into arrays (1 slice for each year)
    for y in years:
        N_yr = int(data_by_year[y]['N'])
        year_sizes.append(N_yr)
        year_starts.append(current_idx)
        year_ends.append(current_idx + N_yr - 1)
        current_idx += N_yr

    dist_mats = np.zeros((T, N_max, N_max)) 
    wind_mats = np.zeros((T, N_max, N_max))

    for i, y in enumerate(years):
        N_yr = int(data_by_year[y]['N'])
        dist_mats[i, :N_yr, :N_yr] = data_by_year[y]['distance']
        wind_mats[i, :N_yr, :N_yr] = data_by_year[y][f'wind_{lag_month}']

    #get factorized ids
    factorized_ids, unique_ids = pd.factorize(stacked_data['field_id'].flatten())

    #Format for stan
    stan_data = {
        "T": T, "N_total": sum(year_sizes), "N_max": N_max, 
        "N_yards": len(unique_ids),
        "yard_ids": (factorized_ids + 1).tolist(), 
        "year_starts": year_starts, "year_ends": year_ends, "year_sizes": year_sizes,
        "y": stacked_data[f'y_{analysis_month}'].flatten().astype(int).tolist(), 
        "n": stacked_data[f'n_{analysis_month}'].flatten().astype(int).tolist(),
        "y_lag": stacked_data[f'y_{lag_month}'].flatten(),
        "n_lag": stacked_data[f'n_{lag_month}'].flatten(),
        "s_lag": stacked_data[f's_{lag_month}'].flatten(),
        "sI1_lag": stacked_data[f'sI1_{lag_month}'].flatten(),
        "a_lag": stacked_data[f'a_{lag_month}'].flatten(),
        "cultivar": stacked_data['tI1'].flatten(),
        "dist_mats": dist_mats, "wind_mats": wind_mats,
        "J": len(np.unique(stacked_data['field_id'].flatten())),
        **prior_config 
    }
    return stan_data

# --- Specify Priors ---
prior_scenarios = {
    "weakly_informative": {
        # Global baseline intercept (Unconstrained linear regressor)
        "beta_mu": 0,       "beta_sigma": 2.5, 
        
        # Auto-infection & Dispersal (Clamped to prevent logit overflow)
        "delta_mu": 0,      "delta_sigma": 2.5, 
        "gamma_mu": 0,      "gamma_sigma": 2.5, 
        
        # Distance decay (Anchored tightly around 1.0)
        "alpha_mu": 0,      "alpha_sigma": 2.5, 
        
        # Spray decays (Moderately regularized)
        "eta1_mu": 0,       "eta1_sigma": 2.5, 
        "eta2_mu": 0,       "eta2_sigma": 2.5,
        "phi_mu": 0,        "phi_sigma": 2.5
    }
}

# Set up years and time transitions
years = [2014, 2015, 2016, 2017]
months = ['may', 'jun', 'jul']




#Select/Compile the model (one of "binomial", "zero_inflated_binomial", "zero_inflated_beta_binomial") and prep the data
models = ["binomial", "zero_inflated_binomial", "zero_inflated_beta_binomial"]
data_by_year, stacked = load_and_preprocess_hop_data(years)

#Covert the stacked data to a dataframe for ease
stacked_flat = {k: np.asarray(v).ravel() for k, v in stacked.items()}
stacked_df = pd.DataFrame(stacked_flat)


#Save the formatted data
stacked_df.to_csv("data/processed/stacked_data.csv")
np.savez("data/processed/stacked_data_test.npz", stacked)
np.savez("data/processed/data_by_year_test.npz", data_by_year)

# Prepare an empty list to serve as our lightweight index container
results_list = []

for model in models:
    #Compile the model
    mod_path = str("src/dispersal_mod_") + str(model) + str(".stan")
    stan_model = cmdstanpy.CmdStanModel(stan_file= mod_path)
    for month in months:
        for scenario, priors in prior_scenarios.items():
            print("\n=========================================")
            print(f"Running: Model = {model}, Month = {month}, Prior = {scenario}")
            print("=========================================")
            
            # Pull and compile the stan data
            stan_data = prepare_stan_inputs(month, data_by_year, stacked, years, priors)

            # Build a highly descriptive unique prefix for this file combination
            run_output_dir = os.path.join("results", "stan_fits", model, month)
            os.makedirs(run_output_dir, exist_ok=True)

            try:
                # Fit the model
                fit = stan_model.sample(
                    data=stan_data,
                    output_dir=run_output_dir,
                    time_fmt="%Y%m%d",
                    show_progress=True
                )
                
                # Quick diagnostic checks
                diagnose_output = fit.diagnose()
                has_divergences = "No divergent transitions" in diagnose_output
                
                # Capture the file paths to the permanent output CSVs
                csv_files = fit.runset.csv_files
                
                # Append metadata and file references to our container
                results_list.append({
                    "month": month,
                    "prior_scenario": scenario,
                    "csv_paths": ",".join(csv_files),  # Stored as a comma-separated string
                    "divergences": not has_divergences,
                    "status": "success"
                })
                
            except Exception as e:
                print(f"!!! Optimization or Sampling Failed for {month} ({scenario}): {e}")
                results_list.append({
                    "month": month,
                    "prior_scenario": scenario,
                    "csv_paths": None,
                    "divergences": None,
                    "status": f"failed: {str(e)}"
                })
                
            # CRITICAL: Drop the fit object and clear RAM before the next iteration
            if 'fit' in locals():
                del fit

# Save your master index mapping metadata to file locations
results_list_df = pd.DataFrame(results_list)
print(results_list_df)
print("\nAll loops completed!")
