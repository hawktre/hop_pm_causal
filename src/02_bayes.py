import numpy as np
import pandas as pd
import cmdstanpy
import arviz as az

#Function to read in the data and separate it into objects stacked (across years) and by year
def load_and_preprocess_hop_data(years, data_path_template='data/processed/data_{year}.npz'):
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

    #Format for stan
    stan_data = {
        "T": T, "N_total": sum(year_sizes), "N_max": N_max,
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
        "field_id": pd.factorize(stacked_data['field_id'].flatten())[0] + 1,
        "J": len(np.unique(stacked_data['field_id'].flatten())),
        **prior_config 
    }
    return stan_data

# --- Specify Priors ---
prior_scenarios = {
    "towards_zero": {
        "beta_mu": 0, "beta_sigma": 10, #log-normal
        "delta_mu": 0, "delta_sigma": 10, #log-normal
        "gamma_mu": 0, "gamma_sigma": 10, #log-normal
        "alpha_mu": 1, "alpha_sigma": 10, #log-normal
        "eta1_mu": 0, "eta1_sigma": 10, #log-normal
        "eta2_mu": 0, "eta2_sigma": 10, #log-normal
    },
    "informed_priors": {
    "beta_mu": -2.5, "beta_sigma": 10, #log-normal
    "delta_mu": 1, "delta_sigma": 1, #log-normal
    "gamma_mu": 7.5, "gamma_sigma": 1, #log-normal
    "alpha_mu": 1, "alpha_sigma": 1, #log-normal
    "eta1_mu": 0, "eta1_sigma": 1, #log-normal
    "eta2_mu": 0, "eta2_sigma": 1, #log-normal
    }
}

years = [2014, 2015, 2016, 2017]
months = ['may', 'jun', 'jul']

#Compile the model and prep the data
stan_model = cmdstanpy.CmdStanModel(stan_file="src/dispersal_mod.stan")
data_by_year, stacked = load_and_preprocess_hop_data(years)
params = ['beta', 'delta', 'gamma', 'alpha', 'eta1', 'eta2']
test = stan_model.sample(prepare_stan_inputs('may', data_by_year, stacked, years, prior_scenarios['towards_zero']))
test.summary().iloc[1:10]


idata = az.from_cmdstanpy(posterior=test)

az.plot_dist(idata, var_names=params)
az.plot_trace(idata, var_names=params)
az.plot_pair(idata, var_names=params)
