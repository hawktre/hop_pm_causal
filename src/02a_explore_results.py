import   cmdstanpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

# Read in the raw data
data_by_year = np.load("data/processed/data_by_year.npz", allow_pickle=True)['arr_0'].item()
stacked = np.load("data/processed/stacked_data.npz", allow_pickle=True)['arr_0'].item()

def prepare_stan_inputs(analysis_month, data_by_year, stacked_data, years):
    month_map = {'may': 'apr', 'jun': 'may', 'jul': 'jun'}
    lag_month = month_map[analysis_month]
    N_max = int(max(data_by_year[y]['N'] for y in years))
    T = len(years)

    year_starts, year_ends, year_sizes = [], [], []
    current_idx = 1
    year_ids_list = []

    for idx, y in enumerate(years):
        N_yr = int(data_by_year[y]['N'])
        year_sizes.append(N_yr)
        year_starts.append(current_idx)
        year_ends.append(current_idx + N_yr - 1)
        current_idx += N_yr
        
        # Create 1-indexed year IDs for this year's block of observations
        year_ids_list.append(np.repeat(y, N_yr))

    # Combine into a single flat array matching N_total length
    year_id = np.concatenate(year_ids_list).astype(int).tolist()

    dist_mats = np.zeros((T, N_max, N_max))
    wind_mats = np.zeros((T, N_max, N_max))

    for i, y in enumerate(years):
        N_yr = int(data_by_year[y]['N'])
        dist_mats[i, :N_yr, :N_yr] = data_by_year[y]['distance']
        wind_mats[i, :N_yr, :N_yr] = data_by_year[y][f'wind_{lag_month}']

    factorized_ids, unique_ids = pd.factorize(stacked_data['field_id'].flatten())

    stan_data = {
        "T": T, "N_total": sum(year_sizes), "N_max": N_max,
        "N_yards": len(unique_ids),
        "yard_ids": (factorized_ids + 1).tolist(),
        "year_id": year_id, # Added vector tracking the year index for every observation row
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
        "J": len(np.unique(stacked_data['field_id'].flatten()))
    }
    return stan_data, unique_ids

years = [2014, 2015, 2016, 2017]
months = ['may', 'jun', 'jul']


# Read in the Stan fit results for the month of interest
month = "jul"
time = "2026_07_09_1904"
fit = cmdstanpy.from_csv(f"results/stan_fits/{month}/weakly_informative/{time}/")
# Extract the posterior draws for plotting
params = ['beta', 'delta', 'gamma', 'alpha', 'eta1', 'eta2', 'zi', 'phi']
# Force observed data to integer type
y_observed = stacked[f'y_{month}'].flatten().astype(int)

# Re-build your idata object using the raw year_id values
idata = az.from_cmdstanpy(
    posterior=fit,
    observed_data={"y_rep": y_observed},     # same name as posterior_predictive now
    posterior_predictive="y_rep",
    dims={"y_rep": ["obs"]},
    coords={"obs": stan_data['year_id']},
    dtypes={"y_rep": int},
)

# Generate trace plots for selected parameters
az.plot_trace(idata, var_names=params)
plt.suptitle(f"Trace Plots for Month: {month}", fontsize=16)
plt.savefig(f"output/figures/hmc_diagnostics/{month}_trace_plots.png", dpi=300, bbox_inches='tight')

# Generate autocorrelation plots for selected parameters
az.plot_autocorr(idata, var_names=params)
plt.suptitle(f"Autocorrelation Plots for Month: {month}", fontsize=16)
plt.savefig(f"output/figures/hmc_diagnostics/{month}_autocorr_plots.png", dpi=300, bbox_inches='tight')

# Generate Distribution Plots for selected parameters
az.plot_dist(idata, var_names=params)
plt.suptitle(f"Posterior Distributions for Month: {month}", fontsize=16)
plt.savefig(f"output/figures/hmc_diagnostics/{month}_posterior_distributions.png", dpi=300, bbox_inches='tight')

# Generate Posterior Predictive Checks (PPC) for the month of interest
az.plot_ppc_rootogram(idata)
plt.xlim([-1,50])
plt.title(f"Posterior Predictive Checks (Rootogram) for Month: {month}", fontsize=16)
plt.savefig(f"output/figures/hmc_diagnostics/{month}_ppc_rootogram.png", dpi=300, bbox_inches='tight')

#Generate a table summary of the posterior estimates for the parameters of interest
posterior_summary = az.summary(idata, var_names=params)
posterior_summary.to_csv(f"output/figures/hmc_diagnostics/{month}_posterior_summary.csv")
