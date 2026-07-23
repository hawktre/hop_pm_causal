import pandas as pd
import numpy as np
import cmdstanpy
import matplotlib.pyplot as plt
import os

#Import the data for the model 
stacked = np.load('data/processed/stacked_data_test.npz', allow_pickle=True)['arr_0'].item()
data_by_year = np.load('data/processed/data_by_year_test.npz', allow_pickle=True)['arr_0'].item()

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
        "T": T, "N_total": sum(year_sizes), 
        "N_max": N_max, 
        "N_yards": len(unique_ids),
        "yard_ids": (factorized_ids + 1).tolist(), 
        "year_starts": year_starts, 
        "year_ends": year_ends, 
        "year_sizes": year_sizes,
        "y": stacked_data[f'y_{analysis_month}'].flatten().astype(int).tolist(), 
        "n": stacked_data[f'n_{analysis_month}'].flatten().astype(int).tolist(),
        "y_lag": stacked_data[f'y_{lag_month}'].flatten(),
        "n_lag": stacked_data[f'n_{lag_month}'].flatten(),
        "s_lag": stacked_data[f's_{lag_month}'].flatten(),
        "sI1_lag": stacked_data[f'sI1_{lag_month}'].flatten(),
        "a_lag": stacked_data[f'a_{lag_month}'].flatten(),
        "cultivar": stacked_data['tI1'].flatten(),
        "dist_mats": dist_mats, "wind_mats": wind_mats,
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

#Prepare the data
stan_data = prepare_stan_inputs('may', data_by_year, stacked, years, prior_scenarios['weakly_informative'])

# Build the grid — span each parameter's plausible prior range
alpha_grid = np.linspace(0.1, 20, 30)
gamma_grid = np.linspace(0.0, 20, 30)

profile_data = dict(stan_data)  # copy existing data
profile_data["N_alpha_grid"] = len(alpha_grid)
profile_data["N_gamma_grid"] = len(gamma_grid)
profile_data["alpha_grid"] = alpha_grid.tolist()
profile_data["gamma_grid"] = gamma_grid.tolist()

#Compile the model 
profile_mod = cmdstanpy.CmdStanModel(stan_file='src/dispersal_mod_binomial_profile.stan')

#Run the model 
profile_fit = profile_mod.sample(profile_data)

# =====================================================================
# PLOTS: alpha/gamma profile log-likelihood diagnostic
# =====================================================================

output_dir = "results/profile_diagnostic/may"
os.makedirs(output_dir, exist_ok=True)

loglik_draws = profile_fit.stan_variable("loglik_grid")  # (n_draws, N_alpha_grid, N_gamma_grid)
loglik_grid_mean = loglik_draws.mean(axis=0)

A, G = np.meshgrid(alpha_grid, gamma_grid, indexing="ij")

# --- 1. Mean log-likelihood surface ---
fig, ax = plt.subplots(figsize=(7, 6))
cf = ax.contourf(A, G, loglik_grid_mean, levels=30, cmap="viridis")
fig.colorbar(cf, label="Mean log-likelihood")
ax.set_xlabel("alpha")
ax.set_ylabel("gamma")
ax.set_title("Profile log-likelihood surface (averaged over posterior draws)")
plt.tight_layout()
plt.savefig(f"{output_dir}/01_loglik_surface.png", dpi=200)
plt.close()

# --- 2. Delta-from-max contour ---
delta_loglik = loglik_grid_mean - loglik_grid_mean.max()

fig, ax = plt.subplots(figsize=(7, 6))
levels = [-16, -8, -4, -2, -1, -0.5, 0]
cf = ax.contourf(A, G, delta_loglik, levels=levels, cmap="viridis_r")
fig.colorbar(cf, label="Log-likelihood drop from max")
cs = ax.contour(A, G, delta_loglik, levels=[-4], colors="red", linewidths=2)
ax.clabel(cs, fmt={-4: "-4"})
ax.set_xlabel("alpha")
ax.set_ylabel("gamma")
ax.set_title("Log-likelihood drop from maximum\n(red line ~ approx. profile-likelihood region)")
plt.tight_layout()
plt.savefig(f"{output_dir}/02_delta_loglik.png", dpi=200)
plt.close()

# --- 3. Posterior draws (alpha, gamma) overlaid on the delta-loglik contour ---
alpha_draws = profile_fit.stan_variable("alpha")
gamma_draws = profile_fit.stan_variable("gamma")

fig, ax = plt.subplots(figsize=(7, 6))
cf = ax.contourf(A, G, delta_loglik, levels=levels, cmap="viridis_r")
fig.colorbar(cf, label="Log-likelihood drop from max")
ax.scatter(alpha_draws, gamma_draws, s=8, color="white", alpha=0.4,
           edgecolor="black", linewidth=0.2, label="Posterior draws")
ax.set_xlabel("alpha")
ax.set_ylabel("gamma")
ax.set_title("Posterior draws over the profile likelihood surface")
ax.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/03_posterior_overlay.png", dpi=200)
plt.close()

# --- 4. 1D profile slices ---
profile_alpha = loglik_grid_mean.max(axis=1)  # best gamma for each alpha
profile_gamma = loglik_grid_mean.max(axis=0)  # best alpha for each gamma

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(alpha_grid, profile_alpha - profile_alpha.max(), marker="o")
axes[0].set_xlabel("alpha")
axes[0].set_ylabel("Profile log-likelihood (relative)")
axes[0].set_title("Profile over alpha\n(max over gamma at each alpha)")
axes[0].axhline(-2, color="red", linestyle="--", linewidth=1)

axes[1].plot(gamma_grid, profile_gamma - profile_gamma.max(), marker="o")
axes[1].set_xlabel("gamma")
axes[1].set_ylabel("Profile log-likelihood (relative)")
axes[1].set_title("Profile over gamma\n(max over alpha at each gamma)")
axes[1].axhline(-2, color="red", linestyle="--", linewidth=1)

plt.tight_layout()
plt.savefig(f"{output_dir}/04_1d_profiles.png", dpi=200)
plt.close()

# --- 5. Draw-to-draw stability: small multiples of individual draws ---
n_show = min(6, loglik_draws.shape[0])
idx = np.linspace(0, loglik_draws.shape[0] - 1, n_show).astype(int)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, i in zip(axes.ravel(), idx):
    surf = loglik_draws[i] - loglik_draws[i].max()
    cf = ax.contourf(A, G, surf, levels=20, cmap="viridis_r")
    ax.set_title(f"Draw {i}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("gamma")
plt.tight_layout()
plt.savefig(f"{output_dir}/05_draw_stability.png", dpi=200)
plt.close()

print(f"Saved 5 diagnostic plots to {output_dir}/")