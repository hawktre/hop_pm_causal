import numpy as np
import pandas as pd
import cmdstanpy

# --- 1. DATA PRE-PROCESSING ---
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

    for year in years:
        path = data_path_template.format(year=year)
        try:
            raw_data = np.load(path)
            N_yr = int(raw_data['N'])
            processed = {'N': N_yr}
            for key in vector_keys:
                if key in raw_data:
                    processed[key] = raw_data[key].reshape(N_yr, 1)
            for key in matrix_keys:
                if key in raw_data:
                    processed[key] = raw_data[key]
            data_by_year[year] = processed
        except FileNotFoundError:
            continue

    stacked_data = {}
    for key in vector_keys:
        available_arrays = [data_by_year[y][key] for y in data_by_year if key in data_by_year[y]]
        if available_arrays:
            stacked_data[key] = np.vstack(available_arrays)

    return data_by_year, stacked_data

# --- 2. STAN INPUT PREPARATION ---
def prepare_stan_inputs(analysis_month, data_by_year, stacked_data, years, prior_config):
    month_map = {'may': 'apr', 'jun': 'may', 'jul': 'jun'}
    lag_month = month_map[analysis_month]
    N_max = int(max(data_by_year[y]['N'] for y in years))
    T = len(years)
    
    year_starts, year_ends, year_sizes = [], [], []
    current_idx = 1 
    
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

# --- 3. CONFIGURATIONS ---
prior_scenarios = {
    "No Priors": {
        "beta_mu": 0, "beta_sigma": 100, "delta_mu": 0, "delta_sigma": 10,
        "gamma_mu": 0, "gamma_sigma": 10, "alpha_mu": 1, "alpha_sigma": 10,
        "eta1_mu": 0, "eta1_sigma": 10, "eta2_mu": 0, "eta2_sigma": 10
    },
    "Vague Priors": {
        "beta_mu": 0, "beta_sigma": 5, "delta_mu": 0, "delta_sigma": 1.5,
        "gamma_mu": 0, "gamma_sigma": 1.0, "alpha_mu": 1, "alpha_sigma": 0.5,
        "eta1_mu": 0, "eta1_sigma": 1.0, "eta2_mu": 0, "eta2_sigma": 0.1 
    },
    "Strong Priors": {
        "beta_mu": 0, "beta_sigma": 2, "delta_mu": 0, "delta_sigma": 0.5,
        "gamma_mu": 0, "gamma_sigma": 0.5, "alpha_mu": 1, "alpha_sigma": 0.2,
        "eta1_mu": 0, "eta1_sigma": 0.2, "eta2_mu": 0, "eta2_sigma": 0.05 
    }
}

years = [2014, 2015, 2016, 2017]
months = ['may', 'jun', 'jul']
seeds = range(100, 110)
stan_model = cmdstanpy.CmdStanModel(stan_file="src/dispersal_mod.stan")
data_by_year, stacked = load_and_preprocess_hop_data(years)

all_tidy_rows = []
all_preds = {}
edge_weights = {}
# --- MAIN EXECUTION LOOP ---
for month in months:
    for scenario_name, config in prior_scenarios.items():
        print(f"--- Processing {month} | {scenario_name} ---")
        stan_data = prepare_stan_inputs(month, data_by_year, stacked, years, config)
        
        # We no longer track a single "best_lp" for the whole loop
        # Instead, we process every successful fit
        for s in seeds:
            try:
                current_fit = stan_model.optimize(data=stan_data, seed=s, jacobian=False)
                current_lp = current_fit.optimized_params_dict['lp__']
                
                # --- Laplace Approximation for each successful fit ---
                params = ['beta', 'delta', 'gamma', 'alpha', 'eta1', 'eta2']
                try:
                    fit_laplace = stan_model.laplace_sample(data=stan_data, mode=current_fit, jacobian=False, seed=124)
                    samples = fit_laplace.draws_pd()
                    
                    if not samples.empty:
                        std_errors = samples[params].std(axis=0).to_numpy()
                    else:
                        std_errors = np.full(len(params), np.nan)
                        
                except (RuntimeError, ValueError, Exception) as e:
                    print(f"Laplace failed for Seed {s} in {month}/{scenario_name}: {e}")
                    std_errors = np.full(len(params), np.nan)

                # --- Wrangle results for THIS specific seed ---
                mle_vals = current_fit.optimized_params_pd[params].to_numpy()[0]
                
                # Use a unique key for predictions/edges that includes the seed
                run_id = f"{month}_{scenario_name.lower().replace(' ', '_')}_seed{s}"
                
                all_preds[run_id] = current_fit.optimized_params_pd.filter(like="logit_p").to_numpy()[0]
                # Added deviance_resid to capture the update from the Stan code
                # deviance_resids[run_id] = current_fit.optimized_params_pd.filter(like="deviance_resid").to_numpy()[0]
                
                for i, p_name in enumerate(params):
                    all_tidy_rows.append({
                        'month': month,
                        'scenario': scenario_name,
                        'seed': s,
                        'parameter': p_name,
                        'estimate': mle_vals[i],
                        'std_error': std_errors[i],
                        'lp__': current_lp
                    })
                    
            except (RuntimeError, ValueError) as e:
                print(f"Seed {s} failed for {month} in {scenario_name}: {e}")
                continue

# --- 5. RESULTS ---
df_results = pd.DataFrame(all_tidy_rows)
df_results['lower_95'] = df_results['estimate'] - (1.96 * df_results['std_error']) 
df_results['upper_95'] = df_results['estimate'] + (1.96 * df_results['std_error'])
pd.set_option('display.max_rows', None)
print(df_results)

#Save out MLE estimates and Predicted Values
df_results.to_csv('data/processed/mle_results.csv', index=False)

prediction_list = []
for key, logits in all_preds.items():
    # Split the key: 'jul_strong_priors' -> ['jul', 'strong_priors']
    month, scenario = key.split('_', 1)
    
    # Retrieve the edge weights for this specific configuration
    # We use the same 'jul_strong_priors' key
    current_edge_weights = edge_weights.get(key, np.full(len(logits), np.nan))
    
    # Convert logit_p to probability scale
    probs = 1 / (1 + np.exp(-logits))
    
    # Loop through the array and give each value a row
    for yard_idx, pred in enumerate(probs):
        prediction_list.append({
            'field_id': stacked['field_id'][yard_idx][0],
            'year': stacked['year_vec'][yard_idx][0],
            'month': month,
            'scenario': scenario,
            'pred_prob': pred,
            'logit_p': logits[yard_idx],
            'edge_weight': current_edge_weights[yard_idx]
        })

# Create final DataFrame
df_preds = pd.DataFrame(prediction_list)

# Pivot so each scenario (No Priors, Vague, Strong) gets its own columns
df_wide = df_preds.pivot(
    index=['field_id', 'year', 'month'], 
    columns='scenario', 
    values=['pred_prob', 'logit_p', 'edge_weight']
)

# flatten the multi-index columns (e.g., 'pred_prob_strong_priors')
df_wide.columns = [f"{val}_{col.lower().replace(' ', '_')}" for val, col in df_wide.columns]

# Reset index to make field_id, year, and month regular columns again
df_wide = df_wide.reset_index()
df_wide.to_csv("data/processed/mle_preds.csv")
