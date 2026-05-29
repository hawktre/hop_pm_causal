import numpy as np
import pandas as pd
import cmdstanpy
import matplotlib.pyplot as plt
import seaborn as sns


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
    "towards_zero": {
        "beta_mu": 0, "beta_sigma": 10, #log-normal
        "delta_mu": 0, "delta_sigma": 1, #log-normal
        "gamma_mu": 0, "gamma_sigma": 1, #log-normal
        "alpha_mu": 1, "alpha_sigma": 1, #log-normal
        "eta1_mu": 0, "eta1_sigma": 1, #log-normal
        "eta2_mu": 0, "eta2_sigma": 1, #log-normal
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
seeds = range(100, 110)
stan_model = cmdstanpy.CmdStanModel(stan_file="src/dispersal_mod.stan")
data_by_year, stacked = load_and_preprocess_hop_data(years)

# --- MAIN EXECUTION LOOP ---
all_tidy_rows = []
prediction_list = []
edge_weight_list = []

for month in months:
    for scenario_name, config in prior_scenarios.items():
        print(f"--- Processing {month} | {scenario_name} ---")
        stan_data = prepare_stan_inputs(month, data_by_year, stacked, years, config)
        
        seed_results = []
        params = ['beta', 'delta', 'gamma', 'alpha', 'eta1', 'eta2']
        
        for s in seeds:
            try:
                current_fit = stan_model.optimize(data=stan_data, seed=s, jacobian=False)
                mle_vals = current_fit.optimized_params_pd[params].to_numpy()[0]
                lp = current_fit.optimized_params_dict['lp__']
                seed_results.append({
                    'seed': s,
                    'fit': current_fit,
                    'lp__': lp,
                    'params': mle_vals
                })
            except (RuntimeError, ValueError) as e:
                print(f"Seed {s} failed for {month} in {scenario_name}: {e}")

        if not seed_results:
            print(f"No successful fits for {month} | {scenario_name}")
            continue

        # Check parameter consistency across seeds
        all_param_vals = np.array([res['params'] for res in seed_results])
        param_means = np.mean(all_param_vals, axis=0)
        max_diffs = np.max(np.abs(all_param_vals - param_means), axis=0)
        print(f"Parameter consistency (max diff from mean): {dict(zip(params, np.round(max_diffs, 4)))}")

        # Pick the best fit based on log-likelihood (lp__)
        best_res = max(seed_results, key=lambda x: x['lp__'])
        best_fit = best_res['fit']
        best_seed = best_res['seed']
        best_lp = best_res['lp__']
        
        # Count how many seeds converged to the "same" maximum (within 0.1 lp units)
        convergence_tol = 0.1
        n_converged = sum(1 for res in seed_results if np.abs(res['lp__'] - best_lp) < convergence_tol)
        n_success = len(seed_results)
        
        print(f"Selected Seed {best_seed} with lp__ = {best_lp:.2f}")
        print(f"Convergence: {n_converged}/{n_success} seeds reached best lp__ (tol={convergence_tol})")

        # Laplace Approximation for standard errors (only for the best fit)
        try:
            fit_laplace = stan_model.laplace_sample(data=stan_data, mode=best_fit, jacobian=False, seed=124)
            samples = fit_laplace.draws_pd()
            if not samples.empty:
                std_errors = samples[params].std(axis=0).to_numpy()
            else:
                std_errors = np.full(len(params), np.nan)
        except Exception as e:
            print(f"Laplace failed for {month}/{scenario_name}: {e}")
            std_errors = np.full(len(params), np.nan)

        # Record tidy results for the best fit
        for i, p_name in enumerate(params):
            all_tidy_rows.append({
                'month': month,
                'scenario': scenario_name,
                'seed': best_seed,
                'parameter': p_name,
                'estimate': best_res['params'][i],
                'std_error': std_errors[i],
                'lp__': best_res['lp__'],
                'n_converged': n_converged,
                'n_success': n_success
            })

        # Extract fitted values, edge_weights, and deviance residuals from the best fit
        logit_p = best_fit.stan_variable('logit_p')
        edge_weights_padded = best_fit.stan_variable('edge_weights') # (T, N_max, N_max)
        deviance_resids = best_fit.stan_variable('deviance_resid')

        # Aggregate results by year and yard
        for t, year in enumerate(years):
            start = stan_data['year_starts'][t] - 1
            end = stan_data['year_ends'][t]
            N_yr = stan_data['year_sizes'][t]
            
            # Trim the padded zeros to get the actual N_yr x N_yr matrix for this year
            year_logit_p = logit_p[start:end]
            year_edges = edge_weights_padded[t][:N_yr, :N_yr]
            year_resids = deviance_resids[start:end]
            y_true = np.array(stan_data['y'])[start:end]
            n_true = np.array(stan_data['n'])[start:end]
            
            # Save the trimmed pairwise matrices directly as CSV
            scenario_slug = scenario_name.lower().replace(' ', '_')
            edge_csv_path = f"data/processed/results/edge_weights/edge_weights_{month}_{scenario_slug}_{year}.csv"
            np.savetxt(edge_csv_path, year_edges, delimiter=",")

            # Sum incoming edge weights (columns) for each target yard (rows)
            # This represents the total neighborhood pressure on each yard
            total_neighborhood_pressure = np.sum(year_edges, axis=1)
            probs = 1 / (1 + np.exp(-year_logit_p))
            
            for i in range(N_yr):
                yard_idx = start + i
                prediction_list.append({
                    'field_id': stacked['field_id'][yard_idx][0],
                    'year': year,
                    'month': month,
                    'scenario': scenario_name,
                    'y_true': y_true[i],
                    'n_true': n_true[i],
                    'true_prob': y_true[i] / n_true[i] if n_true[i] > 0 else 0,
                    'pred_prob': probs[i],
                    'logit_p': year_logit_p[i],
                    'deviance_resid': year_resids[i],
                    'edge_weight': total_neighborhood_pressure[i]
                })

                # Save the full trimmed pairwise matrices (long format)
                for j in range(N_yr):
                    edge_weight_list.append({
                        'month': month,
                        'scenario': scenario_name,
                        'year': year,
                        'target_id': stacked['field_id'][yard_idx][0],
                        'source_id': stacked['field_id'][start + j][0],
                        'weight': year_edges[i, j]
                    })

# --- 5. RESULTS ---
df_results = pd.DataFrame(all_tidy_rows)
df_results['lower_95'] = df_results['estimate'] - (1.96 * df_results['std_error']) 
df_results['upper_95'] = df_results['estimate'] + (1.96 * df_results['std_error'])
pd.set_option('display.max_rows', None)
print("\n--- MLE Parameter Results ---")
print(df_results)

# Save MLE results
df_results.to_csv('data/processed/results/mle_results.csv', index=False)

# Save Predictions (Pivoted by Scenario)
df_preds = pd.DataFrame(prediction_list)
df_preds.to_csv("data/processed/results/mle_preds.csv", index=False)

# Save Edge Weights (Full pairwise matrices)
df_edges = pd.DataFrame(edge_weight_list)
df_edges.to_csv("data/processed/results/mle_edges_long.csv", index=False)
print(f"\nSaved results to data/processed/")

# --- 6. DIAGNOSTIC PLOTS ---
print("\n--- Generating Diagnostic Plots ---")
sns.set_theme(style="whitegrid")

# 1. Observed vs Predicted probabilities
g = sns.FacetGrid(df_preds, col="scenario", row="month", height=4, aspect=1.2, 
                  sharex=True, sharey=True, margin_titles=True)

def add_identity_line(*args, **kwargs):
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.5)

g.map(add_identity_line)
g.map_dataframe(sns.scatterplot, x="pred_prob", y="true_prob", alpha=0.6, s=40)
g.set_axis_labels("Predicted Probability", "Observed Proportion (y/n)")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.tight_layout()
g.savefig('output/mle_diagnostic_fit.png')

# 2. Residuals vs Fitted
r = sns.FacetGrid(df_preds, col="scenario", row="month", height=4, aspect=1.2, 
                  sharex=True, sharey=False, margin_titles=True)

def add_zero_line(*args, **kwargs):
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)

r.map(add_zero_line)
r.map_dataframe(sns.scatterplot, x="pred_prob", y="deviance_resid", alpha=0.6, s=40)
r.set_axis_labels("Predicted Probability", "Deviance Residual")
r.set_titles(col_template="{col_name}", row_template="{row_name}")
r.tight_layout()
r.savefig('output/mle_diagnostic_residuals.png')

print(f"Diagnostic plots saved to output/ folder")
plt.close('all')

