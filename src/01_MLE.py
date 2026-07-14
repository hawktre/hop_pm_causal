import os
import numpy as np
import pandas as pd
import cmdstanpy
import glob

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

years = [2014, 2015, 2016, 2017]
months = ['may', 'jun', 'jul']
seeds = range(100, 110)

data_by_year, stacked = load_and_preprocess_hop_data(years)

models = {
    'binomial': 'src/dispersal_mod_binomial.stan',
    'zero_inflated_binomial': 'src/dispersal_mod_zero_inflated_binomial.stan',
    'zero_inflated_beta_binomial': 'src/dispersal_mod_zero_inflated_beta_binomial.stan'
}

# --- Fit the models ---
for model_name, stan_path in models.items():
    print(f"\n=========================================")
    print(f"Running Optimization for Model: {model_name}")
    print("=========================================")

    # Create output directories if they do not exist
    os.makedirs(f"results/mle/{model_name}/edge_weights", exist_ok=True)

    stan_model = cmdstanpy.CmdStanModel(stan_file=stan_path)

    # Compile a temporary model without the generated quantities block for Laplace standard errors
    dir_name, file_name = os.path.split(stan_path)
    temp_stan_path = os.path.join(dir_name, f"temp_laplace_{file_name}")
    
    with open(stan_path, 'r') as f:
        lines = f.readlines()
        
    clean_lines = []
    for line in lines:
        if 'generated quantities' in line.lower() and '{' in line:
            break
        clean_lines.append(line)
        
    with open(temp_stan_path, 'w') as f:
        f.writelines(clean_lines)
        
    laplace_model = cmdstanpy.CmdStanModel(stan_file=temp_stan_path)

    all_tidy_rows = []
    prediction_list = []
    edge_weight_list = []

    for month in months:
        for scenario_name, config in prior_scenarios.items():
            print(f"\n--- Processing {model_name} | {month} | {scenario_name} ---")
            stan_data = prepare_stan_inputs(month, data_by_year, stacked, years, config)
            
            # Determine parameters based on model type
            if 'zero_inflated_beta_binomial' in model_name:
                params = ['beta', 'delta', 'gamma', 'alpha', 'eta1', 'eta2', 'pi', 'phi']
            elif 'zero_inflated_binomial' in model_name:
                params = ['beta', 'delta', 'gamma', 'alpha', 'eta1', 'eta2', 'pi']
            else:  # binomial
                params = ['beta', 'delta', 'gamma', 'alpha', 'eta1', 'eta2']

            seed_results = []
            
            # Fit the model for multiple inits and see if they converge to the same thing
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
                    print(f"Seed {s} failed for {model_name}/{month} in {scenario_name}: {e}")

            if not seed_results:
                print(f"No successful fits for {model_name} | {month} | {scenario_name}")
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
                # Run optimization on the Laplace-only model first to get a compatible mode
                laplace_best_fit = laplace_model.optimize(data=stan_data, seed=best_seed, jacobian=False)
                fit_laplace = laplace_model.laplace_sample(data=stan_data, mode=laplace_best_fit, jacobian=False, seed=124)
                samples = fit_laplace.draws_pd()
                if not samples.empty:
                    std_errors = samples[params].std(axis=0).to_numpy()
                else:
                    std_errors = np.full(len(params), np.nan)
            except Exception as e:
                print(f"Laplace failed for {model_name}/{month}/{scenario_name}: {e}")
                std_errors = np.full(len(params), np.nan)

            # Record results for the best fit
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

            # Extract fitted values, edge_weights from the best fit
            logit_p = best_fit.stan_variable('logit_p')
            edge_weights_padded = best_fit.stan_variable('edge_weights') # (T, N_max, N_max)

            # Compute deviance residuals locally
            y_observed = np.array(stan_data['y'])
            n_observed = np.array(stan_data['n'])
            
            p = 1.0 / (1.0 + np.exp(-logit_p))
            p = np.clip(p, 1e-9, 1.0 - 1e-9)
            y_hat = n_observed * p
            d2 = np.zeros_like(y_observed, dtype=float)
            
            idx1 = y_observed > 0
            d2[idx1] += 2 * y_observed[idx1] * np.log(y_observed[idx1] / y_hat[idx1])
            
            idx2 = (n_observed - y_observed) > 0
            d2[idx2] += 2 * (n_observed - y_observed)[idx2] * np.log((n_observed - y_observed)[idx2] / (n_observed - y_hat)[idx2])
            
            deviance_resids = np.where(y_observed > y_hat, 1, -1) * np.sqrt(d2)

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
                edge_csv_path = f"results/mle/{model_name}/edge_weights/edge_weights_{month}_{scenario_slug}_{year}.csv"
                np.savetxt(edge_csv_path, year_edges, delimiter=",")

                # Sum incoming edge weights (columns) for each target yard (rows)
                total_neighborhood_pressure = np.sum(year_edges, axis=1)
                probs = 1 / (1 + np.exp(-year_logit_p))
                
                for i in range(N_yr):
                    yard_idx = start + i
                    prediction_list.append({
                        'field_id': stacked['field_id'][yard_idx][0],
                        'cultivar': stacked['tI1'][yard_idx][0],
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

    # Compute Confidence Intervals and Save Results
    df_results = pd.DataFrame(all_tidy_rows)
    df_results['lower_95'] = df_results['estimate'] - (1.96 * df_results['std_error']) 
    df_results['upper_95'] = df_results['estimate'] + (1.96 * df_results['std_error'])
    
    # Save MLE results
    df_results.to_csv(f'results/mle/{model_name}/mle_results.csv', index=False)

    # Save Predictions 
    df_preds = pd.DataFrame(prediction_list)
    df_preds.to_csv(f"results/mle/{model_name}/mle_preds.csv", index=False)

    # Save Edge Weights 
    df_edges = pd.DataFrame(edge_weight_list)
    df_edges.to_csv(f"results/mle/{model_name}/mle_edges_long.csv", index=False)
    print(f"\nFinished saving results to results/mle/{model_name}/")

    # Cleanup temp files for this model
    base_temp_name = f"temp_laplace_{os.path.splitext(file_name)[0]}"
    for f in glob.glob(os.path.join(dir_name, f"{base_temp_name}*")):
        try:
            os.remove(f)
        except Exception:
            pass
