import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Import my MLEs
stan_mles = pd.read_csv("data/processed/results/mle_results.csv")
stan_preds = pd.read_csv("data/processed/results/mle_preds.csv")

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
def prepare_stan_inputs(analysis_month, data_by_year, stacked_data, years):
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
        "J": len(np.unique(stacked_data['field_id'].flatten()))
    }
    return stan_data


# Run the code
years = [2014, 2015, 2016, 2017]
months = ['may', 'jun', 'jul']
data_by_year, stacked = load_and_preprocess_hop_data(years)

# Hard code seperate MLEs
seperate_mle = {
    "may": {
        "r6": {
            "beta_1": -7.29,
            "delta_1": 75.27,
            "gamma_1": 0.002,
            "alpha_1": 24.23,
            "eta_11": 0.00001,
            "eta_12": 2.25,
        },
        "non_r6": {
            "beta_2": -6.35,
            "delta_2": 192.26,
            "gamma_2": 12.40,
            "alpha_2": 14.13,
            "eta_21": 0.00001,
            "eta_22": 3.62,
        },
    },
    "jun": {
        "r6": {
            "beta_1": -2.06,
            "delta_1": 2074.35,
            "gamma_1": 34358.91,
            "alpha_1": 0.71,
            "eta_11": 3.51,
            "eta_12": 1.62,
        },
        "non_r6": {
            "beta_2": -4.15,
            "delta_2": 120.46,
            "gamma_2": 14536.03,
            "alpha_2": 2.67,
            "eta_21": 2.9,
            "eta_22": 0.27,
        },
    },
    "jul": {
        "r6": {
            "beta_1": -2.79,
            "delta_1": 2.94,
            "gamma_1": 1390.71,
            "alpha_1": 1.0,
            "eta_11": 0.02,
            "eta_12": 0.31,
        },
        "non_r6": {
            "beta_2": -3.88,
            "delta_2": 8.04,
            "gamma_2": 286.8,
            "alpha_2": 2.04,
            "eta_21": 0.95,
            "eta_22": 0.37,
        },
    },
}

def compute_predictions_by_year(stan_data, t, month_key, seperate_mle):
    """
    Computes logit_p and the edge weight matrix for a specific year and month
    using the dual-cultivar parameter dictionary structure.
    """
    # Slice year-specific indices (Stan is 1-indexed, Python is 0-indexed)
    start = stan_data['year_starts'][t] - 1
    end = stan_data['year_ends'][t]
    N_yr = stan_data['year_sizes'][t]
    
    # Extract year-specific arrays
    y_lag = stan_data['y_lag'][start:end]
    n_lag = stan_data['n_lag'][start:end]
    s_lag = stan_data['s_lag'][start:end]
    a_lag = stan_data['a_lag'][start:end]
    cultivar = stan_data['cultivar'][start:end]
    
    # Compute source strength matching Stan: a_lag * (y_lag / n_lag)
    source_strength = a_lag * (y_lag/n_lag)
    
    # Slice localized matrices out of padded data structures
    dist_local = stan_data['dist_mats'][t][:N_yr, :N_yr]
    wind_local = stan_data['wind_mats'][t][:N_yr, :N_yr]
    
    # Initialize parameter vectors for each yard in this year slice
    beta = np.zeros(N_yr)
    delta = np.zeros(N_yr)
    gamma = np.zeros(N_yr)
    alpha = np.zeros(N_yr)
    eta1 = np.zeros(N_yr)
    eta2 = np.zeros(N_yr)
    
    # Vectorize parameter selection using cultivar mappings
    p_r6 = seperate_mle[month_key]['r6']
    p_non = seperate_mle[month_key]['non_r6']
    is_r6 = (cultivar == 1)
    
    beta = np.where(is_r6, p_r6['beta_1'], p_non['beta_2'])
    delta = np.where(is_r6, p_r6['delta_1'], p_non['delta_2'])
    gamma = np.where(is_r6, p_r6['gamma_1'], p_non['gamma_2'])
    alpha = np.where(is_r6, p_r6['alpha_1'], p_non['alpha_2'])
    eta1 = np.where(is_r6, p_r6['eta_11'], p_non['eta_21'])
    eta2 = np.where(is_r6, p_r6['eta_12'], p_non['eta_22'])
    
    # Build pairwise network dispersal matrix (N_yr x N_yr)
    # matching calc_dispersal_mat from Stan
    kernel = np.zeros((N_yr, N_yr))
    for i in range(N_yr):
        for j in range(N_yr):
            if i == j:
                kernel[i, j] = 0.0
            else:
                # Dispersal decay parameter alpha depends on focal yard i
                kernel[i, j] = pow(1.0 + dist_local[i, j], -alpha[i])
                
    # Scale source strength by source yard specific spray history decay (eta2)
    effective_source = source_strength * np.exp(-eta2 * s_lag)
    effective_source_mat = np.tile(effective_source, (N_yr, 1)) # Replicate for elements
    
    # Combine wind, spatial power-law kernel, and effective source volume
    dispersal_matrix = wind_local * kernel * effective_source_mat
    
    # Final edge weight matrix scaled by global spatial magnitude (gamma) for focal yard i
    edge_weights_yr = gamma[:, np.newaxis] * dispersal_matrix
    
    # Sum across columns to find total pressure vector entering target yard rows
    disp_vec = np.sum(dispersal_matrix, axis=1)
    
    # Linear predictor mapping transformed parameters block
    logit_p_yr = beta + (delta * (y_lag/n_lag) * np.exp(-eta1 * s_lag)) + (gamma * disp_vec)
    
    return logit_p_yr, edge_weights_yr


def compute_deviance_residuals(y, n, logit_p):
    """Calculates binomial deviance residuals matching Stan's generated quantities."""
    p = 1.0 / (1.0 + np.exp(-logit_p))
    y_hat = n * p
    
    d2 = np.zeros_like(y_hat)
    
    # Term 1: y > 0
    idx1 = y > 0
    d2[idx1] += 2.0 * y[idx1] * np.log(y[idx1] / y_hat[idx1])
    
    # Term 2: n - y > 0
    idx2 = (n - y) > 0
    d2[idx2] += 2.0 * (n[idx2] - y[idx2]) * np.log((n[idx2] - y[idx2]) / (n[idx2] - y_hat[idx2]))
    
    sign = np.where(y > y_hat, 1.0, -1.0)
    return sign * np.sqrt(d2)

# --- EXECUTION BLOCK FOR NEW MLE PREDICTIONS ---
seperate_prediction_list = []
seperate_edge_weight_list = []

print("\n--- Generating Predictions for Cultivar-Specific MLEs ---")

for month in months:
    print(f"Processing evaluation metrics for: {month}")
    # Prior config does not alter values since we bypass optimization, passing 'towards_zero' as placeholder
    stan_data = prepare_stan_inputs(month, data_by_year, stacked, years)
    
    for t, year in enumerate(years):
        start = stan_data['year_starts'][t] - 1
        end = stan_data['year_ends'][t]
        N_yr = stan_data['year_sizes'][t]
        
        # Calculate logit scale probabilities and structural network edge weights
        year_logit_p, year_edges = compute_predictions_by_year(stan_data, t, month, seperate_mle)
        
        y_true = np.array(stan_data['y'])[start:end]
        n_true = np.array(stan_data['n'])[start:end]
        year_resids = compute_deviance_residuals(y_true, n_true, year_logit_p)
        
        # Save structural network matrix files
        edge_csv_path = f"data/processed/results/edge_weights/edge_weights_{month}_seperate_mle_{year}.csv"
        np.savetxt(edge_csv_path, year_edges, delimiter=",")
        
        # Calculate inbound row sum network pressure
        total_neighborhood_pressure = np.sum(year_edges, axis=1)
        probs = 1.0 / (1.0 + np.exp(-year_logit_p))
        
        for i in range(N_yr):
            yard_idx = start + i
            seperate_prediction_list.append({
                'field_id': stacked['field_id'][yard_idx][0],
                'cultivar': stacked['tI1'][yard_idx][0],
                'year': year,
                'month': month,
                'scenario': 'seperate_mle',
                'y_true': y_true[i],
                'n_true': n_true[i],
                'true_prob': y_true[i] / n_true[i] if n_true[i] > 0 else 0,
                'pred_prob': probs[i],
                'logit_p': year_logit_p[i],
                'deviance_resid': year_resids[i],
                'edge_weight': total_neighborhood_pressure[i]
            })
            
            # Unpack full edge weights into flat network list
            for j in range(N_yr):
                seperate_edge_weight_list.append({
                    'month': month,
                    'scenario': 'seperate_mle',
                    'year': year,
                    'target_id': stacked['field_id'][yard_idx][0],
                    'source_id': stacked['field_id'][start + j][0],
                    'weight': year_edges[i, j]
                })

# Convert data arrays to dataframes and output results
seperate_preds = pd.DataFrame(seperate_prediction_list)
seperate_edges = pd.DataFrame(seperate_edge_weight_list)

seperate_preds.to_csv("data/processed/results/seperate_mle_preds.csv", index=False)
seperate_edges.to_csv("data/processed/results/seperate_mle_edges_long.csv", index=False)
print("Cultivar-specific evaluation datasets written successfully!")
# --- 6. DIAGNOSTIC PLOTS ---
all_preds = pd.concat([stan_preds, seperate_preds], ignore_index=True)
print("\n--- Generating Diagnostic Plots ---")
sns.set_theme(style="whitegrid")

# 1. Observed vs Predicted probabilities
g = sns.FacetGrid(all_preds, col="scenario", row="month", height=4, aspect=1.2, 
                  sharex=True, sharey=True, margin_titles=True)

def add_identity_line(*args, **kwargs):
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.5)

g.map(add_identity_line)
g.map_dataframe(sns.scatterplot, x="pred_prob", y="true_prob", alpha=0.6, s=40, hue = "cultivar")
g.set_axis_labels("Predicted Probability", "Observed Proportion (y/n)")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.tight_layout()
g.savefig('output/mle_diagnostic_fit.png')

# 2. Residuals vs Fitted
r = sns.FacetGrid(all_preds, col="scenario", row="month", height=4, aspect=1.2, 
                  sharex=True, sharey=False, margin_titles=True)

def add_zero_line(*args, **kwargs):
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)

r.map(add_zero_line)
r.map_dataframe(sns.scatterplot, x="pred_prob", y="deviance_resid", alpha=0.6, s=40, hue = "cultivar")
r.set_axis_labels("Predicted Probability", "Deviance Residual")
r.set_titles(col_template="{col_name}", row_template="{row_name}")
r.tight_layout()
r.savefig('output/mle_diagnostic_residuals.png')

print("Diagnostic plots saved to output/ folder")
plt.close('all')

