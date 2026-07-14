// Stan model for Maximum Likelihood Estimation of Hop Model Parameters
// Data is assumed to be stacked across all years and estimation is performed for each monthly transition

functions {
  /**
   * Calculates neighborhood dispersal pressure summed over all source yards.
   * Returns a vector of length N.
   */
  vector calc_dispersal_vec(matrix dist,
                            matrix wind,
                            vector source_strength,
                            vector initial_strain,
                            vector sprays_prev,
                            vector cultivar_vec,
                            real alpha,
                            real eta2) {
    
    int N = size(cultivar_vec);
    matrix[N, N] kernel;
    
    matrix[N, N] wind_local = wind[1 : N, 1 : N];
    matrix[N, N] dist_local = dist[1 : N, 1 : N];
    
    // Build the power-law dispersal kernel with cultivar matching
    for (i in 1 : N) {
      for (j in 1 : N) {
        if (i == j) {
          kernel[i, j] = 0;
        } else {
          kernel[i, j] = pow(1 + dist_local[i, j], -alpha)
                         * (cultivar_vec[i] == cultivar_vec[j])
                         * initial_strain[j];
        }
      }
    }
    
    // Adjust neighbor source strength by their specific spray history
    vector[N] effective_source = source_strength .* exp(-eta2 * sprays_prev);
    
    // Multiply and sum over the columns (dot product)
    vector[N] dispersal_pressure = (wind_local .* kernel) * effective_source;
    
    return dispersal_pressure;
  }

  /**
   * Calculates pairwise neighborhood dispersal pressure between all yards.
   * Returns an N x N matrix.
   */
  matrix calc_dispersal_mat(matrix dist,
                            matrix wind,
                            vector source_strength,
                            vector initial_strain,
                            vector sprays_prev,
                            vector cultivar_vec,
                            real alpha,
                            real eta2) {
    
    int N = size(cultivar_vec);
    matrix[N, N] kernel;
    
    matrix[N, N] wind_local = wind[1 : N, 1 : N];
    matrix[N, N] dist_local = dist[1 : N, 1 : N];
    
    // Build the power-law dispersal kernel with cultivar matching
    for (i in 1 : N) {
      for (j in 1 : N) {
        if (i == j) {
          kernel[i, j] = 0;
        } else {
          kernel[i, j] = pow(1 + dist_local[i, j], -alpha)
                         * (cultivar_vec[i] == cultivar_vec[j])
                         * initial_strain[j];
        }
      }
    }
    
    // Adjust neighbor source strength by their specific spray history
    vector[N] effective_source = source_strength .* exp(-eta2 * sprays_prev);
    
    // Replicate effective source column vector into an N x N matrix for element-wise multiplication
    matrix[N, N] effective_source_mat = rep_matrix(effective_source, N);
    
    // Element-wise multiplication results in an N x N matrix
    matrix[N, N] dispersal_pressure_mat = wind_local .* kernel .* effective_source_mat;
    
    return dispersal_pressure_mat;
  }
}
data {
  int<lower=1> T; // Number of years
  int<lower=1> N_total; // Total stacked yards
  int<lower=1> N_max; // Max yards in any single year (for matrix padding)
  int<lower=1> N_yards; // Total number of unique yards
  array[N_total] int <lower=1> yard_ids; // Unique yard IDs

  
  array[T] int year_starts;
  array[T] int year_ends;
  array[T] int year_sizes;
  
  // Outcomes (Integer arrays)
  array[N_total] int y; // Infected plants (Outcome)
  array[N_total] int n; // Total plants (Outcome)
  
  // Lagged predictors
  vector[N_total] y_lag; // Infected last month
  vector[N_total] n_lag; // Total last month
  vector[N_total] s_lag; // Sprays last month
  vector[N_total] sI1_lag; // initial strain
  vector[N_total] a_lag; // Yard-specific area
  
  // Cultivar indicator
  vector[N_total] cultivar;
  
  // Year-specific matrices (Padded to N_max x N_max)
  array[T] matrix[N_max, N_max] dist_mats;
  array[T] matrix[N_max, N_max] wind_mats;

  // Prior Hyperparameters
  real beta_mu;    real beta_sigma;
  real delta_mu;   real delta_sigma;
  real gamma_mu;   real gamma_sigma;
  real alpha_mu;   real alpha_sigma;
  real eta1_mu;    real eta1_sigma;
  real eta2_mu;    real eta2_sigma;
}
transformed data {
  //Pre compute the weighted incidence
  vector[N_total] source_strength = a_lag .* (y_lag ./ n_lag);
}
parameters {
  real beta; // Global Intercept
  real <lower = 0> delta; // Global Auto-infection magnitude
  real <lower = 0> gamma; // Global Dispersal magnitude
  real <lower = 0> alpha; // Dispersal kernel decay
  real <lower = 0> eta1; // Auto-infection spray decay
  real <lower = 0> eta2; // Neighborhood spray decay
  real <lower = 0, upper = 1> pi; 
}

transformed parameters {
  
  // Linear predictor for all observations
  vector[N_total] logit_p;
  
  for (t in 1 : T) {
    int start = year_starts[t];
    int end = year_ends[t];
    int N_yr = year_sizes[t];
    
    // Year-specific slices
    vector[N_yr] ss_y = source_strength[start : end];
    vector[N_yr] s_y = s_lag[start : end];
    vector[N_yr] sI1_y = sI1_lag[start : end];
    vector[N_yr] cult_y = cultivar[start : end];
    vector[N_yr] prop_y = y_lag[start:end] ./ n_lag[start:end];
    array[N_yr] int id_y = yard_ids[start:end];
    
    // Neighborhood dispersal for this year block
    vector[N_yr] disp_yr = calc_dispersal_vec(dist_mats[t], wind_mats[t], ss_y,
                                          sI1_y, s_y, cult_y, alpha, eta2);
    
    // Linear Predictor 
    
    logit_p[start:end] = beta + (delta * prop_y .* exp(-eta1 * s_y))
                           + gamma * disp_yr;
    
  }
}

model {
  beta ~ normal(beta_mu, beta_sigma);
  delta ~ normal(delta_mu, delta_sigma);
  gamma ~ normal(gamma_mu, gamma_sigma);
  alpha ~ normal(alpha_mu, alpha_sigma);
  eta1 ~ normal(eta1_mu, eta1_sigma);
  eta2 ~ normal(eta2_mu, eta2_sigma);
  pi ~ beta(1, 1); 

  for (i in 1 : N_total) {
    real p_i = inv_logit(logit_p[i]);

    if (y[i] == 0) {
      target += log_mix(pi, 0, binomial_lpmf(0 | n[i], p_i));
    } else {
      target += log1m(pi) + binomial_lpmf(y[i] | n[i], p_i);
    }
  }
}

generated quantities {
  array[T] matrix[N_max, N_max] edge_weights;
  
  for (t in 1 : T) {
    int start = year_starts[t];
    int end = year_ends[t];
    int N_yr = year_sizes[t];

    // Year-specific slices
    vector[N_yr] ss_y = source_strength[start : end];
    vector[N_yr] s_y = s_lag[start : end];
    vector[N_yr] sI1_y = sI1_lag[start : end];
    vector[N_yr] cult_y = cultivar[start : end];
    
    // Initialize this year's matrix with zeros
    edge_weights[t] = rep_matrix(0.0, N_max, N_max);
    
    // Calculate the smaller local dispersal matrix (N_yr x N_yr)
    matrix[N_yr, N_yr] disp_yr = calc_dispersal_mat(dist_mats[t], wind_mats[t], ss_y,
                                                    sI1_y, s_y, cult_y, alpha, eta2);
    
    // Assign to our edge weights matrix
    edge_weights[t][1 : N_yr, 1 : N_yr] = gamma * disp_yr;
  }

  // generate y-star for posterior predictive checks
  array[N_total] int y_rep;
  for (i in 1 : N_total) {
    real p_i = inv_logit(logit_p[i]);

    if (bernoulli_rng(pi)) {
      y_rep[i] = 0;
    } else {
      y_rep[i] = binomial_rng(n[i], p_i);
    }
  }
}
