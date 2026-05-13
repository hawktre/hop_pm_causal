// Stan model for Maximum Likelihood Estimation of Hop Model Parameters
// Data is assumed to be stacked across all years and estimation is performed for each monthly transition

functions {
  /**
   * Calculates neighborhood dispersal pressure for a specific year block.
   * Ensures no cross-year infection and excludes self-infection (j != i).
   * cultivar_vec is used to ensure only yards of the same cultivar infect each other.
   */
  vector calc_dispersal(matrix dist,
                        matrix wind,
                        vector source_strength,
                        vector initial_strain,
                        vector sprays_prev,
                        vector cultivar_vec,
                        real alpha,
                        real eta2) {
    //Set sample size for current year                    
    int N = size(cultivar_vec);
    
    //Initialize storage objects
    matrix[N, N] kernel;
    vector[N] dispersal_pressure;
    
    //Subset distance and wind to only the entries we need (not padded zeros)
    matrix[N, N] wind_local = wind[1 : N, 1 : N];
    matrix[N, N] dist_local = dist[1 : N, 1 : N];
    // Build the power-law dispersal kernel with cultivar matching
    for (i in 1 : N) {
      for (j in 1 : N) {
        if (i == j) {
          kernel[i, j] = 0;
        } else {
          // Kernel weighted by cultivar matching (1 if same, 0 if different, 1 if initial_strain = "V6")
          kernel[i, j] = pow(1 + dist_local[i, j], -alpha)
                         * (cultivar_vec[i] == cultivar_vec[j])
                         * initial_strain[j];
        }
      }
    }
    
    // Adjust neighbor source strength by their specific spray history
    vector[N] effective_source = source_strength .* exp(-eta2 * sprays_prev);
    
    // Weight by wind and calculate arriving pressure
    dispersal_pressure = (wind_local .* kernel) * effective_source;
    
    return dispersal_pressure;
  }
}
data {
  int<lower=1> T; // Number of years
  int<lower=1> N_total; // Total stacked yards
  int<lower=1> N_max; // Max yards in any single year (for matrix padding)

  
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
  vector[N_total] source_strength = a_lag .* (y_lag ./ n_lag);
}
parameters {
  real beta; // Global Intercept
  real<lower=0> delta; // Global Auto-infection magnitude
  real<lower=0> gamma; // Global Dispersal magnitude
  real<lower=0> alpha; // Dispersal kernel decay
  real<lower=0> eta1; // Auto-infection spray decay
  real<lower=0> eta2; // Neighborhood spray decay

}

transformed parameters {
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
    
    // Neighborhood dispersal for this year block
    vector[N_yr] disp_yr = calc_dispersal(dist_mats[t], wind_mats[t], ss_y,
                                          sI1_y, s_y, cult_y, alpha, eta2);
    
    // Linear Predictor 
    logit_p[start : end] = beta + (delta * ss_y .* exp(-eta1 * s_y))
                           + gamma * disp_yr;
  }
}

model {
  // Baseline risk
  beta ~ normal(beta_mu, beta_sigma);
  
  // Auto-infection (Log-Normal)
  delta ~ lognormal(delta_mu, delta_sigma);
  
  // Dispersal magnitude (Log-Normal)
  gamma ~ lognormal(gamma_mu, gamma_sigma);
  
  // Distance decay 
  alpha ~ lognormal(alpha_mu, alpha_sigma);
  
  // Sprays decay 
  eta1 ~ lognormal(eta1_mu, eta1_sigma);
  eta2 ~ lognormal(eta2_mu, eta2_sigma);

  // Likelihood
  y ~ binomial_logit(n, logit_p);
}

generated quantities {
  vector[N_total] edge_weight;
  
  for (t in 1 : T) {
    int start = year_starts[t];
    int end = year_ends[t];
    int N_yr = year_sizes[t];
    
    // Year-specific slices
    vector[N_yr] ss_y = source_strength[start : end];
    vector[N_yr] s_y = s_lag[start : end];
    vector[N_yr] sI1_y = sI1_lag[start : end];
    vector[N_yr] cult_y = cultivar[start : end];
    
    // Neighborhood dispersal for this year block
    vector[N_yr] disp_yr = calc_dispersal(dist_mats[t], wind_mats[t], ss_y,
                                          sI1_y, s_y, cult_y, alpha, eta2);
    
    // Linear Predictor 
    edge_weight[start : end] = gamma * disp_yr;
  }
}
