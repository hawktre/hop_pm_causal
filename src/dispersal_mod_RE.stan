functions {
  /**
   * Calculates neighborhood dispersal pressure for a specific year block.
   */
  vector calc_dispersal(matrix dist,
                        matrix wind,
                        vector source_strength,
                        vector initial_strain,
                        vector sprays_prev,
                        vector cultivar_vec,
                        real alpha,
                        real eta2) {
    int N = size(cultivar_vec);
    matrix[N, N] kernel;
    vector[N] dispersal_pressure;
    
    matrix[N, N] wind_local = wind[1 : N, 1 : N];
    matrix[N, N] dist_local = dist[1 : N, 1 : N];

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
    
    vector[N] effective_source = source_strength .* exp(-eta2 * sprays_prev);
    dispersal_pressure = (wind_local .* kernel) * effective_source;
    
    return dispersal_pressure;
  }
}

data {
  int<lower=1> T; 
  int<lower=1> N_total; 
  int<lower=1> N_max; 
  int<lower=1> J;                 // Total number of unique fields/yards

  array[T] int year_starts;
  array[T] int year_ends;
  array[T] int year_sizes;
  array[N_total] int<lower=1> field_id; // Field ID for each row
  
  array[N_total] int y; 
  array[N_total] int n; 
  
  vector[N_total] y_lag; 
  vector[N_total] n_lag; 
  vector[N_total] s_lag; 
  vector[N_total] sI1_lag; 
  vector[N_total] a_lag; 
  vector[N_total] cultivar;
  
  array[T] matrix[N_max, N_max] dist_mats;
  array[T] matrix[N_max, N_max] wind_mats;

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
  real beta; 
  real<lower=0> delta; 
  real<lower=0> gamma; 
  real<lower=0> alpha; 
  real<lower=0> eta1; 
  real<lower=0> eta2; 
  
  // Random Effects (Non-centered)
  vector[J] yard_z;                
  real<lower=0> sigma_yard;        
}

transformed parameters {
  vector[J] yard_effects = yard_z * sigma_yard;
  vector[N_total] logit_p;
  
  for (t in 1 : T) {
    int start = year_starts[t];
    int end = year_ends[t];
    int N_yr = year_sizes[t];
    
    vector[N_yr] ss_y = source_strength[start : end];
    vector[N_yr] s_y = s_lag[start : end];
    vector[N_yr] sI1_y = sI1_lag[start : end];
    vector[N_yr] cult_y = cultivar[start : end];
    
    // Get yard effects for this specific year's observations
    vector[N_yr] yard_eff_yr = yard_effects[field_id[start : end]];
    
    vector[N_yr] disp_yr = calc_dispersal(dist_mats[t], wind_mats[t], ss_y,
                                          sI1_y, s_y, cult_y, alpha, eta2);
    
    // Add yard_eff_yr to the linear predictor
    logit_p[start : end] = beta + yard_eff_yr 
                           + (delta * ss_y .* exp(-eta1 * s_y))
                           + gamma * disp_yr;
  }
}

model {
  beta ~ normal(beta_mu, beta_sigma);
  delta ~ lognormal(delta_mu, delta_sigma);
  gamma ~ lognormal(gamma_mu, gamma_sigma);
  alpha ~ normal(alpha_mu, alpha_sigma);
  eta1 ~ normal(eta1_mu, eta1_sigma);
  eta2 ~ normal(eta2_mu, eta2_sigma);

  // Random Effect Priors
  yard_z ~ std_normal();
  sigma_yard ~ exponential(1);

  y ~ binomial_logit(n, logit_p);
}

generated quantities {
  vector[N_total] edge_weight;
  vector[N_total] y_rep;
  
  for (t in 1 : T) {
    int start = year_starts[t];
    int end = year_ends[t];
    int N_yr = year_sizes[t];
    
    vector[N_yr] ss_y = source_strength[start : end];
    vector[N_yr] s_y = s_lag[start : end];
    vector[N_yr] sI1_y = sI1_lag[start : end];
    vector[N_yr] cult_y = cultivar[start : end];
    
    vector[N_yr] disp_yr = calc_dispersal(dist_mats[t], wind_mats[t], ss_y,
                                          sI1_y, s_y, cult_y, alpha, eta2);
    
    edge_weight[start : end] = gamma * disp_yr;
  }
  
  for (i in 1:N_total) {
    y_rep[i] = binomial_rng(n[i], inv_logit(logit_p[i]));
  }
}