// Profile log-likelihood diagnostic for the binomial dispersal model.
// Evaluates the log-likelihood over a grid of (alpha, gamma) values,
// holding beta, delta, eta1, eta2 fixed at whatever posterior draw is
// supplied via `previous_fit` in cmdstanpy's generate_quantities().
//
// This model has NO likelihood/model block on purpose: it is only ever
// run via generate_quantities against draws from the real fitted model,
// never sampled directly.

functions {
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
    vector[N] dispersal_pressure = (wind_local .* kernel) * effective_source;
    return dispersal_pressure;
  }
}
data {
  int<lower=1> T;
  int<lower=1> N_total;
  int<lower=1> N_max;
  int<lower=1> N_yards;
  array[N_total] int<lower=1> yard_ids;

  array[T] int year_starts;
  array[T] int year_ends;
  array[T] int year_sizes;

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

  // Grid points for the profile log-likelihood diagnostic
  int<lower=1> N_alpha_grid;
  int<lower=1> N_gamma_grid;
  vector[N_alpha_grid] alpha_grid;
  vector[N_gamma_grid] gamma_grid;
}
transformed data {
  vector[N_total] source_strength = a_lag .* (y_lag ./ n_lag);
}
parameters {
  // These must match the real model's parameter block exactly (names +
  // shapes + constraints) so that generate_quantities can line up draws.
  real beta;
  real<lower=0> delta;
  real<lower=0> gamma;
  real<lower=0> alpha;
  real<lower=0> eta1;
  real<lower=0> eta2;
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

  y~binomial_logit(n, logit_p);
}
generated quantities {
  matrix[N_alpha_grid, N_gamma_grid] loglik_grid;

  for (ag in 1 : N_alpha_grid) {
    real alpha_g = alpha_grid[ag];

    // Dispersal term depends on alpha_g but not on gamma_grid, so it's
    // computed once per alpha_g value and reused across all gamma_grid values.
    vector[N_total] disp_full;

    for (t in 1 : T) {
      int start = year_starts[t];
      int end = year_ends[t];
      int N_yr = year_sizes[t];

      vector[N_yr] ss_y = source_strength[start : end];
      vector[N_yr] s_y = s_lag[start : end];
      vector[N_yr] sI1_y = sI1_lag[start : end];
      vector[N_yr] cult_y = cultivar[start : end];

      disp_full[start : end] = calc_dispersal_vec(dist_mats[t], wind_mats[t],
                                                   ss_y, sI1_y, s_y, cult_y,
                                                   alpha_g, eta2);
    }

    for (gg in 1 : N_gamma_grid) {
      real gamma_g = gamma_grid[gg];
      real ll = 0;
      vector[N_total] logit_p_g;  // full length so indices line up globally

      for (t in 1 : T) {
        int start = year_starts[t];
        int end = year_ends[t];
        vector[end - start + 1] prop_y = y_lag[start : end] ./ n_lag[start : end];
        vector[end - start + 1] s_y = s_lag[start : end];

        logit_p_g[start : end] = beta
                                  + (delta * prop_y .* exp(-eta1 * s_y))
                                  + gamma_g * disp_full[start : end];
      }

      for (i in 1 : N_total) {
        ll += binomial_logit_lpmf(y[i] | n[i], logit_p_g[i]);
      }

      loglik_grid[ag, gg] = ll;
    }
  }
}
