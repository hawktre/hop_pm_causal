vector[N_total] deviance_resid;
  
  //Compute the deviance residual for the binomial
  {
    vector[N_total] p = inv_logit(logit_p);
    for (i in 1 : N_total) {
      real y_hat = n[i] * p[i];
      real d2 = 0;
      if (y[i] > 0) 
        d2 += 2 * y[i] * log(y[i] / y_hat);
      if (n[i] - y[i] > 0)
        d2 += 2 * (n[i] - y[i]) * log((n[i] - y[i]) / (n[i] - y_hat));
      deviance_resid[i] = (y[i] > y_hat ? 1 : -1) * sqrt(d2);
    }
  }