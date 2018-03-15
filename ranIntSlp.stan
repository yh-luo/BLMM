data {
  int<lower=1> N;                  //number of data points
  real rating[N];                  //rating
  real<lower=0,upper=2> time[N];  //predictor(time)
  int<lower=1> J;                  //number of subjects
  int<lower=1, upper=J> id[N];     //subject id
}

parameters {
  vector[2] beta;                  //intercept and slope
  real<lower=0> sigma_e;           //error sd
  vector<lower=0>[2] sigma_u;      //subj sd
  cholesky_factor_corr[2] L_u;
  matrix[2,J] z_u;
}

transformed parameters{
  matrix[2,J] u;
  u = diag_pre_multiply(sigma_u,L_u) * z_u;	//subj random effects
}

model {
  real mu;
  //priors
  L_u ~ lkj_corr_cholesky(2.0);
  to_vector(z_u) ~ normal(0,1);
  //likelihood
  for (i in 1:N){
    mu = beta[1] + u[1,id[i]] +(beta[2] + u[2,id[i]]) * time[i];
    rating[i] ~ normal(mu,sigma_e);
  }
}
