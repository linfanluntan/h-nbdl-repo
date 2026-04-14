"""
Hierarchical Causal Outcome Model using H-NBDL representations.

Implements the downstream HBM described in Section 5.2 of the paper:
    y_{ij} ~ p(y | T_{ij}, f(r_{ij}), theta_j)
    theta_j ~ N(mu_theta, Sigma_theta)

Uses PyMC for posterior inference via NUTS.
"""

import numpy as np
from typing import Optional, Dict


class HierarchicalCausalModel:
    """Hierarchical Bayesian outcome model for causal effect estimation.

    Takes H-NBDL representations (mean + variance) as structured covariates
    and estimates treatment effects with site-level partial pooling.

    Parameters
    ----------
    representation_dim : int
        Dimensionality of the representation (2 * K_eff for mean + var).
    n_sites : int
        Number of sites.
    outcome_type : str
        "binary" for logistic, "continuous" for Gaussian outcome.
    n_representation_features : int or None
        If set, use PCA to reduce representation dimension before modeling.
    """

    def __init__(
        self,
        representation_dim: int,
        n_sites: int,
        outcome_type: str = "binary",
        n_representation_features: Optional[int] = None,
    ):
        self.representation_dim = representation_dim
        self.n_sites = n_sites
        self.outcome_type = outcome_type
        self.n_features = n_representation_features or representation_dim
        self.trace_ = None
        self.pca_ = None

    def _reduce_features(self, R: np.ndarray) -> np.ndarray:
        """Optionally reduce representation dimensionality via PCA."""
        if self.n_features >= R.shape[1]:
            return R
        from sklearn.decomposition import PCA
        if self.pca_ is None:
            self.pca_ = PCA(n_components=self.n_features)
            return self.pca_.fit_transform(R)
        return self.pca_.transform(R)

    def fit(
        self,
        r_mean: np.ndarray,
        r_var: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        site_ids: np.ndarray,
        n_samples: int = 2000,
        n_tune: int = 1000,
        n_chains: int = 4,
        seed: int = 42,
    ) -> Dict:
        """Fit the hierarchical causal model.

        Parameters
        ----------
        r_mean : np.ndarray, shape (N, K)
            Posterior mean of H-NBDL representation.
        r_var : np.ndarray, shape (N, K)
            Posterior variance of H-NBDL representation.
        treatment : np.ndarray, shape (N,)
            Treatment assignments (binary or continuous).
        outcome : np.ndarray, shape (N,)
            Observed outcomes.
        site_ids : np.ndarray, shape (N,)
            Site assignments.
        n_samples, n_tune, n_chains : int
            MCMC settings.

        Returns
        -------
        trace : dict
            Posterior samples (or ArviZ InferenceData if PyMC available).
        """
        try:
            return self._fit_pymc(
                r_mean, r_var, treatment, outcome, site_ids,
                n_samples, n_tune, n_chains, seed
            )
        except ImportError:
            print("PyMC not installed. Using simple MAP estimation.")
            return self._fit_map(r_mean, r_var, treatment, outcome, site_ids)

    def _fit_pymc(self, r_mean, r_var, treatment, outcome, site_ids,
                  n_samples, n_tune, n_chains, seed):
        """Full Bayesian inference via PyMC."""
        import pymc as pm

        # Combine mean and variance into feature matrix
        R = np.hstack([r_mean, r_var])
        R = self._reduce_features(R)
        N, P = R.shape

        with pm.Model() as model:
            # Population-level coefficients
            mu_beta = pm.Normal("mu_beta", mu=0, sigma=1, shape=P)
            sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.5, shape=P)

            # Site-level coefficients (partial pooling)
            beta_site = pm.Normal(
                "beta_site", mu=mu_beta, sigma=sigma_beta,
                shape=(self.n_sites, P)
            )

            # Treatment effect (hierarchical)
            mu_tau = pm.Normal("mu_tau", mu=0, sigma=1)
            sigma_tau = pm.HalfNormal("sigma_tau", sigma=0.5)
            tau_site = pm.Normal(
                "tau_site", mu=mu_tau, sigma=sigma_tau,
                shape=self.n_sites
            )

            # Intercept
            alpha_site = pm.Normal("alpha_site", mu=0, sigma=2, shape=self.n_sites)

            # Linear predictor
            eta = (
                alpha_site[site_ids]
                + (R * beta_site[site_ids]).sum(axis=1)
                + tau_site[site_ids] * treatment
            )

            # Outcome model
            if self.outcome_type == "binary":
                pm.Bernoulli("y_obs", logit_p=eta, observed=outcome)
            else:
                sigma_y = pm.HalfNormal("sigma_y", sigma=1)
                pm.Normal("y_obs", mu=eta, sigma=sigma_y, observed=outcome)

            # Sample
            trace = pm.sample(
                n_samples, tune=n_tune, chains=n_chains,
                random_seed=seed, return_inferencedata=True,
                target_accept=0.9,
            )

        self.trace_ = trace
        return trace

    def _fit_map(self, r_mean, r_var, treatment, outcome, site_ids):
        """Simple MAP estimation fallback without PyMC."""
        from sklearn.linear_model import LogisticRegression, Ridge

        R = np.hstack([r_mean, r_var, treatment.reshape(-1, 1)])
        R = self._reduce_features(R) if R.shape[1] > self.n_features + 1 else R

        if self.outcome_type == "binary":
            model = LogisticRegression(C=1.0, max_iter=1000)
        else:
            model = Ridge(alpha=1.0)

        model.fit(R, outcome)
        self.trace_ = {"coefficients": model.coef_, "intercept": model.intercept_}
        return self.trace_

    def average_treatment_effect(self, trace=None) -> Dict[str, float]:
        """Compute the Average Treatment Effect (ATE) from posterior samples.

        Returns
        -------
        ate : dict with keys 'mean', 'std', 'ci_lower', 'ci_upper'.
        """
        if trace is None:
            trace = self.trace_

        try:
            # ArviZ InferenceData
            tau_samples = trace.posterior["mu_tau"].values.flatten()
        except (AttributeError, KeyError):
            # MAP fallback
            return {"mean": float(trace.get("coefficients", [[0]])[-1][-1]),
                    "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}

        return {
            "mean": float(np.mean(tau_samples)),
            "std": float(np.std(tau_samples)),
            "ci_lower": float(np.percentile(tau_samples, 2.5)),
            "ci_upper": float(np.percentile(tau_samples, 97.5)),
        }
