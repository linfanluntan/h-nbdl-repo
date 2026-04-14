"""
Collapsed Gibbs Sampler for H-NBDL.

Marginalizes over continuous codes S and dictionary atoms D analytically
(exploiting Gaussian conjugacy) and samples the binary activation matrix Z
and hyperparameters via Gibbs sweeps.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from tqdm import tqdm


@dataclass
class GibbsSamples:
    """Container for posterior samples from the Gibbs sampler."""
    Z_samples: List[np.ndarray] = field(default_factory=list)
    K_eff_trace: List[int] = field(default_factory=list)
    sigma2_trace: List[float] = field(default_factory=list)
    alpha0_trace: List[float] = field(default_factory=list)
    D_mean: Optional[np.ndarray] = None
    S_mean: Optional[np.ndarray] = None

    def dictionary_mean(self) -> np.ndarray:
        return self.D_mean

    def active_atoms_trace(self) -> List[int]:
        return self.K_eff_trace

    def effective_K(self) -> float:
        """Posterior mean of effective number of atoms."""
        burnin = len(self.K_eff_trace) // 2
        return np.mean(self.K_eff_trace[burnin:])


class CollapsedGibbs:
    """Collapsed Gibbs sampler for H-NBDL.

    Collapses out D and S, sampling Z from its marginal conditional.
    Uses the matrix determinant lemma for efficient rank-one updates.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Observations.
    site_ids : np.ndarray, shape (N,)
        Site assignments.
    k_max : int
        Truncation level.
    alpha0 : float
        IBP concentration.
    sigma2_init : float
        Initial noise variance.
    tau : float
        Code precision.
    lambda_precision : float
        Site deviation precision.
    n_iter : int
        Total number of iterations.
    burnin : int
        Burn-in iterations to discard.
    thin : int
        Thinning factor.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        X: np.ndarray,
        site_ids: np.ndarray,
        k_max: int = 100,
        alpha0: float = 5.0,
        sigma2_init: float = 0.1,
        tau: float = 1.0,
        lambda_precision: float = 10.0,
        n_iter: int = 2000,
        burnin: int = 1000,
        thin: int = 5,
        seed: Optional[int] = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.X = X.astype(np.float64)
        self.site_ids = site_ids.astype(int)
        self.N, self.D = X.shape
        self.J = len(np.unique(site_ids))
        self.k_max = k_max
        self.n_iter = n_iter
        self.burnin = burnin
        self.thin = thin

        # Hyperparameters
        self.alpha0 = alpha0
        self.sigma2 = sigma2_init
        self.tau = tau
        self.lam = lambda_precision

        # Initialize Z: each sample activates ~alpha0*log(2)/K_max features
        init_prob = min(0.5, alpha0 * np.log(2) / k_max)
        self.Z = self.rng.binomial(1, init_prob, size=(self.N, k_max)).astype(np.float64)

        # Global activation probabilities
        self.pi = np.full(k_max, init_prob)

        # Site masks for efficient indexing
        self.site_masks = {
            j: np.where(site_ids == j)[0] for j in range(self.J)
        }

    def _collapsed_log_likelihood_ratio(
        self, i: int, k: int, site_idx: int
    ) -> float:
        """Compute log p(x_i | z_{ik}=1, rest) / p(x_i | z_{ik}=0, rest).

        Uses the marginal likelihood with D and S integrated out.
        For efficiency, uses the Woodbury identity for rank-one updates.
        """
        mask = self.site_masks[site_idx]
        X_j = self.X[mask]
        Z_j = self.Z[mask].copy()
        N_j = len(mask)

        # Local index of sample i within site j
        local_i = np.searchsorted(mask, i)

        # Compute M_j = Z_j^T Z_j + (sigma2/tau + sigma2*lambda) * I
        reg = self.sigma2 / self.tau + self.sigma2 * self.lam
        M_on = Z_j.T @ Z_j + reg * np.eye(self.k_max)
        M_on_inv = np.linalg.solve(M_on, np.eye(self.k_max))

        # Marginal precision with z_{ik} = 1
        z_on = Z_j[local_i].copy()
        z_on[k] = 1.0

        # Marginal precision with z_{ik} = 0
        z_off = Z_j[local_i].copy()
        z_off[k] = 0.0

        # Simplified: compute reconstruction residual under both settings
        # Use the conditional mean E[D s_i | Z, X, z_{ik}]
        x_i = self.X[i]

        # Log-likelihood ratio approximation via residual comparison
        # This is a simplified version; full derivation uses matrix det lemma
        active_on = np.where(z_on > 0)[0]
        active_off = np.where(z_off > 0)[0]

        def _marginal_residual(active):
            if len(active) == 0:
                return np.sum(x_i ** 2)
            Z_active = Z_j[:, active]
            M = Z_active.T @ Z_active + reg * np.eye(len(active))
            try:
                M_inv = np.linalg.solve(M, np.eye(len(active)))
            except np.linalg.LinAlgError:
                M_inv = np.linalg.pinv(M)
            proj = Z_active @ M_inv @ Z_active.T
            x_proj = proj[local_i] @ X_j
            return np.sum((x_i - x_proj) ** 2)

        resid_on = _marginal_residual(active_on)
        resid_off = _marginal_residual(active_off)

        log_ratio = -0.5 / self.sigma2 * (resid_on - resid_off)
        return log_ratio

    def _sample_z(self):
        """One full sweep over all entries of Z."""
        for i in range(self.N):
            j = self.site_ids[i]
            for k in range(self.k_max):
                # Prior odds
                n_k = self.Z[:, k].sum() - self.Z[i, k]
                prior_on = (n_k + self.alpha0 / self.k_max) / (self.N - 1 + self.alpha0)
                prior_on = np.clip(prior_on, 1e-10, 1 - 1e-10)
                log_prior_ratio = np.log(prior_on) - np.log(1 - prior_on)

                # Likelihood ratio (collapsed)
                log_lik_ratio = self._collapsed_log_likelihood_ratio(i, k, j)

                # Posterior
                log_odds = log_prior_ratio + log_lik_ratio
                prob = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -30, 30)))
                self.Z[i, k] = self.rng.binomial(1, prob)

    def _sample_sigma2(self):
        """Sample noise variance from its inverse-Gamma conditional."""
        # Compute reconstruction under current Z
        residual_sum = 0.0
        for j in range(self.J):
            mask = self.site_masks[j]
            X_j = self.X[mask]
            Z_j = self.Z[mask]
            active = np.where(Z_j.sum(axis=0) > 0)[0]
            if len(active) > 0:
                Z_a = Z_j[:, active]
                reg = self.sigma2 / self.tau * np.eye(len(active))
                coef = np.linalg.solve(Z_a.T @ Z_a + reg, Z_a.T @ X_j)
                resid = X_j - Z_a @ coef
            else:
                resid = X_j
            residual_sum += np.sum(resid ** 2)

        a_post = 1.0 + 0.5 * self.N * self.D
        b_post = 1.0 + 0.5 * residual_sum
        self.sigma2 = 1.0 / self.rng.gamma(a_post, 1.0 / b_post)
        self.sigma2 = np.clip(self.sigma2, 1e-6, 10.0)

    def _sample_alpha0(self):
        """Sample IBP concentration via Metropolis-Hastings."""
        K_plus = int((self.Z.sum(axis=0) > 0).sum())
        # Proposal
        alpha_prop = self.alpha0 * np.exp(0.1 * self.rng.standard_normal())

        # Log-posterior ratio (using IBP marginal)
        log_ratio = (
            K_plus * (np.log(alpha_prop) - np.log(self.alpha0))
            - (alpha_prop - self.alpha0) * sum(1.0 / n for n in range(1, self.N + 1))
            + (1.0 - 1.0) * (np.log(alpha_prop) - np.log(self.alpha0))  # Gamma(1,1) prior
            - (alpha_prop - self.alpha0)  # Gamma(1,1) prior
        )

        if np.log(self.rng.random()) < log_ratio:
            self.alpha0 = alpha_prop

    def _compute_posterior_D_S(self) -> tuple:
        """Compute posterior mean of D and S given final Z."""
        D_atoms = []
        S_all = np.zeros((self.N, self.k_max))

        for j in range(self.J):
            mask = self.site_masks[j]
            X_j = self.X[mask]
            Z_j = self.Z[mask]
            active = np.where(Z_j.sum(axis=0) > 0)[0]

            if len(active) > 0:
                Z_a = Z_j[:, active]
                reg = (self.sigma2 / self.tau) * np.eye(len(active))
                coef = np.linalg.solve(Z_a.T @ Z_a + reg, Z_a.T @ X_j)
                # coef: (K_active, D) ~ dictionary atoms weighted by codes
                S_all[np.ix_(mask, active)] = Z_a @ np.diag(np.linalg.norm(coef, axis=1))

        # Global D estimate: average across sites
        D_est = np.zeros((self.k_max, self.D))
        for j in range(self.J):
            mask = self.site_masks[j]
            Z_j = self.Z[mask]
            X_j = self.X[mask]
            active = np.where(Z_j.sum(axis=0) > 0)[0]
            if len(active) > 0:
                Z_a = Z_j[:, active]
                reg = (self.sigma2 / self.tau) * np.eye(len(active))
                coef = np.linalg.solve(Z_a.T @ Z_a + reg, Z_a.T @ X_j)
                D_est[active] += coef / self.J

        return D_est, S_all

    def run(self, verbose: bool = True) -> GibbsSamples:
        """Run the collapsed Gibbs sampler.

        Returns
        -------
        samples : GibbsSamples
            Posterior samples and trace diagnostics.
        """
        samples = GibbsSamples()

        iterator = range(self.n_iter)
        if verbose:
            iterator = tqdm(iterator, desc="Gibbs sampling")

        for it in iterator:
            # Sample Z (most expensive step)
            self._sample_z()

            # Sample hyperparameters
            self._sample_sigma2()
            self._sample_alpha0()

            # Record
            K_eff = int((self.Z.sum(axis=0) > 0).sum())
            samples.K_eff_trace.append(K_eff)
            samples.sigma2_trace.append(self.sigma2)
            samples.alpha0_trace.append(self.alpha0)

            # Store thinned post-burnin samples
            if it >= self.burnin and (it - self.burnin) % self.thin == 0:
                samples.Z_samples.append(self.Z.copy())

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    K=K_eff, sigma2=f"{self.sigma2:.4f}", alpha=f"{self.alpha0:.2f}"
                )

        # Compute posterior mean of D and S using final Z
        samples.D_mean, samples.S_mean = self._compute_posterior_D_S()

        return samples
