"""
Generative model for H-NBDL.

Provides functions to sample synthetic data from the full hierarchical
generative process, useful for testing and validation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SyntheticData:
    """Container for synthetic H-NBDL data with ground truth."""
    X: np.ndarray           # (N_total, D) observations
    site_ids: np.ndarray    # (N_total,) site assignments
    Z_true: np.ndarray      # (N_total, K_true) true activations
    S_true: np.ndarray      # (N_total, K_true) true codes
    D_global: np.ndarray    # (K_true, D) global dictionary
    D_sites: list           # list of (K_true, D) site dictionaries
    pi_global: np.ndarray   # (K_true,) global activation probs
    pi_sites: list          # list of (K_true,) site activation probs
    sigma2: float           # noise variance
    K_true: int             # true number of atoms


def generate_hierarchical_data(
    n_sites: int = 5,
    n_per_site: int = 200,
    data_dim: int = 50,
    k_true: int = 15,
    sigma2: float = 0.1,
    lambda_inv: float = 0.1,
    alpha0: float = 5.0,
    site_concentration: float = 10.0,
    shared_atoms: int = 3,
    seed: Optional[int] = None,
) -> SyntheticData:
    """Generate synthetic data from the H-NBDL generative model.

    Creates a dataset with known ground truth for validation, including
    atoms that are shared globally, shared by subsets, and site-specific.

    Parameters
    ----------
    n_sites : int
        Number of sites J.
    n_per_site : int
        Number of samples per site.
    data_dim : int
        Observation dimensionality D.
    k_true : int
        True number of active atoms.
    sigma2 : float
        Observation noise variance.
    lambda_inv : float
        Site-deviation variance (1/lambda).
    alpha0 : float
        IBP concentration (controls global sparsity).
    site_concentration : float
        Beta concentration for site-level activation.
    shared_atoms : int
        Number of atoms shared by all sites.
    seed : int or None
        Random seed.

    Returns
    -------
    SyntheticData
        Container with observations and all ground-truth latent variables.
    """
    rng = np.random.default_rng(seed)

    # 1. Global dictionary atoms
    D_global = rng.standard_normal((k_true, data_dim))
    D_global /= np.linalg.norm(D_global, axis=1, keepdims=True)  # unit-norm atoms

    # 2. Site-specific dictionaries
    D_sites = []
    for j in range(n_sites):
        delta = rng.standard_normal((k_true, data_dim)) * np.sqrt(lambda_inv)
        D_sites.append(D_global + delta)

    # 3. Global activation probabilities (structured sparsity pattern)
    pi_global = np.zeros(k_true)
    # Shared atoms: high probability
    pi_global[:shared_atoms] = rng.beta(8, 2, size=shared_atoms)
    # Subset atoms
    n_subset = (k_true - shared_atoms) // 2
    pi_global[shared_atoms:shared_atoms + n_subset] = rng.beta(4, 4, size=n_subset)
    # Site-specific atoms: moderate probability globally, high in specific sites
    n_specific = k_true - shared_atoms - n_subset
    pi_global[shared_atoms + n_subset:] = rng.beta(1, 5, size=n_specific)

    # 4. Site-level activation probabilities
    pi_sites = []
    for j in range(n_sites):
        a = site_concentration * pi_global
        b = site_concentration * (1 - pi_global)
        # Avoid degenerate parameters
        a = np.maximum(a, 0.01)
        b = np.maximum(b, 0.01)
        pi_j = rng.beta(a, b)

        # Make some atoms site-specific
        specific_start = shared_atoms + n_subset
        for k in range(specific_start, k_true):
            if (k - specific_start) % n_sites == j:
                pi_j[k] = rng.beta(8, 2)  # high in this site
            else:
                pi_j[k] = rng.beta(1, 10)  # low elsewhere

        pi_sites.append(pi_j)

    # 5. Generate observations
    X_list, Z_list, S_list, site_list = [], [], [], []

    for j in range(n_sites):
        for i in range(n_per_site):
            # Binary activations
            z = rng.binomial(1, pi_sites[j])

            # Continuous codes (only where active)
            s = z * rng.standard_normal(k_true)

            # Observation
            x = D_sites[j].T @ s + rng.standard_normal(data_dim) * np.sqrt(sigma2)

            X_list.append(x)
            Z_list.append(z)
            S_list.append(s)
            site_list.append(j)

    return SyntheticData(
        X=np.array(X_list),
        site_ids=np.array(site_list),
        Z_true=np.array(Z_list),
        S_true=np.array(S_list),
        D_global=D_global,
        D_sites=D_sites,
        pi_global=pi_global,
        pi_sites=pi_sites,
        sigma2=sigma2,
        K_true=k_true,
    )
