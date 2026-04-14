"""
Evaluation metrics for dictionary learning.

Includes Amari distance for dictionary recovery, calibration scores
for posterior credible intervals, and reconstruction metrics.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def amari_distance(D_true: np.ndarray, D_est: np.ndarray) -> float:
    """Amari distance between true and estimated dictionaries.

    Measures how well the estimated dictionary recovers the true one,
    up to permutation and sign ambiguity.

    Parameters
    ----------
    D_true : np.ndarray, shape (K_true, D)
        True dictionary atoms (rows).
    D_est : np.ndarray, shape (K_est, D)
        Estimated dictionary atoms (rows).

    Returns
    -------
    distance : float
        Amari distance in [0, 1]. Lower is better.
    """
    K_true, D = D_true.shape
    K_est = D_est.shape[0]
    K = min(K_true, K_est)

    # Normalize atoms
    D_true_n = D_true / (np.linalg.norm(D_true, axis=1, keepdims=True) + 1e-10)
    D_est_n = D_est / (np.linalg.norm(D_est, axis=1, keepdims=True) + 1e-10)

    # Absolute correlation matrix (handles sign ambiguity)
    C = np.abs(D_true_n @ D_est_n.T)  # (K_true, K_est)

    # Hungarian algorithm for optimal permutation
    cost = 1.0 - C[:K, :K] if K_true <= K_est else 1.0 - C[:K, :K]
    row_ind, col_ind = linear_sum_assignment(cost)

    # Amari distance
    matched_corr = C[row_ind, col_ind]
    distance = 1.0 - np.mean(matched_corr)
    return float(distance)


def calibration_score(
    true_values: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_std: np.ndarray,
    level: float = 0.95,
) -> float:
    """Empirical coverage of Bayesian credible intervals.

    Parameters
    ----------
    true_values : np.ndarray
        Ground-truth values.
    posterior_mean : np.ndarray
        Posterior mean estimates.
    posterior_std : np.ndarray
        Posterior standard deviation estimates.
    level : float
        Nominal credible level (e.g., 0.95).

    Returns
    -------
    coverage : float
        Fraction of true values within the credible interval.
    """
    from scipy import stats
    z = stats.norm.ppf(0.5 + level / 2)
    lower = posterior_mean - z * posterior_std
    upper = posterior_mean + z * posterior_std
    covered = np.logical_and(true_values >= lower, true_values <= upper)
    return float(np.mean(covered))


def reconstruction_mse(X: np.ndarray, X_hat: np.ndarray) -> float:
    """Mean squared error for reconstruction quality."""
    return float(np.mean((X - X_hat) ** 2))


def effective_dimension(activation_probs: np.ndarray, threshold: float = 0.1) -> int:
    """Count atoms with mean activation probability above threshold."""
    if activation_probs.ndim == 2:
        mean_probs = activation_probs.mean(axis=0)
    else:
        mean_probs = activation_probs
    return int(np.sum(mean_probs > threshold))


def sparsity_ratio(Z: np.ndarray) -> float:
    """Fraction of zero entries in the activation matrix."""
    return float(1.0 - np.mean(Z > 0))
