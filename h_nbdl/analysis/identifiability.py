"""
Identifiability diagnostics for H-NBDL (Proposition 1 of the paper).

Provides tools to verify and visualize the global-local decomposition
identifiability conditions and measure decomposition quality.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple


def check_identifiability_conditions(
    model,
    site_ids: np.ndarray,
) -> Dict[str, bool]:
    """Check the three identifiability conditions from Proposition 1.

    (a) Linear independence of global atoms
    (b) At least one atom active in >= 2 sites
    (c) J >= 2

    Parameters
    ----------
    model : HierarchicalNBDL
        Trained model.
    site_ids : np.ndarray
        Site assignments.

    Returns
    -------
    conditions : dict
        Status of each condition.
    """
    J = len(np.unique(site_ids))
    D_global = model.dictionary_prior.D_global.detach().cpu().numpy()
    k_eff = model.effective_atoms(threshold=0.1)

    # (a) Linear independence: check rank of active atoms
    active_mask = np.arange(D_global.shape[0]) < k_eff
    D_active = D_global[active_mask]
    if D_active.shape[0] == 0:
        rank = 0
    else:
        rank = np.linalg.matrix_rank(D_active, tol=1e-6)
    cond_a = rank == D_active.shape[0]

    # (b) At least one atom active in >= 2 sites
    shared_count = 0
    for k in range(min(k_eff, model.k_max)):
        sites_active = 0
        for j in range(model.n_sites):
            prob = model.activation_prior.q_site_mean(j)[k].item()
            if prob > 0.1:
                sites_active += 1
        if sites_active >= 2:
            shared_count += 1
    cond_b = shared_count >= 1

    # (c) J >= 2
    cond_c = J >= 2

    return {
        "condition_a_independent": cond_a,
        "condition_a_rank": int(rank),
        "condition_a_n_active": int(D_active.shape[0]),
        "condition_b_shared_atoms": cond_b,
        "condition_b_n_shared": shared_count,
        "condition_c_multiple_sites": cond_c,
        "condition_c_J": J,
        "all_satisfied": cond_a and cond_b and cond_c,
    }


def decomposition_quality(
    model,
    D_global_true: Optional[np.ndarray] = None,
    D_sites_true: Optional[list] = None,
) -> Dict[str, float]:
    """Measure quality of the global-local decomposition.

    If ground truth is available, computes correlation between
    true and estimated global/local components.

    Parameters
    ----------
    model : HierarchicalNBDL
    D_global_true : np.ndarray or None
        True global atoms, shape (K_true, D).
    D_sites_true : list of np.ndarray or None
        True site-specific atoms.

    Returns
    -------
    quality : dict
    """
    D0_est = model.dictionary_prior.D_global.detach().cpu().numpy()
    offsets_est = model.dictionary_prior.D_site_offsets.detach().cpu().numpy()
    lam = model.dictionary_prior.lam.item()

    result = {
        "lambda_learned": lam,
        "site_deviation_std": 1.0 / np.sqrt(lam) if lam > 0 else float("inf"),
        "global_atom_norms": np.linalg.norm(D0_est, axis=1).tolist(),
        "offset_norms_by_site": [
            np.linalg.norm(offsets_est[j], axis=1).mean()
            for j in range(offsets_est.shape[0])
        ],
        "pooling_ratio": float(
            np.linalg.norm(D0_est) /
            (np.linalg.norm(D0_est) + np.linalg.norm(offsets_est) + 1e-10)
        ),
    }

    if D_global_true is not None:
        from h_nbdl.utils.metrics import amari_distance
        K = min(D_global_true.shape[0], D0_est.shape[0])
        result["amari_global"] = amari_distance(D_global_true, D0_est[:K])

    if D_sites_true is not None:
        site_amaris = []
        for j, D_j_true in enumerate(D_sites_true):
            D_j_est = D0_est + offsets_est[j] if j < offsets_est.shape[0] else D0_est
            K = min(D_j_true.shape[0], D_j_est.shape[0])
            site_amaris.append(amari_distance(D_j_true, D_j_est[:K]))
        result["amari_per_site"] = site_amaris
        result["amari_site_mean"] = float(np.mean(site_amaris))

    return result


def shared_vs_specific_analysis(model) -> Dict:
    """Analyze which atoms are shared vs site-specific.

    Categorizes atoms based on activation probability patterns
    across sites, matching the visualization in Figure 4.

    Returns
    -------
    analysis : dict
        Atom categorization and statistics.
    """
    n_sites = model.n_sites
    k_max = model.k_max

    probs = np.zeros((n_sites, k_max))
    for j in range(n_sites):
        probs[j] = model.activation_prior.q_site_mean(j).detach().cpu().numpy()

    # Categorize atoms
    active_threshold = 0.1
    shared = []     # active in all sites
    subset = []     # active in 2+ but not all sites
    specific = []   # active in exactly 1 site
    inactive = []   # active in 0 sites

    for k in range(k_max):
        n_active = np.sum(probs[:, k] > active_threshold)
        if n_active == 0:
            inactive.append(k)
        elif n_active == 1:
            specific.append(k)
        elif n_active < n_sites:
            subset.append(k)
        else:
            shared.append(k)

    return {
        "n_shared": len(shared),
        "n_subset": len(subset),
        "n_specific": len(specific),
        "n_inactive": len(inactive),
        "k_effective": len(shared) + len(subset) + len(specific),
        "shared_atoms": shared,
        "subset_atoms": subset,
        "specific_atoms": specific,
        "activation_probs": probs,
    }
