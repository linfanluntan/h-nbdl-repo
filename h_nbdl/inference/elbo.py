"""
ELBO computation utilities and diagnostics.

Provides standalone functions for computing individual ELBO terms,
monitoring convergence, and assessing the variational approximation quality.
"""

import torch
import numpy as np
from typing import Dict, List


def reconstruction_nll(
    x: torch.Tensor, x_hat: torch.Tensor, sigma2: torch.Tensor
) -> torch.Tensor:
    """Gaussian negative log-likelihood for reconstruction.

    Parameters
    ----------
    x : Tensor, shape (batch, D)
    x_hat : Tensor, shape (batch, D)
    sigma2 : Tensor (scalar)

    Returns
    -------
    nll : Tensor (scalar), averaged over batch.
    """
    D = x.shape[-1]
    mse = ((x - x_hat) ** 2).sum(dim=-1)  # (batch,)
    nll = 0.5 * (mse / sigma2 + D * torch.log(sigma2) + D * np.log(2 * np.pi))
    return nll.mean()


def kl_gaussian(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    prior_mu: torch.Tensor = None,
    prior_logvar: torch.Tensor = None,
) -> torch.Tensor:
    """KL divergence between two diagonal Gaussians.

    KL(N(mu, diag(exp(logvar))) || N(prior_mu, diag(exp(prior_logvar))))

    If prior_mu and prior_logvar are None, computes KL to N(0, I).
    """
    if prior_mu is None:
        prior_mu = torch.zeros_like(mu)
    if prior_logvar is None:
        prior_logvar = torch.zeros_like(logvar)

    var = logvar.exp()
    prior_var = prior_logvar.exp()

    kl = 0.5 * (
        prior_logvar - logvar
        + var / prior_var
        + (mu - prior_mu) ** 2 / prior_var
        - 1
    )
    return kl.sum(dim=-1).mean()


def kl_bernoulli(q_prob: torch.Tensor, p_prob: torch.Tensor) -> torch.Tensor:
    """KL divergence between two Bernoulli distributions.

    Parameters
    ----------
    q_prob, p_prob : Tensor
        Probabilities, clamped to avoid log(0).

    Returns
    -------
    kl : Tensor, summed over last dim, averaged over batch.
    """
    q = q_prob.clamp(1e-6, 1 - 1e-6)
    p = p_prob.clamp(1e-6, 1 - 1e-6)
    kl = q * (q.log() - p.log()) + (1 - q) * ((1 - q).log() - (1 - p).log())
    return kl.sum(dim=-1).mean()


def assess_convergence(history: List[Dict[str, float]], window: int = 20) -> Dict:
    """Assess training convergence from history.

    Parameters
    ----------
    history : list of dict
        Training history from AmortizedVI.fit().
    window : int
        Window size for moving average.

    Returns
    -------
    diagnostics : dict
        Convergence metrics.
    """
    losses = [h["loss"] for h in history]
    k_effs = [h.get("k_effective", 0) for h in history]

    if len(losses) < 2 * window:
        return {"converged": False, "reason": "too_few_epochs"}

    recent = np.mean(losses[-window:])
    earlier = np.mean(losses[-2 * window:-window])
    relative_change = abs(recent - earlier) / (abs(earlier) + 1e-8)

    k_recent = np.mean(k_effs[-window:])
    k_std = np.std(k_effs[-window:])

    return {
        "converged": relative_change < 0.01 and k_std < 2.0,
        "final_loss": losses[-1],
        "loss_relative_change": relative_change,
        "k_effective_mean": k_recent,
        "k_effective_std": k_std,
    }
