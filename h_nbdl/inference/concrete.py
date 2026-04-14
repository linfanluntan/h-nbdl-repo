"""
Concrete (Gumbel-Softmax) distribution utilities.

Provides sampling and KL divergence for the continuous relaxation
of Bernoulli random variables used in the H-NBDL activation model.
"""

import torch
import math


def concrete_sample(
    logits: torch.Tensor,
    temperature: float = 0.5,
    hard: bool = False,
) -> torch.Tensor:
    """Sample from the Binary Concrete distribution.

    Parameters
    ----------
    logits : Tensor
        Log-odds (logit of the Bernoulli parameter).
    temperature : float
        Temperature; lower = closer to discrete.
    hard : bool
        If True, use straight-through estimator for hard samples.

    Returns
    -------
    samples : Tensor
        Samples in (0, 1), same shape as logits.
    """
    u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
    gumbel_noise = torch.log(u) - torch.log(1 - u)
    y = torch.sigmoid((logits + gumbel_noise) / temperature)

    if hard:
        y_hard = (y > 0.5).float()
        y = y_hard - y.detach() + y  # straight-through

    return y


def concrete_kl(
    q_logits: torch.Tensor,
    p_logits: torch.Tensor,
) -> torch.Tensor:
    """Approximate KL between two Binary Concrete distributions.

    Uses the KL between the corresponding Bernoulli distributions
    as an approximation (exact as temperature -> 0).

    Parameters
    ----------
    q_logits, p_logits : Tensor
        Logits of the approximate posterior and prior.

    Returns
    -------
    kl : Tensor
        KL divergence, summed over the last dimension.
    """
    q_prob = torch.sigmoid(q_logits).clamp(1e-6, 1 - 1e-6)
    p_prob = torch.sigmoid(p_logits).clamp(1e-6, 1 - 1e-6)

    kl = (
        q_prob * (q_prob.log() - p_prob.log())
        + (1 - q_prob) * ((1 - q_prob).log() - (1 - p_prob).log())
    )
    return kl.sum(dim=-1)


def temperature_schedule(
    epoch: int,
    total_epochs: int,
    temp_init: float = 1.0,
    temp_final: float = 0.1,
    anneal_fraction: float = 0.5,
) -> float:
    """Linear temperature annealing schedule.

    Parameters
    ----------
    epoch : int
        Current epoch (0-indexed).
    total_epochs : int
        Total number of training epochs.
    temp_init : float
        Starting temperature.
    temp_final : float
        Final temperature.
    anneal_fraction : float
        Fraction of training over which to anneal.

    Returns
    -------
    temperature : float
    """
    anneal_epochs = int(total_epochs * anneal_fraction)
    if epoch >= anneal_epochs:
        return temp_final
    t = epoch / anneal_epochs
    return temp_init + (temp_final - temp_init) * t
