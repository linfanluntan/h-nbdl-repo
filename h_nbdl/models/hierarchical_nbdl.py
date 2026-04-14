"""
Hierarchical Nonparametric Bayesian Dictionary Learning (H-NBDL).

Main model class that combines the dictionary prior, activation prior,
encoder, and decoder into a complete generative + inference model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from h_nbdl.models.priors import (
    IBPPrior,
    HierarchicalBetaBernoulli,
    HierarchicalDictionaryPrior,
)
from h_nbdl.models.encoder import SiteAwareEncoder


class HierarchicalNBDL(nn.Module):
    """Full H-NBDL model for amortized variational inference.

    Generative model:
        d^0_k ~ N(0, alpha^{-1} I)
        d_{jk} | d^0_k ~ N(d^0_k, lambda^{-1} I)
        pi_k ~ Beta(alpha0/K, 1)    [IBP]
        pi_{jk} | pi_k ~ Beta(a*pi_k, a*(1-pi_k))
        z_{ijk} ~ Bernoulli(pi_{jk})
        s_{ijk} | z_{ijk}=1 ~ N(0, tau_k^{-1})
        x_{ij} | D^(j), s, z ~ N(D^(j) (z ⊙ s), sigma^2 I)

    Inference:
        Amortized VI with Concrete relaxation for z, reparameterized
        Gaussian for s, and learned encoder conditioned on (x, site_id).

    Parameters
    ----------
    data_dim : int
        Observation dimensionality D.
    k_max : int
        Truncation level for IBP.
    n_sites : int
        Number of sites J.
    encoder_hidden : list of int
        Hidden layer widths for the encoder.
    site_embed_dim : int
        Site embedding dimensionality.
    alpha0 : float
        IBP concentration.
    concentration : float
        Site-level Beta concentration.
    """

    def __init__(
        self,
        data_dim: int,
        k_max: int = 100,
        n_sites: int = 1,
        encoder_hidden: Optional[List[int]] = None,
        site_embed_dim: int = 32,
        alpha0: float = 5.0,
        concentration: float = 10.0,
    ):
        super().__init__()
        if encoder_hidden is None:
            encoder_hidden = [256, 128]

        self.data_dim = data_dim
        self.k_max = k_max
        self.n_sites = n_sites

        # Components
        self.ibp_prior = IBPPrior(k_max, alpha0)
        self.activation_prior = HierarchicalBetaBernoulli(k_max, n_sites, concentration)
        self.dictionary_prior = HierarchicalDictionaryPrior(data_dim, k_max, n_sites)
        self.encoder = SiteAwareEncoder(
            data_dim, k_max, n_sites, encoder_hidden, site_embed_dim
        )

        # Learnable noise precision
        self.log_sigma2 = nn.Parameter(torch.tensor(math.log(0.1)))

        # Learnable code precision per atom
        self.log_tau = nn.Parameter(torch.zeros(k_max))

    @property
    def sigma2(self) -> torch.Tensor:
        return self.log_sigma2.exp()

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def concrete_sample(
        self, logit_z: torch.Tensor, temperature: float = 0.5
    ) -> torch.Tensor:
        """Sample from the Concrete (Gumbel-Softmax) distribution.

        Parameters
        ----------
        logit_z : Tensor, shape (..., k_max)
            Pre-sigmoid logits.
        temperature : float
            Concrete temperature (annealed during training).

        Returns
        -------
        z_relaxed : Tensor, shape (..., k_max)
            Relaxed binary activations in (0, 1).
        """
        if self.training:
            # Gumbel noise for reparameterization
            u = torch.rand_like(logit_z).clamp(1e-8, 1 - 1e-8)
            gumbel = torch.log(u) - torch.log(1 - u)
            z_relaxed = torch.sigmoid((logit_z + gumbel) / temperature)
        else:
            # Hard thresholding at test time
            z_relaxed = (logit_z > 0).float()
        return z_relaxed

    def decode(
        self, z: torch.Tensor, s: torch.Tensor, site_ids: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct observations from codes and activations.

        Parameters
        ----------
        z : Tensor, shape (batch, k_max)
            Binary (or relaxed) activation.
        s : Tensor, shape (batch, k_max)
            Continuous codes.
        site_ids : Tensor, shape (batch,)
            Site indices.

        Returns
        -------
        x_hat : Tensor, shape (batch, data_dim)
        """
        # Elementwise gating
        effective_code = z * s  # (batch, k_max)

        # Site-specific reconstruction
        x_hat = torch.zeros(z.shape[0], self.data_dim, device=z.device)
        for j in range(self.n_sites):
            mask = site_ids == j
            if mask.any():
                D_j = self.dictionary_prior.get_site_dictionary(j)  # (k_max, data_dim)
                x_hat[mask] = effective_code[mask] @ D_j  # (n_j, data_dim)
        return x_hat

    def forward(
        self,
        x: torch.Tensor,
        site_ids: torch.Tensor,
        temperature: float = 0.5,
    ) -> dict:
        """Full forward pass: encode, sample, decode.

        Returns a dict with reconstruction, codes, activations,
        and all quantities needed for ELBO computation.
        """
        batch_size = x.shape[0]

        # Encode
        mu, logvar, logit_z = self.encoder(x, site_ids)

        # Sample codes via reparameterization
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        s = mu + std * eps

        # Sample activations via Concrete relaxation
        z = self.concrete_sample(logit_z, temperature)

        # Decode
        x_hat = self.decode(z, s, site_ids)

        return {
            "x_hat": x_hat,
            "mu": mu,
            "logvar": logvar,
            "logit_z": logit_z,
            "s": s,
            "z": z,
        }

    def elbo(
        self,
        x: torch.Tensor,
        site_ids: torch.Tensor,
        fwd: dict,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the Evidence Lower Bound.

        Parameters
        ----------
        x : Tensor, shape (batch, data_dim)
        site_ids : Tensor, shape (batch,)
        fwd : dict
            Output of self.forward().
        beta : float
            KL weight (for beta-VAE annealing).

        Returns
        -------
        loss : Tensor (scalar)
            Negative ELBO (to be minimized).
        diagnostics : dict
            Breakdown of loss components.
        """
        batch_size = x.shape[0]

        # 1. Reconstruction loss: -E_q[log p(x | D, s, z)]
        recon_loss = (
            0.5 / self.sigma2 * ((x - fwd["x_hat"]) ** 2).sum(dim=-1).mean()
            + 0.5 * self.data_dim * self.log_sigma2
        )

        # 2. KL for codes: KL(q(s) || p(s | z))
        # Prior: s_k | z_k=1 ~ N(0, tau_k^{-1}), s_k | z_k=0 = 0
        # Approximate: KL for the Gaussian part, weighted by z
        prior_var = 1.0 / self.tau  # (k_max,)
        kl_s = 0.5 * fwd["z"] * (
            fwd["logvar"].exp() / prior_var
            + fwd["mu"] ** 2 / prior_var
            - 1
            - fwd["logvar"]
            + prior_var.log()
        )
        kl_s = kl_s.sum(dim=-1).mean()

        # 3. KL for activations: KL(q(z) || p(z | pi))
        # Using the Concrete approximation; the KL between Concrete and
        # Bernoulli is approximated by the KL between the
        # corresponding Bernoulli distributions
        q_z_prob = torch.sigmoid(fwd["logit_z"]).clamp(1e-6, 1 - 1e-6)

        # Get site-level prior probabilities
        p_z_prob = torch.zeros_like(q_z_prob)
        for j in range(self.n_sites):
            mask = site_ids == j
            if mask.any():
                p_z_prob[mask] = self.activation_prior.q_site_mean(j).unsqueeze(0)
        p_z_prob = p_z_prob.clamp(1e-6, 1 - 1e-6)

        kl_z = (
            q_z_prob * (q_z_prob.log() - p_z_prob.log())
            + (1 - q_z_prob) * ((1 - q_z_prob).log() - (1 - p_z_prob).log())
        ).sum(dim=-1).mean()

        # 4. KL for dictionary
        kl_dict = self.dictionary_prior.kl_dictionary() / batch_size

        # Total loss
        loss = recon_loss + beta * (kl_s + kl_z + kl_dict)

        diagnostics = {
            "recon_loss": recon_loss.item(),
            "kl_s": kl_s.item(),
            "kl_z": kl_z.item(),
            "kl_dict": kl_dict.item(),
            "sigma2": self.sigma2.item(),
            "k_effective": (q_z_prob.mean(dim=0) > 0.1).sum().item(),
        }

        return loss, diagnostics

    def encode(
        self, x: torch.Tensor, site_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract uncertainty-aware representations for downstream use.

        Returns the posterior mean and variance of z ⊙ s.

        Parameters
        ----------
        x : Tensor, shape (batch, data_dim)
        site_ids : Tensor, shape (batch,)

        Returns
        -------
        r_mean : Tensor, shape (batch, k_max)
            E[z * s] = E[z] * E[s] (mean-field).
        r_var : Tensor, shape (batch, k_max)
            Var[z * s] approximation.
        """
        self.eval()
        with torch.no_grad():
            mu, logvar, logit_z = self.encoder(x, site_ids)
            z_prob = torch.sigmoid(logit_z)
            s_var = logvar.exp()

            r_mean = z_prob * mu
            # Var(z*s) = E[z^2]*E[s^2] - (E[z]*E[s])^2
            # = E[z]*(var_s + mu^2) - (E[z]*mu)^2  [since z^2 = z for Bernoulli]
            r_var = z_prob * (s_var + mu ** 2) - (z_prob * mu) ** 2

        return r_mean, r_var

    def effective_atoms(self, threshold: float = 0.1) -> int:
        """Count the number of effectively active atoms.

        An atom is considered active if its average activation probability
        (across sites) exceeds the threshold.
        """
        with torch.no_grad():
            avg_prob = torch.stack(
                [self.activation_prior.q_site_mean(j) for j in range(self.n_sites)]
            ).mean(dim=0)
            return (avg_prob > threshold).sum().item()
