"""
Prior distributions for H-NBDL.

Implements the Indian Buffet Process (via finite Beta-Bernoulli approximation),
hierarchical Beta-Bernoulli activation priors, and Gaussian dictionary priors.
"""

import math
import torch
import torch.nn as nn
from torch.distributions import Beta, Bernoulli, Gamma, Normal


class IBPPrior(nn.Module):
    """Finite approximation to the Indian Buffet Process.

    In the IBP(alpha0) limit K -> inf, the expected number of active features
    grows as alpha0 * log(N). We work with a truncated approximation at K_max
    and let the posterior prune unused atoms via the Beta-Bernoulli mechanism.

    Parameters
    ----------
    k_max : int
        Truncation level (maximum number of atoms).
    alpha0 : float
        IBP concentration parameter.
    """

    def __init__(self, k_max: int = 100, alpha0: float = 5.0):
        super().__init__()
        self.k_max = k_max
        # Log-space parameterization for positivity
        self.log_alpha0 = nn.Parameter(torch.tensor(math.log(alpha0)))

    @property
    def alpha0(self) -> torch.Tensor:
        return self.log_alpha0.exp()

    def prior_pi(self) -> Beta:
        """Beta(alpha0/K, 1) prior on global activation probabilities."""
        a = self.alpha0 / self.k_max
        return Beta(a.expand(self.k_max), torch.ones(self.k_max, device=a.device))

    def expected_active_atoms(self, n: int) -> float:
        """Expected number of active features under the IBP for N observations."""
        return self.alpha0.item() * math.log(n)

    def kl_pi(self, q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
        """KL(q(pi) || p(pi)) where q(pi) = Beta(q_a, q_b)."""
        q_dist = Beta(q_a, q_b)
        p_dist = self.prior_pi()
        return torch.distributions.kl_divergence(q_dist, p_dist).sum()


class HierarchicalBetaBernoulli(nn.Module):
    """Two-level Beta-Bernoulli activation prior.

    Global:  pi_k     ~ Beta(alpha0/K, 1)  [IBP limit]
    Site:    pi_{jk}  ~ Beta(a * pi_k, a * (1 - pi_k))
    Sample:  z_{ijk}  ~ Bernoulli(pi_{jk})

    Parameters
    ----------
    k_max : int
        Maximum number of atoms.
    n_sites : int
        Number of sites/groups.
    concentration : float
        Site-level concentration parameter `a`. High = sites close to global.
    """

    def __init__(self, k_max: int, n_sites: int, concentration: float = 10.0):
        super().__init__()
        self.k_max = k_max
        self.n_sites = n_sites

        # Variational parameters for global pi: Beta(q_a_global, q_b_global)
        self.q_a_global = nn.Parameter(torch.ones(k_max))
        self.q_b_global = nn.Parameter(torch.ones(k_max))

        # Variational parameters for site-level pi: Beta(q_a_site, q_b_site)
        self.q_a_site = nn.Parameter(torch.ones(n_sites, k_max))
        self.q_b_site = nn.Parameter(torch.ones(n_sites, k_max))

        self.log_concentration = nn.Parameter(torch.tensor(math.log(concentration)))

    @property
    def concentration(self) -> torch.Tensor:
        return self.log_concentration.exp()

    def q_global_mean(self) -> torch.Tensor:
        """E[pi_k] under q(pi_k) = Beta(a, b)."""
        a = torch.nn.functional.softplus(self.q_a_global)
        b = torch.nn.functional.softplus(self.q_b_global)
        return a / (a + b)

    def q_site_mean(self, site_idx: int) -> torch.Tensor:
        """E[pi_{jk}] under q(pi_{jk})."""
        a = torch.nn.functional.softplus(self.q_a_site[site_idx])
        b = torch.nn.functional.softplus(self.q_b_site[site_idx])
        return a / (a + b)

    def sample_activation_probs(self, site_idx: int) -> torch.Tensor:
        """Sample site-level activation probabilities from the variational posterior."""
        a = torch.nn.functional.softplus(self.q_a_site[site_idx])
        b = torch.nn.functional.softplus(self.q_b_site[site_idx])
        return Beta(a, b).rsample()


class HierarchicalDictionaryPrior(nn.Module):
    """Gaussian hierarchical prior over dictionary atoms.

    Global:  d^0_k ~ N(0, alpha^{-1} I)
    Site:    d_{jk} | d^0_k ~ N(d^0_k, lambda^{-1} I)

    Parameters
    ----------
    data_dim : int
        Dimensionality of each atom.
    k_max : int
        Maximum number of atoms.
    n_sites : int
        Number of sites.
    alpha : float
        Precision for global atom prior.
    """

    def __init__(self, data_dim: int, k_max: int, n_sites: int, alpha: float = 1.0):
        super().__init__()
        self.data_dim = data_dim
        self.k_max = k_max
        self.n_sites = n_sites

        # Global dictionary atoms D^0: (k_max, data_dim)
        self.D_global = nn.Parameter(torch.randn(k_max, data_dim) * (1.0 / math.sqrt(alpha)))

        # Site-specific offsets delta_j: D^(j) = D^0 + delta_j
        self.D_site_offsets = nn.Parameter(torch.zeros(n_sites, k_max, data_dim))

        # Learnable precision for site deviations
        self.log_lambda = nn.Parameter(torch.tensor(math.log(10.0)))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha)))

    @property
    def lam(self) -> torch.Tensor:
        return self.log_lambda.exp()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def get_site_dictionary(self, site_idx: int) -> torch.Tensor:
        """Return D^(j) = D^0 + delta_j for site j. Shape: (k_max, data_dim)."""
        return self.D_global + self.D_site_offsets[site_idx]

    def kl_dictionary(self) -> torch.Tensor:
        """KL divergence for global atoms and site offsets.

        KL(q(D^0) || p(D^0)) + sum_j KL(q(D^j) || p(D^j | D^0))

        Since we use point estimates (MAP) for D, this reduces to the
        log-prior penalty terms.
        """
        # Global prior: -alpha/2 * ||D^0||^2
        kl_global = 0.5 * self.alpha * (self.D_global ** 2).sum()

        # Site prior: -lambda/2 * ||delta_j||^2
        kl_site = 0.5 * self.lam * (self.D_site_offsets ** 2).sum()

        return kl_global + kl_site
