"""
Site-aware neural encoder for amortized variational inference.

Maps (x_{ij}, site_id_j) -> (mu, log_sigma, logit_z) for approximate
posterior q(s_{ij}, z_{ij} | x_{ij}, j).
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class SiteAwareEncoder(nn.Module):
    """Encoder network conditioned on observation and site embedding.

    Architecture: MLP with GELU activations and LayerNorm, producing
    three output heads: code mean, code log-variance, and activation logits.

    Parameters
    ----------
    data_dim : int
        Input observation dimensionality.
    k_max : int
        Maximum number of dictionary atoms (output dimension per head).
    n_sites : int
        Number of sites.
    hidden_dims : list of int
        Hidden layer widths.
    site_embed_dim : int
        Dimensionality of the learned site embedding.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        data_dim: int,
        k_max: int,
        n_sites: int,
        hidden_dims: List[int] = None,
        site_embed_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.k_max = k_max

        # Site embedding table
        self.site_embedding = nn.Embedding(n_sites, site_embed_dim)

        # Build MLP backbone
        input_dim = data_dim + site_embed_dim
        layers = []
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            input_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        # Output heads
        self.head_mu = nn.Linear(hidden_dims[-1], k_max)       # code mean
        self.head_logvar = nn.Linear(hidden_dims[-1], k_max)    # code log-variance
        self.head_logit_z = nn.Linear(hidden_dims[-1], k_max)   # activation logits

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize logit head bias to slightly negative (sparse prior)
        nn.init.constant_(self.head_logit_z.bias, -1.0)

    def forward(
        self, x: torch.Tensor, site_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode observations into variational parameters.

        Parameters
        ----------
        x : Tensor, shape (batch, data_dim)
            Input observations.
        site_ids : Tensor, shape (batch,), dtype long
            Site index for each observation.

        Returns
        -------
        mu : Tensor, shape (batch, k_max)
            Posterior mean of continuous codes.
        logvar : Tensor, shape (batch, k_max)
            Posterior log-variance of continuous codes.
        logit_z : Tensor, shape (batch, k_max)
            Pre-sigmoid logits for activation probabilities.
        """
        e_j = self.site_embedding(site_ids)       # (batch, site_embed_dim)
        h = torch.cat([x, e_j], dim=-1)           # (batch, data_dim + site_embed_dim)
        h = self.backbone(h)                       # (batch, hidden_dims[-1])

        mu = self.head_mu(h)
        logvar = self.head_logvar(h)
        logit_z = self.head_logit_z(h)

        return mu, logvar, logit_z
