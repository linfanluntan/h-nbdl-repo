"""
Amortized Variational Inference for H-NBDL.

Trains the encoder and generative model end-to-end by maximizing
the Evidence Lower Bound (ELBO) via stochastic gradient ascent.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm
import math

from h_nbdl.models.hierarchical_nbdl import HierarchicalNBDL
from h_nbdl.inference.concrete import temperature_schedule


class AmortizedVI:
    """Amortized variational inference trainer for H-NBDL.

    Parameters
    ----------
    model : HierarchicalNBDL
        The model to train.
    lr : float
        Learning rate.
    weight_decay : float
        AdamW weight decay.
    temp_anneal : tuple of (float, float, int)
        (initial_temp, final_temp, anneal_epochs).
    beta_anneal : tuple of (float, float, int) or None
        Optional KL annealing (beta-VAE style).
    device : str
        Device to train on.
    """

    def __init__(
        self,
        model: HierarchicalNBDL,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        temp_anneal: Tuple[float, float, int] = (1.0, 0.1, 100),
        beta_anneal: Optional[Tuple[float, float, int]] = None,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.temp_init, self.temp_final, self.temp_anneal_epochs = temp_anneal
        self.beta_anneal = beta_anneal
        self.history: List[Dict[str, float]] = []

    def _get_temperature(self, epoch: int, total_epochs: int) -> float:
        return temperature_schedule(
            epoch, total_epochs,
            self.temp_init, self.temp_final,
            self.temp_anneal_epochs / total_epochs if total_epochs > 0 else 0.5,
        )

    def _get_beta(self, epoch: int, total_epochs: int) -> float:
        if self.beta_anneal is None:
            return 1.0
        b_init, b_final, b_epochs = self.beta_anneal
        if epoch >= b_epochs:
            return b_final
        t = epoch / b_epochs
        return b_init + (b_final - b_init) * t

    def fit(
        self,
        X: torch.Tensor,
        site_ids: torch.Tensor,
        epochs: int = 200,
        batch_size: int = 256,
        val_X: Optional[torch.Tensor] = None,
        val_site_ids: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """Train the model.

        Parameters
        ----------
        X : Tensor, shape (N, D)
            Training observations.
        site_ids : Tensor, shape (N,), dtype long
            Site assignments.
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size.
        val_X, val_site_ids : Tensor or None
            Optional validation set.
        verbose : bool
            Print progress.

        Returns
        -------
        history : list of dict
            Per-epoch training diagnostics.
        """
        X = X.float().to(self.device)
        site_ids = site_ids.long().to(self.device)

        dataset = TensorDataset(X, site_ids)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        self.history = []
        iterator = range(epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training H-NBDL")

        for epoch in iterator:
            self.model.train()
            temperature = self._get_temperature(epoch, epochs)
            beta = self._get_beta(epoch, epochs)

            epoch_loss = 0.0
            epoch_diag = {}
            n_batches = 0

            for x_batch, sid_batch in loader:
                self.optimizer.zero_grad()

                fwd = self.model(x_batch, sid_batch, temperature=temperature)
                loss, diag = self.model.elbo(x_batch, sid_batch, fwd, beta=beta)

                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                for k, v in diag.items():
                    epoch_diag[k] = epoch_diag.get(k, 0.0) + v
                n_batches += 1

            scheduler.step()

            # Average diagnostics
            epoch_loss /= n_batches
            for k in epoch_diag:
                epoch_diag[k] /= n_batches
            epoch_diag["loss"] = epoch_loss
            epoch_diag["temperature"] = temperature
            epoch_diag["beta"] = beta
            epoch_diag["lr"] = scheduler.get_last_lr()[0]

            # Validation
            if val_X is not None:
                val_loss = self._validate(val_X, val_site_ids, temperature, beta)
                epoch_diag["val_loss"] = val_loss

            self.history.append(epoch_diag)

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    loss=f"{epoch_loss:.4f}",
                    K=f"{epoch_diag['k_effective']:.0f}",
                    T=f"{temperature:.3f}",
                )

        return self.history

    @torch.no_grad()
    def _validate(
        self,
        X: torch.Tensor,
        site_ids: torch.Tensor,
        temperature: float,
        beta: float,
    ) -> float:
        self.model.eval()
        X = X.float().to(self.device)
        site_ids = site_ids.long().to(self.device)
        fwd = self.model(X, site_ids, temperature=temperature)
        loss, _ = self.model.elbo(X, site_ids, fwd, beta=beta)
        return loss.item()

    def get_representations(
        self, X: torch.Tensor, site_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract representations from the trained model.

        Returns
        -------
        r_mean : Tensor, shape (N, k_max)
        r_var : Tensor, shape (N, k_max)
        """
        X = X.float().to(self.device)
        site_ids = site_ids.long().to(self.device)
        return self.model.encode(X, site_ids)
