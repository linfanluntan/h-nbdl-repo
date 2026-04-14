"""
Data loading and preprocessing utilities.
"""

import numpy as np
import torch
from typing import Tuple, Optional


def prepare_data(
    X: np.ndarray,
    site_ids: np.ndarray,
    normalize: bool = True,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare data for H-NBDL training.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
    site_ids : np.ndarray, shape (N,)
    normalize : bool
        If True, standardize features per-site (zero mean, unit variance).
    device : str

    Returns
    -------
    X_tensor : Tensor
    site_tensor : Tensor
    """
    X = X.astype(np.float32).copy()
    site_ids = site_ids.astype(int)

    if normalize:
        for j in np.unique(site_ids):
            mask = site_ids == j
            X[mask] -= X[mask].mean(axis=0)
            std = X[mask].std(axis=0)
            std[std < 1e-8] = 1.0
            X[mask] /= std

    return (
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(site_ids, dtype=torch.long, device=device),
    )


def train_val_split(
    X: np.ndarray,
    site_ids: np.ndarray,
    val_fraction: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/validation split preserving site proportions.

    Returns
    -------
    X_train, site_train, X_val, site_val
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for j in np.unique(site_ids):
        idx_j = np.where(site_ids == j)[0]
        rng.shuffle(idx_j)
        n_val = max(1, int(len(idx_j) * val_fraction))
        val_idx.extend(idx_j[:n_val])
        train_idx.extend(idx_j[n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    return X[train_idx], site_ids[train_idx], X[val_idx], site_ids[val_idx]
