"""
Visualization utilities for H-NBDL.

Plotting functions for dictionary atoms, activation patterns,
posterior diagnostics, and training curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


def plot_dictionary_atoms(
    D: np.ndarray,
    n_cols: int = 5,
    atom_shape: Optional[tuple] = None,
    title: str = "Learned Dictionary Atoms",
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Plot dictionary atoms as small images or bar plots.

    Parameters
    ----------
    D : np.ndarray, shape (K, D)
        Dictionary atoms.
    n_cols : int
        Number of columns in the grid.
    atom_shape : tuple or None
        If provided, reshape each atom to this 2D shape for image display.
    title : str
    figsize : tuple or None
    """
    K = D.shape[0]
    n_rows = int(np.ceil(K / n_cols))
    if figsize is None:
        figsize = (n_cols * 2, n_rows * 2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        if idx < K:
            atom = D[idx]
            if atom_shape is not None:
                ax.imshow(atom.reshape(atom_shape), cmap="RdBu_r", aspect="auto")
            else:
                ax.bar(range(len(atom)), atom, color="steelblue", width=1.0)
            ax.set_title(f"Atom {idx}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_activation_heatmap(
    Z: np.ndarray,
    site_ids: np.ndarray,
    title: str = "Feature Activation Matrix",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot the binary activation matrix Z as a heatmap, sorted by site.

    Parameters
    ----------
    Z : np.ndarray, shape (N, K)
    site_ids : np.ndarray, shape (N,)
    """
    sort_idx = np.argsort(site_ids)
    Z_sorted = Z[sort_idx]
    sites_sorted = site_ids[sort_idx]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(Z_sorted.T, aspect="auto", cmap="Greys", interpolation="nearest")

    # Mark site boundaries
    boundaries = np.where(np.diff(sites_sorted))[0]
    for b in boundaries:
        ax.axvline(b + 0.5, color="red", linewidth=0.5, alpha=0.7)

    ax.set_xlabel("Samples (sorted by site)")
    ax.set_ylabel("Dictionary Atoms")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    return fig


def plot_training_curves(
    history: List[Dict[str, float]],
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """Plot training diagnostics from AmortizedVI.

    Shows loss, reconstruction, KL terms, effective K, and temperature.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    epochs = range(len(history))

    # Loss
    axes[0, 0].plot(epochs, [h["loss"] for h in history], "b-")
    if "val_loss" in history[0]:
        axes[0, 0].plot(epochs, [h.get("val_loss", np.nan) for h in history], "r--")
        axes[0, 0].legend(["Train", "Val"])
    axes[0, 0].set_title("Total Loss (neg. ELBO)")
    axes[0, 0].set_xlabel("Epoch")

    # Reconstruction
    axes[0, 1].plot(epochs, [h["recon_loss"] for h in history], "g-")
    axes[0, 1].set_title("Reconstruction Loss")
    axes[0, 1].set_xlabel("Epoch")

    # KL terms
    axes[0, 2].plot(epochs, [h["kl_s"] for h in history], label="KL(s)")
    axes[0, 2].plot(epochs, [h["kl_z"] for h in history], label="KL(z)")
    axes[0, 2].plot(epochs, [h["kl_dict"] for h in history], label="KL(D)")
    axes[0, 2].legend()
    axes[0, 2].set_title("KL Divergence Terms")
    axes[0, 2].set_xlabel("Epoch")

    # Effective K
    axes[1, 0].plot(epochs, [h["k_effective"] for h in history], "m-")
    axes[1, 0].set_title("Effective # Atoms")
    axes[1, 0].set_xlabel("Epoch")

    # Sigma^2
    axes[1, 1].plot(epochs, [h["sigma2"] for h in history], "k-")
    axes[1, 1].set_title("Noise Variance σ²")
    axes[1, 1].set_xlabel("Epoch")

    # Temperature
    axes[1, 2].plot(epochs, [h["temperature"] for h in history], "orange")
    axes[1, 2].set_title("Concrete Temperature")
    axes[1, 2].set_xlabel("Epoch")

    plt.tight_layout()
    return fig


def plot_gibbs_trace(
    K_trace: List[int],
    sigma2_trace: List[float],
    alpha0_trace: List[float],
    burnin: int = 0,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """Plot trace plots for Gibbs sampler diagnostics."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].plot(K_trace, "b-", alpha=0.7)
    axes[0].axvline(burnin, color="red", linestyle="--", label="Burn-in")
    axes[0].set_title("Active Atoms K")
    axes[0].legend()

    axes[1].plot(sigma2_trace, "g-", alpha=0.7)
    axes[1].axvline(burnin, color="red", linestyle="--")
    axes[1].set_title("Noise Variance σ²")

    axes[2].plot(alpha0_trace, "m-", alpha=0.7)
    axes[2].axvline(burnin, color="red", linestyle="--")
    axes[2].set_title("IBP Concentration α₀")

    plt.tight_layout()
    return fig
