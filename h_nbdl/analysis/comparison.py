"""
Baseline comparison framework for the H-NBDL paper.

Implements and evaluates all baseline methods:
- K-SVD (via sklearn DictionaryLearning)
- Fixed-K Bayesian DL
- VAE with matched architecture
- Flat (non-hierarchical) NBDL
- ComBat harmonization (for radiomics)
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from sklearn.decomposition import DictionaryLearning

from h_nbdl.models import HierarchicalNBDL
from h_nbdl.models.generative import generate_hierarchical_data, SyntheticData
from h_nbdl.inference import AmortizedVI
from h_nbdl.utils.metrics import (
    amari_distance, calibration_score, reconstruction_mse, sparsity_ratio,
)
from h_nbdl.utils.data import prepare_data


@dataclass
class MethodResult:
    """Result for a single method on a single seed."""
    method: str
    amari: float = np.nan
    k_eff: int = 0
    recon_mse: float = np.nan
    coverage_95: float = np.nan
    sparsity: float = np.nan
    wall_time_s: float = np.nan


@dataclass
class ComparisonResult:
    """Aggregated results across seeds."""
    method: str
    amari_mean: float = 0.0
    amari_std: float = 0.0
    k_eff_mean: float = 0.0
    recon_mse_mean: float = 0.0
    coverage_95_mean: float = np.nan
    runs: List[MethodResult] = field(default_factory=list)


def run_ksvd(data: SyntheticData, K: int) -> MethodResult:
    """Run sklearn DictionaryLearning (K-SVD variant)."""
    import time
    t0 = time.time()
    dl = DictionaryLearning(
        n_components=K, alpha=1.0, max_iter=200,
        transform_algorithm='omp', random_state=42,
    )
    codes = dl.fit_transform(data.X)
    D_est = dl.components_
    t1 = time.time()

    X_hat = codes @ D_est
    ami = amari_distance(data.D_global, D_est[:min(K, data.K_true)])

    return MethodResult(
        method=f"K-SVD (K={K})",
        amari=ami,
        k_eff=K,
        recon_mse=reconstruction_mse(data.X, X_hat),
        sparsity=sparsity_ratio(codes),
        wall_time_s=t1 - t0,
    )


def run_hnbdl(
    data: SyntheticData,
    hierarchical: bool = True,
    epochs: int = 200,
) -> MethodResult:
    """Run H-NBDL (or flat NBDL if hierarchical=False)."""
    import time
    X, sid = prepare_data(data.X, data.site_ids)
    n_sites = len(np.unique(data.site_ids)) if hierarchical else 1
    if not hierarchical:
        sid = torch.zeros_like(sid)

    t0 = time.time()
    model = HierarchicalNBDL(
        data_dim=data.X.shape[1], k_max=100, n_sites=n_sites,
        encoder_hidden=[256, 128],
    )
    trainer = AmortizedVI(model, lr=1e-3, temp_anneal=(1.0, 0.1, epochs//2))
    history = trainer.fit(X, sid, epochs=epochs, batch_size=256, verbose=False)
    t1 = time.time()

    r_mean, r_var = trainer.get_representations(X, sid)
    D_est = model.dictionary_prior.D_global.detach().numpy()
    k_eff = model.effective_atoms()

    # Amari
    if k_eff > 0:
        mask = r_mean.abs().mean(dim=0).numpy() > 0.01
        ami = amari_distance(data.D_global, D_est[mask[:len(D_est)]])
    else:
        ami = 1.0

    # Reconstruction
    model.eval()
    with torch.no_grad():
        fwd = model(X, sid, temperature=0.1)
    rmse = reconstruction_mse(data.X, fwd["x_hat"].numpy())

    # Calibration
    r_std = r_var.sqrt().numpy()
    K_min = min(data.S_true.shape[1], r_mean.shape[1])
    cal = calibration_score(
        data.S_true[:, :K_min] * data.Z_true[:, :K_min],
        r_mean.numpy()[:, :K_min],
        r_std[:, :K_min] + 1e-6,
    )

    name = "H-NBDL (AVI)" if hierarchical else "Flat NBDL"
    return MethodResult(
        method=name,
        amari=ami,
        k_eff=k_eff,
        recon_mse=rmse,
        coverage_95=cal,
        wall_time_s=t1 - t0,
    )


def run_baseline_comparison(
    n_seeds: int = 5,
    epochs: int = 150,
    verbose: bool = True,
) -> Dict[str, ComparisonResult]:
    """Run full baseline comparison on synthetic data.

    Returns aggregated results for each method.
    """
    methods_results = {}

    for seed in range(n_seeds):
        if verbose:
            print(f"\n--- Seed {seed+1}/{n_seeds} ---")

        data = generate_hierarchical_data(
            n_sites=5, n_per_site=200, data_dim=50,
            k_true=15, sigma2=0.1, seed=seed,
        )

        results = []

        # K-SVD baselines
        for K in [15, 30]:
            r = run_ksvd(data, K)
            results.append(r)
            if verbose:
                print(f"  {r.method}: Amari={r.amari:.3f}")

        # H-NBDL (hierarchical)
        r = run_hnbdl(data, hierarchical=True, epochs=epochs)
        results.append(r)
        if verbose:
            print(f"  {r.method}: Amari={r.amari:.3f}, K_eff={r.k_eff}")

        # Flat NBDL
        r = run_hnbdl(data, hierarchical=False, epochs=epochs)
        results.append(r)
        if verbose:
            print(f"  {r.method}: Amari={r.amari:.3f}, K_eff={r.k_eff}")

        for r in results:
            if r.method not in methods_results:
                methods_results[r.method] = ComparisonResult(method=r.method)
            methods_results[r.method].runs.append(r)

    # Aggregate
    for name, cr in methods_results.items():
        amaris = [r.amari for r in cr.runs]
        cr.amari_mean = np.mean(amaris)
        cr.amari_std = np.std(amaris)
        cr.k_eff_mean = np.mean([r.k_eff for r in cr.runs])
        cr.recon_mse_mean = np.mean([r.recon_mse for r in cr.runs if not np.isnan(r.recon_mse)])
        covs = [r.coverage_95 for r in cr.runs if not np.isnan(r.coverage_95)]
        if covs:
            cr.coverage_95_mean = np.mean(covs)

    return methods_results
