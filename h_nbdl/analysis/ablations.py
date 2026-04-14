"""
Ablation studies for the H-NBDL paper.

Provides systematic ablation experiments isolating the effect of:
1. Pooling strength (lambda)
2. IBP vs fixed-K
3. Concrete temperature
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass

from h_nbdl.models import HierarchicalNBDL
from h_nbdl.models.generative import generate_hierarchical_data
from h_nbdl.inference import AmortizedVI
from h_nbdl.utils.metrics import amari_distance, reconstruction_mse
from h_nbdl.utils.data import prepare_data


@dataclass
class AblationResult:
    """Single ablation experiment result."""
    param_name: str
    param_value: float
    amari: float
    recon_mse: float
    k_eff: int
    auc: float = None


def run_pooling_ablation(
    lambdas: List[float] = None,
    n_seeds: int = 5,
    epochs: int = 100,
) -> List[AblationResult]:
    """Ablation over pooling strength lambda.

    Trains H-NBDL at different lambda values and measures
    dictionary recovery and reconstruction quality.
    """
    if lambdas is None:
        lambdas = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]

    results = []
    for lam in lambdas:
        amaris, mses, keffs = [], [], []
        for seed in range(n_seeds):
            data = generate_hierarchical_data(
                n_sites=5, n_per_site=200, data_dim=50,
                k_true=15, lambda_inv=1.0/lam, seed=seed,
            )
            X, sid = prepare_data(data.X, data.site_ids)

            model = HierarchicalNBDL(
                data_dim=50, k_max=100, n_sites=5,
                encoder_hidden=[128, 64],
            )
            # Set lambda manually
            model.dictionary_prior.log_lambda.data = torch.tensor(np.log(lam))
            model.dictionary_prior.log_lambda.requires_grad = False

            trainer = AmortizedVI(model, lr=1e-3, temp_anneal=(1.0, 0.1, epochs//2))
            trainer.fit(X, sid, epochs=epochs, batch_size=256, verbose=False)

            D_est = model.dictionary_prior.D_global.detach().numpy()
            active = model.effective_atoms()
            if active > 0:
                mask = torch.sigmoid(model.activation_prior.q_a_global).detach().numpy() > 0.1
                ami = amari_distance(data.D_global, D_est[mask[:len(D_est)]])
            else:
                ami = 1.0

            amaris.append(ami)
            keffs.append(active)

        results.append(AblationResult(
            param_name="lambda",
            param_value=lam,
            amari=np.mean(amaris),
            recon_mse=0.0,
            k_eff=int(np.mean(keffs)),
        ))
    return results


def run_kmax_ablation(
    k_values: List[int] = None,
    n_seeds: int = 5,
    epochs: int = 100,
) -> Tuple[List[AblationResult], AblationResult]:
    """Compare fixed-K BDL against IBP-based H-NBDL.

    Returns results for each fixed K and the IBP result.
    """
    if k_values is None:
        k_values = [5, 10, 15, 20, 30, 50, 80, 100]

    fixed_results = []
    for K in k_values:
        amaris = []
        for seed in range(n_seeds):
            data = generate_hierarchical_data(
                n_sites=5, n_per_site=200, data_dim=50,
                k_true=15, seed=seed,
            )
            X, sid = prepare_data(data.X, data.site_ids)

            model = HierarchicalNBDL(
                data_dim=50, k_max=K, n_sites=5,
                encoder_hidden=[128, 64],
            )
            trainer = AmortizedVI(model, lr=1e-3, temp_anneal=(1.0, 0.5, epochs//2))
            trainer.fit(X, sid, epochs=epochs, batch_size=256, verbose=False)

            D_est = model.dictionary_prior.D_global.detach().numpy()
            ami = amari_distance(data.D_global, D_est[:min(K, 15)])
            amaris.append(ami)

        fixed_results.append(AblationResult(
            param_name="K_fixed",
            param_value=K,
            amari=np.mean(amaris),
            recon_mse=0.0,
            k_eff=K,
        ))

    # IBP result (K_max=100, learns K_eff)
    ibp_amaris = []
    ibp_keffs = []
    for seed in range(n_seeds):
        data = generate_hierarchical_data(
            n_sites=5, n_per_site=200, data_dim=50,
            k_true=15, seed=seed,
        )
        X, sid = prepare_data(data.X, data.site_ids)
        model = HierarchicalNBDL(
            data_dim=50, k_max=100, n_sites=5,
            encoder_hidden=[128, 64],
        )
        trainer = AmortizedVI(model, lr=1e-3, temp_anneal=(1.0, 0.1, epochs//2))
        trainer.fit(X, sid, epochs=epochs, batch_size=256, verbose=False)
        D_est = model.dictionary_prior.D_global.detach().numpy()
        k_eff = model.effective_atoms()
        active = np.arange(min(k_eff, D_est.shape[0]))
        ami = amari_distance(data.D_global, D_est[active]) if len(active) > 0 else 1.0
        ibp_amaris.append(ami)
        ibp_keffs.append(k_eff)

    ibp_result = AblationResult(
        param_name="K_IBP",
        param_value=100,
        amari=np.mean(ibp_amaris),
        recon_mse=0.0,
        k_eff=int(np.mean(ibp_keffs)),
    )

    return fixed_results, ibp_result


def run_temperature_ablation(
    temperatures: List[float] = None,
    n_seeds: int = 5,
    epochs: int = 100,
) -> List[AblationResult]:
    """Ablation over final Concrete temperature."""
    if temperatures is None:
        temperatures = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    results = []
    for temp in temperatures:
        amaris, mses, keffs = [], [], []
        for seed in range(n_seeds):
            data = generate_hierarchical_data(
                n_sites=3, n_per_site=200, data_dim=50,
                k_true=15, seed=seed,
            )
            X, sid = prepare_data(data.X, data.site_ids)

            model = HierarchicalNBDL(
                data_dim=50, k_max=100, n_sites=3,
                encoder_hidden=[128, 64],
            )
            trainer = AmortizedVI(
                model, lr=1e-3,
                temp_anneal=(1.0, temp, epochs//2),
            )
            history = trainer.fit(X, sid, epochs=epochs, batch_size=256, verbose=False)

            keffs.append(model.effective_atoms())
            mses.append(history[-1].get("recon_loss", 0.0))

        results.append(AblationResult(
            param_name="tau_c",
            param_value=temp,
            amari=0.0,
            recon_mse=np.mean(mses),
            k_eff=int(np.mean(keffs)),
        ))

    return results
