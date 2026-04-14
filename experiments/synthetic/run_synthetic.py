"""
Synthetic Benchmark Experiment for H-NBDL.

Generates data from the H-NBDL generative model with known ground truth
and evaluates dictionary recovery, calibration, and reconstruction under
both Gibbs and AVI inference.

Usage:
    python experiments/synthetic/run_synthetic.py --config configs/synthetic.yaml
"""

import argparse
import yaml
import numpy as np
import torch
import json
from pathlib import Path

from h_nbdl.models.generative import generate_hierarchical_data
from h_nbdl.models import HierarchicalNBDL
from h_nbdl.inference import AmortizedVI, CollapsedGibbs
from h_nbdl.utils.metrics import amari_distance, calibration_score, reconstruction_mse
from h_nbdl.utils.data import prepare_data, train_val_split


def run_experiment(config: dict) -> dict:
    """Run the synthetic benchmark experiment.

    Parameters
    ----------
    config : dict
        Experiment configuration.

    Returns
    -------
    results : dict
        Dictionary of all evaluation metrics.
    """
    seed = config.get("seed", 42)
    results = {}

    # ── Generate synthetic data ──
    print("Generating synthetic data...")
    data = generate_hierarchical_data(
        n_sites=config["data"]["n_sites"],
        n_per_site=config["data"]["n_per_site"],
        data_dim=config["data"]["data_dim"],
        k_true=config["data"]["k_true"],
        sigma2=config["data"]["sigma2"],
        lambda_inv=config["data"]["lambda_inv"],
        seed=seed,
    )
    print(f"  N={data.X.shape[0]}, D={data.X.shape[1]}, K_true={data.K_true}")
    print(f"  Sites: {data.X.shape[0] // config['data']['n_per_site']}")

    # ── Train/val split ──
    X_train, sid_train, X_val, sid_val = train_val_split(
        data.X, data.site_ids, val_fraction=0.15, seed=seed
    )

    # ── Amortized Variational Inference ──
    if config.get("run_avi", True):
        print("\n── Amortized Variational Inference ──")
        avi_cfg = config["avi"]

        model = HierarchicalNBDL(
            data_dim=config["data"]["data_dim"],
            k_max=avi_cfg["k_max"],
            n_sites=config["data"]["n_sites"],
            encoder_hidden=avi_cfg["encoder_hidden"],
            site_embed_dim=avi_cfg["site_embed_dim"],
            alpha0=avi_cfg["alpha0"],
        )

        X_t, sid_t = prepare_data(X_train, sid_train)
        X_v, sid_v = prepare_data(X_val, sid_val)

        trainer = AmortizedVI(
            model,
            lr=avi_cfg["lr"],
            temp_anneal=(
                avi_cfg["temp_init"],
                avi_cfg["temp_final"],
                avi_cfg["temp_anneal_epochs"],
            ),
        )

        history = trainer.fit(
            X_t, sid_t,
            epochs=avi_cfg["epochs"],
            batch_size=avi_cfg["batch_size"],
            val_X=X_v, val_site_ids=sid_v,
        )

        # Evaluate
        r_mean, r_var = trainer.get_representations(
            torch.tensor(data.X, dtype=torch.float32),
            torch.tensor(data.site_ids, dtype=torch.long),
        )

        # Dictionary recovery
        D_est = model.dictionary_prior.D_global.detach().cpu().numpy()
        active_mask = r_mean.abs().mean(dim=0).numpy() > 0.01
        D_est_active = D_est[active_mask]

        if D_est_active.shape[0] > 0:
            ami = amari_distance(data.D_global, D_est_active)
        else:
            ami = 1.0

        # Reconstruction
        model.eval()
        with torch.no_grad():
            fwd = model(
                torch.tensor(data.X, dtype=torch.float32),
                torch.tensor(data.site_ids, dtype=torch.long),
                temperature=0.1,
            )
        recon_mse = reconstruction_mse(data.X, fwd["x_hat"].numpy())

        # Calibration
        r_std = r_var.sqrt().numpy()
        r_mean_np = r_mean.numpy()
        # Calibration of code estimates vs true codes
        K_min = min(data.S_true.shape[1], r_mean_np.shape[1])
        cal = calibration_score(
            data.S_true[:, :K_min] * data.Z_true[:, :K_min],
            r_mean_np[:, :K_min],
            r_std[:, :K_min] + 1e-6,
        )

        k_eff = model.effective_atoms()

        results["avi"] = {
            "amari_distance": float(ami),
            "reconstruction_mse": float(recon_mse),
            "calibration_95": float(cal),
            "k_effective": int(k_eff),
            "k_true": data.K_true,
            "final_loss": history[-1]["loss"],
        }
        print(f"  Amari distance: {ami:.4f}")
        print(f"  Reconstruction MSE: {recon_mse:.4f}")
        print(f"  95% CI coverage: {cal:.3f}")
        print(f"  K_effective: {k_eff} (true: {data.K_true})")

    # ── Collapsed Gibbs Sampling ──
    if config.get("run_gibbs", True):
        print("\n── Collapsed Gibbs Sampling ──")
        gibbs_cfg = config["gibbs"]

        sampler = CollapsedGibbs(
            data.X, data.site_ids,
            k_max=gibbs_cfg["k_max"],
            alpha0=gibbs_cfg["alpha0"],
            n_iter=gibbs_cfg["n_iter"],
            burnin=gibbs_cfg["burnin"],
            thin=gibbs_cfg["thin"],
            seed=seed,
        )

        samples = sampler.run()

        k_eff_gibbs = samples.effective_K()
        D_gibbs = samples.dictionary_mean()
        active_gibbs = np.where(np.abs(D_gibbs).sum(axis=1) > 0.01)[0]

        if len(active_gibbs) > 0:
            ami_gibbs = amari_distance(data.D_global, D_gibbs[active_gibbs])
        else:
            ami_gibbs = 1.0

        results["gibbs"] = {
            "amari_distance": float(ami_gibbs),
            "k_effective_mean": float(k_eff_gibbs),
            "k_true": data.K_true,
            "final_sigma2": float(samples.sigma2_trace[-1]),
        }
        print(f"  Amari distance: {ami_gibbs:.4f}")
        print(f"  K_effective (posterior mean): {k_eff_gibbs:.1f} (true: {data.K_true})")

    return results


def main():
    parser = argparse.ArgumentParser(description="H-NBDL Synthetic Benchmark")
    parser.add_argument("--config", type=str, default="configs/synthetic.yaml")
    parser.add_argument("--output", type=str, default="results/synthetic_results.json")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            "seed": 42,
            "run_avi": True,
            "run_gibbs": True,
            "data": {
                "n_sites": 5,
                "n_per_site": 200,
                "data_dim": 50,
                "k_true": 15,
                "sigma2": 0.1,
                "lambda_inv": 0.1,
            },
            "avi": {
                "k_max": 100,
                "alpha0": 5.0,
                "encoder_hidden": [256, 128],
                "site_embed_dim": 32,
                "lr": 1e-3,
                "epochs": 200,
                "batch_size": 256,
                "temp_init": 1.0,
                "temp_final": 0.1,
                "temp_anneal_epochs": 100,
            },
            "gibbs": {
                "k_max": 100,
                "alpha0": 5.0,
                "n_iter": 500,
                "burnin": 250,
                "thin": 5,
            },
        }

    results = run_experiment(config)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
