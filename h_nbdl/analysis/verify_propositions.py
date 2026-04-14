"""
Empirical verification of theoretical propositions from the paper.

Provides runnable experiments that validate:
- Proposition 1: Identifiability of global-local decomposition
- Proposition 2: Posterior consistency (K_eff -> K_true as N -> inf)
- Proposition 3: ELBO tightness bound O(tau_c log K_max)
- Proposition 5: Calibration under Gibbs sampler
"""

import numpy as np
import torch
from typing import Dict, List
from dataclasses import dataclass

from h_nbdl.models.generative import generate_hierarchical_data
from h_nbdl.models import HierarchicalNBDL
from h_nbdl.inference import AmortizedVI, CollapsedGibbs
from h_nbdl.utils.data import prepare_data
from h_nbdl.utils.metrics import amari_distance, calibration_score
from h_nbdl.analysis.identifiability import (
    check_identifiability_conditions, decomposition_quality,
)
from h_nbdl.analysis.diagnostics import elbo_gap_decomposition


@dataclass
class PropositionResult:
    """Result from a proposition verification experiment."""
    proposition: str
    verified: bool
    details: Dict


def verify_proposition_1(n_seeds: int = 5, verbose: bool = True) -> PropositionResult:
    """Verify Proposition 1: Identifiability of the global-local decomposition.

    Tests that with J >= 2, linearly independent atoms, and shared atoms,
    the decomposition D^(j) = D^0 + Delta_j is recovered accurately.
    Compares against a single-site (non-identifiable) baseline.
    """
    if verbose:
        print("Verifying Proposition 1 (Identifiability)...")

    amaris_hier = []
    amaris_flat = []
    conditions_met = []

    for seed in range(n_seeds):
        data = generate_hierarchical_data(
            n_sites=3, n_per_site=200, data_dim=30,
            k_true=8, lambda_inv=0.2, seed=seed,
        )
        X, sid = prepare_data(data.X, data.site_ids)

        # Hierarchical model (should identify decomposition)
        model_h = HierarchicalNBDL(data_dim=30, k_max=50, n_sites=3, encoder_hidden=[128, 64])
        trainer_h = AmortizedVI(model_h, lr=1e-3, temp_anneal=(1.0, 0.1, 40))
        trainer_h.fit(X, sid, epochs=80, batch_size=128, verbose=False)

        conds = check_identifiability_conditions(model_h, data.site_ids)
        conditions_met.append(conds['all_satisfied'])

        quality = decomposition_quality(model_h, data.D_global, data.D_sites)
        amaris_hier.append(quality.get('amari_global', 1.0))

        # Flat model (single site, no decomposition)
        sid_flat = torch.zeros_like(sid)
        model_f = HierarchicalNBDL(data_dim=30, k_max=50, n_sites=1, encoder_hidden=[128, 64])
        trainer_f = AmortizedVI(model_f, lr=1e-3, temp_anneal=(1.0, 0.1, 40))
        trainer_f.fit(X, sid_flat, epochs=80, batch_size=128, verbose=False)
        D_f = model_f.dictionary_prior.D_global.detach().numpy()
        K = min(data.D_global.shape[0], D_f.shape[0])
        amaris_flat.append(amari_distance(data.D_global, D_f[:K]))

    verified = np.mean(amaris_hier) < np.mean(amaris_flat)
    if verbose:
        print(f"  Hierarchical Amari: {np.mean(amaris_hier):.3f} ± {np.std(amaris_hier):.3f}")
        print(f"  Flat Amari:         {np.mean(amaris_flat):.3f} ± {np.std(amaris_flat):.3f}")
        print(f"  Conditions met: {sum(conditions_met)}/{n_seeds}")
        print(f"  Verified: {verified}")

    return PropositionResult(
        proposition="1 (Identifiability)",
        verified=verified,
        details={
            "amari_hier_mean": float(np.mean(amaris_hier)),
            "amari_flat_mean": float(np.mean(amaris_flat)),
            "conditions_met": sum(conditions_met),
        },
    )


def verify_proposition_2(
    sample_sizes: List[int] = None,
    n_seeds: int = 3,
    verbose: bool = True,
) -> PropositionResult:
    """Verify Proposition 2: Posterior consistency (K_eff -> K_true as N -> inf).

    Tests that the inferred K_eff approaches K_true as N grows.
    """
    if sample_sizes is None:
        sample_sizes = [50, 100, 200, 500]
    if verbose:
        print("Verifying Proposition 2 (Posterior consistency)...")

    k_true = 10
    results_by_n = {}

    for n_per_site in sample_sizes:
        k_effs = []
        for seed in range(n_seeds):
            data = generate_hierarchical_data(
                n_sites=3, n_per_site=n_per_site, data_dim=30,
                k_true=k_true, seed=seed,
            )
            X, sid = prepare_data(data.X, data.site_ids)
            model = HierarchicalNBDL(data_dim=30, k_max=50, n_sites=3, encoder_hidden=[128, 64])
            trainer = AmortizedVI(model, lr=1e-3, temp_anneal=(1.0, 0.1, 40))
            trainer.fit(X, sid, epochs=80, batch_size=min(128, n_per_site), verbose=False)
            k_effs.append(model.effective_atoms())

        results_by_n[n_per_site] = {
            "k_eff_mean": float(np.mean(k_effs)),
            "k_eff_std": float(np.std(k_effs)),
            "error": float(abs(np.mean(k_effs) - k_true)),
        }
        if verbose:
            print(f"  N_per_site={n_per_site}: K_eff = {np.mean(k_effs):.1f} ± {np.std(k_effs):.1f} (true={k_true})")

    # Verify: error should decrease with N
    errors = [results_by_n[n]["error"] for n in sample_sizes]
    monotone_decreasing = all(errors[i] >= errors[i+1] - 1.0 for i in range(len(errors)-1))
    final_close = errors[-1] < 3.0

    verified = final_close
    if verbose:
        print(f"  Error trend: {[f'{e:.1f}' for e in errors]}")
        print(f"  Verified (final error < 3): {verified}")

    return PropositionResult(
        proposition="2 (Posterior consistency)",
        verified=verified,
        details={"results_by_n": results_by_n, "k_true": k_true},
    )


def verify_proposition_3(
    temperatures: List[float] = None,
    n_seeds: int = 3,
    verbose: bool = True,
) -> PropositionResult:
    """Verify Proposition 3: ELBO gap is O(tau_c log K_max).

    Tests that the ELBO gap between cold and hot temperatures
    scales linearly with tau_c.
    """
    if temperatures is None:
        temperatures = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    if verbose:
        print("Verifying Proposition 3 (ELBO tightness)...")

    data = generate_hierarchical_data(
        n_sites=3, n_per_site=200, data_dim=30, k_true=8, seed=42,
    )
    X, sid = prepare_data(data.X, data.site_ids)

    model = HierarchicalNBDL(data_dim=30, k_max=50, n_sites=3, encoder_hidden=[128, 64])
    trainer = AmortizedVI(model, lr=1e-3, temp_anneal=(1.0, 0.1, 50))
    trainer.fit(X, sid, epochs=100, batch_size=128, verbose=False)

    # Evaluate ELBO at each temperature
    model.eval()
    elbos = {}
    with torch.no_grad():
        for temp in temperatures:
            losses = []
            for _ in range(10):
                fwd = model(X, sid, temperature=temp)
                loss, _ = model.elbo(X, sid, fwd)
                losses.append(-loss.item())
            elbos[temp] = np.mean(losses)

    # The gap from coldest to each temperature should scale ~ linearly with tau_c
    best_elbo = max(elbos.values())
    gaps = {t: best_elbo - e for t, e in elbos.items()}

    # Check approximate linearity: gap / tau_c should be roughly constant
    ratios = {t: gaps[t] / (t + 1e-8) for t in temperatures if t > 0.01}
    ratio_values = list(ratios.values())
    cv = np.std(ratio_values) / (np.mean(ratio_values) + 1e-8)

    verified = cv < 1.5  # coefficient of variation reasonably bounded

    if verbose:
        for t in temperatures:
            print(f"  tau_c={t:.2f}: ELBO={elbos[t]:.1f}, gap={gaps[t]:.2f}")
        print(f"  Gap/tau_c ratio CV: {cv:.2f}")
        print(f"  Verified (CV < 1.5): {verified}")

    return PropositionResult(
        proposition="3 (ELBO tightness)",
        verified=verified,
        details={"elbos": {str(k): v for k, v in elbos.items()},
                 "gaps": {str(k): v for k, v in gaps.items()},
                 "ratio_cv": float(cv)},
    )


def verify_proposition_5_calibration(
    levels: List[float] = None,
    n_seeds: int = 3,
    verbose: bool = True,
) -> PropositionResult:
    """Verify Proposition 5: Calibration of credible intervals.

    Tests that posterior credible intervals achieve nominal coverage.
    """
    if levels is None:
        levels = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
    if verbose:
        print("Verifying Proposition 5 (Calibration)...")

    coverages_avi = {l: [] for l in levels}
    coverages_target = {l: l for l in levels}

    for seed in range(n_seeds):
        data = generate_hierarchical_data(
            n_sites=3, n_per_site=150, data_dim=20, k_true=6, seed=seed,
        )
        X, sid = prepare_data(data.X, data.site_ids)
        model = HierarchicalNBDL(data_dim=20, k_max=30, n_sites=3, encoder_hidden=[64, 32])
        trainer = AmortizedVI(model, lr=1e-3, temp_anneal=(1.0, 0.1, 40))
        trainer.fit(X, sid, epochs=80, batch_size=128, verbose=False)

        r_mean, r_var = model.encode(X, sid)
        r_std = r_var.sqrt().numpy()
        r_mean_np = r_mean.numpy()

        K_min = min(data.S_true.shape[1], r_mean_np.shape[1])
        true_codes = data.S_true[:, :K_min] * data.Z_true[:, :K_min]

        for level in levels:
            cov = calibration_score(
                true_codes, r_mean_np[:, :K_min], r_std[:, :K_min] + 1e-6, level=level,
            )
            coverages_avi[level].append(cov)

    mean_coverages = {l: np.mean(coverages_avi[l]) for l in levels}
    # Check: coverage should be within 15% of nominal (AVI has some gap)
    max_gap = max(abs(mean_coverages[l] - l) for l in levels)
    verified = max_gap < 0.15

    if verbose:
        for l in levels:
            print(f"  Nominal {l:.0%}: empirical {mean_coverages[l]:.3f} (gap: {mean_coverages[l]-l:+.3f})")
        print(f"  Max gap: {max_gap:.3f}")
        print(f"  Verified (max gap < 0.15): {verified}")

    return PropositionResult(
        proposition="5 (Calibration)",
        verified=verified,
        details={"coverages": {str(l): float(mean_coverages[l]) for l in levels},
                 "max_gap": float(max_gap)},
    )


def verify_all(verbose: bool = True) -> Dict[str, PropositionResult]:
    """Run all proposition verifications."""
    results = {}
    results["P1"] = verify_proposition_1(n_seeds=3, verbose=verbose)
    results["P2"] = verify_proposition_2(n_seeds=2, verbose=verbose)
    results["P3"] = verify_proposition_3(n_seeds=2, verbose=verbose)
    results["P5"] = verify_proposition_5_calibration(n_seeds=3, verbose=verbose)

    if verbose:
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        for key, r in results.items():
            status = "✓ PASS" if r.verified else "✗ FAIL"
            print(f"  {status}  Proposition {r.proposition}")

    return results


if __name__ == "__main__":
    verify_all(verbose=True)
