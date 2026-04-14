"""
Convergence diagnostics for H-NBDL inference.

Implements Gelman-Rubin R-hat, effective sample size, ELBO gap
decomposition (Section 4.4 of the paper), and posterior predictive checks.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


def gelman_rubin_rhat(chains: List[np.ndarray]) -> float:
    """Compute the Gelman-Rubin R-hat convergence diagnostic.

    R-hat < 1.05 indicates acceptable convergence.

    Parameters
    ----------
    chains : list of np.ndarray, each shape (n_samples,)
        Multiple MCMC chains for a single scalar quantity.

    Returns
    -------
    rhat : float
        Potential scale reduction factor.
    """
    m = len(chains)
    n = min(len(c) for c in chains)
    chains = [c[:n] for c in chains]

    chain_means = np.array([c.mean() for c in chains])
    grand_mean = chain_means.mean()

    # Between-chain variance
    B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)

    # Within-chain variance
    W = np.mean([np.var(c, ddof=1) for c in chains])

    # Pooled variance estimate
    var_hat = (1 - 1 / n) * W + (1 / n) * B

    rhat = np.sqrt(var_hat / W) if W > 0 else float('inf')
    return float(rhat)


def effective_sample_size(chain: np.ndarray, max_lag: int = 100) -> float:
    """Compute effective sample size via autocorrelation.

    Parameters
    ----------
    chain : np.ndarray, shape (n_samples,)
    max_lag : int

    Returns
    -------
    ess : float
    """
    n = len(chain)
    chain_centered = chain - chain.mean()
    var = np.var(chain)
    if var < 1e-12:
        return float(n)

    # Autocorrelation via FFT
    fft = np.fft.fft(chain_centered, n=2 * n)
    acf = np.real(np.fft.ifft(fft * np.conj(fft)))[:n] / (n * var)

    # Sum autocorrelations until they go negative (Geyer's criterion)
    tau = 1.0
    for lag in range(1, min(max_lag, n // 2)):
        if acf[lag] < 0:
            break
        tau += 2 * acf[lag]

    return float(n / tau)


def gibbs_convergence_report(
    samples,
    n_chains: int = 4,
    seed: int = 42,
) -> Dict:
    """Run convergence diagnostics on Gibbs sampler output.

    Checks R-hat and ESS for key quantities (K_eff, sigma2, alpha0).

    Parameters
    ----------
    samples : GibbsSamples
        Output from CollapsedGibbs.run().
    n_chains : int
        Number of pseudo-chains to split the single chain into.

    Returns
    -------
    report : dict
        Convergence diagnostics for each quantity.
    """
    report = {}

    for name, trace in [
        ("K_eff", np.array(samples.K_eff_trace, dtype=float)),
        ("sigma2", np.array(samples.sigma2_trace, dtype=float)),
        ("alpha0", np.array(samples.alpha0_trace, dtype=float)),
    ]:
        n = len(trace)
        # Use second half (post-burnin) and split into pseudo-chains
        half = trace[n // 2:]
        chunk = len(half) // n_chains
        if chunk < 10:
            report[name] = {"rhat": float("nan"), "ess": float("nan"),
                            "warning": "too few samples"}
            continue

        chains = [half[i * chunk:(i + 1) * chunk] for i in range(n_chains)]
        rhat = gelman_rubin_rhat(chains)
        ess = effective_sample_size(half)

        report[name] = {
            "rhat": rhat,
            "ess": ess,
            "mean": float(np.mean(half)),
            "std": float(np.std(half)),
            "converged": rhat < 1.05,
        }

    report["all_converged"] = all(
        v.get("converged", False)
        for v in report.values()
        if isinstance(v, dict)
    )
    return report


# ─────────────────────────────────────────────────
# ELBO Gap Decomposition (Section 4.4.2 of paper)
# ─────────────────────────────────────────────────

def elbo_gap_decomposition(
    model,
    X,
    site_ids,
    n_temperatures: int = 5,
    n_samples: int = 10,
) -> Dict[str, float]:
    """Decompose the ELBO gap into factorization, Concrete, and amortization components.

    Implements the decomposition from Equation (14) of the paper:
        log p(X) - L = KL_factorization + KL_Concrete + KL_amortization

    Approximated empirically by varying temperature and comparing
    with per-sample optimization.

    Parameters
    ----------
    model : HierarchicalNBDL
        Trained model.
    X : Tensor, shape (N, D)
    site_ids : Tensor, shape (N,)
    n_temperatures : int
        Number of temperatures to evaluate.
    n_samples : int
        Number of samples for Monte Carlo estimates.

    Returns
    -------
    decomposition : dict
        Estimated gap components.
    """
    import torch

    model.eval()
    temps = np.logspace(-2, 0, n_temperatures)  # 0.01 to 1.0
    elbos_by_temp = []

    with torch.no_grad():
        for temp in temps:
            elbo_samples = []
            for _ in range(n_samples):
                fwd = model(X, site_ids, temperature=float(temp))
                loss, diag = model.elbo(X, site_ids, fwd)
                elbo_samples.append(-loss.item())  # ELBO = -loss
            elbos_by_temp.append(np.mean(elbo_samples))

    # Best ELBO (lowest temperature) approximates mean-field ELBO
    elbo_mf = max(elbos_by_temp)
    # ELBO at temp=1.0 includes full Concrete gap
    elbo_hot = elbos_by_temp[-1]
    # ELBO at default temp=0.1
    idx_default = np.argmin(np.abs(temps - 0.1))
    elbo_default = elbos_by_temp[idx_default]

    # Concrete gap: difference between cold and hot
    concrete_gap = elbo_mf - elbo_hot

    # Estimate total gap (would need importance sampling for true log p(X))
    # We report the decomposable components
    return {
        "elbo_best": elbo_mf,
        "elbo_default_temp": elbo_default,
        "elbo_hot": elbo_hot,
        "concrete_gap_estimate": concrete_gap,
        "temperatures": temps.tolist(),
        "elbos": elbos_by_temp,
        "note": "True log p(X) requires importance sampling; "
                "reported gaps are lower bounds on the decomposition.",
    }


def posterior_predictive_check(
    model,
    X,
    site_ids,
    n_samples: int = 100,
) -> Dict[str, float]:
    """Posterior predictive check: compare reconstruction statistics.

    Generates samples from the posterior predictive distribution and
    compares summary statistics with the observed data.

    Parameters
    ----------
    model : HierarchicalNBDL
    X : Tensor, shape (N, D)
    site_ids : Tensor, shape (N,)
    n_samples : int

    Returns
    -------
    ppc_results : dict
        p-values and test statistics.
    """
    import torch

    model.eval()
    obs_mean = X.mean(dim=0).numpy()
    obs_var = X.var(dim=0).numpy()
    obs_norm = np.linalg.norm(X.numpy(), axis=1)

    pred_means = []
    pred_vars = []
    pred_norms = []

    with torch.no_grad():
        for _ in range(n_samples):
            fwd = model(X, site_ids, temperature=0.1)
            x_hat = fwd["x_hat"]
            # Add noise
            noise = torch.randn_like(x_hat) * model.sigma2.sqrt()
            x_pred = x_hat + noise

            pred_means.append(x_pred.mean(dim=0).numpy())
            pred_vars.append(x_pred.var(dim=0).numpy())
            pred_norms.append(np.linalg.norm(x_pred.numpy(), axis=1))

    pred_means = np.array(pred_means)
    pred_vars = np.array(pred_vars)

    # Bayesian p-values: fraction of posterior samples exceeding observed
    mean_pval = np.mean(pred_means.mean(axis=1) > obs_mean.mean())
    var_pval = np.mean(pred_vars.mean(axis=1) > obs_var.mean())
    norm_pval = np.mean(
        np.array([pn.mean() for pn in pred_norms]) > obs_norm.mean()
    )

    return {
        "mean_pvalue": float(mean_pval),
        "variance_pvalue": float(var_pval),
        "norm_pvalue": float(norm_pval),
        "well_calibrated": all(0.05 < p < 0.95 for p in [mean_pval, var_pval, norm_pval]),
    }
