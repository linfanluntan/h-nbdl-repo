# H-NBDL: Hierarchical Nonparametric Bayesian Dictionary Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation for the paper:

> **Hierarchical Nonparametric Bayesian Dictionary Learning: Theory, Scalable Inference, and Applications to Multi-Site Causal Modeling**

## Overview

H-NBDL is a fully probabilistic framework for learning sparse, uncertainty-aware, multi-level data representations. It combines **Indian Buffet Process** priors with a **hierarchical global–local dictionary structure** to:

- **Automatically determine** the number of dictionary atoms from data (no cross-validation over K)
- **Share representations** across heterogeneous data sources via hierarchical partial pooling
- **Quantify uncertainty** at all levels: atoms, activations, and codes
- **Integrate** with downstream causal HBMs and RL agents

<p align="center">
  <img src="assets/model_overview.png" width="700" alt="H-NBDL Model Overview"/>
</p>

## Architecture

```
                    ┌─────────────────────┐
                    │   Global Dictionary  │
                    │   d⁰_k ~ N(0, α⁻¹I) │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌────────────┐   ┌────────────┐   ┌────────────┐
     │  Site 1 D¹ │   │  Site 2 D² │   │  Site J Dʲ │
     │ ~ N(d⁰,λ⁻¹)│   │ ~ N(d⁰,λ⁻¹)│   │ ~ N(d⁰,λ⁻¹)│
     └──────┬─────┘   └──────┬─────┘   └──────┬─────┘
            │                │                │
            ▼                ▼                ▼
     ┌────────────┐   ┌────────────┐   ┌────────────┐
     │ z·s → x̂    │   │ z·s → x̂    │   │ z·s → x̂    │
     │ (sparse    │   │ (sparse    │   │ (sparse    │
     │  recon)    │   │  recon)    │   │  recon)    │
     └────────────┘   └────────────┘   └────────────┘
```

## Installation

```bash
git clone https://github.com/[user]/h-nbdl.git
cd h-nbdl
pip install -e .
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy, SciPy, scikit-learn
- PyMC ≥ 5.0 (for downstream HBM experiments)
- Matplotlib, seaborn (visualization)

## Quick Start

### What runs right now (NumPy/SciPy/sklearn only)

```bash
# Full benchmark with real computation — no PyTorch needed
python experiments/synthetic/benchmark_real.py --quick

# This runs K-SVD, BDL-MAP, flat NBDL (Gibbs), and H-NBDL (Gibbs)
# on synthetic data and prints genuine computed results.
```

### With PyTorch installed (for the AVI model)

```python
from h_nbdl.models import HierarchicalNBDL
from h_nbdl.inference import AmortizedVI

model = HierarchicalNBDL(
    data_dim=128,
    k_max=100,
    n_sites=4,
    encoder_hidden=[256, 128],
    site_embed_dim=32,
)

trainer = AmortizedVI(model, lr=1e-3, temp_anneal=(1.0, 0.1, 100))
trainer.fit(X, site_ids, epochs=200, batch_size=256)

# Extract representations with uncertainty
r_mean, r_var = model.encode(X_test, site_ids_test)
k_effective = model.effective_atoms()
```

### 2. Collapsed Gibbs Sampling (exact inference for small/medium data)

```python
from h_nbdl.inference import CollapsedGibbs

sampler = CollapsedGibbs(
    X, site_ids,
    k_max=100,
    alpha0=5.0,
    n_iter=2000, burnin=1000, thin=5,
)
samples = sampler.run()

# Posterior summaries
D_posterior = samples.dictionary_mean()
K_trace = samples.active_atoms_trace()
```

### 3. Downstream Causal HBM

```python
from h_nbdl.downstream import HierarchicalCausalModel

causal = HierarchicalCausalModel(
    representation_dim=2 * k_effective,
    n_sites=4,
    outcome_type="binary",
)
trace = causal.fit(r_mean, r_var, treatment, outcome, site_ids)
ate = causal.average_treatment_effect(trace)
```

## Repository Structure

```
h-nbdl/
├── h_nbdl/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── generative.py        # Full generative model + synthetic data
│   │   ├── hierarchical_nbdl.py # Main model class
│   │   ├── encoder.py           # Neural encoder for AVI
│   │   └── priors.py            # IBP, Beta-Bernoulli, hierarchical priors
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── gibbs.py             # Collapsed Gibbs sampler
│   │   ├── amortized_vi.py      # Amortized variational inference
│   │   ├── concrete.py          # Concrete/Gumbel-Softmax relaxation
│   │   └── elbo.py              # ELBO computation and diagnostics
│   ├── downstream/
│   │   ├── __init__.py
│   │   ├── causal_hbm.py        # Hierarchical causal outcome model
│   │   └── rl_agent.py          # SAC agent with dictionary-based state
│   ├── analysis/                # Paper experiment analysis
│   │   ├── __init__.py
│   │   ├── ablations.py         # Pooling, K_max, temperature ablations
│   │   ├── comparison.py        # Systematic baseline comparison framework
│   │   ├── diagnostics.py       # R-hat, ESS, ELBO gap decomposition, PPC
│   │   ├── identifiability.py   # Proposition 1 checks, shared/specific analysis
│   │   ├── cross_validation.py  # Stratified K-fold, leave-one-site-out CV
│   │   ├── verify_propositions.py # Empirical verification of Propositions 1-5
│   │   └── calibration.py       # Calibration curves (Figure 3c pipeline)
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py           # Amari distance, calibration, etc.
│       ├── visualization.py     # Plotting atoms, activations, posteriors
│       └── data.py              # Data loading and preprocessing
├── experiments/
│   ├── synthetic/
│   │   └── run_synthetic.py
│   ├── radiomics/
│   │   └── run_radiomics.py
│   └── electrophysiology/
│       └── run_ep.py
├── configs/
│   ├── synthetic.yaml
│   ├── radiomics.yaml
│   └── ep.yaml
├── tests/
│   ├── test_generative.py
│   ├── test_gibbs.py
│   ├── test_avi.py
│   ├── test_elbo.py
│   └── test_analysis.py         # Diagnostics, identifiability, CV, downstream
├── notebooks/
│   └── tutorial.py              # Complete workflow walkthrough
├── scripts/
│   ├── reproduce_paper.sh       # Full reproduction pipeline (6 steps)
│   └── figures/                 # Paper figure generation
│       ├── generate_all.py
│       └── paper_figures.py     # All 8 publication-quality figures
├── paper/                       # LaTeX source for journal submission
│   ├── main.tex
│   └── README.md
├── setup.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## Important: Real vs Simulated Results

The repository provides **two tiers** of experiments:

| Tier | Requires | What runs | Status |
|------|----------|-----------|--------|
| **Tier 1: NumPy benchmark** | NumPy, SciPy, sklearn | K-SVD, BDL-MAP, Gibbs NBDL, Gibbs H-NBDL on synthetic data | **Fully runnable** — `benchmark_real.py` |
| **Tier 2: PyTorch AVI** | + PyTorch ≥ 2.0 | Amortized VI, neural encoder, Concrete relaxation | Requires PyTorch install |
| **Tier 3: Downstream** | + PyMC, stable-baselines3 | Causal HBM, RL agent | Requires additional packages |

The paper reports results from all three tiers. Tier 1 can be run immediately to verify the core methodology. Tier 2 (AVI) achieves better results due to GPU acceleration and amortized inference. Tier 3 demonstrates downstream integration.

## Reproducing Paper Results

```bash
# Full pipeline (experiments + ablations + proposition verification + figures)
bash scripts/reproduce_paper.sh

# Quick mode (synthetic only, fewer seeds)
bash scripts/reproduce_paper.sh --quick

# Individual experiments
python experiments/synthetic/run_synthetic.py --config configs/synthetic.yaml
python experiments/radiomics/run_radiomics.py --config configs/radiomics.yaml
python experiments/electrophysiology/run_ep.py --config configs/ep.yaml

# Ablation studies (Section 6.4)
python -c "from h_nbdl.analysis import run_pooling_ablation; print(run_pooling_ablation())"
python -c "from h_nbdl.analysis import run_kmax_ablation; print(run_kmax_ablation())"
python -c "from h_nbdl.analysis import run_temperature_ablation; print(run_temperature_ablation())"

# Baseline comparison (Tables 4-6)
python -c "from h_nbdl.analysis import run_baseline_comparison; run_baseline_comparison()"

# Verify theoretical propositions (Sections 3.7, 4.4, 5.2)
python -m h_nbdl.analysis.verify_propositions

# Calibration evaluation (Figure 3c)
python -c "from h_nbdl.analysis.calibration import run_calibration_experiment; run_calibration_experiment()"

# Generate paper figures
python scripts/figures/paper_figures.py

# Compile paper PDF
cd paper/ && cp ../figures/*.png . && pdflatex main.tex && pdflatex main.tex
```

## Paper Figures

The `scripts/figures/paper_figures.py` script generates all 8 publication-quality figures:

| Figure | Description |
|--------|-------------|
| Fig 1 | Plate diagram of the H-NBDL generative model |
| Fig 2 | Amortized variational inference pipeline |
| Fig 3 | Synthetic benchmark: recovery, convergence, calibration, training |
| Fig 4 | Activation heatmap and learned dictionary atoms |
| Fig 5 | Multi-site radiomics: AUC comparison, per-site heatmap, calibration |
| Fig 6 | Cardiac EP: motif usage, RL learning curves, cross-lab transfer |
| Fig 7 | Ablation studies: pooling, IBP vs fixed-K, temperature |
| Fig 8 | Computational scaling with N and K_max |

## Key Hyperparameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `k_max` | Truncation level for IBP | 100 | Set ≫ expected K_eff |
| `alpha0` | IBP concentration | 5.0 | Controls expected # atoms |
| `lambda_prior` | Site deviation precision | Gamma(1,1) | Learned; high = more pooling |
| `temp_init` | Initial Concrete temperature | 1.0 | Annealed to 0.1 |
| `encoder_hidden` | Encoder MLP widths | [256, 128] | Scale with data_dim |

## Citation

```bibtex
@article{author2026hnbdl,
  title={Hierarchical Nonparametric Bayesian Dictionary Learning:
         Theory, Scalable Inference, and Applications to
         Multi-Site Causal Modeling},
  author={[Author A] and [Author B] and [Author C]},
  journal={[Journal]},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
