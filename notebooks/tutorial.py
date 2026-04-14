#!/usr/bin/env python
"""
H-NBDL Tutorial
================

This script demonstrates the complete H-NBDL workflow:
1. Generate synthetic multi-site data
2. Train with amortized variational inference
3. Check identifiability and convergence diagnostics
4. Extract uncertainty-aware representations
5. Downstream causal effect estimation
6. Visualize results

Run:
    python notebooks/tutorial.py
"""

import numpy as np
import torch

print("=" * 60)
print("  H-NBDL Tutorial: Complete Workflow")
print("=" * 60)

# ─── Step 1: Generate Synthetic Data ───
print("\n[1/6] Generating synthetic multi-site data...")
from h_nbdl.models.generative import generate_hierarchical_data

data = generate_hierarchical_data(
    n_sites=3,
    n_per_site=200,
    data_dim=30,
    k_true=8,
    sigma2=0.1,
    lambda_inv=0.2,
    seed=42,
)
print(f"  N={data.X.shape[0]}, D={data.X.shape[1]}, K_true={data.K_true}")
print(f"  Sites: {len(np.unique(data.site_ids))}, samples/site: 200")

# ─── Step 2: Train H-NBDL ───
print("\n[2/6] Training H-NBDL with amortized VI...")
from h_nbdl.models import HierarchicalNBDL
from h_nbdl.inference import AmortizedVI
from h_nbdl.utils.data import prepare_data, train_val_split

X_train, sid_train, X_val, sid_val = train_val_split(
    data.X, data.site_ids, val_fraction=0.15, seed=42
)
X_t, sid_t = prepare_data(X_train, sid_train)
X_v, sid_v = prepare_data(X_val, sid_val)

model = HierarchicalNBDL(
    data_dim=30,
    k_max=50,
    n_sites=3,
    encoder_hidden=[128, 64],
    site_embed_dim=16,
    alpha0=4.0,
)

trainer = AmortizedVI(
    model, lr=1e-3,
    temp_anneal=(1.0, 0.1, 50),
)
history = trainer.fit(
    X_t, sid_t, epochs=100, batch_size=128,
    val_X=X_v, val_site_ids=sid_v,
    verbose=True,
)

k_eff = model.effective_atoms()
print(f"\n  Final loss: {history[-1]['loss']:.4f}")
print(f"  K_effective: {k_eff} (true: {data.K_true})")
print(f"  Learned sigma^2: {model.sigma2.item():.4f} (true: {data.sigma2})")

# ─── Step 3: Diagnostics ───
print("\n[3/6] Running diagnostics...")
from h_nbdl.analysis.identifiability import (
    check_identifiability_conditions,
    decomposition_quality,
    shared_vs_specific_analysis,
)

conds = check_identifiability_conditions(model, data.site_ids)
print(f"  Identifiability conditions satisfied: {conds['all_satisfied']}")
print(f"    (a) Independent atoms: {conds['condition_a_independent']} (rank {conds['condition_a_rank']}/{conds['condition_a_n_active']})")
print(f"    (b) Shared atoms exist: {conds['condition_b_shared_atoms']} ({conds['condition_b_n_shared']} shared)")
print(f"    (c) Multiple sites: {conds['condition_c_multiple_sites']}")

quality = decomposition_quality(model, data.D_global, data.D_sites)
print(f"\n  Decomposition quality:")
print(f"    Lambda (pooling): {quality['lambda_learned']:.2f}")
print(f"    Site deviation std: {quality['site_deviation_std']:.3f}")
print(f"    Pooling ratio: {quality['pooling_ratio']:.3f}")
if "amari_global" in quality:
    print(f"    Amari (global atoms): {quality['amari_global']:.4f}")

analysis = shared_vs_specific_analysis(model)
print(f"\n  Atom categories:")
print(f"    Shared: {analysis['n_shared']}, Subset: {analysis['n_subset']}, "
      f"Specific: {analysis['n_specific']}, Inactive: {analysis['n_inactive']}")

# ─── Step 4: Extract Representations ───
print("\n[4/6] Extracting uncertainty-aware representations...")
X_all = torch.tensor(data.X, dtype=torch.float32)
sid_all = torch.tensor(data.site_ids, dtype=torch.long)
r_mean, r_var = model.encode(X_all, sid_all)

print(f"  Representation shape: {r_mean.shape}")
print(f"  Mean activation: {(r_mean.abs() > 0.01).float().mean():.3f}")
print(f"  Mean uncertainty: {r_var.mean():.4f}")

# ─── Step 5: Downstream Causal Estimation ───
print("\n[5/6] Downstream causal effect estimation...")
# Simulate treatment and outcome
rng = np.random.default_rng(42)
treatment = rng.binomial(1, 0.5, len(data.X)).astype(float)
true_ate = 0.3
outcome = (
    r_mean[:, 0].numpy() * 0.5
    + treatment * true_ate
    + rng.normal(0, 0.5, len(data.X))
    > 0
).astype(float)

from h_nbdl.downstream import HierarchicalCausalModel

causal = HierarchicalCausalModel(
    representation_dim=k_eff, n_sites=3, outcome_type="binary",
)
trace = causal.fit(
    r_mean.numpy()[:, :k_eff],
    r_var.numpy()[:, :k_eff],
    treatment, outcome, data.site_ids,
)
ate = causal.average_treatment_effect(trace)
print(f"  Estimated ATE: {ate['mean']:.3f}")
if not np.isnan(ate.get('ci_lower', np.nan)):
    print(f"  95% CI: [{ate['ci_lower']:.3f}, {ate['ci_upper']:.3f}]")
print(f"  True ATE: {true_ate}")

# ─── Step 6: Summary ───
print("\n[6/6] Summary")
print("=" * 60)
print(f"  Model: H-NBDL with K_max={model.k_max}, learned K_eff={k_eff}")
print(f"  Inference: AVI, 100 epochs, final loss {history[-1]['loss']:.4f}")
print(f"  Identifiability: {'PASS' if conds['all_satisfied'] else 'FAIL'}")
print(f"  Downstream ATE: {ate['mean']:.3f} (true: {true_ate})")
print("=" * 60)
print("\nTutorial complete! See the paper and README for full details.")
