"""Tests for analysis modules: diagnostics, identifiability, CV, downstream."""

import numpy as np
import torch
import pytest
from h_nbdl.models import HierarchicalNBDL
from h_nbdl.models.generative import generate_hierarchical_data
from h_nbdl.inference import AmortizedVI
from h_nbdl.analysis.diagnostics import (
    gelman_rubin_rhat, effective_sample_size, gibbs_convergence_report,
)
from h_nbdl.analysis.identifiability import (
    check_identifiability_conditions, decomposition_quality, shared_vs_specific_analysis,
)
from h_nbdl.analysis.cross_validation import (
    stratified_site_kfold, leave_one_site_out, cv_summary,
)


# ── Diagnostics ──

class TestGelmanRubin:
    def test_identical_chains(self):
        chains = [np.ones(100) * 5.0 + np.random.normal(0, 0.01, 100) for _ in range(4)]
        rhat = gelman_rubin_rhat(chains)
        assert rhat < 1.1

    def test_divergent_chains(self):
        chains = [np.ones(100) * i for i in range(4)]
        rhat = gelman_rubin_rhat(chains)
        assert rhat > 1.5

    def test_converging_chains(self):
        rng = np.random.default_rng(42)
        chains = [rng.normal(0, 1, 500) for _ in range(4)]
        rhat = gelman_rubin_rhat(chains)
        assert 0.9 < rhat < 1.1


class TestESS:
    def test_iid_samples(self):
        chain = np.random.normal(0, 1, 1000)
        ess = effective_sample_size(chain)
        # ESS should be close to N for iid
        assert ess > 500

    def test_correlated_samples(self):
        chain = np.cumsum(np.random.normal(0, 0.1, 1000))
        ess = effective_sample_size(chain)
        assert ess < 500  # Should be much less than N


# ── Identifiability ──

class TestIdentifiability:
    @pytest.fixture
    def trained_model(self):
        data = generate_hierarchical_data(
            n_sites=3, n_per_site=50, data_dim=20, k_true=5, seed=0
        )
        X = torch.tensor(data.X, dtype=torch.float32)
        sid = torch.tensor(data.site_ids, dtype=torch.long)
        model = HierarchicalNBDL(
            data_dim=20, k_max=30, n_sites=3,
            encoder_hidden=[64, 32],
        )
        trainer = AmortizedVI(model, lr=1e-3, temp_anneal=(1.0, 0.5, 5))
        trainer.fit(X, sid, epochs=10, batch_size=64, verbose=False)
        return model, data

    def test_check_conditions(self, trained_model):
        model, data = trained_model
        conds = check_identifiability_conditions(model, data.site_ids)
        assert "condition_a_independent" in conds
        assert "condition_b_shared_atoms" in conds
        assert "condition_c_multiple_sites" in conds
        assert conds["condition_c_J"] == 3

    def test_decomposition_quality(self, trained_model):
        model, data = trained_model
        quality = decomposition_quality(model, data.D_global, data.D_sites)
        assert "lambda_learned" in quality
        assert "pooling_ratio" in quality
        assert 0 <= quality["pooling_ratio"] <= 1

    def test_shared_vs_specific(self, trained_model):
        model, _ = trained_model
        analysis = shared_vs_specific_analysis(model)
        assert analysis["k_effective"] >= 0
        assert analysis["n_shared"] + analysis["n_subset"] + analysis["n_specific"] == analysis["k_effective"]


# ── Cross-Validation ──

class TestCrossValidation:
    def test_stratified_folds(self):
        site_ids = np.repeat(np.arange(4), 50)  # 200 samples, 4 sites
        folds = stratified_site_kfold(site_ids, n_folds=5, seed=42)
        assert len(folds) == 5
        for train_idx, val_idx in folds:
            assert len(train_idx) + len(val_idx) == 200
            # Check no overlap
            assert len(set(train_idx) & set(val_idx)) == 0
            # Check all sites present in val
            val_sites = np.unique(site_ids[val_idx])
            assert len(val_sites) == 4

    def test_leave_one_site_out(self):
        site_ids = np.repeat(np.arange(3), 30)
        splits = leave_one_site_out(site_ids)
        assert len(splits) == 3
        for train_idx, test_idx, held_out in splits:
            assert held_out not in site_ids[train_idx]
            assert np.all(site_ids[test_idx] == held_out)

    def test_cv_summary(self):
        from h_nbdl.analysis.cross_validation import CVFoldResult
        results = [
            CVFoldResult(fold=i, train_loss=1.0-i*0.1, val_loss=1.1-i*0.1,
                         k_eff=15, metrics={"auc": 0.75 + i*0.01})
            for i in range(5)
        ]
        summary = cv_summary(results)
        assert "auc_mean" in summary
        assert "auc_std" in summary


# ── Downstream ──

class TestDownstream:
    def test_causal_hbm_map_fallback(self):
        """Test causal HBM with MAP fallback (no PyMC required)."""
        from h_nbdl.downstream import HierarchicalCausalModel
        rng = np.random.default_rng(42)
        N = 100
        r_mean = rng.standard_normal((N, 10))
        r_var = np.abs(rng.standard_normal((N, 10))) * 0.1
        treatment = rng.binomial(1, 0.5, N).astype(float)
        outcome = (r_mean[:, 0] + treatment * 0.5 + rng.normal(0, 0.5, N) > 0).astype(float)
        site_ids = rng.choice(3, N)

        model = HierarchicalCausalModel(
            representation_dim=10, n_sites=3, outcome_type="binary"
        )
        trace = model._fit_map(r_mean, r_var, treatment, outcome, site_ids)
        ate = model.average_treatment_effect(trace)
        assert "mean" in ate
