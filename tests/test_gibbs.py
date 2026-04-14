"""Tests for the collapsed Gibbs sampler."""

import numpy as np
import pytest
from h_nbdl.inference.gibbs import CollapsedGibbs, GibbsSamples
from h_nbdl.models.generative import generate_hierarchical_data


class TestCollapsedGibbs:
    @pytest.fixture
    def small_data(self):
        return generate_hierarchical_data(
            n_sites=2, n_per_site=30, data_dim=10, k_true=3,
            sigma2=0.05, seed=42,
        )

    def test_initialization(self, small_data):
        sampler = CollapsedGibbs(
            small_data.X, small_data.site_ids,
            k_max=20, n_iter=10, burnin=5, thin=1, seed=0,
        )
        assert sampler.Z.shape == (60, 20)
        assert sampler.N == 60
        assert sampler.D == 10
        assert sampler.J == 2

    def test_run_returns_samples(self, small_data):
        sampler = CollapsedGibbs(
            small_data.X, small_data.site_ids,
            k_max=20, n_iter=20, burnin=10, thin=2, seed=0,
        )
        samples = sampler.run(verbose=False)
        assert isinstance(samples, GibbsSamples)
        assert len(samples.K_eff_trace) == 20
        assert len(samples.Z_samples) == 5  # (20-10)/2
        assert samples.D_mean is not None
        assert samples.D_mean.shape == (20, 10)

    def test_k_eff_trace_positive(self, small_data):
        sampler = CollapsedGibbs(
            small_data.X, small_data.site_ids,
            k_max=20, n_iter=15, burnin=5, thin=1, seed=0,
        )
        samples = sampler.run(verbose=False)
        assert all(k >= 0 for k in samples.K_eff_trace)

    def test_sigma2_stays_bounded(self, small_data):
        sampler = CollapsedGibbs(
            small_data.X, small_data.site_ids,
            k_max=20, n_iter=15, burnin=5, thin=1, seed=0,
        )
        samples = sampler.run(verbose=False)
        assert all(0 < s < 100 for s in samples.sigma2_trace)

    def test_effective_K(self, small_data):
        sampler = CollapsedGibbs(
            small_data.X, small_data.site_ids,
            k_max=20, n_iter=20, burnin=10, thin=2, seed=0,
        )
        samples = sampler.run(verbose=False)
        k = samples.effective_K()
        assert 0 < k <= 20
