"""Tests for the H-NBDL generative model and synthetic data generation."""

import numpy as np
import pytest
from h_nbdl.models.generative import generate_hierarchical_data


class TestSyntheticData:
    def test_shapes(self):
        data = generate_hierarchical_data(
            n_sites=3, n_per_site=50, data_dim=20, k_true=5, seed=0
        )
        assert data.X.shape == (150, 20)
        assert data.site_ids.shape == (150,)
        assert data.Z_true.shape == (150, 5)
        assert data.S_true.shape == (150, 5)
        assert data.D_global.shape == (5, 20)
        assert len(data.D_sites) == 3

    def test_site_ids(self):
        data = generate_hierarchical_data(n_sites=4, n_per_site=30, seed=1)
        for j in range(4):
            assert np.sum(data.site_ids == j) == 30

    def test_sparsity(self):
        data = generate_hierarchical_data(
            n_sites=2, n_per_site=100, k_true=10, seed=2
        )
        # Z should be sparse (more zeros than ones)
        assert np.mean(data.Z_true) < 0.7

    def test_reproducibility(self):
        d1 = generate_hierarchical_data(seed=42)
        d2 = generate_hierarchical_data(seed=42)
        np.testing.assert_array_equal(d1.X, d2.X)
        np.testing.assert_array_equal(d1.Z_true, d2.Z_true)

    def test_global_atoms_unit_norm(self):
        data = generate_hierarchical_data(seed=3)
        norms = np.linalg.norm(data.D_global, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)
