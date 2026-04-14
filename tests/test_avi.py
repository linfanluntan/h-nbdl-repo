"""Tests for the H-NBDL amortized variational inference."""

import torch
import numpy as np
import pytest
from h_nbdl.models import HierarchicalNBDL
from h_nbdl.inference import AmortizedVI
from h_nbdl.models.generative import generate_hierarchical_data


class TestHierarchicalNBDL:
    @pytest.fixture
    def model(self):
        return HierarchicalNBDL(
            data_dim=20, k_max=30, n_sites=3,
            encoder_hidden=[64, 32], site_embed_dim=8,
        )

    @pytest.fixture
    def data(self):
        syn = generate_hierarchical_data(
            n_sites=3, n_per_site=50, data_dim=20, k_true=5, seed=0
        )
        X = torch.tensor(syn.X, dtype=torch.float32)
        sid = torch.tensor(syn.site_ids, dtype=torch.long)
        return X, sid

    def test_forward_shapes(self, model, data):
        X, sid = data
        fwd = model(X, sid, temperature=0.5)
        assert fwd["x_hat"].shape == X.shape
        assert fwd["mu"].shape == (X.shape[0], 30)
        assert fwd["logvar"].shape == (X.shape[0], 30)
        assert fwd["z"].shape == (X.shape[0], 30)

    def test_elbo_returns_scalar(self, model, data):
        X, sid = data
        fwd = model(X, sid)
        loss, diag = model.elbo(X, sid, fwd)
        assert loss.dim() == 0
        assert "recon_loss" in diag
        assert "kl_z" in diag

    def test_encode_shapes(self, model, data):
        X, sid = data
        r_mean, r_var = model.encode(X, sid)
        assert r_mean.shape == (X.shape[0], 30)
        assert r_var.shape == (X.shape[0], 30)
        assert (r_var >= 0).all()

    def test_concrete_deterministic_at_eval(self, model):
        logits = torch.randn(10, 30)
        model.eval()
        z1 = model.concrete_sample(logits, temperature=0.1)
        z2 = model.concrete_sample(logits, temperature=0.1)
        # At eval, should be deterministic (hard thresholding)
        torch.testing.assert_close(z1, z2)


class TestAmortizedVI:
    def test_training_reduces_loss(self):
        syn = generate_hierarchical_data(
            n_sites=2, n_per_site=100, data_dim=20, k_true=5, seed=0
        )
        X = torch.tensor(syn.X, dtype=torch.float32)
        sid = torch.tensor(syn.site_ids, dtype=torch.long)

        model = HierarchicalNBDL(
            data_dim=20, k_max=30, n_sites=2,
            encoder_hidden=[64, 32], site_embed_dim=8,
        )
        trainer = AmortizedVI(model, lr=1e-3, temp_anneal=(1.0, 0.5, 5))
        history = trainer.fit(X, sid, epochs=10, batch_size=64, verbose=False)

        assert history[-1]["loss"] < history[0]["loss"]
