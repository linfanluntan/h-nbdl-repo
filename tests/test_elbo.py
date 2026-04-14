"""Tests for ELBO computation utilities."""

import torch
import pytest
from h_nbdl.inference.elbo import (
    reconstruction_nll, kl_gaussian, kl_bernoulli, assess_convergence,
)


class TestELBOComponents:
    def test_recon_nll_zero_when_perfect(self):
        x = torch.randn(32, 10)
        sigma2 = torch.tensor(0.1)
        nll = reconstruction_nll(x, x, sigma2)
        # Should be just the normalizing constant
        assert nll.item() > 0  # log-det term

    def test_kl_gaussian_zero_for_standard_normal(self):
        mu = torch.zeros(32, 10)
        logvar = torch.zeros(32, 10)
        kl = kl_gaussian(mu, logvar)
        assert abs(kl.item()) < 1e-5

    def test_kl_gaussian_positive(self):
        mu = torch.randn(32, 10)
        logvar = torch.randn(32, 10)
        kl = kl_gaussian(mu, logvar)
        assert kl.item() >= -1e-5  # Should be non-negative

    def test_kl_bernoulli_zero_for_equal(self):
        p = torch.full((32, 10), 0.5)
        kl = kl_bernoulli(p, p)
        assert abs(kl.item()) < 1e-5

    def test_kl_bernoulli_positive(self):
        q = torch.full((32, 10), 0.3)
        p = torch.full((32, 10), 0.7)
        kl = kl_bernoulli(q, p)
        assert kl.item() > 0

    def test_assess_convergence(self):
        history = [{"loss": 100 - i * 0.1, "k_effective": 15} for i in range(100)]
        result = assess_convergence(history, window=10)
        assert "converged" in result
        assert "final_loss" in result
