"""
Cross-validation utilities for H-NBDL experiments.

Implements stratified K-fold CV preserving site proportions,
and cross-site leave-one-site-out evaluation.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from h_nbdl.models import HierarchicalNBDL
from h_nbdl.inference import AmortizedVI
from h_nbdl.utils.data import prepare_data


@dataclass
class CVFoldResult:
    """Result for a single CV fold."""
    fold: int
    train_loss: float
    val_loss: float
    k_eff: int
    metrics: Dict  # task-specific metrics (AUC, MSE, etc.)


def stratified_site_kfold(
    site_ids: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate stratified K-fold splits preserving site proportions.

    Parameters
    ----------
    site_ids : np.ndarray, shape (N,)
    n_folds : int
    seed : int

    Returns
    -------
    folds : list of (train_idx, val_idx) tuples
    """
    rng = np.random.default_rng(seed)
    sites = np.unique(site_ids)
    N = len(site_ids)

    # Collect indices per site, shuffle
    site_indices = {}
    for j in sites:
        idx = np.where(site_ids == j)[0]
        rng.shuffle(idx)
        site_indices[j] = idx

    # Assign to folds round-robin within each site
    fold_assignments = np.zeros(N, dtype=int)
    for j in sites:
        idx = site_indices[j]
        for i, sample_idx in enumerate(idx):
            fold_assignments[sample_idx] = i % n_folds

    folds = []
    for f in range(n_folds):
        val_mask = fold_assignments == f
        train_mask = ~val_mask
        folds.append((np.where(train_mask)[0], np.where(val_mask)[0]))

    return folds


def leave_one_site_out(
    site_ids: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Generate leave-one-site-out splits for cross-site evaluation.

    Returns
    -------
    splits : list of (train_idx, test_idx, held_out_site)
    """
    sites = np.unique(site_ids)
    splits = []
    for held_out in sites:
        test_idx = np.where(site_ids == held_out)[0]
        train_idx = np.where(site_ids != held_out)[0]
        splits.append((train_idx, test_idx, int(held_out)))
    return splits


def run_cv_experiment(
    X: np.ndarray,
    site_ids: np.ndarray,
    outcome: Optional[np.ndarray] = None,
    treatment: Optional[np.ndarray] = None,
    n_folds: int = 5,
    model_kwargs: Optional[Dict] = None,
    train_kwargs: Optional[Dict] = None,
    seed: int = 42,
) -> List[CVFoldResult]:
    """Run a complete cross-validation experiment.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
    site_ids : np.ndarray, shape (N,)
    outcome : np.ndarray or None
        Binary or continuous outcome for downstream evaluation.
    treatment : np.ndarray or None
        Treatment assignments.
    n_folds : int
    model_kwargs : dict
        Arguments for HierarchicalNBDL constructor.
    train_kwargs : dict
        Arguments for AmortizedVI.fit().

    Returns
    -------
    results : list of CVFoldResult
    """
    if model_kwargs is None:
        model_kwargs = {}
    if train_kwargs is None:
        train_kwargs = {"epochs": 150, "batch_size": 256}

    data_dim = X.shape[1]
    n_sites = len(np.unique(site_ids))
    folds = stratified_site_kfold(site_ids, n_folds, seed)

    results = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}/{n_folds}...")

        X_train, sid_train = prepare_data(X[train_idx], site_ids[train_idx])
        X_val, sid_val = prepare_data(X[val_idx], site_ids[val_idx])

        model = HierarchicalNBDL(
            data_dim=data_dim, n_sites=n_sites,
            **{k: v for k, v in model_kwargs.items() if k not in ["data_dim", "n_sites"]},
        )
        trainer = AmortizedVI(model, lr=train_kwargs.get("lr", 1e-3))
        history = trainer.fit(
            X_train, sid_train,
            epochs=train_kwargs.get("epochs", 150),
            batch_size=train_kwargs.get("batch_size", 256),
            val_X=X_val, val_site_ids=sid_val,
            verbose=False,
        )

        fold_metrics = {
            "train_loss": history[-1]["loss"],
            "val_loss": history[-1].get("val_loss", float("nan")),
            "k_eff": model.effective_atoms(),
        }

        # Downstream evaluation if outcome provided
        if outcome is not None:
            r_mean, r_var = trainer.get_representations(X_val, sid_val)
            r_np = r_mean.cpu().numpy()

            if len(np.unique(outcome)) == 2:
                # Binary classification
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import roc_auc_score
                clf = LogisticRegression(C=1.0, max_iter=1000)
                clf.fit(
                    trainer.get_representations(X_train, sid_train)[0].cpu().numpy(),
                    outcome[train_idx],
                )
                pred_prob = clf.predict_proba(r_np)[:, 1]
                fold_metrics["auc"] = roc_auc_score(outcome[val_idx], pred_prob)
            else:
                # Regression
                from sklearn.linear_model import Ridge
                from sklearn.metrics import mean_squared_error
                reg = Ridge(alpha=1.0)
                reg.fit(
                    trainer.get_representations(X_train, sid_train)[0].cpu().numpy(),
                    outcome[train_idx],
                )
                pred = reg.predict(r_np)
                fold_metrics["mse"] = mean_squared_error(outcome[val_idx], pred)

        results.append(CVFoldResult(
            fold=fold_idx,
            train_loss=fold_metrics["train_loss"],
            val_loss=fold_metrics["val_loss"],
            k_eff=fold_metrics["k_eff"],
            metrics=fold_metrics,
        ))

    return results


def cv_summary(results: List[CVFoldResult]) -> Dict:
    """Aggregate CV results into mean/std summary."""
    metrics_keys = set()
    for r in results:
        metrics_keys.update(r.metrics.keys())

    summary = {}
    for key in metrics_keys:
        vals = [r.metrics.get(key, float("nan")) for r in results]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))

    return summary
