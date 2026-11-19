"""Deterministic L2-regularized logistic risk model and exact ROC-AUC evaluation.

Exists to measure how much apparent offline ROC-AUC lift evaporates when a
model trained on leaky batch-snapshot features is deployed onto honest
point-in-time online serving features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


class ModelEvalError(ValueError):
    """Raised when classification targets or feature matrices are invalid."""


@dataclass(frozen=True)
class LeakageAuditResult:
    """Complete offline-vs-online AUC inflation and row-corruption audit."""

    batch_offline_auc: float
    batch_model_on_online_auc: float
    pit_honest_auc: float
    auc_inflation_points: float
    corrupted_row_count: int
    corrupted_row_pct: float
    risk_alert_leak_rows: int
    gate_passed: bool


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute exact ROC-AUC via the Mann-Whitney U rank statistic (ties averaged)."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or y.shape != s.shape or y.size == 0:
        raise ModelEvalError("y_true and y_score must be non-empty 1D arrays of equal length")
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        raise ModelEvalError("ROC-AUC requires at least one positive and one negative label")

    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(s.size, dtype=float)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    pos_rank_sum = float(np.sum(ranks[y == 1]))
    u_stat = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


class LogisticRiskModel:
    """Deterministic L2-regularized logistic regression solved with SciPy L-BFGS-B."""

    def __init__(self, l2_reg: float = 1.0) -> None:
        if l2_reg <= 0.0:
            raise ModelEvalError(f"l2_reg must be positive, got {l2_reg}")
        self.l2_reg = l2_reg
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.bias_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRiskModel:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if X_arr.ndim != 2 or y_arr.ndim != 1 or X_arr.shape[0] != y_arr.shape[0]:
            raise ModelEvalError("X must be 2D and y must be 1D with matching row counts")
        self.mean_ = np.mean(X_arr, axis=0)
        std = np.std(X_arr, axis=0)
        self.scale_ = np.where(std > 1e-8, std, 1.0)
        Xn = (X_arr - self.mean_) / self.scale_

        n_samples, n_features = Xn.shape

        def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            w = params[:n_features]
            b = params[n_features]
            logits = np.clip(Xn @ w + b, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            nll = -np.mean(
                y_arr * np.log(np.clip(probs, 1e-12, 1.0))
                + (1.0 - y_arr) * np.log(np.clip(1.0 - probs, 1e-12, 1.0))
            )
            reg = 0.5 * self.l2_reg * float(np.sum(w * w)) / n_samples
            diff = (probs - y_arr) / n_samples
            grad_w = Xn.T @ diff + (self.l2_reg / n_samples) * w
            grad_b = float(np.sum(diff))
            return float(nll + reg), np.append(grad_w, grad_b)

        res = minimize(
            loss_and_grad,
            x0=np.zeros(n_features + 1, dtype=float),
            jac=True,
            method="L-BFGS-B",
        )
        self.weights_ = res.x[:n_features]
        self.bias_ = float(res.x[n_features])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.bias_ is None or self.mean_ is None or self.scale_ is None:
            raise ModelEvalError("LogisticRiskModel must be fitted before predict_proba")
        Xn = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        logits = np.clip(Xn @ self.weights_ + self.bias_, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))


def evaluate_leakage_gap(
    X_batch: np.ndarray,
    X_pit: np.ndarray,
    labels: np.ndarray,
    corrupted_row_count: int,
    corrupted_row_pct: float,
    risk_alert_leak_rows: int,
    max_auc_inflation: float = 0.02,
    max_corrupted_row_pct: float = 5.0,
) -> LeakageAuditResult:
    """Train/test split comparing leaky batch evaluation vs honest online serving AUC."""
    n = labels.size
    split_idx = int(n * 0.65)
    y_tr, y_te = labels[:split_idx], labels[split_idx:]

    batch_model = LogisticRiskModel(l2_reg=1.0).fit(X_batch[:split_idx], y_tr)
    pit_model = LogisticRiskModel(l2_reg=1.0).fit(X_pit[:split_idx], y_tr)

    batch_offline_auc = roc_auc_score(y_te, batch_model.predict_proba(X_batch[split_idx:]))
    batch_on_online_auc = roc_auc_score(y_te, batch_model.predict_proba(X_pit[split_idx:]))
    pit_honest_auc = roc_auc_score(y_te, pit_model.predict_proba(X_pit[split_idx:]))

    inflation = batch_offline_auc - batch_on_online_auc
    gate_passed = (inflation <= max_auc_inflation) and (
        corrupted_row_pct <= max_corrupted_row_pct
    )

    return LeakageAuditResult(
        batch_offline_auc=round(batch_offline_auc, 4),
        batch_model_on_online_auc=round(batch_on_online_auc, 4),
        pit_honest_auc=round(pit_honest_auc, 4),
        auc_inflation_points=round(inflation * 100.0, 1),
        corrupted_row_count=corrupted_row_count,
        corrupted_row_pct=corrupted_row_pct,
        risk_alert_leak_rows=risk_alert_leak_rows,
        gate_passed=gate_passed,
    )
