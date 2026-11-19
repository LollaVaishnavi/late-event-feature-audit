"""Property tests for ROC-AUC, logistic risk model, and offline-vs-online leakage gap."""

from __future__ import annotations

import unittest

import numpy as np

from late_event_feature_audit.config import StreamConfig
from late_event_feature_audit.features import extract_batch_and_pit_features
from late_event_feature_audit.model import (
    LogisticRiskModel,
    ModelEvalError,
    evaluate_leakage_gap,
    roc_auc_score,
)
from late_event_feature_audit.stream import generate_stream_corpus


class TestRocAucAndLogisticModel(unittest.TestCase):
    def test_roc_auc_equals_one_for_perfect_ranking_and_zero_for_inverted_ranking(self) -> None:
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(roc_auc_score(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertAlmostEqual(roc_auc_score(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)

    def test_roc_auc_equals_half_for_constant_tied_scores(self) -> None:
        y = np.array([0, 1, 0, 1, 0, 1])
        self.assertAlmostEqual(roc_auc_score(y, np.full(6, 0.5)), 0.5)

    def test_roc_auc_is_invariant_to_strictly_increasing_score_transforms(self) -> None:
        y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        s = np.array([0.1, 0.4, 0.3, 0.7, 0.6, 0.2, 0.5, 0.9])
        self.assertAlmostEqual(roc_auc_score(y, s), roc_auc_score(y, np.exp(3.0 * s + 1.0)))

    def test_roc_auc_rejects_single_class_or_mismatched_arrays(self) -> None:
        with self.assertRaises(ModelEvalError):
            roc_auc_score(np.zeros(4), np.ones(4))
        with self.assertRaises(ModelEvalError):
            roc_auc_score(np.ones(4), np.ones(4))
        with self.assertRaises(ModelEvalError):
            roc_auc_score(np.array([0, 1]), np.array([0.5]))

    def test_logistic_model_achieves_high_auc_on_linearly_separable_features(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(200, 3))
        y = (X[:, 0] + 0.8 * X[:, 1] > 0.0).astype(int)
        model = LogisticRiskModel(l2_reg=0.5).fit(X, y)
        probs = model.predict_proba(X)
        self.assertGreater(roc_auc_score(y, probs), 0.97)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_logistic_model_is_deterministic_across_repeated_fits(self) -> None:
        rng = np.random.default_rng(11)
        X = rng.normal(size=(100, 4))
        y = (X[:, 0] > 0.2).astype(int)
        p1 = LogisticRiskModel(l2_reg=1.0).fit(X, y).predict_proba(X)
        p2 = LogisticRiskModel(l2_reg=1.0).fit(X, y).predict_proba(X)
        np.testing.assert_allclose(p1, p2)

    def test_logistic_model_handles_constant_zero_variance_feature_column(self) -> None:
        X = np.column_stack([np.ones(40), np.linspace(-2.0, 2.0, 40)])
        y = (X[:, 1] > 0.0).astype(int)
        probs = LogisticRiskModel().fit(X, y).predict_proba(X)
        self.assertTrue(np.all(np.isfinite(probs)))

    def test_logistic_model_rejects_invalid_reg_and_unfitted_predict(self) -> None:
        with self.assertRaises(ModelEvalError):
            LogisticRiskModel(l2_reg=0.0)
        m = LogisticRiskModel()
        with self.assertRaises(ModelEvalError):
            m.predict_proba(np.ones((5, 2)))
        with self.assertRaises(ModelEvalError):
            m.fit(np.ones((5, 2)), np.ones(4))

    def test_evaluate_leakage_gap_detects_positive_auc_inflation_under_late_events(self) -> None:
        corpus = generate_stream_corpus(StreamConfig(seed=20260619, p95_lag_minutes=120.0))
        mats = extract_batch_and_pit_features(corpus.events, corpus.predictions)
        res = evaluate_leakage_gap(
            mats.X_batch,
            mats.X_pit,
            mats.labels,
            mats.corrupted_row_count,
            mats.corrupted_row_pct,
            mats.risk_alert_leak_rows,
        )
        self.assertGreater(res.auc_inflation_points, 2.0)
        self.assertFalse(res.gate_passed)

    def test_evaluate_leakage_gap_passes_when_late_event_rate_is_zero(self) -> None:
        corpus = generate_stream_corpus(StreamConfig(seed=20260619, late_event_rate=0.0))
        mats = extract_batch_and_pit_features(corpus.events, corpus.predictions)
        res = evaluate_leakage_gap(
            mats.X_batch,
            mats.X_pit,
            mats.labels,
            mats.corrupted_row_count,
            mats.corrupted_row_pct,
            mats.risk_alert_leak_rows,
        )
        self.assertLessEqual(abs(res.auc_inflation_points), 1.0)
        self.assertLess(res.corrupted_row_pct, 1.0)
        self.assertTrue(res.gate_passed)


if __name__ == "__main__":
    unittest.main()
