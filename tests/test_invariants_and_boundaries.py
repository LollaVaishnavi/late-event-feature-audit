"""Additional bitemporal, multi-cloud, watermark, and ROC-AUC boundary property tests."""

from __future__ import annotations

import unittest

import numpy as np

from late_event_feature_audit.config import StreamConfig
from late_event_feature_audit.features import FEATURE_NAMES, extract_batch_and_pit_features
from late_event_feature_audit.manifests import build_cloud_manifest
from late_event_feature_audit.model import LogisticRiskModel, evaluate_leakage_gap, roc_auc_score
from late_event_feature_audit.stream import (
    PredictionPoint,
    StreamEvent,
    apply_spark_watermark,
    generate_stream_corpus,
)


class TestBitemporalAndManifestInvariants(unittest.TestCase):
    def test_feature_names_tuple_matches_five_sql_output_columns(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), 5)
        corpus = generate_stream_corpus(StreamConfig(seed=3, entity_count=15, prediction_count=60))
        mats = extract_batch_and_pit_features(corpus.events, corpus.predictions)
        self.assertEqual(mats.X_batch.shape[1], len(FEATURE_NAMES))
        self.assertEqual(mats.X_pit.shape[1], len(FEATURE_NAMES))

    def test_prediction_ids_are_returned_in_deterministic_sorted_order(self) -> None:
        preds = [
            PredictionPoint("P-003", "A1", 200.0, 0),
            PredictionPoint("P-001", "A1", 200.0, 1),
            PredictionPoint("P-002", "A1", 200.0, 0),
        ]
        mats = extract_batch_and_pit_features([], preds)
        self.assertEqual(mats.prediction_ids, ("P-001", "P-002", "P-003"))

    def test_event_landing_exactly_at_prediction_ts_is_included_in_both_batch_and_pit(self) -> None:
        ev = StreamEvent("E1", "A1", 0, 0, 100.0, 150.0, "order_burst", 80.0, 0)
        pred = PredictionPoint("P1", "A1", 150.0, 1)
        mats = extract_batch_and_pit_features([ev], [pred])
        self.assertEqual(mats.X_batch[0, 0], 1.0)
        self.assertEqual(mats.X_pit[0, 0], 1.0)
        self.assertEqual(mats.corrupted_row_count, 0)

    def test_event_landing_one_minute_after_prediction_ts_is_excluded_from_pit_only(self) -> None:
        ev = StreamEvent("E1", "A1", 0, 0, 100.0, 151.0, "order_burst", 80.0, 0)
        pred = PredictionPoint("P1", "A1", 150.0, 1)
        mats = extract_batch_and_pit_features([ev], [pred])
        self.assertEqual(mats.X_batch[0, 0], 1.0)
        self.assertEqual(mats.X_pit[0, 0], 0.0)
        self.assertEqual(mats.corrupted_row_count, 1)

    def test_events_for_other_entities_never_contaminate_target_entity_features(self) -> None:
        ev = StreamEvent("E1", "OTHER", 0, 0, 100.0, 101.0, "gateway_risk_alert", 900.0, 1)
        pred = PredictionPoint("P1", "TARGET", 150.0, 0)
        mats = extract_batch_and_pit_features([ev], [pred])
        np.testing.assert_array_equal(mats.X_batch[0], np.zeros(5))
        np.testing.assert_array_equal(mats.X_pit[0], np.zeros(5))

    def test_amount_sum_and_max_amount_aggregate_correctly_across_multiple_events(self) -> None:
        e1 = StreamEvent("E1", "A1", 0, 0, 100.0, 101.0, "page_checkout", 40.0, 0)
        e2 = StreamEvent("E2", "A1", 0, 1, 110.0, 111.0, "order_burst", 160.0, 0)
        pred = PredictionPoint("P1", "A1", 150.0, 1)
        mats = extract_batch_and_pit_features([e1, e2], [pred])
        self.assertAlmostEqual(mats.X_pit[0, 0], 2.0)
        self.assertAlmostEqual(mats.X_pit[0, 1], 200.0)
        self.assertAlmostEqual(mats.X_pit[0, 2], 1.0)
        self.assertAlmostEqual(mats.X_pit[0, 4], 160.0)

    def test_corrupted_row_pct_is_bounded_between_0_and_100(self) -> None:
        cfg = StreamConfig(seed=44, entity_count=20, prediction_count=100)
        corpus = generate_stream_corpus(cfg)
        mats = extract_batch_and_pit_features(corpus.events, corpus.predictions)
        self.assertGreaterEqual(mats.corrupted_row_pct, 0.0)
        self.assertLessEqual(mats.corrupted_row_pct, 100.0)

    def test_risk_alert_leak_rows_never_exceed_total_corrupted_rows(self) -> None:
        cfg = StreamConfig(seed=44, entity_count=20, prediction_count=100)
        corpus = generate_stream_corpus(cfg)
        mats = extract_batch_and_pit_features(corpus.events, corpus.predictions)
        self.assertLessEqual(mats.risk_alert_leak_rows, mats.corrupted_row_count)

    def test_manifest_digest_changes_when_a_single_event_amount_changes(self) -> None:
        e1 = StreamEvent("E1", "A1", 0, 0, 100.0, 101.0, "page_checkout", 40.0, 0)
        e2 = StreamEvent("E1", "A1", 0, 0, 100.0, 101.0, "page_checkout", 40.01, 0)
        m1 = build_cloud_manifest([e1], "s3://b")
        m2 = build_cloud_manifest([e2], "s3://b")
        self.assertNotEqual(m1.manifest_sha256, m2.manifest_sha256)

    def test_manifest_digest_is_invariant_to_input_event_list_shuffling(self) -> None:
        e1 = StreamEvent("E1", "A1", 0, 0, 100.0, 101.0, "page_checkout", 40.0, 0)
        e2 = StreamEvent("E2", "A1", 0, 1, 105.0, 106.0, "page_checkout", 50.0, 0)
        m1 = build_cloud_manifest([e1, e2], "gs://b")
        m2 = build_cloud_manifest([e2, e1], "gs://b")
        self.assertEqual(m1.manifest_sha256, m2.manifest_sha256)

    def test_azure_adls_abfss_scheme_matches_aws_s3_manifest_digest(self) -> None:
        corpus = generate_stream_corpus(StreamConfig(seed=8, entity_count=15, prediction_count=60))
        m_s3 = build_cloud_manifest(corpus.events, "s3://bucket")
        m_adls = build_cloud_manifest(corpus.events, "abfss://fs@acct.dfs.core.windows.net/events")
        self.assertEqual(m_s3.manifest_sha256, m_adls.manifest_sha256)
        self.assertEqual(m_adls.provider, "azure_adls")

    def test_manifest_partition_event_counts_sum_to_total_events(self) -> None:
        corpus = generate_stream_corpus(StreamConfig(seed=8, entity_count=20, prediction_count=80))
        m = build_cloud_manifest(corpus.events, "az://container/events")
        self.assertEqual(sum(e.event_count for e in m.entries), m.total_events)

    def test_manifest_strips_trailing_slash_from_bucket_root(self) -> None:
        e1 = StreamEvent("E1", "A1", 0, 0, 100.0, 101.0, "page_checkout", 40.0, 0)
        m = build_cloud_manifest([e1], "gs://my-bucket///")
        self.assertEqual(m.bucket_root, "gs://my-bucket")
        self.assertNotIn("///", m.entries[0].uri)

    def test_watermark_zero_drops_all_strictly_out_of_order_events(self) -> None:
        e1 = StreamEvent("E1", "A1", 0, 0, 200.0, 201.0, "page_checkout", 10.0, 0)
        e2 = StreamEvent("E2", "A1", 0, 1, 199.0, 202.0, "page_checkout", 10.0, 0)
        wm = apply_spark_watermark([e1, e2], watermark_minutes=0.0)
        self.assertEqual(wm.dropped_events_count, 1)

    def test_watermark_accepts_event_at_exact_cutoff_boundary(self) -> None:
        e1 = StreamEvent("E1", "A1", 0, 0, 200.0, 201.0, "page_checkout", 10.0, 0)
        e2 = StreamEvent("E2", "A1", 0, 1, 185.0, 202.0, "page_checkout", 10.0, 0)
        wm = apply_spark_watermark([e1, e2], watermark_minutes=15.0)
        self.assertEqual(wm.dropped_events_count, 0)

    def test_longer_p95_lag_produces_more_or_equal_corrupted_rows_than_short_lag(self) -> None:
        c_short = generate_stream_corpus(
            StreamConfig(seed=20260619, entity_count=40, prediction_count=300, p95_lag_minutes=5.0)
        )
        c_long = generate_stream_corpus(
            StreamConfig(
                seed=20260619, entity_count=40, prediction_count=300, p95_lag_minutes=360.0
            )
        )
        m_short = extract_batch_and_pit_features(c_short.events, c_short.predictions)
        m_long = extract_batch_and_pit_features(c_long.events, c_long.predictions)
        self.assertGreater(m_long.corrupted_row_count, m_short.corrupted_row_count)

    def test_roc_auc_with_partial_ties_computes_exact_mann_whitney_value(self) -> None:
        y = np.array([0, 0, 1, 1])
        s = np.array([0.2, 0.5, 0.5, 0.9])
        # Pairs: (0.2<0.5)=1, (0.2<0.9)=1, (0.5==0.5)=0.5, (0.5<0.9)=1 => 3.5 / 4 = 0.875
        self.assertAlmostEqual(roc_auc_score(y, s), 0.875, places=10)

    def test_logistic_model_stronger_regularization_shrinks_weight_l2_norm(self) -> None:
        rng = np.random.default_rng(5)
        X = rng.normal(size=(150, 4))
        y = (X[:, 0] - 0.5 * X[:, 1] > 0.0).astype(int)
        m_weak = LogisticRiskModel(l2_reg=0.1).fit(X, y)
        m_strong = LogisticRiskModel(l2_reg=50.0).fit(X, y)
        self.assertLess(
            float(np.linalg.norm(m_strong.weights_)),
            float(np.linalg.norm(m_weak.weights_)),
        )

    def test_evaluate_leakage_gap_fails_gate_when_corrupted_rows_breach_threshold(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 5))
        y = (X[:, 0] > 0.0).astype(int)
        res = evaluate_leakage_gap(
            X, X, y, corrupted_row_count=12, corrupted_row_pct=12.0, risk_alert_leak_rows=0,
            max_auc_inflation=0.50, max_corrupted_row_pct=5.0,
        )
        self.assertFalse(res.gate_passed)

    def test_evaluate_leakage_gap_reports_zero_inflation_when_batch_equals_pit(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 5))
        y = (X[:, 0] > 0.0).astype(int)
        res = evaluate_leakage_gap(X, X, y, 0, 0.0, 0)
        self.assertAlmostEqual(res.auc_inflation_points, 0.0)
        self.assertEqual(res.batch_offline_auc, res.batch_model_on_online_auc)
        self.assertTrue(res.gate_passed)

    def test_corpus_partition_assignment_covers_all_configured_partitions(self) -> None:
        corpus = generate_stream_corpus(
            StreamConfig(seed=20260619, entity_count=40, prediction_count=100, partitions=4)
        )
        self.assertEqual({e.partition for e in corpus.events}, {0, 1, 2, 3})

    def test_corpus_event_ids_and_prediction_ids_are_unique(self) -> None:
        cfg = StreamConfig(seed=20260619, entity_count=30, prediction_count=90)
        corpus = generate_stream_corpus(cfg)
        self.assertEqual(len({e.event_id for e in corpus.events}), len(corpus.events))
        pred_ids = {p.prediction_id for p in corpus.predictions}
        self.assertEqual(len(pred_ids), len(corpus.predictions))

    def test_corpus_labels_are_strictly_binary_0_or_1(self) -> None:
        cfg = StreamConfig(seed=20260619, entity_count=30, prediction_count=150)
        corpus = generate_stream_corpus(cfg)
        self.assertEqual({p.label for p in corpus.predictions}, {0, 1})

    def test_corpus_amounts_are_strictly_positive(self) -> None:
        cfg = StreamConfig(seed=20260619, entity_count=20, prediction_count=80)
        corpus = generate_stream_corpus(cfg)
        self.assertTrue(all(e.amount > 0.0 for e in corpus.events))

    def test_watermark_replay_preserves_ingestion_order_of_accepted_events(self) -> None:
        cfg = StreamConfig(seed=20260619, entity_count=20, prediction_count=80)
        corpus = generate_stream_corpus(cfg)
        wm = apply_spark_watermark(corpus.events, watermark_minutes=30.0)
        ing_ts = [e.ingestion_ts_min for e in wm.accepted_events]
        self.assertEqual(ing_ts, sorted(ing_ts))

    def test_watermark_dropped_late_share_is_zero_when_zero_events_dropped(self) -> None:
        e1 = StreamEvent("E1", "A1", 0, 0, 100.0, 120.0, "page_checkout", 10.0, 0)
        wm = apply_spark_watermark([e1], watermark_minutes=60.0)
        self.assertEqual(wm.dropped_events_count, 0)
        self.assertEqual(wm.dropped_late_share_pct, 0.0)

    def test_manifest_min_event_ts_is_less_or_equal_to_max_ingestion_ts(self) -> None:
        cfg = StreamConfig(seed=20260619, entity_count=20, prediction_count=80)
        corpus = generate_stream_corpus(cfg)
        m = build_cloud_manifest(corpus.events, "s3://prod-events")
        for entry in m.entries:
            self.assertLessEqual(entry.min_event_ts, entry.max_ingestion_ts)

    def test_pit_honest_model_predictions_are_bounded_probabilities(self) -> None:
        cfg = StreamConfig(seed=20260619, entity_count=40, prediction_count=200)
        corpus = generate_stream_corpus(cfg)
        mats = extract_batch_and_pit_features(corpus.events, corpus.predictions)
        m = LogisticRiskModel().fit(mats.X_pit[:120], mats.labels[:120])
        probs = m.predict_proba(mats.X_pit[120:])
        self.assertTrue(np.all((probs > 0.0) & (probs < 1.0)))

    def test_roc_auc_rejects_empty_arrays(self) -> None:
        with self.assertRaises(Exception):
            roc_auc_score(np.array([]), np.array([]))

    def test_logistic_model_rejects_1d_feature_matrix(self) -> None:
        with self.assertRaises(Exception):
            LogisticRiskModel().fit(np.ones(10), np.ones(10))

    def test_stream_config_accepts_zero_watermark_and_zero_late_rate(self) -> None:
        cfg = StreamConfig(late_event_rate=0.0, watermark_minutes=0.0)
        self.assertEqual(cfg.late_event_rate, 0.0)
        self.assertEqual(cfg.watermark_minutes, 0.0)

    def test_event_exactly_24h_old_is_excluded_by_strict_lower_window_bound(self) -> None:
        ev = StreamEvent("E1", "A1", 0, 0, 100.0, 101.0, "order_burst", 90.0, 0)
        pred = PredictionPoint("P1", "A1", 1540.0, 0)
        mats = extract_batch_and_pit_features([ev], [pred])
        self.assertEqual(mats.X_pit[0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
