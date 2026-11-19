"""Property tests for multi-cloud manifests (S3/GCS/Azure) and bitemporal SQLite features."""

from __future__ import annotations

import unittest

import numpy as np

from late_event_feature_audit.config import StreamConfig
from late_event_feature_audit.features import extract_batch_and_pit_features
from late_event_feature_audit.manifests import (
    ManifestError,
    build_cloud_manifest,
    detect_cloud_provider,
)
from late_event_feature_audit.stream import (
    PredictionPoint,
    StreamEvent,
    generate_stream_corpus,
)


class TestMultiCloudManifests(unittest.TestCase):
    def test_detect_cloud_provider_identifies_s3_gcs_azure_blob_and_adls(self) -> None:
        self.assertEqual(detect_cloud_provider("s3://bucket/path"), "aws_s3")
        self.assertEqual(detect_cloud_provider("gs://bucket/path"), "gcp_gcs")
        self.assertEqual(detect_cloud_provider("az://container/path"), "azure_blob")
        self.assertEqual(detect_cloud_provider("abfss://fs@acct/path"), "azure_adls")

    def test_detect_cloud_provider_rejects_unsupported_schemes(self) -> None:
        for bad in ("http://example.com", "file:///tmp/x", "hdfs://cluster/data"):
            with self.assertRaises(ManifestError):
                detect_cloud_provider(bad)

    def test_s3_gcs_and_azure_manifests_produce_identical_sha256_for_same_stream(self) -> None:
        corpus = generate_stream_corpus(StreamConfig(seed=13, entity_count=20, prediction_count=80))
        m_s3 = build_cloud_manifest(corpus.events, "s3://prod-events/")
        m_gs = build_cloud_manifest(corpus.events, "gs://prod-events")
        m_az = build_cloud_manifest(corpus.events, "az://prod-events")
        self.assertEqual(m_s3.manifest_sha256, m_gs.manifest_sha256)
        self.assertEqual(m_gs.manifest_sha256, m_az.manifest_sha256)
        self.assertEqual(m_s3.total_events, len(corpus.events))

    def test_manifest_entries_include_hive_date_and_partition_paths(self) -> None:
        corpus = generate_stream_corpus(StreamConfig(seed=13, entity_count=20, prediction_count=80))
        m_az = build_cloud_manifest(corpus.events, "az://prod-events", date_partition="2026-06-20")
        for entry in m_az.entries:
            self.assertIn("dt=2026-06-20/partition=", entry.uri)
            self.assertEqual(entry.provider, "azure_blob")

    def test_build_cloud_manifest_rejects_empty_event_stream(self) -> None:
        with self.assertRaises(ManifestError):
            build_cloud_manifest([], "s3://prod-events")


class TestBitemporalSQLiteFeatures(unittest.TestCase):
    def test_pit_join_excludes_late_event_that_arrived_after_prediction_ts(self) -> None:
        ev_on_time = StreamEvent("E1", "A1", 0, 0, 100.0, 101.0, "page_checkout", 50.0, 0)
        ev_late = StreamEvent("E2", "A1", 0, 1, 110.0, 200.0, "gateway_risk_alert", 300.0, 1)
        pred = PredictionPoint("P1", "A1", 120.0, 1)
        mats = extract_batch_and_pit_features([ev_on_time, ev_late], [pred])
        # Batch sees both events (count=2, risk_alerts=1); PIT sees only E1 (count=1, risk_alerts=0)
        self.assertEqual(mats.X_batch[0, 0], 2.0)
        self.assertEqual(mats.X_batch[0, 3], 1.0)
        self.assertEqual(mats.X_pit[0, 0], 1.0)
        self.assertEqual(mats.X_pit[0, 3], 0.0)
        self.assertEqual(mats.corrupted_row_count, 1)
        self.assertEqual(mats.risk_alert_leak_rows, 1)

    def test_batch_and_pit_features_are_identical_when_zero_events_arrive_late(self) -> None:
        ev1 = StreamEvent("E1", "A1", 0, 0, 100.0, 101.0, "page_checkout", 50.0, 0)
        ev2 = StreamEvent("E2", "A1", 0, 1, 110.0, 112.0, "gateway_risk_alert", 300.0, 1)
        pred = PredictionPoint("P1", "A1", 120.0, 1)
        mats = extract_batch_and_pit_features([ev1, ev2], [pred])
        np.testing.assert_array_equal(mats.X_batch, mats.X_pit)
        self.assertEqual(mats.corrupted_row_count, 0)

    def test_events_older_than_24h_window_are_excluded_by_both_batch_and_pit_sql(self) -> None:
        ev_old = StreamEvent("E1", "A1", 0, 0, 10.0, 11.0, "order_burst", 999.0, 1)
        pred = PredictionPoint("P1", "A1", 2000.0, 0)
        mats = extract_batch_and_pit_features([ev_old], [pred])
        np.testing.assert_array_equal(mats.X_batch[0], np.zeros(5))
        np.testing.assert_array_equal(mats.X_pit[0], np.zeros(5))

    def test_future_events_with_event_ts_after_prediction_ts_are_excluded_by_both_queries(self) -> None:
        ev_future = StreamEvent("E1", "A1", 0, 0, 200.0, 201.0, "gateway_risk_alert", 500.0, 1)
        pred = PredictionPoint("P1", "A1", 150.0, 1)
        mats = extract_batch_and_pit_features([ev_future], [pred])
        np.testing.assert_array_equal(mats.X_batch[0], np.zeros(5))
        np.testing.assert_array_equal(mats.X_pit[0], np.zeros(5))

    def test_pit_features_are_always_elementwise_less_or_equal_to_batch_features(self) -> None:
        corpus = generate_stream_corpus(StreamConfig(seed=20260619, entity_count=25, prediction_count=100))
        mats = extract_batch_and_pit_features(corpus.events, corpus.predictions)
        self.assertTrue(np.all(mats.X_pit <= mats.X_batch + 1e-9))

    def test_extract_features_rejects_empty_predictions_list(self) -> None:
        with self.assertRaises(ValueError):
            extract_batch_and_pit_features([], [])

    def test_extract_features_rejects_impossible_ingestion_before_event_timestamp(self) -> None:
        """Pin clock-skew boundary bug: ingestion_ts < event_ts is physically impossible
        and must raise ValueError rather than silently corrupting bitemporal joins."""
        bad_ev = StreamEvent("E1", "A1", 0, 0, 150.0, 120.0, "page_checkout", 20.0, 0)
        pred = PredictionPoint("P1", "A1", 200.0, 0)
        with self.assertRaises(ValueError):
            extract_batch_and_pit_features([bad_ev], [pred])


if __name__ == "__main__":
    unittest.main()
