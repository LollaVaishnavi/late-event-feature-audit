"""Property tests for StreamConfig, LeakagePolicy, Kafka replay, and Spark watermarks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from late_event_feature_audit.config import LeakagePolicy, StreamConfig, StreamConfigError
from late_event_feature_audit.stream import (
    StreamEvent,
    apply_spark_watermark,
    generate_stream_corpus,
)


class TestConfigValidation(unittest.TestCase):
    def test_leakage_policy_round_trips_through_json(self) -> None:
        policy = LeakagePolicy(
            max_auc_inflation=0.03,
            max_corrupted_row_pct=4.5,
            max_watermark_drop_pct=15.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "policy.json"
            p.write_text(json.dumps(policy.to_dict()), encoding="utf-8")
            self.assertEqual(LeakagePolicy.from_json(p), policy)

    def test_leakage_policy_rejects_negative_or_out_of_range_thresholds(self) -> None:
        with self.assertRaises(StreamConfigError):
            LeakagePolicy(max_auc_inflation=-0.01)
        with self.assertRaises(StreamConfigError):
            LeakagePolicy(max_corrupted_row_pct=101.0)
        with self.assertRaises(StreamConfigError):
            LeakagePolicy(max_watermark_drop_pct=-1.0)

    def test_stream_config_rejects_invalid_bounds(self) -> None:
        with self.assertRaises(StreamConfigError):
            StreamConfig(entity_count=5)
        with self.assertRaises(StreamConfigError):
            StreamConfig(prediction_count=10)
        with self.assertRaises(StreamConfigError):
            StreamConfig(events_per_entity=2)
        with self.assertRaises(StreamConfigError):
            StreamConfig(partitions=0)
        with self.assertRaises(StreamConfigError):
            StreamConfig(late_event_rate=1.5)
        with self.assertRaises(StreamConfigError):
            StreamConfig(p95_lag_minutes=0.0)
        with self.assertRaises(StreamConfigError):
            StreamConfig(watermark_minutes=-5.0)


class TestStreamReplayAndWatermarks(unittest.TestCase):
    def test_stream_corpus_generator_is_strictly_deterministic_for_same_seed(self) -> None:
        cfg = StreamConfig(seed=42, entity_count=20, prediction_count=100, events_per_entity=10)
        c1 = generate_stream_corpus(cfg)
        c2 = generate_stream_corpus(cfg)
        self.assertEqual(c1.events, c2.events)
        self.assertEqual(c1.predictions, c2.predictions)

    def test_different_seeds_produce_distinct_event_streams(self) -> None:
        c1 = generate_stream_corpus(StreamConfig(seed=1, entity_count=20, prediction_count=80))
        c2 = generate_stream_corpus(StreamConfig(seed=2, entity_count=20, prediction_count=80))
        self.assertNotEqual(c1.events[0], c2.events[0])

    def test_every_generated_event_has_ingestion_ts_at_or_after_event_ts(self) -> None:
        corpus = generate_stream_corpus(StreamConfig(seed=9, entity_count=20, prediction_count=80))
        for ev in corpus.events:
            self.assertGreaterEqual(ev.ingestion_ts_min, ev.event_ts_min)

    def test_kafka_offsets_are_strictly_contiguous_within_each_partition(self) -> None:
        cfg = StreamConfig(seed=9, entity_count=20, prediction_count=80, partitions=4)
        corpus = generate_stream_corpus(cfg)
        by_part: dict[int, list[int]] = {p: [] for p in range(4)}
        for ev in corpus.events:
            by_part[ev.partition].append(ev.kafka_offset)
        for p_id, offsets in by_part.items():
            self.assertEqual(offsets, list(range(len(offsets))), f"Gap in partition {p_id}")

    def test_lag_percentiles_are_ordered_p50_le_p95_le_p99(self) -> None:
        corpus = generate_stream_corpus(StreamConfig(seed=17, entity_count=20, prediction_count=80))
        self.assertLessEqual(corpus.p50_lag_minutes, corpus.p95_lag_minutes)
        self.assertLessEqual(corpus.p95_lag_minutes, corpus.p99_lag_minutes)

    def test_wider_spark_watermark_drops_fewer_or_equal_events(self) -> None:
        corpus = generate_stream_corpus(StreamConfig(seed=20260619, entity_count=30, prediction_count=120))
        wm5 = apply_spark_watermark(corpus.events, watermark_minutes=5.0)
        wm30 = apply_spark_watermark(corpus.events, watermark_minutes=30.0)
        wm240 = apply_spark_watermark(corpus.events, watermark_minutes=240.0)
        self.assertGreaterEqual(wm5.dropped_events_count, wm30.dropped_events_count)
        self.assertGreaterEqual(wm30.dropped_events_count, wm240.dropped_events_count)

    def test_watermark_accepted_plus_dropped_equals_total_events(self) -> None:
        corpus = generate_stream_corpus(StreamConfig(seed=5, entity_count=20, prediction_count=80))
        wm = apply_spark_watermark(corpus.events, watermark_minutes=15.0)
        self.assertEqual(len(wm.accepted_events) + wm.dropped_events_count, wm.total_events)

    def test_huge_watermark_accepts_every_event_with_zero_drops(self) -> None:
        corpus = generate_stream_corpus(StreamConfig(seed=5, entity_count=20, prediction_count=80))
        wm = apply_spark_watermark(corpus.events, watermark_minutes=100000.0)
        self.assertEqual(wm.dropped_events_count, 0)
        self.assertEqual(len(wm.accepted_events), len(corpus.events))

    def test_watermark_evicts_explicit_late_event_beyond_cutoff(self) -> None:
        e1 = StreamEvent("E1", "A1", 0, 0, 100.0, 101.0, "page_checkout", 10.0, 0)
        e2 = StreamEvent("E2", "A1", 0, 1, 50.0, 102.0, "gateway_risk_alert", 200.0, 1)
        wm = apply_spark_watermark([e1, e2], watermark_minutes=20.0)
        self.assertEqual(wm.dropped_events_count, 1)
        self.assertEqual([e.event_id for e in wm.accepted_events], ["E1"])

    def test_watermark_rejects_negative_cutoff(self) -> None:
        with self.assertRaises(ValueError):
            apply_spark_watermark([], watermark_minutes=-1.0)

    def test_watermark_handles_empty_event_list_cleanly(self) -> None:
        wm = apply_spark_watermark([], watermark_minutes=15.0)
        self.assertEqual(wm.total_events, 0)
        self.assertEqual(wm.dropped_events_count, 0)
        self.assertEqual(wm.dropped_late_share_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
