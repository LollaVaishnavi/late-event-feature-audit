# ADR-004: Unified Multi-Cloud Partition Manifest Contract (`s3://`, `gs://`, `az://`)

## Status
Accepted

## Context
Enterprise data platforms often split streaming sinks and ML training pipelines across AWS S3, Google Cloud Storage (GCS), and Microsoft Azure Blob / ADLS Gen2 (`az://`, `abfss://`). Separate cloud-specific ingestion scripts easily drift in partition ordering, date-path formatting, or arrival-timestamp handling.

## Decision
Define a unified `CloudManifest` schema (`manifests.py`) that parses `s3://`, `gs://`, `az://`, and `abfss://` roots into normalized Hive-partitioned entries (`dt=YYYY-MM-DD/partition=N/part-NNNN.jsonl`) with canonical SHA-256 content digests sorted by `(partition, kafka_offset)`.

## Alternatives Considered
1. **Cloud-specific SDK loaders (`boto3`, `google-cloud-storage`, `azure-storage-blob`):** Couples offline unit tests to cloud auth libraries and makes cross-cloud parity checks impossible without live credentials.

## Consequences
* Proves byte-identical manifest digests across all 3 cloud providers (`aws_s3`, `gcp_gcs`, `azure_blob`) over the same 4800 streaming events.
* Rejects unsupported URI schemes (`http://`, `ftp://`, `hdfs://`) explicitly at the boundary with `ManifestError`.
