import pytest
import pandas as pd
import numpy as np

from tinyai.data.samples import get_sample_dataset, list_sample_datasets
from tinyai.data.ingestion import DatasetIngestionEngine
from tinyai.data.validation import DatasetValidator
from tinyai.data.splitter import DataSplitter


def test_sample_datasets():
    samples = list_sample_datasets()
    assert len(samples) >= 2
    fraud_df = get_sample_dataset("fraud_detection_benchmark")
    assert len(fraud_df) > 1000
    assert "is_fraud" in fraud_df.columns
    assert "transaction_amount" in fraud_df.columns


def test_dataset_ingestion_and_fingerprinting():
    df = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0],
        "feature_b": ["x", "y", "x", "z"],
        "target": [0, 1, 0, 1]
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    fp1 = DatasetIngestionEngine.compute_fingerprint(csv_bytes)
    fp2 = DatasetIngestionEngine.compute_fingerprint(csv_bytes)
    assert fp1 == fp2
    assert len(fp1) == 64

    parsed_df = DatasetIngestionEngine.parse_data(csv_bytes, "test.csv")
    assert len(parsed_df) == 4
    meta = DatasetIngestionEngine.extract_metadata(parsed_df, "ds_1", "Test", fp1, "csv")
    assert meta.row_count == 4
    assert meta.column_count == 3
    assert "feature_a" in meta.numeric_columns
    assert "feature_b" in meta.categorical_columns


def test_dataset_validation():
    df = get_sample_dataset("fraud_detection_benchmark")
    report = DatasetValidator.validate(df, target_column="is_fraud", task_type="classification")
    assert report.is_valid is True
    assert report.task_type == "classification"
    assert report.imbalance_ratio is not None
    assert report.imbalance_ratio > 5.0  # Demonstrates class imbalance detection


def test_leakage_free_splitter():
    df = get_sample_dataset("fraud_detection_benchmark")
    splits = DataSplitter.split(df, target_column="is_fraud", random_seed=42)
    assert len(splits.X_train) > len(splits.X_val)
    assert len(splits.X_train) > len(splits.X_test)
    assert splits.X_train.shape[1] == splits.X_val.shape[1] == splits.X_test.shape[1]
    # Check that preprocessor produces clean non-nan arrays
    assert not np.isnan(splits.X_train).any()
    assert not np.isnan(splits.X_test).any()
