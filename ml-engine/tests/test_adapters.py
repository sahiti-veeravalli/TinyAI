import os
import pytest
import numpy as np

from tinyai.data.samples import get_sample_dataset
from tinyai.data.splitter import DataSplitter
from tinyai.models.baseline import BaselineTrainer
from tinyai.adapters.quantization import QuantizationAdapter
from tinyai.adapters.pruning import PruningAdapter
from tinyai.adapters.distillation import DistillationAdapter
from tinyai.adapters.model_selection import ModelSelectionAdapter


@pytest.fixture(scope="module")
def dataset_splits():
    df = get_sample_dataset("fraud_detection_benchmark").head(600)  # fast slice for unit test
    return DataSplitter.split(df, target_column="is_fraud", random_seed=42)


@pytest.fixture(scope="module")
def lightgbm_baseline(dataset_splits):
    return BaselineTrainer.train_and_evaluate("lightgbm", dataset_splits, seed=42)


def test_quantization_adapter(dataset_splits, lightgbm_baseline, tmp_path):
    adapter = QuantizationAdapter()
    assert adapter.supports(lightgbm_baseline, "classification", {})
    ok, _ = adapter.validate_configuration({"precision": "float16"})
    assert ok is True

    res = adapter.optimize(
        model_wrapper=lightgbm_baseline,
        splits=dataset_splits,
        constraints={"precision": "float16"},
        output_dir=str(tmp_path)
    )
    assert res.is_successful is True
    assert res.disk_size_bytes > 0
    assert "accuracy" in res.test_metrics


def test_pruning_adapter(dataset_splits, lightgbm_baseline, tmp_path):
    adapter = PruningAdapter()
    assert adapter.supports(lightgbm_baseline, "classification", {})
    ok, _ = adapter.validate_configuration({"target_sparsity": 0.5})
    assert ok is True

    res = adapter.optimize(
        model_wrapper=lightgbm_baseline,
        splits=dataset_splits,
        constraints={"target_sparsity": 0.5},
        output_dir=str(tmp_path)
    )
    assert res.is_successful is True
    assert res.disk_size_bytes > 0
    assert res.sparsity_ratio > 0.0


def test_distillation_adapter(dataset_splits, lightgbm_baseline, tmp_path):
    adapter = DistillationAdapter()
    assert adapter.supports(lightgbm_baseline, "classification", {})
    ok, _ = adapter.validate_configuration({"temperature": 2.0, "alpha": 0.5})
    assert ok is True

    res = adapter.optimize(
        model_wrapper=lightgbm_baseline,
        splits=dataset_splits,
        constraints={"temperature": 2.0, "alpha": 0.6},
        output_dir=str(tmp_path)
    )
    assert res.is_successful is True
    assert res.disk_size_bytes > 0


def test_model_selection_adapter(dataset_splits, lightgbm_baseline, tmp_path):
    adapter = ModelSelectionAdapter()
    assert adapter.supports(lightgbm_baseline, "classification", {})
    ok, _ = adapter.validate_configuration({"candidate_type": "shallow_tree"})
    assert ok is True

    res = adapter.optimize(
        model_wrapper=lightgbm_baseline,
        splits=dataset_splits,
        constraints={"candidate_type": "shallow_tree"},
        output_dir=str(tmp_path)
    )
    assert res.is_successful is True
    assert res.disk_size_bytes > 0
