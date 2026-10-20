import os
import pytest
import pickle
import numpy as np

from tinyai.data.samples import get_sample_dataset
from tinyai.data.splitter import DataSplitter
from tinyai.models.baseline import BaselineTrainer
from tinyai.export.bundle import DeploymentPackager


def test_bundle_packager(tmp_path):
    df = get_sample_dataset("fraud_detection_benchmark").head(400)
    splits = DataSplitter.split(df, target_column="is_fraud", random_seed=42)
    wrapper = BaselineTrainer.train_and_evaluate("logistic_regression", splits, seed=42)

    # Save mock artifact
    model_path = os.path.join(tmp_path, "orig_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(wrapper.model, f)

    bundle_dir = os.path.join(tmp_path, "bundle_test")
    bundle_files = DeploymentPackager.generate_bundle(
        bundle_dir=bundle_dir,
        model_artifact_path=model_path,
        preprocessor=splits.preprocessor,
        feature_names=splits.feature_names,
        model_name="Test LR",
        model_family="logistic_regression",
        quality_score=0.90,
        latency_ms=0.25,
        disk_size_kb=15.0,
        target_column="is_fraud"
    )

    # Check generated files
    assert os.path.exists(bundle_files["model_path"])
    assert os.path.exists(bundle_files["preprocessor_path"])
    assert os.path.exists(bundle_files["app_path"])
    assert os.path.exists(bundle_files["dockerfile_path"])
    assert os.path.exists(bundle_files["requirements_path"])
    assert os.path.exists(bundle_files["model_card_path"])
    assert os.path.exists(bundle_files["readme_path"])

    # Test that the exported model and preprocessor can actually be reloaded and run predictions!
    with open(bundle_files["model_path"], "rb") as f:
        reloaded_model = pickle.load(f)
    with open(bundle_files["preprocessor_path"], "rb") as f:
        reloaded_prep = pickle.load(f)

    sample_row = df.head(1).drop(columns=["is_fraud"])
    X = reloaded_prep.transform(sample_row)
    preds = reloaded_model.predict(X)
    assert len(preds) == 1
    assert preds[0] in [0, 1]
