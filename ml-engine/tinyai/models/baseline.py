import os
import time
import pickle
from typing import Dict, Any, Optional, Tuple
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb

from .metrics import calculate_classification_metrics, calculate_regression_metrics
from ..data.splitter import DatasetSplits


class BaselineModelWrapper:
    """Wraps scikit-learn, LightGBM, and XGBoost models with unified training, prediction, and serialization."""

    def __init__(self, model_family: str, model_instance: Any, hyperparams: Dict[str, Any]):
        self.model_family = model_family
        self.model = model_instance
        self.hyperparams = hyperparams
        self.training_time_sec: float = 0.0
        self.validation_metrics: Dict[str, float] = {}
        self.test_metrics: Dict[str, float] = {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def save(self, filepath: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "model_family": self.model_family,
                "hyperparams": self.hyperparams,
                "model": self.model,
                "training_time_sec": self.training_time_sec,
                "validation_metrics": self.validation_metrics,
                "test_metrics": self.test_metrics
            }, f)
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "BaselineModelWrapper":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        wrapper = cls(data["model_family"], data["model"], data["hyperparams"])
        wrapper.training_time_sec = data.get("training_time_sec", 0.0)
        wrapper.validation_metrics = data.get("validation_metrics", {})
        wrapper.test_metrics = data.get("test_metrics", {})
        return wrapper


class BaselineTrainer:
    """Trains and benchmarks baseline tabular models across diverse algorithmic paradigms."""

    SUPPORTED_FAMILIES = [
        "logistic_regression",
        "random_forest",
        "lightgbm",
        "xgboost",
        "tabular_mlp"
    ]

    @classmethod
    def create_model(cls, family: str, seed: int = 42, custom_params: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        params = custom_params or {}

        if family == "logistic_regression":
            merged = {"C": 1.0, "max_iter": 500, "random_state": seed, "solver": "lbfgs"}
            merged.update(params)
            return LogisticRegression(**merged), merged

        elif family == "random_forest":
            merged = {"n_estimators": 100, "max_depth": 12, "random_state": seed, "n_jobs": -1}
            merged.update(params)
            return RandomForestClassifier(**merged), merged

        elif family == "lightgbm":
            merged = {
                "n_estimators": 120,
                "max_depth": 6,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "random_state": seed,
                "verbosity": -1,
                "n_jobs": -1
            }
            merged.update(params)
            return lgb.LGBMClassifier(**merged), merged

        elif family == "xgboost":
            merged = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.05,
                "random_state": seed,
                "eval_metric": "logloss",
                "n_jobs": -1
            }
            merged.update(params)
            return xgb.XGBClassifier(**merged), merged

        elif family == "tabular_mlp":
            merged = {
                "hidden_layer_sizes": (64, 32),
                "activation": "relu",
                "max_iter": 150,
                "random_state": seed,
                "early_stopping": True
            }
            merged.update(params)
            return MLPClassifier(**merged), merged

        else:
            raise ValueError(f"Unsupported model family '{family}'. Supported: {cls.SUPPORTED_FAMILIES}")

    @classmethod
    def train_and_evaluate(
        cls,
        family: str,
        splits: DatasetSplits,
        seed: int = 42,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> BaselineModelWrapper:
        model, hparams = cls.create_model(family, seed=seed, custom_params=custom_params)

        # Train model with timing
        t0 = time.perf_counter()
        model.fit(splits.X_train, splits.y_train)
        train_duration = time.perf_counter() - t0

        wrapper = BaselineModelWrapper(family, model, hparams)
        wrapper.training_time_sec = float(train_duration)

        # Evaluate on validation set
        val_pred = wrapper.predict(splits.X_val)
        val_prob = wrapper.predict_proba(splits.X_val)
        wrapper.validation_metrics = calculate_classification_metrics(splits.y_val, val_pred, val_prob)

        # Evaluate on test set
        test_pred = wrapper.predict(splits.X_test)
        test_prob = wrapper.predict_proba(splits.X_test)
        wrapper.test_metrics = calculate_classification_metrics(splits.y_test, test_pred, test_prob)

        return wrapper
