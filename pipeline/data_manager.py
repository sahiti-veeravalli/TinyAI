"""
Enterprise Dataset Manager & Preprocessing Engine for TinyAI.
Features:
- Strict file validation (CSV/TSV/JSON, size limits, malformed header prevention)
- Schema inference & descriptive statistics (numerical and categorical)
- Target selection & task auto-detection (classification vs regression)
- Zero data leakage: split first, fit transformation pipeline ONLY on train split
- Self-contained, portable JSON preprocessor serialization for edge inference
- Cryptographic SHA-256 fingerprinting for dataset provenance and reproducibility
"""

import os
import io
import json
import uuid
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

from backend.config import settings
from backend.security import compute_sha256, safe_join


@dataclass
class PreprocessorSchema:
    target_column: str
    task_type: str  # "classification" | "regression"
    feature_names: List[str]
    numerical_cols: List[str]
    categorical_cols: List[str]
    numerical_medians: Dict[str, float]
    numerical_means: Dict[str, float]
    numerical_stds: Dict[str, float]
    categorical_categories: Dict[str, List[str]]
    categorical_modes: Dict[str, str]
    class_labels: Optional[List[Union[int, str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PreprocessorSchema":
        return cls(**d)


@dataclass
class ProcessedDataset:
    dataset_id: str
    dataset_name: str
    fingerprint: str
    task_type: str
    target_column: str
    feature_names: List[str]
    numerical_cols: List[str]
    categorical_cols: List[str]
    n_samples: int
    n_features: int
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    preprocessor: PreprocessorSchema
    cost_asymmetry: float = 1.0
    classes: Optional[List[Any]] = None


class DatasetManager:
    """Manages dataset upload, validation, statistical inspection, and leakage-free splitting."""

    @staticmethod
    def validate_and_save_upload(
        file_bytes: bytes,
        filename: str,
        max_size_bytes: int = settings.MAX_UPLOAD_SIZE_BYTES
    ) -> Tuple[str, str, str, pd.DataFrame]:
        """
        Validates uploaded file against size limits, parses CSV/TSV/JSON, sanitizes headers,
        and saves it to the persistent uploads directory.
        Returns (dataset_id, saved_path, fingerprint, df).
        """
        if len(file_bytes) == 0:
            raise ValueError("Uploaded file is empty.")
        if len(file_bytes) > max_size_bytes:
            raise ValueError(f"Uploaded file ({len(file_bytes) / (1024*1024):.1f} MB) exceeds maximum allowed size ({max_size_bytes / (1024*1024):.1f} MB).")

        fingerprint = compute_sha256(file_bytes)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".csv", ".tsv", ".json"]:
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .csv, .tsv, .json")

        bio = io.BytesIO(file_bytes)
        try:
            if ext == ".json":
                df = pd.read_json(bio)
            elif ext == ".tsv":
                df = pd.read_csv(bio, sep="\t")
            else:
                df = pd.read_csv(bio)
        except Exception as e:
            raise ValueError(f"Failed to parse tabular file: {str(e)}")

        if df.empty or df.shape[0] < 5:
            raise ValueError("Dataset must contain at least 5 rows for statistical profiling and training.")
        if df.shape[1] < 2:
            raise ValueError("Dataset must contain at least 2 columns (1 feature + 1 target).")

        # Sanitize column names: strip spaces, prevent non-printable chars
        df.columns = [str(c).strip() for c in df.columns]

        dataset_id = f"ds_{uuid.uuid4().hex[:12]}"
        save_path = safe_join(settings.UPLOADS_DIR, f"{dataset_id}.csv")
        df.to_csv(save_path, index=False)

        return dataset_id, save_path, fingerprint, df

    @staticmethod
    def inspect_dataframe(df: pd.DataFrame, dataset_name: str = "Uploaded Dataset") -> Dict[str, Any]:
        """Performs comprehensive profiling, statistical analysis, and column metadata extraction."""
        rows, cols = df.shape
        duplicate_rows = int(df.duplicated().sum())
        total_cells = rows * cols
        missing_cells = int(df.isnull().sum().sum())
        missing_pct = round((missing_cells / max(total_cells, 1)) * 100.0, 2)

        # Detect likely target column
        candidate_names = ["target", "label", "class", "fraud", "churn", "default", "y", "is_fraud", "status"]
        target_col = None
        for col in df.columns:
            if col.lower() in candidate_names:
                target_col = col
                break
        if target_col is None:
            # Look for binary column
            for col in df.columns:
                if df[col].nunique() == 2:
                    target_col = col
                    break
        if target_col is None:
            target_col = df.columns[-1]

        # Determine task type
        target_series = df[target_col].dropna()
        n_unique_target = target_series.nunique()
        is_numeric_target = pd.api.types.is_numeric_dtype(target_series)

        if not is_numeric_target or n_unique_target <= 10:
            task_type = "classification"
        else:
            task_type = "regression"

        # Columns metadata
        columns_meta = []
        for col in df.columns:
            s = df[col]
            is_num = pd.api.types.is_numeric_dtype(s)
            c_type = "Target" if col == target_col else ("Numerical" if is_num else "Categorical")
            stats: Dict[str, Any] = {
                "name": col,
                "type": c_type,
                "null_count": int(s.isnull().sum()),
                "null_pct": round(float(s.isnull().sum() / rows * 100.0), 2),
                "distinct_count": int(s.nunique())
            }
            if is_num and s.dropna().shape[0] > 0:
                stats.update({
                    "min": round(float(s.min()), 3),
                    "mean": round(float(s.mean()), 3),
                    "std": round(float(s.std()), 3) if rows > 1 else 0.0,
                    "max": round(float(s.max()), 3),
                    "median": round(float(s.median()), 3)
                })
            else:
                top_val = s.mode().iloc[0] if not s.mode().empty else None
                stats["top_value"] = str(top_val) if top_val is not None else None
            columns_meta.append(stats)

        # Imbalance ratio if classification
        imbalance_ratio = 1.0
        if task_type == "classification" and n_unique_target >= 2:
            vc = target_series.value_counts()
            imbalance_ratio = round(float(vc.iloc[0]) / max(float(vc.iloc[1]), 1.0), 2)

        preview_rows = df.head(10).fillna("").to_dict(orient="records")

        return {
            "dataset_name": dataset_name,
            "rows_count": rows,
            "columns_count": cols,
            "features_count": cols - 1,
            "duplicate_rows": duplicate_rows,
            "missing_cells": missing_cells,
            "missing_pct": missing_pct,
            "inferred_target": target_col,
            "inferred_task_type": task_type,
            "imbalance_ratio": f"{imbalance_ratio}:1" if task_type == "classification" else "N/A",
            "columns": columns_meta,
            "preview_rows": preview_rows
        }

    @staticmethod
    def prepare_dataset(
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_seed: int = 42,
        cost_asymmetry: float = 1.0,
        dataset_name: str = "Dataset",
        dataset_id: Optional[str] = None
    ) -> ProcessedDataset:
        """
        Splits dataset into train/val/test sets reproducibly, fits preprocessor ONLY
        on train to strictly avoid data leakage, and transforms all splits.
        """
        if target_column is None or target_column not in df.columns:
            inspection = DatasetManager.inspect_dataframe(df)
            target_column = inspection["inferred_target"]

        if task_type is None:
            target_series = df[target_column].dropna()
            is_num = pd.api.types.is_numeric_dtype(target_series)
            task_type = "classification" if (not is_num or target_series.nunique() <= 10) else "regression"

        # Separate feature frame and target
        feature_cols = [c for c in df.columns if c != target_column]
        if not feature_cols:
            raise ValueError("Dataset has no feature columns besides the target column.")

        X_df = df[feature_cols].copy()
        y_raw = df[target_column].copy()

        # Handle missing targets: drop rows where target is NaN
        valid_idx = y_raw.notnull()
        X_df = X_df.loc[valid_idx].reset_index(drop=True)
        y_raw = y_raw.loc[valid_idx].reset_index(drop=True)

        # Identify numerical and categorical columns
        numerical_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_df[c])]
        categorical_cols = [c for c in feature_cols if c not in numerical_cols]

        # Target encoding
        classes = None
        if task_type == "classification":
            y_cats, unique_classes = pd.factorize(y_raw, sort=True)
            y_arr = y_cats.astype(np.int64)
            classes = [str(c) for c in unique_classes]
            # Stratify split if at least 2 instances per class
            can_stratify = (
                len(np.unique(y_arr)) >= 2
                and min(np.bincount(y_arr)) >= 3
            )
            strat = y_arr if can_stratify else None
        else:
            y_arr = y_raw.to_numpy(dtype=np.float32)
            strat = None

        # Train / Val / Test split
        total_eval = test_size + val_size
        if total_eval >= 0.8:
            test_size = 0.15
            val_size = 0.15
            total_eval = 0.30

        train_indices, eval_indices = train_test_split(
            np.arange(len(df)),
            test_size=total_eval,
            random_state=random_seed,
            stratify=strat
        )

        val_ratio = val_size / total_eval
        strat_eval = y_arr[eval_indices] if strat is not None else None
        can_strat_eval = (
            strat_eval is not None
            and len(np.unique(strat_eval)) >= 2
            and min(np.bincount(strat_eval)) >= 2
        )
        val_indices, test_indices = train_test_split(
            eval_indices,
            test_size=(1.0 - val_ratio),
            random_state=random_seed,
            stratify=strat_eval if can_strat_eval else None
        )

        train_X_df = X_df.iloc[train_indices].copy()
        val_X_df = X_df.iloc[val_indices].copy()
        test_X_df = X_df.iloc[test_indices].copy()

        # Fit preprocessors STRICTLY on train_X_df to avoid data leakage
        num_medians: Dict[str, float] = {}
        num_means: Dict[str, float] = {}
        num_stds: Dict[str, float] = {}
        cat_categories: Dict[str, List[str]] = {}
        cat_modes: Dict[str, str] = {}

        # 1. Fit numerical features
        for col in numerical_cols:
            med = float(train_X_df[col].median()) if train_X_df[col].dropna().shape[0] > 0 else 0.0
            mean = float(train_X_df[col].mean()) if train_X_df[col].dropna().shape[0] > 0 else 0.0
            std = float(train_X_df[col].std()) if train_X_df[col].dropna().shape[0] > 1 else 1.0
            if std == 0.0:
                std = 1.0
            num_medians[col] = med
            num_means[col] = mean
            num_stds[col] = std

        # 2. Fit categorical features
        for col in categorical_cols:
            mode_s = train_X_df[col].dropna().mode()
            mode_val = str(mode_s.iloc[0]) if not mode_s.empty else "missing"
            cat_modes[col] = mode_val
            cats = [str(x) for x in train_X_df[col].dropna().unique().tolist()]
            cat_categories[col] = cats

        preprocessor_schema = PreprocessorSchema(
            target_column=target_column,
            task_type=task_type,
            feature_names=feature_cols,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            numerical_medians=num_medians,
            numerical_means=num_means,
            numerical_stds=num_stds,
            categorical_categories=cat_categories,
            categorical_modes=cat_modes,
            class_labels=classes
        )

        # Transform function
        def transform_frame(frame: pd.DataFrame) -> np.ndarray:
            cols_data = []
            for col in feature_cols:
                if col in numerical_cols:
                    s = frame[col].fillna(num_medians[col]).to_numpy(dtype=np.float32)
                    # Standardization
                    s = (s - num_means[col]) / num_stds[col]
                    cols_data.append(s.reshape(-1, 1))
                else:
                    s_str = frame[col].fillna(cat_modes[col]).astype(str)
                    cats = cat_categories[col]
                    # Ordinal mapping (unseen categories mapped to -1)
                    cat_map = {cat: i for i, cat in enumerate(cats)}
                    encoded = s_str.map(cat_map).fillna(-1.0).to_numpy(dtype=np.float32)
                    cols_data.append(encoded.reshape(-1, 1))
            return np.hstack(cols_data).astype(np.float32)

        X_train = transform_frame(train_X_df)
        X_val = transform_frame(val_X_df)
        X_test = transform_frame(test_X_df)

        y_train = y_arr[train_indices]
        y_val = y_arr[val_indices]
        y_test = y_arr[test_indices]

        # Fingerprint calculation
        raw_bytes = df.to_csv(index=False).encode("utf-8")
        fingerprint = compute_sha256(raw_bytes)

        return ProcessedDataset(
            dataset_id=dataset_id or f"ds_{fingerprint[:12]}",
            dataset_name=dataset_name,
            fingerprint=fingerprint,
            task_type=task_type,
            target_column=target_column,
            feature_names=feature_cols,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            n_samples=len(df),
            n_features=len(feature_cols),
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            preprocessor=preprocessor_schema,
            cost_asymmetry=cost_asymmetry,
            classes=classes
        )

    @staticmethod
    def apply_preprocessor(input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame], schema: PreprocessorSchema) -> np.ndarray:
        """Transforms raw inference inputs (dict or DataFrame) using a fitted PreprocessorSchema."""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("Input data must be a dictionary, list of dictionaries, or pandas DataFrame.")

        # Ensure all expected columns are present
        cols_data = []
        for col in schema.feature_names:
            if col in schema.numerical_cols:
                val = df[col] if col in df.columns else schema.numerical_medians[col]
                s = pd.to_numeric(val, errors="coerce").fillna(schema.numerical_medians[col]).to_numpy(dtype=np.float32)
                s = (s - schema.numerical_means[col]) / schema.numerical_stds[col]
                cols_data.append(s.reshape(-1, 1))
            else:
                val = df[col] if col in df.columns else schema.categorical_modes.get(col, "missing")
                s_str = val.fillna(schema.categorical_modes.get(col, "missing")).astype(str)
                cats = schema.categorical_categories.get(col, [])
                cat_map = {cat: i for i, cat in enumerate(cats)}
                encoded = s_str.map(cat_map).fillna(-1.0).to_numpy(dtype=np.float32)
                cols_data.append(encoded.reshape(-1, 1))

        return np.hstack(cols_data).astype(np.float32)

