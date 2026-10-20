from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer


class DatasetSplits:
    def __init__(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        preprocessor: Any,
        metadata: Dict[str, Any]
    ):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.feature_names = feature_names
        self.preprocessor = preprocessor
        self.metadata = metadata


class TabularPreprocessor:
    """Preprocesses tabular data with leakage-free train fitting: imputation, one-hot encoding, and scaling."""

    def __init__(self, numeric_cols: List[str], categorical_cols: List[str]):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.cat_categories: Dict[str, List[Any]] = {}
        self.feature_names_out: List[str] = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        self.feature_names_out = []

        # Fit numeric
        if self.numeric_cols:
            num_data = df[self.numeric_cols].values
            self.num_imputer.fit(num_data)
            imputed_num = self.num_imputer.transform(num_data)
            self.scaler.fit(imputed_num)
            self.feature_names_out.extend(self.numeric_cols)

        # Fit categoricals (extract top categories for one-hot)
        if self.categorical_cols:
            for col in self.categorical_cols:
                top_cats = df[col].dropna().value_counts().head(10).index.tolist()
                self.cat_categories[col] = top_cats
                for cat in top_cats:
                    self.feature_names_out.append(f"{col}_{cat}")
                self.feature_names_out.append(f"{col}_OTHER")

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")

        parts: List[np.ndarray] = []

        # Numeric transform
        if self.numeric_cols:
            # Handle missing columns if any
            for c in self.numeric_cols:
                if c not in df.columns:
                    df[c] = np.nan
            num_data = df[self.numeric_cols].values
            imputed = self.num_imputer.transform(num_data)
            scaled = self.scaler.transform(imputed)
            parts.append(scaled)

        # Categorical transform
        if self.categorical_cols:
            cat_arrays = []
            for col in self.categorical_cols:
                series = df[col] if col in df.columns else pd.Series([np.nan] * len(df))
                categories = self.cat_categories[col]
                # Matrix of [N, len(categories) + 1]
                encoded = np.zeros((len(df), len(categories) + 1), dtype=np.float32)
                for idx, cat in enumerate(categories):
                    encoded[:, idx] = (series == cat).astype(np.float32)
                # OTHER column
                is_known = np.zeros(len(df), dtype=bool)
                for cat in categories:
                    is_known |= (series == cat).values
                encoded[:, -1] = (~is_known).astype(np.float32)
                cat_arrays.append(encoded)

            if cat_arrays:
                parts.append(np.hstack(cat_arrays))

        if not parts:
            raise ValueError("No valid features transformed.")

        return np.hstack(parts).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


class DataSplitter:
    """Manages reproducible train/val/test splits without data leakage."""

    @classmethod
    def split(
        cls,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_seed: int = 42
    ) -> DatasetSplits:
        # Drop rows where target is missing
        clean_df = df.dropna(subset=[target_column]).copy()

        # Separate target
        y_series = clean_df[target_column]
        if task_type == "classification":
            # Map classes to clean integer indices
            unique_classes = sorted(y_series.unique().tolist())
            class_map = {val: idx for idx, val in enumerate(unique_classes)}
            y = y_series.map(class_map).values.astype(np.int64)
        else:
            y = y_series.values.astype(np.float32)

        X_df = clean_df.drop(columns=[target_column])

        # Feature separation
        numeric_cols = X_df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X_df.select_dtypes(exclude=["number"]).columns.tolist()

        # Stratification flag
        stratify = y if task_type == "classification" and len(np.unique(y)) > 1 else None

        # Split 1: Train+Val vs Test
        X_train_val, X_test_df, y_train_val, y_test = train_test_split(
            X_df, y, test_size=test_size, random_state=random_seed, stratify=stratify
        )

        # Split 2: Train vs Val
        stratify_val = y_train_val if stratify is not None else None
        relative_val_size = val_size / (1.0 - test_size)
        X_train_df, X_val_df, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=relative_val_size, random_state=random_seed, stratify=stratify_val
        )

        # Preprocessing: FIT ONLY ON TRAIN TO AVOID LEAKAGE
        preprocessor = TabularPreprocessor(numeric_cols, categorical_cols)
        preprocessor.fit(X_train_df)

        X_train = preprocessor.transform(X_train_df)
        X_val = preprocessor.transform(X_val_df)
        X_test = preprocessor.transform(X_test_df)

        metadata = {
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
            "feature_dim": int(X_train.shape[1]),
            "random_seed": random_seed,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "target_column": target_column,
            "task_type": task_type,
        }

        return DatasetSplits(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=preprocessor.feature_names_out,
            preprocessor=preprocessor,
            metadata=metadata
        )
