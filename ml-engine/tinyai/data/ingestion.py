import hashlib
import io
import os
from typing import Optional, Dict, Any, List
import pandas as pd
from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    dataset_id: str
    name: str
    fingerprint: str
    format: str
    row_count: int
    column_count: int
    memory_bytes: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    all_columns: List[str]
    preview_rows: List[Dict[str, Any]]
    summary_stats: Dict[str, Dict[str, Any]]


class DatasetIngestionEngine:
    """Safe ingestion engine for CSV and Parquet tabular data with size quotas and fingerprinting."""

    MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB default limit for memory safety
    MAX_ROWS = 500_000

    @classmethod
    def compute_fingerprint(cls, content: bytes) -> str:
        """Computes deterministic SHA-256 hash of dataset content."""
        hasher = hashlib.sha256()
        hasher.update(content)
        return hasher.hexdigest()

    @classmethod
    def parse_data(
        cls,
        content: bytes,
        filename: str,
        max_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """Safely parses CSV or Parquet bytes into a pandas DataFrame."""
        if len(content) > cls.MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File size {len(content) / (1024*1024):.2f}MB exceeds safety limit of {cls.MAX_FILE_SIZE_BYTES / (1024*1024):.2f}MB"
            )

        lower_name = filename.lower()
        if lower_name.endswith(".parquet") or lower_name.endswith(".pq"):
            df = pd.read_parquet(io.BytesIO(content))
        elif lower_name.endswith(".csv") or lower_name.endswith(".txt"):
            # Enforce safe CSV parsing without executing any formulas or code
            df = pd.read_csv(io.BytesIO(content), nrows=max_rows or cls.MAX_ROWS)
        else:
            raise ValueError(f"Unsupported file format '{filename}'. Allowed formats: .csv, .parquet")

        if len(df) == 0:
            raise ValueError("Dataset is empty (0 rows).")

        return df

    @classmethod
    def extract_metadata(
        cls,
        df: pd.DataFrame,
        dataset_id: str,
        name: str,
        fingerprint: str,
        fmt: str
    ) -> DatasetMetadata:
        """Extracts structural metadata, column types, and statistical summaries."""
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
        all_cols = df.columns.tolist()

        # Preview: 10 rows converted to json-compatible dicts
        preview = df.head(10).replace({float("nan"): None}).to_dict(orient="records")

        # Basic summary stats for numeric columns
        stats: Dict[str, Dict[str, Any]] = {}
        for col in numeric_cols[:25]:  # limit to first 25 columns for concise metadata
            series = df[col].dropna()
            if len(series) > 0:
                stats[col] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()) if len(series) > 1 else 0.0,
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                    "null_count": int(df[col].isna().sum()),
                }

        return DatasetMetadata(
            dataset_id=dataset_id,
            name=name,
            fingerprint=fingerprint,
            format=fmt,
            row_count=len(df),
            column_count=len(df.columns),
            memory_bytes=int(df.memory_usage(deep=True).sum()),
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            all_columns=all_cols,
            preview_rows=preview,
            summary_stats=stats,
        )
