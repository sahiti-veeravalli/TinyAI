from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    severity: str  # "ERROR", "WARNING", "INFO"
    code: str
    column: Optional[str] = None
    message: str
    recommendation: str


class ValidationReport(BaseModel):
    is_valid: bool
    task_type: str  # "classification" or "regression"
    target_column: str
    row_count: int
    column_count: int
    missing_value_count: int
    duplicate_rows_count: int
    class_distribution: Optional[Dict[str, int]] = None
    imbalance_ratio: Optional[float] = None
    issues: List[ValidationIssue] = Field(default_factory=list)
    feature_columns: List[str] = Field(default_factory=list)


class DatasetValidator:
    """Validates tabular datasets, verifies target feasibility, inspects class imbalance, and checks for data leakage."""

    @classmethod
    def validate(
        cls,
        df: pd.DataFrame,
        target_column: str,
        task_type: Optional[str] = None
    ) -> ValidationReport:
        issues: List[ValidationIssue] = []

        # 1. Target column existence
        if target_column not in df.columns:
            return ValidationReport(
                is_valid=False,
                task_type=task_type or "unknown",
                target_column=target_column,
                row_count=len(df),
                column_count=len(df.columns),
                missing_value_count=int(df.isna().sum().sum()),
                duplicate_rows_count=int(df.duplicated().sum()),
                issues=[
                    ValidationIssue(
                        severity="ERROR",
                        code="MISSING_TARGET",
                        column=target_column,
                        message=f"Target column '{target_column}' does not exist in dataset.",
                        recommendation="Select an existing column as the optimization target."
                    )
                ]
            )

        target_series = df[target_column]

        # 2. Target null checks
        target_nulls = int(target_series.isna().sum())
        if target_nulls > 0:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    code="TARGET_HAS_NULLS",
                    column=target_column,
                    message=f"Target column contains {target_nulls} missing values ({(target_nulls / len(df))*100:.1f}%).",
                    recommendation="Missing target rows will be dropped during splitting."
                )
            )

        # 3. Detect task type if not explicitly supplied
        unique_target_count = target_series.nunique(dropna=True)
        inferred_task = task_type
        if not inferred_task:
            if unique_target_count == 2:
                inferred_task = "classification"
            elif unique_target_count < 20 and (target_series.dtype == "object" or target_series.dtype.name == "category"):
                inferred_task = "classification"
            elif pd.api.types.is_numeric_dtype(target_series) and unique_target_count > 50:
                inferred_task = "regression"
            else:
                inferred_task = "classification"

        # 4. Class distribution & Imbalance analysis for classification
        class_dist: Optional[Dict[str, int]] = None
        imbalance_ratio: Optional[float] = None
        if inferred_task == "classification":
            val_counts = target_series.dropna().value_counts()
            class_dist = {str(k): int(v) for k, v in val_counts.items()}
            if len(val_counts) >= 2:
                max_count = val_counts.iloc[0]
                min_count = val_counts.iloc[-1]
                imbalance_ratio = float(max_count / max(min_count, 1))

                if imbalance_ratio > 10.0:
                    minority_pct = (min_count / len(df)) * 100
                    issues.append(
                        ValidationIssue(
                            severity="WARNING",
                            code="HIGH_CLASS_IMBALANCE",
                            column=target_column,
                            message=f"Severe class imbalance detected: ratio {imbalance_ratio:.1f}:1 (minority class represents {minority_pct:.2f}% of rows).",
                            recommendation="Do not rely on standard accuracy. Optimize for PR-AUC / ROC-AUC, and use stratified splitting."
                        )
                    )

        # 5. Missing values per feature
        total_missing = int(df.isna().sum().sum())
        for col in df.columns:
            if col == target_column:
                continue
            col_missing = int(df[col].isna().sum())
            if col_missing > 0:
                missing_pct = (col_missing / len(df)) * 100
                if missing_pct > 60.0:
                    issues.append(
                        ValidationIssue(
                            severity="WARNING",
                            code="HIGH_MISSING_FEATURE",
                            column=col,
                            message=f"Feature '{col}' has {col_missing} missing values ({missing_pct:.1f}%).",
                            recommendation="High-missingness columns should be imputed or dropped to prevent instability."
                        )
                    )

        # 6. Duplicate rows check
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            issues.append(
                ValidationIssue(
                    severity="INFO",
                    code="DUPLICATE_ROWS",
                    message=f"Dataset contains {dup_count} exact duplicate rows ({dup_count / len(df)*100:.1f}%).",
                    recommendation="Duplicates can bias validation metrics if present across train and test splits."
                )
            )

        # 7. Data Leakage and High Correlation checks
        feature_cols = [c for c in df.columns if c != target_column]
        for col in feature_cols:
            # Check for ID columns
            col_lower = col.lower()
            if col_lower in ["id", "index", "uuid", "guid", "row_id", "patient_id", "user_id"]:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        code="ID_COLUMN_LEAKAGE",
                        column=col,
                        message=f"Column '{col}' appears to be an identifier.",
                        recommendation="Identifiers often act as arbitrary leak channels. Remove before training."
                    )
                )

            # Check for constant/zero-variance features
            if df[col].nunique(dropna=True) <= 1:
                issues.append(
                    ValidationIssue(
                        severity="INFO",
                        code="ZERO_VARIANCE_FEATURE",
                        column=col,
                        message=f"Column '{col}' has zero variance (constant value).",
                        recommendation="Drop constant features as they provide no discriminative predictive value."
                    )
                )

        has_blocking_errors = any(issue.severity == "ERROR" for issue in issues)

        return ValidationReport(
            is_valid=not has_blocking_errors,
            task_type=inferred_task,
            target_column=target_column,
            row_count=len(df),
            column_count=len(df.columns),
            missing_value_count=total_missing,
            duplicate_rows_count=dup_count,
            class_distribution=class_dist,
            imbalance_ratio=imbalance_ratio,
            issues=issues,
            feature_columns=feature_cols
        )
