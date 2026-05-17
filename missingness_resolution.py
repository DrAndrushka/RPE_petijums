"""
Model-frame builder (missingness + leakage governance, no model training).

This module is kept as the "post-EDA preprocessing" step:
- drop leakage/redundant features based on `feature_decisions_df`
- run simple/advanced imputation based on `missing_action`
- assemble analysis-ready frame: predictors + one/many targets

TODO:
- Add per-column imputation diagnostics export.
- Add optional deterministic train/test split snapshot export.
- Add strict schema validation utility for cross-project portability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# ANSI colors for notebook console summaries.
BOLD = "\033[1m"
BLUE = "\033[38;5;39m"
GREEN = "\033[38;5;28m"
GRAY = "\033[38;5;245m"
RESET = "\033[0m"


def _resolve_targets(
    df: pd.DataFrame,
    feature_decisions_df: pd.DataFrame,
    target_col: str | None,
    target_cols: Sequence[str] | None,
) -> list[str]:
    """Resolve analysis targets from explicit args or feature decision roles."""
    if target_cols is not None:
        targets = list(dict.fromkeys(target_cols))
    elif target_col is not None:
        targets = [target_col]
    else:
        targets = (
            feature_decisions_df.loc[feature_decisions_df.role == "target", "column_name"]
            .dropna()
            .astype(str)
            .tolist()
        )
    targets = [t for t in targets if t in df.columns]
    if not targets:
        raise ValueError("No valid target columns found in dataframe.")
    return targets


def _resolve_predictors(
    df: pd.DataFrame,
    feature_decisions_df: pd.DataFrame,
    targets: Sequence[str],
    predictor_cols: Sequence[str] | None,
) -> list[str]:
    """Resolve predictor set (explicit whitelist has priority)."""
    if predictor_cols is not None:
        predictors = list(dict.fromkeys(predictor_cols))
        missing = [c for c in predictors if c not in df.columns]
        if missing:
            raise ValueError(f"Requested predictors not found: {missing}")
    else:
        predictors = (
            feature_decisions_df.loc[
                (feature_decisions_df.role == "predictor") & (feature_decisions_df.action != "drop"),
                "column_name",
            ]
            .dropna()
            .astype(str)
            .tolist()
        )
    predictors = [c for c in predictors if c in df.columns and c not in set(targets)]
    if not predictors:
        raise ValueError("No predictor columns resolved after filtering.")
    return predictors


def _highlight_feature_row(row: pd.Series) -> list[str]:
    """Color rows in feature decision table for quick visual triage."""
    drop_color = "#818181"
    simple_impute_color = "#CBB255"
    advanced_impute_color = "#D4605C"
    predictor_color = "#1E9101"
    target_color = "#FF8C00"

    if row["action"] == "drop":
        return [f"color: {drop_color};"] * len(row)
    if row["missing_action"] == "simple_impute":
        return [f"color: {simple_impute_color};"] * len(row)
    if row["missing_action"] == "advanced_impute":
        return [f"color: {advanced_impute_color};"] * len(row)
    if row["role"] == "predictor":
        return [f"color: {predictor_color};"] * len(row)
    if row["role"] == "target":
        return [f"color: {target_color};"] * len(row)
    return [""] * len(row)


def _drop_flagged_columns(df: pd.DataFrame, drop_cols: Sequence[str]) -> pd.DataFrame:
    """Drop known-bad columns (leakage/redundancy) if they exist."""
    keep_drop_cols = [c for c in drop_cols if c in df.columns]
    if keep_drop_cols:
        print(f"{BOLD}{GRAY}Dropping flagged columns:{RESET} {keep_drop_cols}")
        return df.drop(columns=keep_drop_cols)
    return df


def _simple_impute(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Simple imputation: mode for categorical-like, median for numeric."""
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        if s.isna().sum() == 0:
            continue
        is_cat = pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
        if is_cat:
            mode_series = s.dropna().mode()
            if mode_series.empty:
                continue
            df[col] = s.fillna(mode_series.iloc[0])
            print(f'{GRAY}- "{col}" imputed with mode{RESET}')
        else:
            median = s.dropna().median()
            df[col] = s.fillna(median)
            print(f'{GRAY}- "{col}" imputed with median{RESET}')
    return df


def _advanced_impute(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Advanced numeric imputation via IterativeImputer + RandomForestRegressor."""
    numeric_cols = []
    fallback_cols = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            numeric_cols.append(col)
        else:
            fallback_cols.append(col)

    if fallback_cols:
        print(f"{GRAY}Advanced-impute fallback to simple for non-numeric: {fallback_cols}{RESET}")
        df = _simple_impute(df, fallback_cols)

    if not numeric_cols:
        return df

    original_dtypes = df.loc[:, numeric_cols].dtypes
    X = df.loc[:, numeric_cols].to_numpy(dtype=float)
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=50,
            random_state=0,
            n_jobs=-1,
        ),
        max_iter=10,
        random_state=0,
    )
    X_imp = imputer.fit_transform(X)
    X_imp_df = pd.DataFrame(X_imp, columns=numeric_cols, index=df.index)

    for col in numeric_cols:
        orig_dtype = original_dtypes[col]
        if pd.api.types.is_integer_dtype(orig_dtype):
            floored = np.floor(X_imp_df[col])
            if pd.api.types.is_extension_array_dtype(orig_dtype):
                df.loc[:, col] = floored.astype("int64")
            else:
                df.loc[:, col] = floored.astype(orig_dtype)
        else:
            df.loc[:, col] = X_imp_df[col]
    return df


def build_model_frame(
    root: Path,
    post_cohort_df: pd.DataFrame,
    feature_decisions_df: pd.DataFrame,
    metadata: dict,
    target_col: str | None = None,
    target_cols: Sequence[str] | None = None,
    predictor_cols: Sequence[str] | None = None,
    export: bool = False,
) -> pd.DataFrame:
    """
    Build analysis-ready dataframe (predictors + targets) after missingness handling.

    Parameters:
    - target_col / target_cols: choose one or many targets.
    - predictor_cols: explicit predictor whitelist (optional).
    - export: save output to `data/processed`.
    """
    display(feature_decisions_df.style.apply(_highlight_feature_row, axis=1))

    df = post_cohort_df.copy()
    targets = _resolve_targets(df, feature_decisions_df, target_col=target_col, target_cols=target_cols)
    predictors = _resolve_predictors(df, feature_decisions_df, targets=targets, predictor_cols=predictor_cols)

    drop_cols = (
        feature_decisions_df.loc[
            (feature_decisions_df.drop_reason == "leakage") | (feature_decisions_df.action == "drop"),
            "column_name",
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    simple_impute_cols = (
        feature_decisions_df.loc[feature_decisions_df.missing_action == "simple_impute", "column_name"]
        .dropna()
        .astype(str)
        .tolist()
    )
    advanced_impute_cols = (
        feature_decisions_df.loc[feature_decisions_df.missing_action == "advanced_impute", "column_name"]
        .dropna()
        .astype(str)
        .tolist()
    )

    print("═" * 90)
    df = _drop_flagged_columns(df, drop_cols)
    df = _simple_impute(df, simple_impute_cols)
    df = _advanced_impute(df, advanced_impute_cols)
    print("═" * 90)

    important_missing_flags = (
        feature_decisions_df.loc[
            (feature_decisions_df.role == "predictor")
            & (feature_decisions_df.action != "drop")
            & (feature_decisions_df.notes.fillna("").str.contains("high_missingness_flag_important", regex=False)),
            "column_name",
        ]
        .dropna()
        .astype(str)
        .map(lambda c: f"{c}_missing")
        .tolist()
    )
    important_missing_flags = [c for c in important_missing_flags if c in df.columns]

    x_cols = list(dict.fromkeys([*predictors, *important_missing_flags]))
    y_cols = [c for c in targets if c in df.columns]
    df_model = pd.concat([df[x_cols], df[y_cols]], axis=1)

    task_name = metadata.get("task_name", "unknown_task")
    positive_cls = metadata.get("positive_class", "—")
    df_rows, df_cols = df_model.shape
    predictor_count = len(x_cols)

    prevalence_lines = []
    for tgt in y_cols:
        if pd.api.types.is_bool_dtype(df_model[tgt]) or df_model[tgt].dropna().nunique() == 2:
            prevalence = float(pd.to_numeric(df_model[tgt], errors="coerce").mean())
            prevalence_lines.append(f"- `{tgt}` prevalence: **{prevalence:.3f}**")
        else:
            prevalence_lines.append(f"- `{tgt}` prevalence: _not binary_")

    display(
        Markdown(
            f"""
# Final Model-Frame Summary

Task: **`{task_name}`**  
Positive class hint: **`{positive_cls}`**

### Cohort Snapshot
- Rows: **{df_rows:,}**
- Columns: **{df_cols}**
- Predictors kept: **{predictor_count}**
- Targets kept: **{len(y_cols)}**

### Target prevalence
{chr(10).join(prevalence_lines) if prevalence_lines else "- _No targets resolved_"}
"""
        )
    )

    print("═" * 70)
    print(f"{BOLD}MODEL FRAME ONLINE{RESET}")
    print("─" * 70)
    total_nans = int(df_model.isna().sum().sum())
    nan_pct = (total_nans / (df_rows * df_cols)) * 100 if df_rows and df_cols else 0.0
    print(f"{GREEN}Shape:{RESET} {df_rows} rows x {df_cols} cols")
    print(f"{GREEN}Remaining NaNs:{RESET} {total_nans} ({nan_pct:0.2f}%)")
    display(df_model.head())

    if export:
        model_frame_path = Path(root) / "data" / "processed" / f"(model_frame){task_name}.pickle"
        model_frame_path.parent.mkdir(parents=True, exist_ok=True)
        df_model.to_pickle(model_frame_path)
        print(f"{GREEN}Saved model frame:{RESET} {BOLD}{model_frame_path}{RESET}")
    else:
        print(f"{GRAY}Set export=True to save model frame in data/processed.{RESET}")

    return df_model
