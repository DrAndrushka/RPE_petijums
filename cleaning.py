"""
cleaning.py
============
Apply a schema to a raw DataFrame:
- coerce dtypes (numeric / categorical / datetime / bool)
- apply null markers and value replacements
- audit / drop duplicates by ID columns
- provide derivation helpers (bin_numeric, bin_datetime, make_missing_flag,
  combine_categories) for the notebook to call freely

Everything here is universal — no project-specific column names.
"""

from __future__ import annotations

from typing import Sequence, Any, Iterable

import numpy as np
import pandas as pd

from schema_infer import ColSpec


# ---------------------------------------------------------------------------
# Schema application
# ---------------------------------------------------------------------------

def apply_schema(df: pd.DataFrame, schema: dict[str, ColSpec]) -> pd.DataFrame:
    """
    Coerce dtypes and apply nulls/replacements according to schema.
    Returns a NEW dataframe (does not mutate input).
    Columns with kind='skip' are dropped from the returned frame.
    """
    out = df.copy()

    for col, spec in schema.items():
        if col not in out.columns:
            continue

        # 1. Replacements first so downstream coercion sees clean values.
        if spec.replace:
            out[col] = out[col].replace(spec.replace)

        # 2. Null markers.
        if spec.nulls:
            out[col] = out[col].replace({v: pd.NA for v in spec.nulls})

        # 3. Type coercion by kind.
        s = out[col]
        if spec.kind == "binary":
            out[col] = _coerce_binary(s)
        elif spec.kind == "ordinal":
            levels = spec.ordered_levels or sorted(s.dropna().unique().tolist())
            out[col] = pd.Categorical(s, categories=levels, ordered=True)
        elif spec.kind == "nominal":
            out[col] = pd.Categorical(s, ordered=False)
        elif spec.kind in ("continuous", "count"):
            out[col] = pd.to_numeric(s, errors="coerce")
        elif spec.kind == "datetime":
            out[col] = pd.to_datetime(s, errors="coerce")
        elif spec.kind == "text":
            out[col] = s.astype("string")
        elif spec.kind == "id":
            out[col] = s.astype("string")
        # "skip" handled below

    drop_cols = [c for c, sp in schema.items() if sp.kind == "skip" and c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    return out


def _coerce_binary(s: pd.Series) -> pd.Series:
    """Map a 2-value column to nullable boolean."""
    truthy = {"true", "t", "yes", "y", "1", "1.0", "positive", "pos"}
    falsy = {"false", "f", "no", "n", "0", "0.0", "negative", "neg"}

    def _to_bool(v):
        if pd.isna(v):
            return pd.NA
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            if v == 1:
                return True
            if v == 0:
                return False
            return pd.NA
        key = str(v).strip().lower()
        if key in truthy:
            return True
        if key in falsy:
            return False
        return pd.NA

    return s.map(_to_bool).astype("boolean")


# ---------------------------------------------------------------------------
# Duplicate auditing
# ---------------------------------------------------------------------------

def audit_duplicates(
    df: pd.DataFrame,
    id_cols: Sequence[str],
    *,
    include_first: bool = True,
    drop: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find duplicates by `id_cols` (string-normalized) and return:
      (audit_df_of_duplicates, cleaned_df)

    - include_first=True : return all rows in any duplicate group.
    - drop=True          : remove duplicates from cleaned_df (keep first).
    """
    missing = [c for c in id_cols if c not in df.columns]
    if missing:
        raise ValueError(f"id_cols not in dataframe: {missing}")

    key = df[list(id_cols)].copy()
    for c in id_cols:
        if pd.api.types.is_object_dtype(key[c]) or pd.api.types.is_string_dtype(key[c]):
            key[c] = key[c].astype("string").str.strip().str.lower().replace("", pd.NA)

    complete = key.notna().all(axis=1)
    keep_kind = False if include_first else "first"
    dup_mask = complete & key.duplicated(keep=keep_kind)

    audit = df.loc[dup_mask].copy().sort_values(by=list(id_cols))

    cleaned = df.copy()
    if drop:
        cleaned = cleaned.loc[~(complete & key.duplicated(keep="first"))].reset_index(drop=True)

    return audit, cleaned


# ---------------------------------------------------------------------------
# Derivation helpers (use in the notebook to make new columns)
# ---------------------------------------------------------------------------

def bin_numeric(
    s: pd.Series,
    bins: Sequence[float],
    labels: Sequence[str] | None = None,
    *,
    right: bool = False,
    ordered: bool = True,
) -> pd.Categorical:
    """
    Bin a numeric series into ordered categorical groups.

    Example:
        df['age_bin'] = bin_numeric(df['age'], [0,50,60,70,120],
                                    labels=['<50','50-59','60-69','70+'])
    """
    out = pd.cut(s, bins=list(bins), labels=labels, right=right, include_lowest=True)
    return pd.Categorical(out, categories=labels if labels else out.cat.categories, ordered=ordered)


def bin_datetime(
    s: pd.Series,
    *,
    unit: str = "year",
) -> pd.Series:
    """
    Extract a coarse calendar unit from a datetime column.
    unit ∈ {"year", "quarter", "month", "week", "weekday", "hour"}
    """
    s = pd.to_datetime(s, errors="coerce")
    if unit == "year":
        return s.dt.year
    if unit == "quarter":
        return s.dt.to_period("Q").astype("string")
    if unit == "month":
        return s.dt.month
    if unit == "week":
        return s.dt.isocalendar().week.astype("Int64")
    if unit == "weekday":
        return s.dt.day_name()
    if unit == "hour":
        return s.dt.hour
    raise ValueError(f"unknown unit: {unit}")


def make_missing_flag(s: pd.Series, suffix: str = "_missing") -> pd.Series:
    """
    Return a boolean Series flagging where `s` is missing.
    Name it explicitly when assigning: df['psa_missing'] = make_missing_flag(df['psa']).
    """
    flag = s.isna().astype("boolean")
    flag.name = (s.name or "value") + suffix
    return flag


def combine_categories(
    s: pd.Series,
    mapping: dict[Any, str],
    *,
    other: str | None = "other",
) -> pd.Categorical:
    """
    Collapse categorical levels via a {original_value: new_label} map.
    Anything not in the map becomes `other` (or NA if other=None).
    """
    def _map(v):
        if pd.isna(v):
            return pd.NA
        if v in mapping:
            return mapping[v]
        return other if other is not None else pd.NA

    new = s.map(_map)
    levels = list(dict.fromkeys([v for v in mapping.values()] + ([other] if other else [])))
    return pd.Categorical(new, categories=levels, ordered=False)


def zscore(s: pd.Series) -> pd.Series:
    """Standard z-score (population sd). NaN-safe."""
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return pd.Series(np.zeros(len(s)), index=s.index, name=(s.name or "value") + "_z")
    out = (s - mu) / sd
    out.name = (s.name or "value") + "_z"
    return out
