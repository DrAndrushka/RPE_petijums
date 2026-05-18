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
# Display formatting helpers (CSV save only — do NOT apply to in-memory dfs
# used by downstream stages, since this converts numerics to display strings
# and would corrupt further math).
#
# Column-aware rules (Andy's spec, 2026-05-18):
#   p-values            : p < 0.001 -> "<0.001";  else 3 decimal places
#   counts (n_*, n)     : integer
#   percentages (*_pct, missing_pct, pct_*)      : 1 decimal
#   means/medians/SD/IQR/quantiles               : 2 decimals
#   correlations / effect sizes / OR / CI bounds : 2 decimals
#   entropy / balance / imbalance metrics        : 2 decimals
#   fallback for any other numeric               : 3 decimals
# Integer-valued floats (e.g. 5.0) always render as int regardless of rule.
# ---------------------------------------------------------------------------

def _classify_column(name: str) -> str:
    """Return the rule key for a column name. Falls back to 'default'.

    Resolution order is deliberate:
      1. p-values (exact match avoids matching every column containing 'p')
      2. counts (exact 'n' or starts/ends with 'n_'/'_n', plus df/count tokens)
      3. entropy/balance/imbalance BEFORE central (so 'max_class_imbalance'
         is not captured by 'max')
      4. percent
      5. effect (OR/CI/coef/SE/stat/VIF)
      6. central tendency / spread
      7. fallback to 'default'

    Tokens like 'mode' or 'first_mode' refer to a category label (string),
    not a numeric statistic, so they intentionally fall through to 'default'.
    """
    n = name.lower()

    # 1. p-value (exact / suffix)
    if n in ("p", "p_fdr", "p_value", "pvalue") or n.endswith("_p") or n.endswith("_pvalue"):
        return "pvalue"

    # 2. count (whole-word / boundary based, not substring — avoids 'continuous'->count)
    count_exact = {"n", "df", "n_rows", "n_cols", "n_cols_analysed", "n_unique",
                   "n_used", "n_models", "n_nonmissing", "n_missing", "count"}
    if n in count_exact or n.startswith("n_") or n.endswith("_n") or n.endswith("_count"):
        return "count"

    # 3. entropy / balance / imbalance (BEFORE central so 'max_class_imbalance' wins)
    for tok in ("entropy", "imbalance", "balance"):
        if tok in n:
            return "entropy"

    # 4. percent
    for tok in ("pct", "percent"):
        if tok in n:
            return "percent"

    # 5. effect / stat / OR / CI
    effect_tokens = ("or_ci_lo", "or_ci_hi", "ci_lo", "ci_hi",
                     "coef", "vif", "r2", "auc", "effect", "stat")
    if n in ("or", "hr", "rr", "se"):
        return "effect"
    for tok in effect_tokens:
        if tok in n:
            return "effect"

    # 6. central tendency / spread — 'mode'/'first_mode'/'second_mode' are category
    # labels (strings), so they are excluded here and stay 'default'.
    central_tokens = ("trimmed_mean", "mean", "median", "std", "sd",
                      "iqr", "min", "max", "p_5th", "p_95th",
                      "cv", "skew", "kurt")
    for tok in central_tokens:
        if tok in n:
            return "central"

    return "default"


def _format_value(x, rule: str) -> object:
    """Format one numeric value according to a rule key."""
    if x is None:
        return x
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    if not isinstance(x, (float, np.floating)):
        return x
    if not np.isfinite(x):
        return x  # NaN / inf preserved

    # p-value special string formatting
    if rule == "pvalue":
        if x < 0.001:
            return "<0.001"
        return float(f"{x:.3f}")

    # counts: round to int (defensive — already int in most cases)
    if rule == "count":
        return int(round(x))

    decimals = {
        "percent": 1,
        "central": 2,
        "effect":  2,
        "entropy": 2,
        "default": 3,
    }.get(rule, 3)

    rounded = round(float(x), decimals)
    # Integer-valued floats render as int (avoids '5.0', '12.00')
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def format_number(x, rule: str = "default") -> object:
    """Format a single value. Use `rule` to override classification."""
    return _format_value(x, rule)


def format_table_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with numeric cells formatted per column-name rules.

    Object/string columns are untouched. Use right before `.to_csv(...)`; never
    assign back into the working DataFrame used by downstream stages.
    """
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        rule = _classify_column(str(col))
        out[col] = out[col].map(lambda v, r=rule: _format_value(v, r))
    return out


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
