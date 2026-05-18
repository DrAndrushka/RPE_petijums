"""
schema_infer.py
================
Auto-infer a column-kind schema for a DataFrame and print it as an editable
Python dict so you can paste it back into the notebook and tweak.

Kinds
-----
- "id"          : high-cardinality unique identifier (skip from analysis)
- "binary"      : exactly 2 non-null unique values (bool/0-1/yes-no/...)
- "ordinal"     : ordered categorical with a small number of levels
- "nominal"     : unordered categorical (incl. low-cardinality strings)
- "continuous"  : numeric with many unique values (or floats)
- "count"       : non-negative integer with moderate range
- "datetime"    : parseable datetime
- "text"        : free text / high-cardinality strings
- "skip"        : explicitly skipped from analysis

Usage
-----
    from schema_infer import infer_schema, print_schema_template
    schema = infer_schema(df)
    print_schema_template(schema)          # prints a dict you can paste back
    # ...edit it in the notebook, then pass into cleaning.apply_schema(df, schema)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional
import re

import numpy as np
import pandas as pd


Kind = Literal[
    "id", "binary", "ordinal", "nominal",
    "continuous", "count", "datetime", "text", "skip",
]


@dataclass
class ColSpec:
    """One column's contract. Everything downstream reads from this."""
    name: str
    kind: Kind
    # Optional declared ordering for ordinal kinds (low -> high).
    ordered_levels: Optional[list[Any]] = None
    # Values to treat as null (in addition to NaN/None).
    nulls: tuple[Any, ...] = ()
    # Value remapping applied before type coercion.
    replace: dict[Any, Any] = field(default_factory=dict)
    # Keep in analysis output (False = computed but hidden, e.g. raw IDs).
    keep: bool = True
    # Free-text note for the schema printout.
    note: str = ""


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

_BINARY_TRUE = {"true", "t", "yes", "y", "1", "1.0", "positive", "pos"}
_BINARY_FALSE = {"false", "f", "no", "n", "0", "0.0", "negative", "neg"}


def _looks_binary(s: pd.Series) -> bool:
    vals = s.dropna().unique()
    if len(vals) != 2:
        return False
    as_str = {str(v).strip().lower() for v in vals}
    if as_str <= (_BINARY_TRUE | _BINARY_FALSE):
        return True
    return set(vals) <= {0, 1} or set(vals) <= {True, False}


def _looks_datetime(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not pd.api.types.is_object_dtype(s):
        return False
    sample = s.dropna().astype(str).head(50)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.9


def _looks_id(s: pd.Series, n_rows: int) -> bool:
    nu = s.nunique(dropna=True)
    if nu < 0.95 * n_rows:
        return False
    name = (s.name or "").lower()
    return bool(re.search(r"(^|_)(id|pk|uuid|guid|code|nr|num)(_|$)", name)) or nu == n_rows


def _infer_one(s: pd.Series, n_rows: int, ordinal_max_levels: int) -> Kind:
    nn = s.dropna()
    if nn.empty:
        return "skip"
    if _looks_id(s, n_rows):
        return "id"
    if _looks_datetime(s):
        return "datetime"
    if _looks_binary(s):
        return "binary"

    nu = nn.nunique()

    # Categorical dtype with declared order -> ordinal.
    if isinstance(s.dtype, pd.CategoricalDtype):
        return "ordinal" if s.dtype.ordered else "nominal"

    if pd.api.types.is_numeric_dtype(s):
        is_int = pd.api.types.is_integer_dtype(s) or (nn % 1 == 0).all()
        if is_int and nu <= ordinal_max_levels and nn.min() >= 0:
            # Small non-negative integer set -> likely ordinal/count.
            return "ordinal" if nu <= 10 else "count"
        return "continuous"

    # Object / string
    if nu <= ordinal_max_levels:
        return "nominal"
    return "text"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_schema(
    df: pd.DataFrame,
    *,
    ordinal_max_levels: int = 15,
    overrides: dict[str, Kind] | None = None,
) -> dict[str, ColSpec]:
    """Return {col_name: ColSpec} with inferred kind for every column."""
    overrides = overrides or {}
    out: dict[str, ColSpec] = {}
    n_rows = len(df)
    for col in df.columns:
        kind = overrides.get(col) or _infer_one(df[col], n_rows, ordinal_max_levels)
        spec = ColSpec(name=col, kind=kind)
        if kind == "ordinal":
            try:
                spec.ordered_levels = sorted(df[col].dropna().unique().tolist())
            except TypeError:
                spec.ordered_levels = list(df[col].dropna().unique())
        out[col] = spec
    return out


def print_schema_template(schema: dict[str, ColSpec]) -> str:
    """Print and return a paste-back-able Python dict literal of the schema."""
    lines = ["schema_overrides = {"]
    for col, spec in schema.items():
        extras = []
        if spec.ordered_levels is not None and spec.kind == "ordinal":
            extras.append(f"ordered_levels={spec.ordered_levels!r}")
        if spec.nulls:
            extras.append(f"nulls={spec.nulls!r}")
        if spec.replace:
            extras.append(f"replace={spec.replace!r}")
        if not spec.keep:
            extras.append("keep=False")
        extras_str = (", " + ", ".join(extras)) if extras else ""
        lines.append(f'    {col!r}: ColSpec(name={col!r}, kind={spec.kind!r}{extras_str}),')
    lines.append("}")
    out = "\n".join(lines)
    print(out)
    return out


def schema_summary(schema: dict[str, ColSpec]) -> pd.DataFrame:
    """Tabular view of the schema (for sanity-checking in the notebook)."""
    rows = []
    for col, spec in schema.items():
        rows.append({
            "column": col,
            "kind": spec.kind,
            "keep": spec.keep,
            "ordered_levels": spec.ordered_levels,
            "nulls": list(spec.nulls) if spec.nulls else None,
            "note": spec.note,
        })
    return pd.DataFrame(rows)
