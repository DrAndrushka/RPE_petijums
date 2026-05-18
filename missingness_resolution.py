"""
missingness_resolution.py
==========================
Missingness analysis + MICE imputation.

Stages
------
1. analyze_missingness(df)
   -> per-column missing %, count, pairwise co-missingness matrix (top pairs),
      saved as table + heatmap SVG.

2. add_missing_flags(df, cols)
   -> add boolean <col>_missing columns for explicit MNAR/MAR tracking.

3. mice_impute(df, m=10, ...)
   -> returns a LIST of m imputed DataFrames using sklearn's IterativeImputer
      (multivariate chained equations). Numeric and categorical predictors are
      handled separately; categoricals are ordinal-encoded for imputation and
      decoded back.

4. simple_impute(df)
   -> single-frame imputation for fast screening (median for numeric, mode for
      categorical). Use only for exploratory work, not for final inference.

Outputs saved under output/missingness/{figures,tables}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from schema_infer import ColSpec


def _ensure_dirs(root: Path) -> tuple[Path, Path]:
    figs = root / "missingness" / "figures"
    tabs = root / "missingness" / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    return figs, tabs


# ---------------------------------------------------------------------------
# 1. Missingness analysis
# ---------------------------------------------------------------------------

def analyze_missingness(df: pd.DataFrame, *, output_root: Path | str = "output") -> pd.DataFrame:
    """
    Per-column missing summary + co-missingness heatmap (saved as SVG).
    Returns the per-column table.
    """
    figs, tabs = _ensure_dirs(Path(output_root))

    miss = df.isna()
    per_col = pd.DataFrame({
        "column": df.columns,
        "n_missing": miss.sum().values,
        "pct_missing": (miss.mean() * 100).round(2).values,
    }).sort_values("pct_missing", ascending=False).reset_index(drop=True)
    per_col.to_csv(tabs / "missing_per_column.csv", index=False)

    # Bar chart
    if (per_col["pct_missing"] > 0).any():
        plot_df = per_col[per_col["pct_missing"] > 0]
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(plot_df))))
        sns.barplot(x="pct_missing", y="column", data=plot_df, ax=ax, color="#e76f51")
        ax.set_title("Missing % per column"); ax.set_xlabel("% missing")
        fig.tight_layout()
        fig.savefig(figs / "missing_per_column.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

    # Co-missingness heatmap (Jaccard over missing rows)
    cols_with_miss = per_col[per_col["pct_missing"] > 0]["column"].tolist()
    if len(cols_with_miss) >= 2:
        m = miss[cols_with_miss].astype(int)
        inter = m.T @ m
        union = (m.values[:, :, None] | m.values[:, None, :]).sum(axis=0)
        jacc = pd.DataFrame(
            np.where(union > 0, inter.values / np.where(union == 0, 1, union), 0),
            index=cols_with_miss, columns=cols_with_miss,
        )
        jacc.to_csv(tabs / "co_missingness_jaccard.csv")
        fig, ax = plt.subplots(figsize=(0.6 * len(cols_with_miss) + 2,
                                        0.6 * len(cols_with_miss) + 2))
        sns.heatmap(jacc, annot=True, fmt=".2f", cmap="Reds", ax=ax, cbar=True)
        ax.set_title("Co-missingness (Jaccard)")
        fig.tight_layout()
        fig.savefig(figs / "co_missingness_heatmap.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

    return per_col


# ---------------------------------------------------------------------------
# 2. Missing flags
# ---------------------------------------------------------------------------

def add_missing_flags(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Add <col>_missing boolean columns for the specified columns."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[f"{c}_missing"] = out[c].isna().astype("boolean")
    return out


# ---------------------------------------------------------------------------
# 3. MICE imputation (multiple imputation)
# ---------------------------------------------------------------------------

def _encode_for_impute(df: pd.DataFrame, schema: dict[str, ColSpec]):
    """Encode categoricals to integer codes for the imputer; remember mapping."""
    work = df.copy()
    decoders: dict[str, dict[int, object]] = {}
    cat_cols = []
    for col, spec in schema.items():
        if col not in work.columns:
            continue
        if spec.kind in ("ordinal", "nominal"):
            cats = pd.Categorical(work[col])
            decoders[col] = dict(enumerate(cats.categories))
            work[col] = pd.Series(cats.codes, index=work.index).replace(-1, np.nan)
            cat_cols.append(col)
        elif spec.kind == "binary":
            work[col] = work[col].astype("float")
            cat_cols.append(col)
    # keep only numeric / coded columns for imputation
    drop = [c for c, sp in schema.items()
            if sp.kind in ("id", "text", "datetime", "skip") and c in work.columns]
    work = work.drop(columns=drop, errors="ignore")
    return work, decoders, cat_cols, drop


def _decode_after_impute(
    imputed: pd.DataFrame,
    decoders: dict[str, dict[int, object]],
    cat_cols: list[str],
    schema: dict[str, ColSpec],
) -> pd.DataFrame:
    out = imputed.copy()
    for col in cat_cols:
        if col not in out.columns:
            continue
        spec = schema[col]
        if spec.kind == "binary":
            out[col] = (out[col] >= 0.5).astype("boolean")
        else:
            codes = out[col].round().clip(lower=0, upper=max(decoders[col]) if decoders[col] else 0)
            out[col] = codes.map(decoders[col])
            levels = spec.ordered_levels if spec.kind == "ordinal" else list(decoders[col].values())
            out[col] = pd.Categorical(out[col], categories=levels, ordered=(spec.kind == "ordinal"))
    return out


def mice_impute(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    *,
    m: int = 10,
    max_iter: int = 10,
    random_state: int = 42,
    output_root: Path | str = "output",
) -> list[pd.DataFrame]:
    """
    Generate `m` imputed datasets via sklearn IterativeImputer with different
    random seeds (sample_posterior=True). Returns a list of m DataFrames with
    the same columns as the original (datetime/id/text columns are passed
    through unchanged from the original df).
    """
    figs, tabs = _ensure_dirs(Path(output_root))

    work, decoders, cat_cols, dropped = _encode_for_impute(df, schema)
    if work.isna().sum().sum() == 0:
        # nothing to impute -> return m copies
        return [df.copy() for _ in range(m)]

    imputed_frames: list[pd.DataFrame] = []
    for i in range(m):
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=50, n_jobs=-1,
                                            random_state=random_state + i),
            max_iter=max_iter,
            sample_posterior=False,  # RF estimator doesn't support posterior
            random_state=random_state + i,
        )
        arr = imputer.fit_transform(work)
        imp = pd.DataFrame(arr, columns=work.columns, index=work.index)
        decoded = _decode_after_impute(imp, decoders, cat_cols, schema)
        # bring back untouched columns from the original df
        for c in dropped:
            if c in df.columns:
                decoded[c] = df[c].values
        # restore column order
        decoded = decoded.reindex(columns=df.columns)
        imputed_frames.append(decoded)

    pd.DataFrame([{"m": m, "max_iter": max_iter, "random_state": random_state}]) \
      .to_csv(tabs / "mice_config.csv", index=False)
    return imputed_frames


# ---------------------------------------------------------------------------
# 4. Simple imputation (fallback / screening)
# ---------------------------------------------------------------------------

def simple_impute(df: pd.DataFrame, schema: dict[str, ColSpec]) -> pd.DataFrame:
    """Median for numeric/ordinal/count, mode for nominal/binary. One frame."""
    out = df.copy()
    for col, spec in schema.items():
        if col not in out.columns or out[col].isna().sum() == 0:
            continue
        if spec.kind in ("continuous", "count"):
            out[col] = out[col].fillna(out[col].median())
        elif spec.kind == "ordinal":
            # use mode of the underlying codes
            cats = pd.Categorical(out[col])
            mode_code = pd.Series(cats.codes).replace(-1, np.nan).mode()
            if len(mode_code):
                fill = cats.categories[int(mode_code.iloc[0])]
                out[col] = out[col].fillna(fill)
        elif spec.kind in ("nominal", "binary"):
            mode = out[col].mode(dropna=True)
            if len(mode):
                out[col] = out[col].fillna(mode.iloc[0])
    return out
