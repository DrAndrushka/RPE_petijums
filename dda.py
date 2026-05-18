"""
dda.py
=======
Descriptive Data Analysis driven by a ColSpec schema.

Per column you get:
- summary row (n, missing %, dtype, kind, kind-specific stats)
- an appropriate plot saved to output/dda/figures/<col>.svg

You also get aggregated overview tables saved to output/dda/tables/.

Stats per kind
--------------
- continuous : n, missing%, mean, sd, median, IQR (Q1,Q3), min, max, skew
- count      : same as continuous + mode
- ordinal    : n, missing%, n_levels, mode, median (by category rank), top freq
- nominal    : n, missing%, n_levels, mode, top 3 freq
- binary     : n, missing%, p(True), n_true, n_false
- datetime   : n, missing%, min, max, span_days
- id/text    : n, missing%, n_unique

Plots per kind (seaborn)
------------------------
- continuous / count : histogram + KDE  AND  boxplot (saved as two files)
- ordinal            : ordered bar chart of counts
- nominal            : bar chart of counts (top 15 + 'other')
- binary             : count plot (True/False)
- datetime           : line of counts-per-month
- id / text / skip   : not plotted
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew

from schema_infer import ColSpec


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _ensure_dirs(root: Path) -> tuple[Path, Path]:
    figs = root / "dda" / "figures"
    tabs = root / "dda" / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    return figs, tabs


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-column stats
# ---------------------------------------------------------------------------

def _stats_continuous(s: pd.Series) -> dict:
    nn = s.dropna()
    q1, q3 = (nn.quantile(0.25), nn.quantile(0.75)) if len(nn) else (np.nan, np.nan)
    return {
        "n": int(s.notna().sum()),
        "missing_pct": round(s.isna().mean() * 100, 2),
        "mean": float(nn.mean()) if len(nn) else np.nan,
        "sd": float(nn.std(ddof=1)) if len(nn) > 1 else np.nan,
        "median": float(nn.median()) if len(nn) else np.nan,
        "q1": float(q1),
        "q3": float(q3),
        "min": float(nn.min()) if len(nn) else np.nan,
        "max": float(nn.max()) if len(nn) else np.nan,
        "skew": float(skew(nn)) if len(nn) > 2 else np.nan,
    }


def _stats_categorical(s: pd.Series, ordered: bool) -> dict:
    nn = s.dropna()
    vc = nn.value_counts()
    top3 = ", ".join(f"{k}={v}" for k, v in vc.head(3).items())
    return {
        "n": int(nn.size),
        "missing_pct": round(s.isna().mean() * 100, 2),
        "n_levels": int(vc.size),
        "mode": vc.index[0] if len(vc) else np.nan,
        "top3": top3,
        "ordered": ordered,
    }


def _stats_binary(s: pd.Series) -> dict:
    nn = s.dropna()
    n_true = int(nn.sum())
    n_false = int(nn.size - n_true)
    return {
        "n": int(nn.size),
        "missing_pct": round(s.isna().mean() * 100, 2),
        "p_true": round(n_true / nn.size, 4) if nn.size else np.nan,
        "n_true": n_true,
        "n_false": n_false,
    }


def _stats_datetime(s: pd.Series) -> dict:
    nn = pd.to_datetime(s, errors="coerce").dropna()
    return {
        "n": int(nn.size),
        "missing_pct": round(s.isna().mean() * 100, 2),
        "min": nn.min() if len(nn) else pd.NaT,
        "max": nn.max() if len(nn) else pd.NaT,
        "span_days": (nn.max() - nn.min()).days if len(nn) else np.nan,
    }


def _stats_id(s: pd.Series) -> dict:
    return {
        "n": int(s.notna().sum()),
        "missing_pct": round(s.isna().mean() * 100, 2),
        "n_unique": int(s.nunique(dropna=True)),
    }


# ---------------------------------------------------------------------------
# Per-column plots
# ---------------------------------------------------------------------------

def _plot_continuous(s: pd.Series, name: str, out_dir: Path) -> list[Path]:
    paths = []
    nn = s.dropna()
    if nn.empty:
        return paths

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(nn, kde=True, ax=ax, color="#3b7ddd")
    ax.set_title(f"Distribution — {name}")
    ax.set_xlabel(name)
    p = out_dir / f"{name}__hist.svg"
    _save_fig(fig, p); paths.append(p)

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(x=nn, ax=ax, color="#3b7ddd")
    ax.set_title(f"Boxplot — {name}")
    ax.set_xlabel(name)
    p = out_dir / f"{name}__box.svg"
    _save_fig(fig, p); paths.append(p)
    return paths


def _plot_ordinal(s: pd.Series, name: str, out_dir: Path) -> list[Path]:
    nn = s.dropna()
    if nn.empty:
        return []
    order = list(s.cat.categories) if isinstance(s.dtype, pd.CategoricalDtype) else sorted(nn.unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(x=nn.astype(str), order=[str(o) for o in order], ax=ax, color="#3b7ddd")
    ax.set_title(f"Ordinal distribution — {name}")
    ax.set_xlabel(name); ax.set_ylabel("count")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    p = out_dir / f"{name}__bar.svg"
    _save_fig(fig, p)
    return [p]


def _plot_nominal(s: pd.Series, name: str, out_dir: Path, top_n: int = 15) -> list[Path]:
    nn = s.dropna()
    if nn.empty:
        return []
    vc = nn.value_counts()
    if len(vc) > top_n:
        top = vc.head(top_n)
        top["(other)"] = vc.iloc[top_n:].sum()
        vc = top
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(vc))))
    sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, color="#3b7ddd")
    ax.set_title(f"Nominal counts — {name}"); ax.set_xlabel("count")
    p = out_dir / f"{name}__bar.svg"
    _save_fig(fig, p)
    return [p]


def _plot_binary(s: pd.Series, name: str, out_dir: Path) -> list[Path]:
    nn = s.dropna()
    if nn.empty:
        return []
    counts = nn.value_counts().reindex([True, False]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bar_df = pd.DataFrame({"value": ["True", "False"], "count": counts.values})
    sns.barplot(data=bar_df, x="value", y="count", hue="value",
                palette=["#2a9d8f", "#e76f51"], legend=False, ax=ax)
    ax.set_title(f"Binary — {name}"); ax.set_ylabel("count")
    p = out_dir / f"{name}__bar.svg"
    _save_fig(fig, p)
    return [p]


def _plot_datetime(s: pd.Series, name: str, out_dir: Path) -> list[Path]:
    nn = pd.to_datetime(s, errors="coerce").dropna()
    if nn.empty:
        return []
    monthly = nn.dt.to_period("M").value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(monthly.index.astype(str), monthly.values, marker="o", color="#3b7ddd")
    ax.set_title(f"Records per month — {name}")
    ax.set_xlabel("month"); ax.set_ylabel("count")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    p = out_dir / f"{name}__timeline.svg"
    _save_fig(fig, p)
    return [p]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_dda(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    *,
    output_root: Path | str = "output",
    skip_cols: Iterable[str] = (),
) -> dict[str, pd.DataFrame]:
    """
    Run DDA on every kept column in the schema.

    Returns a dict of overview tables: {"continuous": ..., "categorical": ...,
    "binary": ..., "datetime": ..., "id": ...}, also saved as CSV.
    All figures saved as SVG in output/dda/figures/.
    """
    output_root = Path(output_root)
    figs_dir, tabs_dir = _ensure_dirs(output_root)
    skip = set(skip_cols)

    rows_cont, rows_cat, rows_bin, rows_dt, rows_id = [], [], [], [], []

    for col, spec in schema.items():
        if col not in df.columns or col in skip or not spec.keep or spec.kind == "skip":
            continue

        s = df[col]
        if spec.kind in ("continuous", "count"):
            row = {"column": col, "kind": spec.kind, **_stats_continuous(s)}
            rows_cont.append(row)
            _plot_continuous(s, col, figs_dir)

        elif spec.kind in ("ordinal", "nominal"):
            row = {"column": col, "kind": spec.kind,
                   **_stats_categorical(s, ordered=(spec.kind == "ordinal"))}
            rows_cat.append(row)
            if spec.kind == "ordinal":
                _plot_ordinal(s, col, figs_dir)
            else:
                _plot_nominal(s, col, figs_dir)

        elif spec.kind == "binary":
            row = {"column": col, "kind": "binary", **_stats_binary(s)}
            rows_bin.append(row)
            _plot_binary(s, col, figs_dir)

        elif spec.kind == "datetime":
            row = {"column": col, "kind": "datetime", **_stats_datetime(s)}
            rows_dt.append(row)
            _plot_datetime(s, col, figs_dir)

        elif spec.kind in ("id", "text"):
            row = {"column": col, "kind": spec.kind, **_stats_id(s)}
            rows_id.append(row)

    tables = {
        "continuous": pd.DataFrame(rows_cont),
        "categorical": pd.DataFrame(rows_cat),
        "binary": pd.DataFrame(rows_bin),
        "datetime": pd.DataFrame(rows_dt),
        "id_text": pd.DataFrame(rows_id),
    }
    for name, tbl in tables.items():
        if not tbl.empty:
            tbl.to_csv(tabs_dir / f"dda_{name}.csv", index=False)

    # Overall dataset overview
    overall = pd.DataFrame([{
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_cols_analysed": sum(not t.empty for t in tables.values()),
        "missing_cells_pct": round(df.isna().mean().mean() * 100, 2),
    }])
    overall.to_csv(tabs_dir / "dda_overall.csv", index=False)
    tables["overall"] = overall
    return tables
