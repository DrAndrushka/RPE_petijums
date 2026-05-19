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
- continuous / count :
    n, n_unique, missing_pct, min, p_5th, median, mean, trimmed_mean,
    p_95th, max, mode, std, cv, iqr, skewness, kurtosis
- ordinal / nominal / binary :
    n, n_unique, missing_pct, ordered, first_mode, first_mode_pct,
    second_mode, second_mode_pct, rarest, max_class_imbalance,
    median_category (ordinal only), balance, entropy_bin
- datetime : n, missing_pct, min, max, span_days
- id/text  : n, missing_pct, n_unique

Definitions
-----------
- trimmed_mean       : 10% symmetric trim (scipy.stats.trim_mean).
- cv                 : std / mean (NaN when |mean| < 1e-12).
- iqr                : Q3 - Q1.
- kurtosis           : Fisher's excess kurtosis (0 = normal).
- max_class_imbalance: first_mode_count / rarest_count
                       (1 = perfect balance, large = degenerate).
- balance            : normalized Shannon entropy H / log2(n_unique)
                       (0 = single class, 1 = uniform).
- entropy_bin        : raw Shannon entropy in BITS.

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
from scipy.stats import skew, kurtosis, trim_mean

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


# Display formatting comes from cleaning.format_table_for_csv (shared helper).
# CSV save only — in-memory DataFrames keep raw floats for downstream stages.
from cleaning import format_table_for_csv as _format_table_for_csv


# ---------------------------------------------------------------------------
# Per-column stats
# ---------------------------------------------------------------------------

def _stats_continuous(s: pd.Series) -> dict:
    nn = s.dropna()
    n = len(nn)
    if n == 0:
        return {"n": 0, "n_unique": 0, "missing_pct": 100.0,
                "min": np.nan, "p_5th": np.nan, "median": np.nan,
                "mean": np.nan, "trimmed_mean": np.nan, "p_95th": np.nan,
                "max": np.nan, "mode": np.nan, "std": np.nan, "cv": np.nan,
                "iqr": np.nan, "skewness": np.nan, "kurtosis": np.nan}

    q1, q3 = float(nn.quantile(0.25)), float(nn.quantile(0.75))
    mean = float(nn.mean())
    std = float(nn.std(ddof=1)) if n > 1 else np.nan
    cv = float(std / mean) if (std == std and abs(mean) > 1e-12) else np.nan
    mode_vals = nn.mode()
    return {
        "n": int(n),
        "n_unique": int(nn.nunique()),
        "missing_pct": round(s.isna().mean() * 100, 2),
        "min": float(nn.min()),
        "p_5th": float(nn.quantile(0.05)),
        "median": float(nn.median()),
        "mean": mean,
        "trimmed_mean": float(trim_mean(nn, 0.1)) if n >= 5 else np.nan,
        "p_95th": float(nn.quantile(0.95)),
        "max": float(nn.max()),
        "mode": float(mode_vals.iloc[0]) if len(mode_vals) == 1 else np.nan,
        "std": std,
        "cv": cv,
        "iqr": q3 - q1,
        "skewness": float(skew(nn)) if n > 2 else np.nan,
        "kurtosis": float(kurtosis(nn, fisher=True)) if n > 3 else np.nan,
    }


def _stats_categorical(s: pd.Series, ordered: bool) -> dict:
    """Shared stats for ordinal, nominal, and binary columns.

    median_category is filled only when `ordered=True`; for nominal/binary it
    is NaN by definition (a median on an unordered set has no meaning).
    """
    nn = s.dropna()
    n = int(nn.size)
    base = {
        "n": n,
        "n_unique": int(nn.nunique()),
        "missing_pct": round(s.isna().mean() * 100, 2),
        "ordered": ordered,
        "first_mode": np.nan, "first_mode_pct": np.nan,
        "second_mode": np.nan, "second_mode_pct": np.nan,
        "rarest": np.nan, "max_class_imbalance": np.nan,
        "median_category": np.nan,
        "balance": np.nan, "entropy_bin": np.nan,
    }
    if n == 0:
        return base

    vc = nn.value_counts()  # sorted descending
    base["first_mode"] = vc.index[0]
    base["first_mode_pct"] = round(float(vc.iloc[0] / n * 100), 2)
    if len(vc) >= 2:
        base["second_mode"] = vc.index[1]
        base["second_mode_pct"] = round(float(vc.iloc[1] / n * 100), 2)
    base["rarest"] = vc.index[-1]
    rarest_count = int(vc.iloc[-1])
    base["max_class_imbalance"] = (round(float(vc.iloc[0] / rarest_count), 2)
                                   if rarest_count > 0 else np.nan)

    # Shannon entropy in bits + normalized balance
    p = vc.values / n
    p = p[p > 0]
    H_bits = float(-(p * np.log2(p)).sum())
    base["entropy_bin"] = round(H_bits, 4)
    k = int((vc > 0).sum())
    base["balance"] = round(H_bits / np.log2(k), 4) if k > 1 else np.nan

    # Median category — only for ordinal
    if ordered:
        if isinstance(s.dtype, pd.CategoricalDtype) and s.dtype.ordered:
            cats = list(s.cat.categories)
            codes = pd.Series(pd.Categorical(nn, categories=cats, ordered=True).codes)
            med_code = codes.median()
            if pd.notna(med_code):
                idx = int(np.floor(med_code))
                idx = max(0, min(idx, len(cats) - 1))
                base["median_category"] = cats[idx]
        else:
            try:
                base["median_category"] = float(nn.median())
            except Exception:
                pass
    return base


def _stats_binary(s: pd.Series) -> dict:
    """Mirror of the categorical schema (ordered=False) for binary columns.

    Binary is treated as a 2-class nominal: same column set as the categorical
    table. p_true / n_true / n_false are derivable from first_mode + counts,
    so they're intentionally omitted to keep the schema uniform.
    """
    return _stats_categorical(s, ordered=False)


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


def _ordinal_bar_order(s: pd.Series, ordered_levels: list | None) -> list:
    obs = set(s.dropna())
    if not obs:
        return []
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [c for c in s.cat.categories if c in obs]
    if ordered_levels:
        head = [lv for lv in ordered_levels if lv in obs]
        tail = sorted((x for x in obs if x not in ordered_levels), key=str)
        return head + tail
    return sorted(obs, key=str)


def _plot_ordinal(
    s: pd.Series, name: str, out_dir: Path, *, ordered_levels: list | None = None,
) -> list[Path]:
    nn = s.dropna()
    if nn.empty:
        return []
    order = _ordinal_bar_order(nn, ordered_levels)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(x=nn.astype(str), order=[str(o) for o in order], ax=ax, color="#3b7ddd")
    ax.set_title(f"Ordinal distribution — {name}")
    ax.set_xlabel(name); ax.set_ylabel("count")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    p = out_dir / f"{name}__bar.svg"
    _save_fig(fig, p)
    return [p]


def plot_distribution_by_year(
    df: pd.DataFrame,
    variable: str,
    year_col: str,
    out_dir: Path | str,
    *,
    min_years: int = 2,
    top_n: int = 15,
) -> Path | None:
    """
    Two-panel SVG: overall category counts, then composition within each year.

    Saved as ``<variable>__bar_by_year.svg``. Returns ``None`` when ``year_col``
    is missing, the variable is absent, or fewer than ``min_years`` distinct years.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if variable not in df.columns or year_col not in df.columns:
        return None

    sub = df[[variable, year_col]].dropna()
    if sub.empty:
        return None

    years = sorted(sub[year_col].astype(str).unique(), key=str)
    if len(years) < min_years:
        return None

    vc = sub[variable].astype(str).value_counts()
    if len(vc) > top_n:
        keep = set(vc.head(top_n).index)
        sub = sub.copy()
        sub[variable] = sub[variable].astype(str).where(
            sub[variable].astype(str).isin(keep), "(other)"
        )
        vc = sub[variable].value_counts()
    cat_order = list(vc.index)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8),
        gridspec_kw={"height_ratios": [1, 1.15]},
    )
    fig.subplots_adjust(hspace=0.42, top=0.96, bottom=0.11, right=0.78)

    sns.countplot(
        data=sub.assign(**{variable: sub[variable].astype(str)}),
        y=variable,
        order=cat_order,
        ax=ax1,
        color="#3b7ddd",
    )
    ax1.set_title(f"Overall counts — {variable}")
    ax1.set_xlabel("count")
    ax1.set_ylabel(variable)

    plot_df = sub.assign(
        **{
            variable: sub[variable].astype(str),
            year_col: sub[year_col].astype(str),
        }
    )
    ct = pd.crosstab(plot_df[year_col], plot_df[variable], normalize="index")
    ct = ct.reindex(index=[str(y) for y in years], columns=cat_order, fill_value=0.0)
    colors = sns.color_palette("husl", n_colors=len(cat_order))
    x = np.arange(len(ct))
    bottom = np.zeros(len(ct))
    for i, cat in enumerate(ct.columns):
        heights = ct[cat].to_numpy()
        ax2.bar(x, heights, bottom=bottom, label=cat, color=colors[i], width=0.65)
        bottom += heights
    ax2.set_xticks(x)
    ax2.set_xticklabels(ct.index)
    ax2.set_ylim(0, 1)
    ax2.set_title(f"Share within each {year_col} (row-normalised)")
    ax2.set_xlabel(year_col)
    ax2.set_ylabel("proportion")
    ax2.legend(title=variable, bbox_to_anchor=(1.02, 1), loc="upper left")

    chi2_note: str | None = None
    try:
        from scipy.stats import chi2_contingency

        raw_ct = pd.crosstab(plot_df[year_col], plot_df[variable])
        if raw_ct.size > 0 and raw_ct.values.sum() > 0:
            _, p, _, expected = chi2_contingency(raw_ct)
            if (expected >= 5).all():
                chi2_note = f"χ² ({year_col} × {variable}): p = {p:.4g}"
    except Exception:
        pass

    if chi2_note:
        # Centered footer — fixed slot, clear of axis labels and legend.
        fig.text(
            0.5, 0.03, chi2_note,
            ha="center", va="center",
            fontsize=8.5, color="#1f2937",
            transform=fig.transFigure,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="#f3f4f6",
                edgecolor="#9ca3af",
                linewidth=0.8,
            ),
        )

    p = out_dir / f"{variable}__bar_by_year.svg"
    fig.savefig(p, format="svg", bbox_inches="tight")
    plt.close(fig)
    return p


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
                _plot_ordinal(s, col, figs_dir, ordered_levels=spec.ordered_levels)
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

    # Canonical column order per table (matches the docstring at top of file).
    CONT_ORDER = ["column", "kind", "n", "n_unique", "missing_pct",
                  "min", "p_5th", "median", "mean", "trimmed_mean",
                  "p_95th", "max", "mode", "std", "cv", "iqr",
                  "skewness", "kurtosis"]
    CAT_ORDER = ["column", "kind", "ordered", "n", "n_unique", "missing_pct",
                 "first_mode", "first_mode_pct",
                 "second_mode", "second_mode_pct",
                 "rarest", "max_class_imbalance",
                 "median_category", "balance", "entropy_bin"]

    def _reorder(df_tbl, order):
        if df_tbl.empty:
            return df_tbl
        cols = [c for c in order if c in df_tbl.columns]
        extras = [c for c in df_tbl.columns if c not in cols]
        return df_tbl[cols + extras]

    tables = {
        "continuous": _reorder(pd.DataFrame(rows_cont), CONT_ORDER),
        "categorical": _reorder(pd.DataFrame(rows_cat), CAT_ORDER),
        "binary": _reorder(pd.DataFrame(rows_bin), CAT_ORDER),
        "datetime": pd.DataFrame(rows_dt),
        "id_text": pd.DataFrame(rows_id),
    }
    for name, tbl in tables.items():
        if not tbl.empty:
            # Format-on-save: integers stay int, fractions -> 3 sig figs.
            # In-memory `tbl` is untouched so downstream code keeps full precision.
            _format_table_for_csv(tbl).to_csv(tabs_dir / f"dda_{name}.csv", index=False)  # display-only rounding

    # Overall dataset overview
    overall = pd.DataFrame([{
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_cols_analysed": sum(not t.empty for t in tables.values()),
        "missing_cells_pct": round(df.isna().mean().mean() * 100, 2),
    }])
    _format_table_for_csv(overall).to_csv(tabs_dir / "dda_overall.csv", index=False)
    tables["overall"] = overall
    return tables
