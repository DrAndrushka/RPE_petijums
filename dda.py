"""
YlivertainenDDA — Descriptive Data Analysis

Two output paths:
  1. Notebook:   .analyse_numerical() / .analyse_categorical() / .analyse_binary()
                 print colored text, show plots inline, display styled tables.
  2. HTML:       .export_all_overviews()
                 re-runs the analyses with ANSI stripped, saves SVG figures to
                 output/figures_svg/, and writes a single self-contained HTML
                 file at output/dda_all_overviews_report.html.

Both paths use the SAME plotting + table-styling helpers, so what you see
in the notebook is what you get in the HTML.

TODO:
- Add optional quiet mode to suppress per-variable console output.
- Add optional JSON export for dashboard pipelines.
- Add configurable thresholds for cardinality-driven plot skipping.
"""

from __future__ import annotations

import base64
import contextlib
import html
import io
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib.ticker import FixedLocator, MaxNLocator
from scipy.stats import entropy, iqr, trim_mean

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colors (notebook only — stripped in HTML)
# ─────────────────────────────────────────────────────────────────────────────
BOLD = "\033[1m"
BLUE = "\033[38;5;39m"
GREEN = "\033[38;5;28m"
YELLOW = "\033[38;5;214m"
ORANGE = "\033[38;5;202m"
RED = "\033[38;5;196m"
RESET = "\033[0m"

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

# ─────────────────────────────────────────────────────────────────────────────
# Global plot style — applied once, used by every graph
# ─────────────────────────────────────────────────────────────────────────────
ACCENT = "#4a86b3"         # single-color bars/boxes — matches reference notebook

sns.set_theme(
    style="white",
    rc={
        "axes.edgecolor": "#222",
        "axes.linewidth": 0.9,
        "axes.labelcolor": "#222",
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.titlepad": 4,
        "axes.grid": False,
        "xtick.color": "#222",
        "ytick.color": "#222",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.size": 9,
        "figure.dpi": 110,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    },
)

MAINTENANCE_COLS = {"included_in_cohort", "row_id", "cohort_exclusion_reason", "target"}


def _format_mode(s: pd.Series, max_items: int = 3) -> str:
    """Compact mode string. If every value is its own mode, just say so."""
    modes = s.mode()
    if len(modes) >= len(s.dropna()):
        return "(all values unique)"
    if len(modes) > max_items:
        head = ", ".join(_fmt_num(m) for m in modes.iloc[:max_items])
        return f"{head}, … (+{len(modes) - max_items} more)"
    return ", ".join(_fmt_num(m) for m in modes)


def _fmt_num(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.2f}"
    return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# Shared plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _style_axis(ax, *, title=None, xlabel=None, ylabel=None, rotate=0):
    if title:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if rotate:
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(rotate)
            lbl.set_ha("right")


def _plot_numerical(series: pd.Series, *, col_name: str):
    """Histogram + boxplot, same look every time. Discrete x for low cardinality."""
    n_unique = series.nunique(dropna=True)
    x = series.dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.4), constrained_layout=True)

    uniques = np.sort(x.unique()) if len(x) else np.array([])
    if n_unique <= 20:
        # discrete: one bar per integer value, ticks ONLY on those values
        sns.histplot(x=x, ax=ax1, stat="density", discrete=True,
                     color=ACCENT, edgecolor="black", linewidth=0.6)
        if np.issubdtype(uniques.dtype, np.number):
            ax1.xaxis.set_major_locator(FixedLocator(uniques))
    else:
        sns.histplot(x=x, ax=ax1, stat="density", kde=True,
                     color=ACCENT, edgecolor="black", linewidth=0.5)
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))

    _style_axis(ax1, xlabel=col_name, ylabel="Density")

    sns.boxplot(x=x, ax=ax2, color=ACCENT, width=0.45,
                fliersize=3, linewidth=1.0)
    if n_unique <= 20 and np.issubdtype(uniques.dtype, np.number):
        ax2.xaxis.set_major_locator(FixedLocator(uniques))
    else:
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    _style_axis(ax2, xlabel=col_name, ylabel="")
    fig._dda_var = col_name
    return fig


def _plot_categorical(series: pd.Series, *, col_name: str):
    """One bar per actual category. No half-step ticks."""
    vc = series.value_counts(normalize=True, dropna=True)
    # respect ordered categoricals' order; otherwise sort by frequency desc
    if pd.api.types.is_categorical_dtype(series) and series.cat.ordered:
        vc = vc.reindex(series.cat.categories, fill_value=0)

    width = max(5.0, min(12.0, 0.55 * len(vc) + 3))
    fig, ax = plt.subplots(figsize=(width, 2.6), constrained_layout=True)
    sns.barplot(
        x=vc.index.astype(str),
        y=vc.values,
        ax=ax,
        color=ACCENT,
        edgecolor="black",
        linewidth=0.6,
    )
    _style_axis(ax, xlabel=col_name, ylabel="Density",
                rotate=25 if len(vc) > 4 else 0)
    fig._dda_var = col_name
    return fig


def _plot_binary(series: pd.Series, *, col_name: str):
    vc = series.value_counts(normalize=True, dropna=True)
    fig, ax = plt.subplots(figsize=(4.2, 2.4), constrained_layout=True)
    sns.barplot(
        x=vc.index.astype(str),
        y=vc.values,
        ax=ax,
        color=ACCENT,
        edgecolor="black",
        linewidth=0.6,
    )
    _style_axis(ax, xlabel=col_name, ylabel="Proportion")
    fig._dda_var = col_name
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Shared table styler — plain, no color coding
# ─────────────────────────────────────────────────────────────────────────────
def _style_overview(df: pd.DataFrame, *, formats: dict):
    if df.empty:
        return df.style
    return (
        df.style
        .format(formats, na_rep="—")
        .set_table_styles([
            {"selector": "th", "props": "background-color:#f3f5fb;color:#1f2a44;font-weight:600;text-align:left;padding:6px 8px;"},
            {"selector": "td", "props": "padding:5px 8px;"},
            {"selector": "tr:hover td", "props": "background-color:#eef3ff;"},
        ])
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────
class YlivertainenDDA:
    """Run descriptive analysis for numerical, categorical, and binary columns."""

    def __init__(self, task, df: pd.DataFrame):
        """Store task context and initialize DDA output containers."""
        self.task = task
        self.DDA = df.copy()
        self.cols_to_not_analyse: list[str] = []

        self.numerical_DDA: pd.DataFrame | None = None
        self.categorical_DDA: pd.DataFrame | None = None
        self.binary_DDA: pd.DataFrame | None = None

    # ── column selection ────────────────────────────────────────────────────
    def keep_from_analysis(self, cols):
        """Mark columns that should be excluded from DDA scans."""
        if cols:
            self.cols_to_not_analyse.extend(cols)
            print(f"{BOLD}🚫 Kept from analysis:{RESET} {cols}")
        else:
            print(f"{BOLD}✅ All columns will be analysed{RESET}")
        return self

    def _numerical_cols(self):
        return [
            c for c in self.DDA.select_dtypes(include="number", exclude="category")
            if self.DDA[c].nunique() != 2
            and c not in MAINTENANCE_COLS
            and c not in self.cols_to_not_analyse
        ]

    def _categorical_cols(self):
        return [
            c for c in self.DDA.select_dtypes(include="category", exclude=["bool", "boolean"])
            if not c.endswith("_missing")
            and c not in MAINTENANCE_COLS
            and len(self.DDA[c].value_counts()) > 2
            and c not in self.cols_to_not_analyse
        ]

    def _binary_cols(self):
        return [
            c for c in self.DDA.columns
            if len(self.DDA[c].value_counts()) == 2
            and c not in MAINTENANCE_COLS
            and c not in self.cols_to_not_analyse
        ]

    # ── overview ────────────────────────────────────────────────────────────
    def dataset_overview(self):
        """Print basic shape and dtype summary."""
        print(f"{BOLD}🟧 Rows & columns{RESET}")
        print(f"  📏 rows:    {len(self.DDA)}")
        print(f"  📐 columns: {len(self.DDA.columns)}")

        print(f"\n{BOLD}🟧 Dtype summary{RESET}")
        pad = max(len(c) for c in self.DDA.columns)
        for c in self.DDA.columns:
            print(f"  🔹 {BLUE}{BOLD}{c:<{pad}}{RESET} : {self.DDA[c].dtype}")
        return self

    # ════════════════════════════════════════════════════════════════════════
    # NUMERICAL
    # ════════════════════════════════════════════════════════════════════════
    def analyse_numerical(self):
        """Profile numerical features and render histogram+boxplot per column."""
        print(f"{BOLD}🔢 ═══════ NUMERICAL ═══════ 🔢{RESET}")
        cols = self._numerical_cols()
        if not cols:
            print("⚠️  No numerical columns to analyse.")
            return self

        rows = []
        for col in cols:
            s = self.DDA[col]
            x = s.dropna().to_numpy(dtype="float64")
            n = int(s.dropna().count())
            n_unique = int(s.nunique())
            missing_pct = float(s.isna().mean() * 100)
            mean = float(s.mean())
            std = float(x.std())
            cv = std / mean if mean else np.nan
            skew_v = float(s.skew())
            kurt_v = float(s.kurt())

            stats = dict(
                column=col, n=n, n_unique=n_unique, **{"missing_%": missing_pct},
                min=float(x.min()), max=float(x.max()),
                median=float(s.median()), mean=mean,
                trimmed_mean=float(trim_mean(x, 0.1)),
                mode=_format_mode(s),
                std=std, cv=cv,
                iqr=float(iqr(x)),
                p_5th=float(np.percentile(x, 5)),
                p_95th=float(np.percentile(x, 95)),
                skewness=skew_v, kurtosis=kurt_v,
            )
            rows.append(stats)

            # ── text ─────────────────────────────────────────────────────
            print(f"\n{BLUE}{BOLD}🟧 {col} 🟧{RESET}")
            print(f"  🧮 n: {n}   🔀 unique: {n_unique}   ❓ missing: {missing_pct:.2f}%")
            print(f"  📉 min: {stats['min']:.2f}   📈 max: {stats['max']:.2f}")
            print(f"  ⚖️  median: {stats['median']:.2f}   📊 mean: {mean:.2f}   ✂️ trimmed: {stats['trimmed_mean']:.2f}")
            print(f"  🏷️  mode: {stats['mode']}")
            print(f"  📏 IQR: {stats['iqr']:.2f}   5%: {stats['p_5th']:.2f}   95%: {stats['p_95th']:.2f}")
            print(f"  📐 std: {std:.2f}")

            # interpret CV / skew / kurtosis with colored verdict + emoji
            if np.isnan(cv):
                print(f"  🌡️  cv: n/a")
            elif cv < 0.10:
                print(f"  🌡️  cv: {cv:.2f} {GREEN}🟢 very stable{RESET}")
            elif cv < 0.30:
                print(f"  🌡️  cv: {cv:.2f} {GREEN}🟢 tight spread{RESET}")
            elif cv < 0.50:
                print(f"  🌡️  cv: {cv:.2f} {YELLOW}🟡 moderate{RESET}")
            elif cv < 1.00:
                print(f"  🌡️  cv: {cv:.2f} {ORANGE}🟠 noisy{RESET}")
            else:
                print(f"  🌡️  cv: {cv:.2f} {RED}🔴 chaotic{RESET}")

            if skew_v < -0.1:
                print(f"  ↩️  skew: {skew_v:.2f} left-skewed (most values high)")
            elif skew_v > 0.1:
                print(f"  ↪️  skew: {skew_v:.2f} right-skewed (most values low)")
            else:
                print(f"  ⚖️  skew: {skew_v:.2f} symmetrical")

            if kurt_v < -0.5:
                print(f"  🏔️  kurtosis: {kurt_v:.2f} platykurtic (thin tails)")
            elif kurt_v > 0.5:
                print(f"  🗻  kurtosis: {kurt_v:.2f} leptokurtic (fat tails / outliers)")
            else:
                print(f"  ⛰️  kurtosis: {kurt_v:.2f} mesokurtic (normal-ish)")

            # ── plot ─────────────────────────────────────────────────────
            _plot_numerical(s, col_name=col)
            plt.show()
            plt.close("all")

        table = pd.DataFrame(rows)
        formats = {
            "n": "{:.0f}", "n_unique": "{:.0f}", "missing_%": "{:.2f}",
            "min": "{:.2f}", "max": "{:.2f}", "median": "{:.2f}",
            "mean": "{:.2f}", "trimmed_mean": "{:.2f}",
            "std": "{:.2f}", "cv": "{:.2f}", "iqr": "{:.2f}",
            "p_5th": "{:.2f}", "p_95th": "{:.2f}",
            "skewness": "{:.2f}", "kurtosis": "{:.2f}",
        }
        print(f"\n{BOLD}📋 Numerical overview table{RESET}")
        display(_style_overview(table, formats=formats))
        self.numerical_DDA = table
        return self

    # ════════════════════════════════════════════════════════════════════════
    # CATEGORICAL
    # ════════════════════════════════════════════════════════════════════════
    def analyse_categorical(self):
        """Profile categorical features with balance/entropy diagnostics."""
        print(f"{BOLD}🚦 ═══════ CATEGORICAL ═══════ 🚦{RESET}")
        cols = self._categorical_cols()
        if not cols:
            print("⚠️  No categorical columns to analyse.")
            return self

        def _balance(s):
            vc = s.value_counts(normalize=True, dropna=True)
            p = vc.values
            if len(p) <= 1:
                return 0.0
            p = p[p > 0]
            ent = -(p * np.log(p)).sum()
            max_ent = np.log(len(p))
            return float(ent / max_ent) if max_ent else 0.0

        def _median_cat(s):
            if not (pd.api.types.is_categorical_dtype(s) and s.cat.ordered):
                return pd.NA
            vc = s.value_counts(normalize=True).reindex(s.cat.categories, fill_value=0)
            cum = vc.cumsum()
            mask = cum >= 0.5
            return cum.index[mask][0] if mask.any() else pd.NA

        rows = []
        for col in cols:
            s = self.DDA[col]
            vc = s.value_counts(normalize=True, dropna=True)
            n = int(s.dropna().count())
            n_unique = int(s.nunique())
            missing_pct = float(s.isna().mean() * 100)
            first_mode = vc.index[0]
            second_mode = vc.index[1] if len(vc) > 1 else pd.NA
            first_pct = float(vc.iloc[0] * 100)
            second_pct = float(vc.iloc[1] * 100) if len(vc) > 1 else np.nan
            imbalance = first_pct / second_pct if second_pct else np.nan
            rarest = f"{vc.index[-1]}: {float(vc.iloc[-1]):.2%}"
            balance = _balance(s)
            ent_bin = float(entropy(vc / vc.sum(), base=2))

            rows.append(dict(
                column=col, n=n, n_unique=n_unique, missing_pct=missing_pct,
                first_mode=str(first_mode), second_mode=str(second_mode),
                rarest=rarest, first_mode_pct=first_pct, second_mode_pct=second_pct,
                max_class_imbalance=imbalance,
                median_category=str(_median_cat(s)),
                balance=balance, entropy_bin=ent_bin,
            ))

            # ── text ─────────────────────────────────────────────────────
            print(f"\n{BLUE}{BOLD}🟧 {col} 🟧{RESET}")
            print(f"  🧮 n: {n}   ❓ missing: {missing_pct:.2f}%")

            if n_unique == 1:
                print(f"  🔢 cardinality: {n_unique} {RED}🔴 constant → drop{RESET}")
            elif n_unique <= 5:
                print(f"  🔢 cardinality: {n_unique} {GREEN}🟢 very low → one-hot ready{RESET}")
            elif n_unique <= 20:
                print(f"  🔢 cardinality: {n_unique} {GREEN}🟢 moderate{RESET}")
            elif n_unique <= 50:
                print(f"  🔢 cardinality: {n_unique} {ORANGE}🟠 high → group rare levels{RESET}")
            elif n_unique <= 200:
                print(f"  🔢 cardinality: {n_unique} {ORANGE}🟠 very high{RESET}")
            else:
                print(f"  🔢 cardinality: {n_unique} {RED}🔴 ID-like → drop{RESET}")

            print(f"  🏆 mode: {first_mode} ({first_pct:.1f}%)")
            print(f"  🥈 second: {second_mode} ({second_pct:.1f}%)" if not np.isnan(second_pct) else "  🥈 second: —")
            print(f"  🐭 rarest: {rarest}")

            if not np.isnan(imbalance):
                if imbalance < 1.5:
                    print(f"  ⚖️  imbalance: {imbalance:.2f} {GREEN}🟢 balanced{RESET}")
                elif imbalance < 3:
                    print(f"  ⚖️  imbalance: {imbalance:.2f} {GREEN}🟢 mild{RESET}")
                elif imbalance < 10:
                    print(f"  ⚖️  imbalance: {imbalance:.2f} {ORANGE}🟠 strong{RESET}")
                else:
                    print(f"  ⚖️  imbalance: {imbalance:.2f} {RED}🔴 extreme{RESET}")

            if ent_bin < 0.1:
                print(f"  🎲 entropy: {ent_bin:.2f} {RED}🔴 flatline → drop{RESET}")
            elif ent_bin < 1.5:
                print(f"  🎲 entropy: {ent_bin:.2f} {ORANGE}🟠 low diversity{RESET}")
            elif ent_bin < 3.5:
                print(f"  🎲 entropy: {ent_bin:.2f} {GREEN}🟢 sweet spot{RESET}")
            elif ent_bin < 5:
                print(f"  🎲 entropy: {ent_bin:.2f} {ORANGE}🟠 high chaos → group{RESET}")
            else:
                print(f"  🎲 entropy: {ent_bin:.2f} {RED}🔴 white noise → drop{RESET}")

            if balance < 0.1:
                print(f"  🪶 balance: {balance:.2f} {RED}🔴 dominated{RESET}")
            elif balance < 0.4:
                print(f"  🪶 balance: {balance:.2f} {ORANGE}🟠 heavy imbalance{RESET}")
            elif balance < 0.7:
                print(f"  🪶 balance: {balance:.2f} {GREEN}🟢 healthy{RESET}")
            elif balance < 0.9:
                print(f"  🪶 balance: {balance:.2f} {ORANGE}🟠 broad{RESET}")
            else:
                print(f"  🪶 balance: {balance:.2f} {ORANGE}🟠 near-uniform{RESET}")

            # ── plot (skip if too many categories to be useful) ─────────
            if n_unique <= 25:
                _plot_categorical(s, col_name=col)
                plt.show()
                plt.close("all")

        table = pd.DataFrame(rows)
        formats = {
            "n": "{:.0f}", "n_unique": "{:.0f}", "missing_pct": "{:.2f}",
            "first_mode_pct": "{:.2f}", "second_mode_pct": "{:.2f}",
            "max_class_imbalance": "{:.2f}",
            "balance": "{:.2f}", "entropy_bin": "{:.2f}",
        }
        print(f"\n{BOLD}📋 Categorical overview table{RESET}")
        display(_style_overview(table, formats=formats))
        self.categorical_DDA = table
        return self

    # ════════════════════════════════════════════════════════════════════════
    # BINARY
    # ════════════════════════════════════════════════════════════════════════
    def analyse_binary(self):
        """Profile binary features with imbalance and entropy diagnostics."""
        print(f"{BOLD}⚖️  ═══════ BINARY ═══════ ⚖️{RESET}")
        cols = self._binary_cols()
        if not cols:
            print("⚠️  No binary columns to analyse.")
            return self

        rows = []
        for col in cols:
            s = self.DDA[col]
            vc = s.value_counts(normalize=True, dropna=True)
            cat1, cat0 = vc.index[0], vc.index[1]
            p1, p0 = float(vc.iloc[0]), float(vc.iloc[1])
            n = int(s.dropna().count())
            missing_pct = pd.NA if "_missing" in col else float(s.isna().mean() * 100)

            ratio = p1 / p0 if p0 else np.inf
            if ratio > 1.1:
                mode, mode_pct = cat1, p1 * 100
            elif ratio < 0.9:
                mode, mode_pct = cat0, p0 * 100
            else:
                mode, mode_pct = "equal", p0 * 100

            balance = 2 * min(p0, p1)
            ent_bin = float(entropy(vc / vc.sum(), base=2))

            rows.append(dict(
                column=col, n=n, missing_pct=missing_pct,
                cat1=str(cat1), p1=p1, cat0=str(cat0), p0=p0,
                mode=str(mode), mode_pct=mode_pct,
                balance=balance, entropy_bin=ent_bin,
            ))

            # ── text ─────────────────────────────────────────────────────
            print(f"\n{BLUE}{BOLD}🟧 {col} 🟧{RESET}")
            print(f"  🧮 n: {n}", end="")
            if "_missing" not in col:
                print(f"   ❓ missing: {missing_pct:.2f}%")
            else:
                print()
            print(f"  🅰️  {cat1}: {p1:.2f}    🅱️  {cat0}: {p0:.2f}")
            print(f"  🏆 mode: {mode} ({mode_pct:.1f}%)")

            if balance < 0.1:
                print(f"  🪶 balance: {balance:.2f} {RED}🔴 essentially constant{RESET}")
            elif balance < 0.3:
                print(f"  🪶 balance: {balance:.2f} {YELLOW}🟡 heavy imbalance{RESET}")
            elif balance < 0.7:
                print(f"  🪶 balance: {balance:.2f} {ORANGE}🟠 moderate{RESET}")
            else:
                print(f"  🪶 balance: {balance:.2f} {GREEN}🟢 healthy{RESET}")

            if ent_bin < 0.05:
                print(f"  🎲 entropy: {ent_bin:.2f} {RED}🔴 near-constant{RESET}")
            elif ent_bin < 0.3:
                print(f"  🎲 entropy: {ent_bin:.2f} {ORANGE}🟠 very skewed{RESET}")
            elif ent_bin < 0.7:
                print(f"  🎲 entropy: {ent_bin:.2f} {YELLOW}🟡 moderate{RESET}")
            else:
                print(f"  🎲 entropy: {ent_bin:.2f} {GREEN}🟢 ~50/50, ideal{RESET}")

            _plot_binary(s, col_name=col)
            plt.show()
            plt.close("all")

        table = pd.DataFrame(rows)
        formats = {
            "n": "{:.0f}", "missing_pct": "{:.2f}",
            "p1": "{:.2f}", "p0": "{:.2f}", "mode_pct": "{:.2f}",
            "balance": "{:.2f}", "entropy_bin": "{:.2f}",
        }
        print(f"\n{BOLD}📋 Binary overview table{RESET}")
        display(_style_overview(table, formats=formats))
        self.binary_DDA = table
        return self

    # ════════════════════════════════════════════════════════════════════════
    # HTML EXPORT — text → graphs → overview table, per category
    # ════════════════════════════════════════════════════════════════════════
    def export_all_overviews(self, output_dir: str = "output"):
        """Export combined DDA HTML report + CSV/XLSX overviews."""
        vector_dir = os.path.join(output_dir, "figures_svg")
        tables_dir = os.path.join(output_dir, "tables")
        os.makedirs(vector_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)

        sections = {
            "numerical":   {"title": "🔢 Numerical",   "vars": {}, "table_html": ""},
            "categorical": {"title": "🚦 Categorical", "vars": {}, "table_html": ""},
            "binary":      {"title": "⚖️ Binary",      "vars": {}, "table_html": ""},
        }

        # one dict per category: column -> {"text_lines": [...], "figures": [{png_b64, svg_rel}]}
        for key in sections:
            sections[key]["vars"] = {}

        current_category: str | None = None
        current_var: str | None = None
        figure_counter = 0
        stdout_buffer = io.StringIO()
        variable_header = re.compile(r"🟧\s*(.+?)\s*🟧")

        def _ensure_var(cat, var):
            if var not in sections[cat]["vars"]:
                sections[cat]["vars"][var] = {"text_lines": [], "figures": []}

        # ── capture display() = the styled overview table ───────────────
        def _capture_display(obj):
            try:
                table_html = obj.to_html() if hasattr(obj, "to_html") else (
                    obj.to_html(index=False) if isinstance(obj, pd.DataFrame)
                    else f"<pre>{html.escape(repr(obj))}</pre>"
                )
            except Exception as exc:
                table_html = f"<pre>table render failed: {html.escape(str(exc))}</pre>"
            if current_category:
                sections[current_category]["table_html"] = table_html

        # ── capture plt.show() = save SVG + embed PNG inline ────────────
        def _safe_name(text):
            s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(text).strip())
            return s.strip("_") or "unnamed"

        def _capture_show(*_, **__):
            nonlocal figure_counter
            fig = plt.gcf()
            if fig is None or not fig.get_axes():
                return
            figure_counter += 1
            cat = current_category or "numerical"
            var = getattr(fig, "_dda_var", None) or f"figure_{figure_counter}"
            _ensure_var(cat, var)

            png_buf = io.BytesIO()
            fig.savefig(png_buf, format="png", dpi=160, bbox_inches="tight",
                        facecolor="white")
            png_b64 = base64.b64encode(png_buf.getvalue()).decode("utf-8")
            png_buf.close()

            svg_name = f"{cat}_{_safe_name(var)}.svg"
            svg_path = os.path.join(vector_dir, svg_name)
            fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor="white")

            sections[cat]["vars"][var]["figures"].append({
                "png_b64": png_b64,
                "svg_rel": os.path.relpath(svg_path, output_dir),
            })

        # ── run the three analyses with redirects ──────────────────────
        global display
        original_display = display
        original_show = plt.show
        try:
            display = _capture_display
            plt.show = _capture_show
            with contextlib.redirect_stdout(stdout_buffer):
                current_category = "numerical"
                self.analyse_numerical()
                current_category = "categorical"
                self.analyse_categorical()
                current_category = "binary"
                self.analyse_binary()
        finally:
            display = original_display
            plt.show = original_show
            current_category = None

        # ── parse captured text into (category, variable) buckets ───────
        clean = ANSI_RE.sub("", stdout_buffer.getvalue())
        parsed_cat = None
        parsed_var = None
        for raw in clean.splitlines():
            line = raw.rstrip()
            if not line.strip():
                continue
            upper = line.upper()
            is_banner = "═══════" in line
            if is_banner and "NUMERICAL" in upper:
                parsed_cat, parsed_var = "numerical", None
                continue
            if is_banner and "CATEGORICAL" in upper:
                parsed_cat, parsed_var = "categorical", None
                continue
            if is_banner and "BINARY" in upper:
                parsed_cat, parsed_var = "binary", None
                continue
            if line.lstrip().startswith("📋"):  # final overview-table header — skip
                parsed_var = None
                continue

            m = variable_header.search(line)
            if m and parsed_cat:
                cand = m.group(1).strip()
                if cand and cand.upper() not in {"NUMERICAL", "CATEGORICAL", "BINARY"}:
                    parsed_var = cand
                    _ensure_var(parsed_cat, parsed_var)
                continue

            if parsed_cat and parsed_var:
                sections[parsed_cat]["vars"][parsed_var]["text_lines"].append(line)

        # ── render HTML ─────────────────────────────────────────────────
        def _render_var_card(var_name, payload):
            text = "\n".join(payload["text_lines"]).strip()
            figs_html = ""
            for i, f in enumerate(payload["figures"], 1):
                figs_html += (
                    f'<div class="fig">'
                    f'  <img src="data:image/png;base64,{f["png_b64"]}" alt="{html.escape(var_name)} fig {i}">'
                    f'  <div class="fig-meta"><a href="{html.escape(f["svg_rel"])}" target="_blank" rel="noopener">⬇ SVG</a></div>'
                    f'</div>'
                )
            text_html = f"<pre>{html.escape(text)}</pre>" if text else ""
            return (
                f'<div class="card">'
                f'  <h3>{html.escape(var_name)}</h3>'
                f'  {text_html}'
                f'  {figs_html}'
                f'</div>'
            )

        def _render_section(cat_key):
            sec = sections[cat_key]
            cards = "\n".join(_render_var_card(v, p) for v, p in sec["vars"].items()
                              if p["text_lines"] or p["figures"])
            table = (f'<div class="card overview"><h3>📋 Overview table</h3>{sec["table_html"]}</div>'
                     if sec["table_html"] else "")
            return f'<section><h2>{sec["title"]}</h2>{cards}{table}</section>'

        report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DDA Report — {html.escape(str(self.task))}</title>
<style>
  :root {{
    --bg: #fafbfd;
    --panel: #ffffff;
    --muted: #6b7280;
    --text: #1f2937;
    --border: #e5e7eb;
    --accent: #3a7ca5;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.55;
  }}
  .wrap {{ max-width: 1400px; margin: 0 auto; padding: 28px 22px 64px; }}
  header.hero {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 22px;
    box-shadow: 0 1px 2px rgba(15,23,42,.04);
  }}
  header.hero h1 {{ margin: 0 0 4px; font-size: 26px; color: var(--accent); }}
  header.hero .meta {{ color: var(--muted); font-size: 13px; }}
  section {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 22px;
    box-shadow: 0 1px 2px rgba(15,23,42,.04);
  }}
  h2 {{ margin: 0 0 14px; font-size: 20px; color: #111827; border-bottom: 2px solid var(--accent); padding-bottom: 6px; display: inline-block; }}
  .card {{
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    margin: 12px 0;
    background: #fdfdff;
  }}
  .card h3 {{ margin: 0 0 10px; font-size: 15px; color: var(--accent); }}
  pre {{
    margin: 0 0 10px;
    white-space: pre-wrap;
    word-wrap: break-word;
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    color: #1f2937;
    font: 12.5px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }}
  .fig {{
    margin-top: 8px;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px;
    background: #fff;
  }}
  .fig img {{ display: block; width: 100%; height: auto; border-radius: 4px; }}
  .fig-meta {{ margin-top: 4px; font-size: 12px; text-align: right; }}
  .fig-meta a {{ color: var(--accent); text-decoration: none; }}
  .fig-meta a:hover {{ text-decoration: underline; }}
  .overview {{ overflow-x: auto; }}
  /* Constrain pandas Styler tables: cells stay compact, gradient renders normally */
  .overview table {{
    border-collapse: collapse;
    width: auto !important;
    font-size: 12.5px;
    table-layout: auto;
  }}
  .overview th, .overview td {{
    border: 1px solid #e5e7eb;
    padding: 4px 8px !important;
    white-space: nowrap;
    text-align: right;
    line-height: 1.35;
    height: 22px;
  }}
  .overview th {{ background: #f3f5fb !important; color: #1f2a44 !important; font-weight: 600; text-align: left; }}
  .overview td:first-child, .overview th:first-child {{ text-align: left; font-weight: 500; }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <h1>🧬 DDA Report</h1>
    <div class="meta">
      Task: <b>{html.escape(str(self.task))}</b>
      · Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
      · SVGs: <code>{html.escape(vector_dir)}</code>
    </div>
  </header>
  {_render_section("numerical")}
  {_render_section("categorical")}
  {_render_section("binary")}
</div>
</body>
</html>
"""

        out_path = os.path.join(output_dir, "dda_all_overviews_report.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_html)

        # ── export tables: CSV per category + one combined Excel workbook ──
        csv_paths = []
        for key, table in (("numerical", self.numerical_DDA),
                           ("categorical", self.categorical_DDA),
                           ("binary", self.binary_DDA)):
            if table is not None and not table.empty:
                p = os.path.join(tables_dir, f"{key}.csv")
                table.to_csv(p, index=False)
                csv_paths.append(p)

        xlsx_path = os.path.join(output_dir, "dda_overview.xlsx")
        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
                wrote_any = False
                for sheet, table in (("numerical", self.numerical_DDA),
                                     ("categorical", self.categorical_DDA),
                                     ("binary", self.binary_DDA)):
                    if table is not None and not table.empty:
                        table.to_excel(xw, sheet_name=sheet, index=False)
                        wrote_any = True
                if not wrote_any:
                    pd.DataFrame({"info": ["no data"]}).to_excel(xw, sheet_name="empty", index=False)
        except ImportError:
            xlsx_path = None
            print("⚠️  openpyxl not installed — skipping Excel export. Install with: pip install openpyxl")

        print(f"✅ HTML report:  {out_path}")
        print(f"✅ SVG figures:  {vector_dir}")
        for p in csv_paths:
            print(f"✅ CSV table:    {p}")
        if xlsx_path:
            print(f"✅ Excel:        {xlsx_path}")
        return self
