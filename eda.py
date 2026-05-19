"""
eda.py
=======
Univariate target × predictor association screening.

Statistical rules (kind-pair → test)
------------------------------------
target kind = binary or ordinal-with-2-levels (we always encode the positive
class as 1, negative as 0 before testing).

predictor kind | test                            | effect size
-------------- | -------------------------------- | -----------
continuous     | Mann–Whitney U (two-sided)       | r = |Z| / √N (rank-biserial)
count          | Mann–Whitney U                   | r = |Z| / √N
ordinal        | Spearman ρ (rank correlation)    | ρ
nominal        | χ² (or Fisher exact if 2×2 and any expected < 5) | Cramér's V
binary         | Fisher exact (2×2)               | odds ratio + Cramér's V
datetime       | converted to days-since-min → Mann–Whitney      | r = |Z| / √N

All p-values per target are corrected with Benjamini–Hochberg (FDR).

Outputs (under output/eda/)
---------------------------
- tables/associations.csv  : long-format (target, predictor, test, stat, p,
                             p_fdr, effect, effect_size, n_used)
- figures/<target>__<predictor>.svg : the appropriate seaborn plot
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import (
    mannwhitneyu, spearmanr, chi2_contingency, fisher_exact, norm,
)
from statsmodels.stats.proportion import proportion_confint

from schema_infer import ColSpec
from cleaning import format_table_for_csv as _format_table_for_csv  # CSV display-only rounding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs(root: Path) -> tuple[Path, Path]:
    figs = root / "eda" / "figures"
    tabs = root / "eda" / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    return figs, tabs


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    """
    Benjamini–Hochberg FDR-adjusted p-values (q-values).
    Implements the step-up procedure: q_i = min over k>=i of p_(k) * m / k.
    """
    p = pd.Series(p).astype(float)
    valid = p.notna()
    pv = p[valid].values
    m = len(pv)
    if m == 0:
        return p.copy()
    order = np.argsort(pv)
    ranks = np.argsort(order) + 1  # 1-based ranks
    raw = pv * m / ranks
    # enforce monotonicity from the largest p downward
    sorted_idx = np.argsort(pv)
    sorted_q = raw[sorted_idx]
    for i in range(len(sorted_q) - 2, -1, -1):
        sorted_q[i] = min(sorted_q[i], sorted_q[i + 1])
    q = np.empty_like(sorted_q)
    q[sorted_idx] = np.clip(sorted_q, 0, 1)
    out = p.copy()
    out.loc[valid] = q
    return out


def _encode_binary_target(y: pd.Series, positive_class) -> pd.Series:
    """Map a binary target to {0,1} with `positive_class` -> 1. Returns float dtype with NaN preserved."""
    if positive_class is None:
        nn = y.dropna().unique()
        if len(nn) != 2:
            raise ValueError(f"Target '{y.name}' is not binary (unique values: {nn})")
        positive_class = True if True in nn else 1 if 1 in nn else sorted(nn, key=str)[-1]
    out = pd.Series(np.where(y.isna(), np.nan, (y == positive_class).astype(float)), index=y.index)
    return out, positive_class


def _cramers_v(table: np.ndarray) -> float:
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.sum()
    if n == 0:
        return np.nan
    r, c = table.shape
    denom = n * (min(r, c) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan


def _mwu_with_effect(x_group1: np.ndarray, x_group0: np.ndarray):
    """Mann–Whitney U two-sided with rank-biserial effect size r = |Z|/√N."""
    n1, n0 = len(x_group1), len(x_group0)
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan, np.nan, n1 + n0
    res = mannwhitneyu(x_group1, x_group0, alternative="two-sided")
    U = float(res.statistic)
    p = float(res.pvalue)
    # Convert U to Z (large-sample approximation, used for effect size only).
    mu = n1 * n0 / 2
    sigma = np.sqrt(n1 * n0 * (n1 + n0 + 1) / 12)
    z = (U - mu) / sigma if sigma > 0 else np.nan
    r = abs(z) / np.sqrt(n1 + n0) if sigma > 0 else np.nan
    return U, p, r, n1 + n0


# ---------------------------------------------------------------------------
# Per-pair plotting
# ---------------------------------------------------------------------------

def _polish_ax(ax: plt.Axes) -> None:
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _categorical_fig_width(n_levels: int) -> float:
    return max(3.2, min(6.0, 1.4 * n_levels + 1.2))


def _errorbar_yerr(props, lo, hi) -> np.ndarray:
    """Matplotlib needs non-negative error bar lengths (Wilson CI vs k/n can disagree slightly)."""
    props_a = np.clip(np.asarray(props, dtype=float), 0.0, 1.0)
    lo_a = np.asarray(lo, dtype=float)
    hi_a = np.asarray(hi, dtype=float)
    return np.vstack([
        np.maximum(0.0, props_a - lo_a),
        np.maximum(0.0, hi_a - props_a),
    ])


def _annotate_above(
    ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, labels: Sequence[str],
) -> None:
    for xi, y, lab in zip(xs, ys, labels):
        ax.annotate(
            lab, xy=(xi, y), xytext=(0, 5),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=9, color="#333333",
        )


def _plot_pair(
    df: pd.DataFrame, target: str, predictor: str,
    pred_kind: str, figs_dir: Path,
) -> None:
    safe = f"{target}__{predictor}"
    sub = df[[target, predictor]].dropna()
    if sub.empty:
        return

    if pred_kind in ("continuous", "count"):
        groups = sorted(sub[target].dropna().unique(), key=str)
        n_g = len(groups)
        fig, ax = plt.subplots(figsize=(_categorical_fig_width(n_g), 4))
        sns.boxplot(
            x=target, y=predictor, data=sub, order=groups, hue=target,
            ax=ax, palette="Set2", legend=False,
            width=0.55, linewidth=1.2, fliersize=3,
        )
        sns.stripplot(
            x=target, y=predictor, data=sub, order=groups, ax=ax,
            color="#333333", size=2.5, alpha=0.35, jitter=0.22,
        )
        labels, tops = [], []
        for i, g in enumerate(groups):
            vals = sub.loc[sub[target] == g, predictor].astype(float)
            n = len(vals)
            med = float(vals.median()) if n else np.nan
            tops.append(float(vals.quantile(0.75)) if n else np.nan)
            labels.append(f"n={n}\nmed={med:.3g}" if n else "n=0")
        _annotate_above(ax, np.arange(n_g), np.asarray(tops), labels)
        y_hi = float(sub[predictor].max())
        if np.isfinite(y_hi):
            pad = (y_hi - float(sub[predictor].min())) * 0.12 or abs(y_hi) * 0.08 or 0.5
            ax.set_ylim(top=y_hi + pad)
        _polish_ax(ax)
        ax.set_xlabel(target)
        ax.set_ylabel(predictor)
        ax.set_title(f"{predictor} by {target}")
    elif pred_kind == "ordinal":
        all_levels = (list(sub[predictor].cat.categories)
                      if isinstance(sub[predictor].dtype, pd.CategoricalDtype)
                      else sorted(sub[predictor].dropna().unique()))
        levels = [lv for lv in all_levels if (sub[predictor] == lv).any()]
        n_lv = len(levels)
        fig, ax = plt.subplots(figsize=(_categorical_fig_width(n_lv), 4))
        y_pos, _ = _encode_binary_target(sub[target], None)
        props, lo, hi, ns = [], [], [], []
        for lv in levels:
            mask = sub[predictor] == lv
            n = int(mask.sum())
            k = int(y_pos.loc[mask].sum())
            ns.append(n)
            p = k / n
            ci_lo, ci_hi = proportion_confint(k, n, alpha=0.05, method="wilson")
            props.append(p)
            lo.append(ci_lo)
            hi.append(ci_hi)
        x = np.arange(n_lv)
        props_a = np.clip(np.asarray(props, dtype=float), 0.0, 1.0)
        lo_a, hi_a = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        ax.errorbar(
            x, props_a, yerr=_errorbar_yerr(props_a, lo_a, hi_a),
            fmt="o-", color="#3b7ddd", markersize=8, capsize=3,
            linewidth=1.4, elinewidth=1.1,
            markeredgecolor="white", markeredgewidth=1.2, zorder=3,
        )
        pad = 0.35 if n_lv <= 4 else 0.5
        ax.set_xlim(-pad, n_lv - 1 + pad)
        ymax = min(1.0, float(np.nanmax(hi_a)) + 0.14)
        ax.set_ylim(0, max(0.35, ymax))
        _annotate_above(
            ax, x, hi_a,
            [f"{p:.0%}\n(n={n})" for p, n in zip(props_a, ns)],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(lv) for lv in levels],
            rotation=30 if n_lv > 4 else 0,
            ha="center" if n_lv <= 4 else "right",
        )
        _polish_ax(ax)
        ax.set_xlabel(predictor)
        ax.set_ylabel(f"P({target}=1)")
        ax.set_title(f"{predictor} → P({target})")
    elif pred_kind in ("nominal", "binary"):
        all_levels = (list(sub[predictor].cat.categories)
                      if isinstance(sub[predictor].dtype, pd.CategoricalDtype)
                      else sorted(sub[predictor].dropna().unique(), key=str))
        levels = [lv for lv in all_levels if (sub[predictor] == lv).any()]
        n_lv = len(levels)
        fig, ax = plt.subplots(figsize=(_categorical_fig_width(n_lv), 4))
        y_pos, _ = _encode_binary_target(sub[target], None)
        props, lo, hi, ns = [], [], [], []
        for lv in levels:
            mask = sub[predictor] == lv
            n = int(mask.sum())
            k = int(y_pos.loc[mask].sum())
            ns.append(n)
            p = k / n
            ci_lo, ci_hi = proportion_confint(k, n, alpha=0.05, method="wilson")
            props.append(p)
            lo.append(ci_lo)
            hi.append(ci_hi)
        x = np.arange(n_lv)
        props_a = np.clip(np.asarray(props, dtype=float), 0.0, 1.0)
        lo_a, hi_a = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        ax.errorbar(
            x, props_a, yerr=_errorbar_yerr(props_a, lo_a, hi_a),
            fmt="o", color="#3b7ddd", markersize=8, capsize=3,
            linewidth=1.4, elinewidth=1.1,
            markeredgecolor="white", markeredgewidth=1.2, zorder=3,
        )
        pad = 0.35 if n_lv <= 4 else 0.5
        ax.set_xlim(-pad, n_lv - 1 + pad)
        ymax = min(1.0, float(np.nanmax(hi_a)) + 0.14)
        ax.set_ylim(0, max(0.35, ymax))
        _annotate_above(
            ax, x, hi_a,
            [f"{p:.0%}\n(n={n})" for p, n in zip(props_a, ns)],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(lv) for lv in levels],
            rotation=30 if n_lv > 4 else 0,
            ha="center" if n_lv <= 4 else "right",
        )
        _polish_ax(ax)
        ax.set_xlabel(predictor)
        ax.set_ylabel(f"P({target}=1)")
        ax.set_title(f"{predictor} → P({target})")
    else:
        return

    fig.tight_layout()
    fig.savefig(figs_dir / f"{safe}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def screen_associations(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    *,
    targets: Sequence[str],
    predictors: Sequence[str] | None = None,
    positive_class: dict | None = None,
    fdr_alpha: float = 0.05,
    output_root: Path | str = "output",
) -> pd.DataFrame:
    """
    Run univariate association tests for each (target × predictor) pair.

    Parameters
    ----------
    df            : the analysis-ready dataframe (after cleaning).
    schema        : the ColSpec schema (drives test selection).
    targets       : binary target column names.
    predictors    : optional whitelist; if None, every kept non-target column
                    with a testable kind is used.
    positive_class: {target: value_that_is_positive} (default: True/1/last).
    fdr_alpha     : threshold for the fdr_significant flag.

    Returns
    -------
    long-format DataFrame with one row per (target, predictor).
    """
    output_root = Path(output_root)
    figs_dir, tabs_dir = _ensure_dirs(output_root)
    positive_class = positive_class or {}

    testable_kinds = {"continuous", "count", "ordinal", "nominal", "binary", "datetime"}
    if predictors is None or len(predictors) == 0:
        predictors = [c for c, sp in schema.items()
                      if c in df.columns and sp.keep and sp.kind in testable_kinds
                      and c not in targets]

    rows = []
    for target in targets:
        if target not in df.columns:
            continue
        y_enc, pos_used = _encode_binary_target(df[target], positive_class.get(target))

        for pred in predictors:
            if pred not in df.columns or pred == target:
                continue
            spec = schema[pred]
            pair = pd.concat([y_enc.rename("_y"), df[pred].rename(pred)], axis=1).dropna()
            n_used = len(pair)
            if n_used < 5:
                rows.append({"target": target, "predictor": pred, "kind": spec.kind,
                             "test": "skip", "stat": np.nan, "p": np.nan,
                             "effect": np.nan, "effect_label": "",
                             "n_used": n_used, "positive_class": pos_used})
                continue

            y_arr = pair["_y"].values

            if spec.kind in ("continuous", "count"):
                x = pair[pred].astype(float).values
                stat, p, eff, n = _mwu_with_effect(x[y_arr == 1], x[y_arr == 0])
                row = {"test": "mann_whitney_u", "stat": stat, "p": p,
                       "effect": eff, "effect_label": "rank_biserial_r"}

            elif spec.kind == "ordinal":
                # Spearman on numeric codes
                codes = pd.Categorical(pair[pred],
                                       categories=spec.ordered_levels
                                       if spec.ordered_levels else None,
                                       ordered=True).codes.astype(float)
                # Guard against constant input (zero variance) — happens when a
                # predictor/target has only one observed level after NA drop.
                # Spearman is undefined; scipy raises ConstantInputWarning and
                # returns NaN. We skip and record n/a explicitly.
                if np.nanstd(codes) == 0 or np.nanstd(y_arr) == 0:
                    row = {"test": "spearman", "stat": np.nan, "p": np.nan,
                           "effect": np.nan, "effect_label": "spearman_rho"}
                else:
                    rho, p = spearmanr(codes, y_arr)
                    row = {"test": "spearman", "stat": float(rho), "p": float(p),
                           "effect": float(rho), "effect_label": "spearman_rho"}

            elif spec.kind == "datetime":
                t = pd.to_datetime(pair[pred], errors="coerce")
                days = (t - t.min()).dt.days.astype(float).values
                stat, p, eff, n = _mwu_with_effect(days[y_arr == 1], days[y_arr == 0])
                row = {"test": "mann_whitney_u_days", "stat": stat, "p": p,
                       "effect": eff, "effect_label": "rank_biserial_r"}

            elif spec.kind in ("nominal", "binary"):
                ct = pd.crosstab(pair[pred], pair["_y"])
                table = ct.values
                if table.shape == (2, 2):
                    # Use Fisher when small expected counts; report OR
                    exp = chi2_contingency(table, correction=False)[3]
                    if (exp < 5).any():
                        odds, p = fisher_exact(table, alternative="two-sided")
                        v = _cramers_v(table)
                        row = {"test": "fisher_exact", "stat": float(odds), "p": float(p),
                               "effect": v, "effect_label": "cramers_v"}
                    else:
                        chi2, p, _, _ = chi2_contingency(table, correction=False)
                        v = _cramers_v(table)
                        row = {"test": "chi2", "stat": float(chi2), "p": float(p),
                               "effect": v, "effect_label": "cramers_v"}
                else:
                    chi2, p, _, _ = chi2_contingency(table, correction=False)
                    v = _cramers_v(table)
                    row = {"test": "chi2", "stat": float(chi2), "p": float(p),
                           "effect": v, "effect_label": "cramers_v"}
            else:
                continue

            row.update({"target": target, "predictor": pred, "kind": spec.kind,
                        "n_used": n_used, "positive_class": pos_used})
            rows.append(row)

            try:
                _plot_pair(df, target, pred, spec.kind, figs_dir)
            except Exception as exc:
                warnings.warn(
                    f"EDA plot skipped for {target} × {pred}: {exc}",
                    stacklevel=2,
                )

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=[
            "target", "predictor", "kind", "test", "stat", "p", "p_fdr",
            "fdr_significant", "effect", "effect_label", "n_used", "positive_class",
        ])
        _format_table_for_csv(out).to_csv(tabs_dir / "associations.csv", index=False)
        return out

    # FDR per target
    out["p_fdr"] = np.nan
    for t in out["target"].unique():
        mask = out["target"] == t
        out.loc[mask, "p_fdr"] = benjamini_hochberg(out.loc[mask, "p"]).values
    out["fdr_significant"] = out["p_fdr"] < fdr_alpha

    cols = ["target", "predictor", "kind", "test", "stat", "p", "p_fdr",
            "fdr_significant", "effect", "effect_label",
            "n_used", "positive_class"]
    out["_eff_abs"] = out["effect"].abs()
    out = (out[cols + ["_eff_abs"]]
           .sort_values(["target", "p_fdr", "_eff_abs"],
                        ascending=[True, True, False])
           .drop(columns="_eff_abs")
           .reset_index(drop=True))
    # display-only rounding: integers stay int, fractions -> 3 sig figs (raw df returned)
    _format_table_for_csv(out).to_csv(tabs_dir / "associations.csv", index=False)
    return out
