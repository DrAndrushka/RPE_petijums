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
                             p_fdr, effect, effect_size, n_used, direction)
- figures/<target>__<predictor>.svg : the appropriate seaborn plot
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import (
    mannwhitneyu, spearmanr, chi2_contingency, fisher_exact, norm,
)

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
    # direction: median diff sign
    direction = float(np.median(x_group1) - np.median(x_group0))
    return U, p, r, n1 + n0, direction


# ---------------------------------------------------------------------------
# Per-pair plotting
# ---------------------------------------------------------------------------

def _plot_pair(
    df: pd.DataFrame, target: str, predictor: str,
    pred_kind: str, figs_dir: Path,
) -> None:
    safe = f"{target}__{predictor}"
    sub = df[[target, predictor]].dropna()
    if sub.empty:
        return

    if pred_kind in ("continuous", "count"):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=target, y=predictor, data=sub, ax=ax,
                    hue=target, palette="Set2", legend=False)
        sns.stripplot(x=target, y=predictor, data=sub, ax=ax,
                      color="black", size=2, alpha=0.4)
        ax.set_title(f"{predictor} by {target}")
    elif pred_kind == "ordinal":
        fig, ax = plt.subplots(figsize=(6, 4))
        order = (list(sub[predictor].cat.categories)
                 if isinstance(sub[predictor].dtype, pd.CategoricalDtype)
                 else sorted(sub[predictor].dropna().unique()))
        ct = pd.crosstab(sub[predictor], sub[target], normalize="index")
        ct = ct.reindex(order)
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
        ax.set_title(f"{predictor} → P({target})")
        ax.set_ylabel("proportion")
    elif pred_kind in ("nominal", "binary"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ct = pd.crosstab(sub[predictor], sub[target])
        ct.plot(kind="bar", ax=ax, colormap="Set2")
        ax.set_title(f"{predictor} × {target}")
        ax.set_ylabel("count")
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
    if predictors is None:
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
                             "direction": np.nan, "n_used": n_used,
                             "positive_class": pos_used})
                continue

            y_arr = pair["_y"].values

            if spec.kind in ("continuous", "count"):
                x = pair[pred].astype(float).values
                stat, p, eff, n, direction = _mwu_with_effect(x[y_arr == 1], x[y_arr == 0])
                row = {"test": "mann_whitney_u", "stat": stat, "p": p,
                       "effect": eff, "effect_label": "rank_biserial_r",
                       "direction": direction}

            elif spec.kind == "ordinal":
                # Spearman on numeric codes
                codes = pd.Categorical(pair[pred],
                                       categories=spec.ordered_levels
                                       if spec.ordered_levels else None,
                                       ordered=True).codes.astype(float)
                rho, p = spearmanr(codes, y_arr)
                row = {"test": "spearman", "stat": float(rho), "p": float(p),
                       "effect": float(rho), "effect_label": "spearman_rho",
                       "direction": float(np.sign(rho))}

            elif spec.kind == "datetime":
                t = pd.to_datetime(pair[pred], errors="coerce")
                days = (t - t.min()).dt.days.astype(float).values
                stat, p, eff, n, direction = _mwu_with_effect(days[y_arr == 1], days[y_arr == 0])
                row = {"test": "mann_whitney_u_days", "stat": stat, "p": p,
                       "effect": eff, "effect_label": "rank_biserial_r",
                       "direction": direction}

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
                               "effect": v, "effect_label": "cramers_v",
                               "direction": float(np.sign(np.log(odds)) if odds > 0 else 0)}
                    else:
                        chi2, p, _, _ = chi2_contingency(table, correction=False)
                        v = _cramers_v(table)
                        row = {"test": "chi2", "stat": float(chi2), "p": float(p),
                               "effect": v, "effect_label": "cramers_v",
                               "direction": np.nan}
                else:
                    chi2, p, _, _ = chi2_contingency(table, correction=False)
                    v = _cramers_v(table)
                    row = {"test": "chi2", "stat": float(chi2), "p": float(p),
                           "effect": v, "effect_label": "cramers_v",
                           "direction": np.nan}
            else:
                continue

            row.update({"target": target, "predictor": pred, "kind": spec.kind,
                        "n_used": n_used, "positive_class": pos_used})
            rows.append(row)

            _plot_pair(df, target, pred, spec.kind, figs_dir)

    out = pd.DataFrame(rows)
    if out.empty:
        _format_table_for_csv(out).to_csv(tabs_dir / "associations.csv", index=False)
        return out

    # FDR per target
    out["p_fdr"] = np.nan
    for t in out["target"].unique():
        mask = out["target"] == t
        out.loc[mask, "p_fdr"] = benjamini_hochberg(out.loc[mask, "p"]).values
    out["fdr_significant"] = out["p_fdr"] < fdr_alpha

    cols = ["target", "predictor", "kind", "test", "stat", "p", "p_fdr",
            "fdr_significant", "effect", "effect_label", "direction",
            "n_used", "positive_class"]
    out = out[cols].sort_values(["target", "p_fdr"]).reset_index(drop=True)
    # display-only rounding: integers stay int, fractions -> 3 sig figs (raw df returned)
    _format_table_for_csv(out).to_csv(tabs_dir / "associations.csv", index=False)
    return out
