"""
eda.py
=======
Univariate target × predictor association screening.

Target kinds (from schema, or inferred from data)
-------------------------------------------------
- **binary**     : 2 levels → encode 0/1, then screening below
- **continuous** : numeric outcome
- **ordinal**    : ordered categories (≥2 levels)
- **nominal**    : unordered categories (≥2 levels)

Tests (target kind × predictor kind)
------------------------------------
| target \\ predictor | continuous / count | ordinal | nominal / binary | datetime |
|--------------------|--------------------|---------|------------------|----------|
| binary             | Mann–Whitney U     | Spearman ρ | χ² / Fisher   | MWU days |
| continuous         | Spearman ρ         | Spearman ρ | Kruskal–Wallis | Spearman |
| ordinal            | Spearman ρ         | Spearman ρ | χ²            | Spearman |
| nominal            | Kruskal–Wallis    | χ²      | χ²            | Kruskal  |

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
    mannwhitneyu, spearmanr, chi2_contingency, fisher_exact, kruskal,
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
    """Mann–Whitney U (two-sided) with signed rank-biserial r = 1 - 2U/(n1·n0)."""
    n1, n0 = len(x_group1), len(x_group0)
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan, np.nan, n1 + n0
    res = mannwhitneyu(x_group1, x_group0, alternative="two-sided")
    U = float(res.statistic)
    p = float(res.pvalue)
    denom = n1 * n0
    r = float(1.0 - (2.0 * U) / denom) if denom > 0 else np.nan
    return U, p, r, n1 + n0


def _infer_target_kind(y: pd.Series, spec: ColSpec | None) -> str:
    """Return binary | continuous | ordinal | nominal | skip."""
    nn = y.dropna()
    if len(nn) == 0 or nn.nunique() < 2:
        return "skip"
    kind = spec.kind if spec is not None else None
    n_u = int(nn.nunique())
    if kind == "binary" or (kind is None and n_u == 2):
        return "binary"
    if kind in ("continuous", "count"):
        return "continuous"
    if kind == "ordinal":
        return "binary" if n_u == 2 else "ordinal"
    if kind == "nominal":
        return "binary" if n_u == 2 else "nominal"
    if pd.api.types.is_numeric_dtype(nn):
        return "continuous" if n_u > 15 else ("binary" if n_u == 2 else "ordinal")
    return "binary" if n_u == 2 else "nominal"


def _prepare_target(
    y_raw: pd.Series,
    target_mode: str,
    spec: ColSpec | None,
    positive_class,
) -> tuple[pd.Series, object | None]:
    if target_mode == "binary":
        return _encode_binary_target(y_raw, positive_class)
    if target_mode == "continuous":
        return pd.to_numeric(y_raw, errors="coerce"), None
    if target_mode == "ordinal":
        if spec and spec.ordered_levels:
            cat = pd.Categorical(y_raw, categories=spec.ordered_levels, ordered=True)
        else:
            cat = pd.Categorical(y_raw, ordered=True)
        codes = pd.Series(cat.codes.astype(float), index=y_raw.index).replace(-1, np.nan)
        return codes, None
    # nominal — string labels for tables / plots
    return y_raw.astype(str), None


def _predictor_values(pair: pd.DataFrame, pred: str, pred_spec: ColSpec) -> tuple[pd.Series, str]:
    """Numeric x for correlation tests; kind label for branching."""
    kind = pred_spec.kind
    if kind in ("continuous", "count"):
        return pair[pred].astype(float), kind
    if kind == "ordinal":
        cats = pd.Categorical(
            pair[pred],
            categories=pred_spec.ordered_levels if pred_spec.ordered_levels else None,
            ordered=True,
        )
        return pd.Series(cats.codes.astype(float), index=pair.index).replace(-1, np.nan), kind
    if kind == "datetime":
        t = pd.to_datetime(pair[pred], errors="coerce")
        days = (t - t.min()).dt.days.astype(float)
        return days, "datetime"
    return pair[pred], kind


def _kruskal_with_effect(groups: list[np.ndarray]) -> tuple[float, float, float]:
    valid = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    if len(valid) < 2:
        return np.nan, np.nan, np.nan
    H, p = kruskal(*valid)
    n = sum(len(g) for g in valid)
    eps2 = float((H - len(valid) + 1) / (n - 1)) if n > 1 else np.nan
    return float(H), float(p), eps2


def _chi2_row(table: np.ndarray) -> dict:
    if table.size == 0 or table.sum() == 0:
        return {"test": "chi2", "stat": np.nan, "p": np.nan,
                "effect": np.nan, "effect_label": "cramers_v"}
    if table.shape == (2, 2):
        exp = chi2_contingency(table, correction=False)[3]
        if (exp < 5).any():
            odds, p = fisher_exact(table, alternative="two-sided")
            return {"test": "fisher_exact", "stat": float(odds), "p": float(p),
                    "effect": _cramers_v(table), "effect_label": "cramers_v"}
    chi2, p, _, _ = chi2_contingency(table, correction=False)
    return {"test": "chi2", "stat": float(chi2), "p": float(p),
            "effect": _cramers_v(table), "effect_label": "cramers_v"}


def _spearman_row(y: np.ndarray, x: np.ndarray) -> dict:
    if np.nanstd(y) == 0 or np.nanstd(x) == 0:
        return {"test": "spearman", "stat": np.nan, "p": np.nan,
                "effect": np.nan, "effect_label": "spearman_rho"}
    rho, p = spearmanr(y, x, nan_policy="omit")
    return {"test": "spearman", "stat": float(rho), "p": float(p),
            "effect": float(rho), "effect_label": "spearman_rho"}


def _association_test(
    target_mode: str,
    pair: pd.DataFrame,
    target: str,
    pred: str,
    pred_spec: ColSpec,
) -> dict:
    """One screening test for a clean (target, predictor) pair."""
    y_col = "_y"
    pred_kind = pred_spec.kind

    if target_mode == "binary":
        y_arr = pair[y_col].values.astype(float)
        if pred_kind in ("continuous", "count"):
            x = pair[pred].astype(float).values
            stat, p, eff, _ = _mwu_with_effect(x[y_arr == 1], x[y_arr == 0])
            return {"test": "mann_whitney_u", "stat": stat, "p": p,
                    "effect": eff, "effect_label": "rank_biserial_r"}
        if pred_kind == "ordinal":
            x, _ = _predictor_values(pair, pred, pred_spec)
            return _spearman_row(y_arr, x.values)
        if pred_kind == "datetime":
            x, _ = _predictor_values(pair, pred, pred_spec)
            stat, p, eff, _ = _mwu_with_effect(
                x.values[y_arr == 1], x.values[y_arr == 0])
            return {"test": "mann_whitney_u_days", "stat": stat, "p": p,
                    "effect": eff, "effect_label": "rank_biserial_r"}
        if pred_kind in ("nominal", "binary"):
            ct = pd.crosstab(pair[pred], pair[y_col])
            return _chi2_row(ct.values)

    y_num = pair[y_col]
    x_num, pk = _predictor_values(pair, pred, pred_spec)

    if target_mode == "continuous":
        yv = y_num.values.astype(float)
        if pk in ("continuous", "count", "ordinal", "datetime"):
            return _spearman_row(yv, x_num.values.astype(float))
        # nominal / binary predictor
        # group order does not affect Kruskal–Wallis p-value
        groups = [yv[pair[pred] == lv].astype(float) for lv in pair[pred].unique()]
        H, p, eps2 = _kruskal_with_effect(groups)
        return {"test": "kruskal_wallis", "stat": H, "p": p,
                "effect": eps2, "effect_label": "epsilon_sq"}

    if target_mode == "ordinal":
        yv = y_num.values.astype(float)
        if pk in ("continuous", "count", "ordinal", "datetime"):
            return _spearman_row(yv, x_num.values.astype(float))
        ct = pd.crosstab(pair[pred], pair["target"])
        return _chi2_row(ct.values)

    # nominal target
    tcol = "target"
    if pk in ("continuous", "count", "datetime"):
        if pk == "datetime":
            x_days = x_num.values.astype(float)
            groups = [
                x_days[pair[tcol].values == lv]
                for lv in sorted(pair[tcol].unique(), key=str)
            ]
        else:
            groups = [
                pair.loc[pair[tcol] == lv, pred].astype(float).values
                for lv in sorted(pair[tcol].unique(), key=str)
            ]
        H, p, eps2 = _kruskal_with_effect(groups)
        return {"test": "kruskal_wallis", "stat": H, "p": p,
                "effect": eps2, "effect_label": "epsilon_sq"}
    ct = pd.crosstab(pair[pred], pair[tcol])
    return _chi2_row(ct.values)


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


def _level_order(s: pd.Series, spec: ColSpec | None) -> list:
    """Observed levels in schema / categorical order (not alphabetical)."""
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    obs = set(s.dropna())
    if not obs:
        return []
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [c for c in s.cat.categories if c in obs]
    if spec and spec.ordered_levels:
        ordered = [lv for lv in spec.ordered_levels if lv in obs]
        extras = sorted((x for x in obs if x not in spec.ordered_levels), key=str)
        return ordered + extras
    return sorted(obs, key=str)


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


def _plot_binary_target_rates(
    ax: plt.Axes,
    sub: pd.DataFrame,
    target: str,
    predictor: str,
    positive_class,
    *,
    pred_levels: Sequence,
) -> None:
    """P(target = positive_class) by predictor level (binary target only)."""
    levels = list(pred_levels)
    n_lv = len(levels)
    y_pos, pos_used = _encode_binary_target(sub[target], positive_class)
    pos_label = str(pos_used) if pos_used is not None else str(positive_class)
    props, lo, hi, ns = [], [], [], []
    for lv in levels:
        mask = sub[predictor] == lv
        n = int(mask.sum())
        k = int(y_pos.loc[mask].sum())
        ns.append(n)
        ci_lo, ci_hi = proportion_confint(k, n, alpha=0.05, method="wilson")
        props.append(k / n if n else 0.0)
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
    _annotate_above(ax, x, hi_a, [f"{p:.0%}\n(n={n})" for p, n in zip(props_a, ns)])
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(lv) for lv in levels],
        rotation=30 if n_lv > 4 else 0,
        ha="center" if n_lv <= 4 else "right",
    )
    ax.set_xlabel(predictor)
    ax.set_ylabel(f"P({target}={pos_label})")
    ax.set_title(f"{predictor} → P({target})")


def _plot_pair(
    df: pd.DataFrame,
    target: str,
    predictor: str,
    *,
    target_mode: str,
    pred_kind: str,
    target_spec: ColSpec | None,
    pred_spec: ColSpec | None,
    positive_class,
    figs_dir: Path,
) -> None:
    safe = f"{target}__{predictor}"
    sub = df[[target, predictor]].dropna()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    # Continuous / count predictor → distribution of predictor by target groups
    if pred_kind in ("continuous", "count", "datetime"):
        if pred_kind == "datetime":
            t = pd.to_datetime(sub[predictor], errors="coerce")
            sub = sub.assign(_x=(t - t.min()).dt.days.astype(float))
            xcol = "_x"
            xlabel = f"{predictor} (days since min)"
        else:
            sub = sub.assign(_x=sub[predictor].astype(float))
            xcol = "_x"
            xlabel = predictor
        groups = _level_order(sub[target], target_spec)
        n_g = len(groups)
        fig.set_size_inches(_categorical_fig_width(n_g), 4)
        sns.boxplot(
            x=target, y=xcol, data=sub, order=groups, hue=target,
            ax=ax, palette="Set2", legend=False,
            width=0.55, linewidth=1.2, fliersize=3,
        )
        sns.stripplot(
            x=target, y=xcol, data=sub, order=groups, ax=ax,
            color="#333333", size=2.5, alpha=0.35, jitter=0.22,
        )
        _polish_ax(ax)
        ax.set_xlabel(target)
        ax.set_ylabel(xlabel)
        ax.set_title(f"{xlabel} by {target}")

    elif target_mode == "binary" and pred_kind in ("ordinal", "nominal", "binary"):
        fig.set_size_inches(_categorical_fig_width(
            sub[predictor].nunique()), 4)
        _plot_binary_target_rates(
            ax, sub, target, predictor, positive_class,
            pred_levels=_level_order(sub[predictor], pred_spec),
        )

    elif target_mode == "continuous" and pred_kind in ("ordinal", "nominal", "binary"):
        groups = _level_order(sub[predictor], pred_spec)
        fig.set_size_inches(_categorical_fig_width(len(groups)), 4)
        sns.boxplot(
            x=predictor, y=target, data=sub, order=groups, hue=predictor,
            ax=ax, palette="Set2", legend=False,
            width=0.55, linewidth=1.2, fliersize=3,
        )
        _polish_ax(ax)
        ax.set_xlabel(predictor)
        ax.set_ylabel(target)
        ax.set_title(f"{target} by {predictor}")

    elif target_mode in ("ordinal", "nominal") and pred_kind in (
        "ordinal", "nominal", "binary",
    ):
        pred_order = _level_order(sub[predictor], pred_spec)
        target_order = _level_order(sub[target], target_spec)
        ct = pd.crosstab(sub[predictor], sub[target], normalize="index")
        ct = ct.reindex(index=pred_order, columns=target_order, fill_value=0.0)
        fig.set_size_inches(max(5, 0.9 * ct.shape[1] + 2), max(3.5, 0.5 * ct.shape[0] + 2))
        sns.heatmap(ct, annot=True, fmt=".0%", cmap="Blues", ax=ax, vmin=0, vmax=1)
        ax.set_xlabel(target)
        ax.set_ylabel(predictor)
        ax.set_title(f"{target} share within {predictor}")

    elif target_mode == "continuous" and pred_kind in ("continuous", "count"):
        sub = sub.assign(_x=sub[predictor].astype(float), _y=sub[target].astype(float))
        sns.regplot(data=sub, x="_x", y="_y", ax=ax,
                    scatter_kws={"alpha": 0.25, "s": 12}, line_kws={"color": "#e76f51"})
        _polish_ax(ax)
        ax.set_xlabel(predictor)
        ax.set_ylabel(target)
        ax.set_title(f"{target} vs {predictor}")

    elif target_mode in ("ordinal", "nominal") and pred_kind in ("continuous", "count"):
        groups = _level_order(sub[target], target_spec)
        fig.set_size_inches(_categorical_fig_width(len(groups)), 4)
        sns.boxplot(
            x=target, y=predictor, data=sub, order=groups, hue=target,
            ax=ax, palette="Set2", legend=False,
        )
        _polish_ax(ax)
        ax.set_xlabel(target)
        ax.set_ylabel(predictor)
        ax.set_title(f"{predictor} by {target}")

    else:
        plt.close(fig)
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
    targets       : outcome column names (binary, continuous, ordinal, or nominal).
    predictors    : optional whitelist; if None, every kept non-target column
                    with a testable kind is used.
    positive_class: for **binary** targets only — {target: value_that_is_positive}.
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
        target_spec = schema.get(target)
        target_mode = _infer_target_kind(df[target], target_spec)
        if target_mode == "skip":
            warnings.warn(f"EDA skipped target '{target}': fewer than 2 non-missing levels.",
                          stacklevel=2)
            continue
        pos_cfg = positive_class.get(target)
        if target_mode != "binary" and pos_cfg is not None:
            warnings.warn(
                f"positive_class for '{target}' ignored (target kind={target_mode}).",
                stacklevel=2,
            )
        y_enc, pos_used = _prepare_target(
            df[target], target_mode, target_spec, pos_cfg)

        for pred in predictors:
            if pred not in df.columns or pred == target:
                continue
            pred_spec = schema.get(pred)
            if pred_spec is None:
                continue
            spec = pred_spec
            pair = pd.concat([
                df[target].rename("target"),
                y_enc.rename("_y"),
                df[pred].rename(pred),
            ], axis=1).dropna()
            n_used = len(pair)
            if n_used < 5:
                rows.append({
                    "target": target, "target_kind": target_mode,
                    "predictor": pred, "kind": spec.kind,
                    "test": "skip", "stat": np.nan, "p": np.nan,
                    "effect": np.nan, "effect_label": "",
                    "n_used": n_used, "positive_class": pos_used,
                })
                continue

            row = _association_test(target_mode, pair, target, pred, spec)
            row.update({
                "target": target, "target_kind": target_mode,
                "predictor": pred, "kind": spec.kind,
                "n_used": n_used, "positive_class": pos_used,
            })
            rows.append(row)

            try:
                _plot_pair(
                    df, target, pred,
                    target_mode=target_mode,
                    pred_kind=spec.kind,
                    target_spec=target_spec,
                    pred_spec=pred_spec,
                    positive_class=pos_used,
                    figs_dir=figs_dir,
                )
            except Exception as exc:
                warnings.warn(
                    f"EDA plot skipped for {target} × {pred}: {exc}",
                    stacklevel=2,
                )

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=[
            "target", "target_kind", "predictor", "kind", "test", "stat", "p", "p_fdr",
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

    cols = ["target", "target_kind", "predictor", "kind", "test", "stat", "p", "p_fdr",
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
