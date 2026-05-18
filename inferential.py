"""
inferential.py
===============
Multivariable logistic regression with MICE pooling (Rubin's rules) + VIF check.

Pipeline per target
-------------------
1. Build design matrix X (one-hot for nominal, ordinal codes kept as numeric,
   continuous standardized to z-scores so coefficients are comparable).
2. Drop predictors with VIF > vif_threshold (default 5), iteratively.
3. Fit a logistic regression on each of the m imputed datasets
   (statsmodels Logit). For each predictor record (coef, se).
4. Pool across imputations with Rubin's rules:
       theta_bar  = mean(theta_i)
       within_var = mean(SE_i^2)
       between_var= var(theta_i, ddof=1)
       total_var  = within_var + (1 + 1/m) * between_var
       SE_pool    = sqrt(total_var)
       df         = (m - 1) * (1 + within_var / ((1+1/m)*between_var))^2   (Barnard–Rubin)
       p          = 2 * (1 - t.cdf(|theta_bar/SE_pool|, df))
       95% CI     = theta_bar ± t.ppf(0.975, df) * SE_pool
5. Report adjusted odds ratios = exp(theta_bar), 95% CI on OR scale.

Outputs (per target) under output/inferential/
----------------------------------------------
- tables/<target>__multivariable.csv : predictor, OR, 95% CI, p, n_models
- tables/<target>__vif.csv           : VIFs after pruning
- figures/<target>__forest.svg       : forest plot of adjusted ORs

Also writes tables/inferential_summary.csv combining all targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import t as t_dist
from statsmodels.stats.outliers_influence import variance_inflation_factor

from schema_infer import ColSpec
from cleaning import format_table_for_csv as _format_table_for_csv  # CSV display-only rounding


def _ensure_dirs(root: Path) -> tuple[Path, Path]:
    figs = root / "inferential" / "figures"
    tabs = root / "inferential" / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    return figs, tabs


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

def _build_design(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    predictors: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Build X with:
      - continuous/count: z-scored numeric
      - ordinal:          numeric ordinal codes (kept as-is, NOT one-hot)
      - nominal:          one-hot, drop_first=True
      - binary:           0/1
    Returns (X, mapping {original_predictor: [columns_in_X]}).
    """
    pieces = []
    mapping: dict[str, list[str]] = {}
    for p in predictors:
        spec = schema[p]
        s = df[p]
        if spec.kind in ("continuous", "count"):
            mu, sd = s.mean(), s.std(ddof=0)
            z = (s - mu) / (sd if sd not in (0, np.nan) else 1.0)
            z.name = p
            pieces.append(z)
            mapping[p] = [p]
        elif spec.kind == "ordinal":
            cats = pd.Categorical(s,
                                  categories=spec.ordered_levels if spec.ordered_levels else None,
                                  ordered=True)
            num = pd.Series(cats.codes.astype(float), index=s.index, name=p).replace(-1, np.nan)
            pieces.append(num)
            mapping[p] = [p]
        elif spec.kind == "binary":
            num = s.astype("float")
            num.name = p
            pieces.append(num)
            mapping[p] = [p]
        elif spec.kind == "nominal":
            dummies = pd.get_dummies(s, prefix=p, drop_first=True, dtype=float)
            pieces.append(dummies)
            mapping[p] = list(dummies.columns)
        # other kinds skipped
    X = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=df.index)
    return X, mapping


def _prune_by_vif(X: pd.DataFrame, threshold: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Iteratively drop the column with the highest VIF until all <= threshold."""
    Xc = sm.add_constant(X, has_constant="add")
    cols = list(X.columns)
    log = []
    while cols:
        Xc = sm.add_constant(X[cols], has_constant="add")
        vifs = []
        for i, c in enumerate(Xc.columns):
            if c == "const":
                continue
            try:
                v = variance_inflation_factor(Xc.values, i)
            except Exception:
                v = np.nan
            vifs.append((c, v))
        vif_df = pd.DataFrame(vifs, columns=["predictor", "vif"])
        worst = vif_df.loc[vif_df["vif"].idxmax()] if vif_df["vif"].notna().any() else None
        if worst is None or worst["vif"] <= threshold or pd.isna(worst["vif"]):
            log.append(vif_df.assign(action="kept"))
            break
        log.append(vif_df.assign(action=lambda d: np.where(d["predictor"] == worst["predictor"], "dropped", "kept")))
        cols.remove(worst["predictor"])
    final_vif = vif_df.copy() if cols else pd.DataFrame(columns=["predictor", "vif"])
    return X[cols], final_vif


# ---------------------------------------------------------------------------
# Rubin pooling
# ---------------------------------------------------------------------------

def _rubin_pool(thetas: np.ndarray, ses: np.ndarray) -> dict[str, float]:
    """
    Rubin's rules pooling for a single coefficient across m imputations.
    thetas, ses: length-m arrays.
    """
    m = len(thetas)
    theta_bar = float(np.mean(thetas))
    within = float(np.mean(ses ** 2))
    between = float(np.var(thetas, ddof=1)) if m > 1 else 0.0
    total = within + (1 + 1 / m) * between
    se = float(np.sqrt(total))
    # Barnard–Rubin degrees of freedom
    if between > 0 and m > 1:
        rel_incr = (1 + 1 / m) * between / within if within > 0 else np.inf
        df = (m - 1) * (1 + 1 / rel_incr) ** 2 if np.isfinite(rel_incr) else np.inf
    else:
        df = np.inf
    if df == np.inf:
        from scipy.stats import norm
        z = theta_bar / se if se > 0 else np.nan
        p = float(2 * (1 - norm.cdf(abs(z)))) if se > 0 else np.nan
        crit = norm.ppf(0.975)
    else:
        tstat = theta_bar / se if se > 0 else np.nan
        p = float(2 * (1 - t_dist.cdf(abs(tstat), df))) if se > 0 else np.nan
        crit = t_dist.ppf(0.975, df)
    ci_lo = theta_bar - crit * se
    ci_hi = theta_bar + crit * se
    return {"coef": theta_bar, "se": se, "df": df, "p": p,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "or": float(np.exp(theta_bar)),
            "or_ci_lo": float(np.exp(ci_lo)),
            "or_ci_hi": float(np.exp(ci_hi))}


# ---------------------------------------------------------------------------
# Per-target multivariable model with MICE pooling
# ---------------------------------------------------------------------------

def _encode_target(y: pd.Series, positive_class) -> tuple[pd.Series, object]:
    if positive_class is None:
        nn = y.dropna().unique()
        if len(nn) != 2:
            raise ValueError(f"Target '{y.name}' not binary; values={nn}")
        positive_class = True if True in nn else 1 if 1 in nn else sorted(nn, key=str)[-1]
    return (y == positive_class).astype(float), positive_class


def fit_multivariable_logistic(
    imputed_frames: list[pd.DataFrame],
    schema: dict[str, ColSpec],
    target: str,
    predictors: Sequence[str],
    *,
    positive_class=None,
    vif_threshold: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run logistic regression on each imputed frame, then Rubin-pool.

    Returns
    -------
    pooled_df : per predictor (column-in-X), OR with 95% CI and p.
    vif_df    : VIF table from the first imputation after pruning.
    """
    # 1. Build X for each imputation, using the SAME columns (use first imp to set design).
    X0, mapping = _build_design(imputed_frames[0], schema, predictors)
    X0_pruned, vif_df = _prune_by_vif(X0, threshold=vif_threshold)
    keep_cols = list(X0_pruned.columns)

    coefs_by_col: dict[str, list[float]] = {c: [] for c in keep_cols}
    ses_by_col: dict[str, list[float]] = {c: [] for c in keep_cols}

    for frame in imputed_frames:
        X, _ = _build_design(frame, schema, predictors)
        X = X[keep_cols].copy()
        y_enc, _ = _encode_target(frame[target], positive_class)
        sub = pd.concat([y_enc.rename("_y"), X], axis=1).dropna()
        if len(sub) < max(20, X.shape[1] + 5):
            continue
        Xc = sm.add_constant(sub[keep_cols], has_constant="add")
        try:
            model = sm.Logit(sub["_y"], Xc).fit(disp=False, method="newton", maxiter=200)
        except Exception:
            try:
                model = sm.Logit(sub["_y"], Xc).fit_regularized(disp=False, alpha=1e-3)
            except Exception:
                continue
        for c in keep_cols:
            if c in model.params.index:
                coefs_by_col[c].append(float(model.params[c]))
                ses_by_col[c].append(float(model.bse[c]))

    rows = []
    for c in keep_cols:
        thetas = np.array(coefs_by_col[c])
        ses = np.array(ses_by_col[c])
        if len(thetas) == 0:
            rows.append({"predictor_col": c, "coef": np.nan, "se": np.nan,
                         "or": np.nan, "or_ci_lo": np.nan, "or_ci_hi": np.nan,
                         "p": np.nan, "n_models": 0})
            continue
        pooled = _rubin_pool(thetas, ses)
        rows.append({"predictor_col": c, **pooled, "n_models": len(thetas)})
    pooled_df = pd.DataFrame(rows)
    pooled_df["target"] = target
    return pooled_df, vif_df


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------

def _forest_plot(pooled: pd.DataFrame, target: str, figs_dir: Path) -> None:
    plot_df = pooled.dropna(subset=["or"]).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("or")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(plot_df))))
    y = np.arange(len(plot_df))
    ax.errorbar(plot_df["or"], y,
                xerr=[plot_df["or"] - plot_df["or_ci_lo"],
                      plot_df["or_ci_hi"] - plot_df["or"]],
                fmt="o", color="#264653", ecolor="#2a9d8f", capsize=3)
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(plot_df["predictor_col"])
    ax.set_xscale("log")
    ax.set_xlabel("Adjusted Odds Ratio (95% CI, log scale)")
    ax.set_title(f"Multivariable logistic — {target}")
    fig.tight_layout()
    fig.savefig(figs_dir / f"{target}__forest.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inferential(
    imputed_frames: list[pd.DataFrame],
    schema: dict[str, ColSpec],
    *,
    targets: Sequence[str],
    predictors: Sequence[str],
    positive_class: dict | None = None,
    vif_threshold: float = 5.0,
    output_root: Path | str = "output",
) -> pd.DataFrame:
    """
    Run multivariable logistic regression per target with Rubin pooling over the
    MICE imputations. Returns combined long-format results and writes tables +
    forest plot SVGs.
    """
    output_root = Path(output_root)
    figs_dir, tabs_dir = _ensure_dirs(output_root)
    positive_class = positive_class or {}

    all_rows = []
    for target in targets:
        pooled_df, vif_df = fit_multivariable_logistic(
            imputed_frames, schema, target, predictors,
            positive_class=positive_class.get(target),
            vif_threshold=vif_threshold,
        )
        # display-only rounding on save; raw df kept for downstream concat/plots
        _format_table_for_csv(pooled_df).to_csv(tabs_dir / f"{target}__multivariable.csv", index=False)
        _format_table_for_csv(vif_df).to_csv(tabs_dir / f"{target}__vif.csv", index=False)
        _forest_plot(pooled_df, target, figs_dir)
        all_rows.append(pooled_df)

    if not all_rows:
        return pd.DataFrame()
    combined = pd.concat(all_rows, ignore_index=True)
    cols = ["target", "predictor_col", "or", "or_ci_lo", "or_ci_hi",
            "coef", "se", "p", "df", "n_models"]
    combined = combined[[c for c in cols if c in combined.columns]]
    _format_table_for_csv(combined).to_csv(tabs_dir / "inferential_summary.csv", index=False)
    return combined
