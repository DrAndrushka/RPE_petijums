"""
the_report.py — Ylivertainen HTML Report Generator
====================================================
Converts all pipeline artifacts (cohort → DDA → EDA → predictive)
into a self-contained, dark-themed clinical HTML report.

Location: ylivertainen/the_report.py

Usage (in notebook):
    from ylivertainen.the_report import YlivertainenTheReport
    report = YlivertainenTheReport(
        root=root,
        metadata=metadata,
        cohorted_df=cohorted_df,
        feature_decisions_df=feature_decisions_df,
        df_model=df_model,
        predictive_outputs=predictive_outputs,
        # optional extras:
        dda_tables={"numerical": project.numerical_DDA,
                     "categorical": project.categorical_DDA,
                     "binary": project.binary_DDA},
        associations_table=project.associations_table,
        feature_importance_df=project.feature_importance_df,
    )
    report.generate()
"""

# ============ imports ============
from __future__ import annotations

import base64
import io
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from jinja2 import Template
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

# ============================================================
#                     COLOUR PALETTE
# ============================================================
# Matches your ANSI aesthetic but in hex for the HTML world
_PAL = {
    "bg":          "#0d1117",
    "bg_card":     "#161b22",
    "bg_sidebar":  "#010409",
    "border":      "#30363d",
    "text":        "#c9d1d9",
    "text_muted":  "#8b949e",
    "heading":     "#e6edf3",
    "accent_green":"#3fb950",
    "accent_blue": "#58a6ff",
    "accent_orange":"#d29922",
    "accent_red":  "#f85149",
    "accent_purple":"#bc8cff",
    "link":        "#58a6ff",
    "table_header":"#21262d",
    "table_row_even":"#0d1117",
    "table_row_odd":"#161b22",
    "highlight":   "#1f6feb22",
}

# ============================================================
#                     CHART HELPERS
# ============================================================
def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=_PAL["bg_card"], edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _style_ax(ax: plt.Axes, title: str = "") -> None:
    """Apply dark-clinical styling to an axes."""
    ax.set_facecolor(_PAL["bg_card"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(_PAL["border"])
    ax.tick_params(colors=_PAL["text_muted"], labelsize=9)
    ax.xaxis.label.set_color(_PAL["text_muted"])
    ax.yaxis.label.set_color(_PAL["text_muted"])
    if title:
        ax.set_title(title, color=_PAL["heading"], fontsize=11, fontweight="bold", pad=10)


# ============================================================
#                   CHART GENERATORS
# ============================================================
def _chart_target_distribution(y: pd.Series, task_name: str) -> str:
    """Bar chart of target class distribution."""
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor(_PAL["bg_card"])
    vc = y.value_counts().sort_index()
    colors = [_PAL["accent_green"] if v == vc.idxmax() else _PAL["accent_blue"] for v in vc.index]
    bars = ax.bar([str(v) for v in vc.index], vc.values, color=colors, edgecolor=_PAL["border"], linewidth=0.5)
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + vc.max() * 0.02,
                str(val), ha="center", va="bottom", color=_PAL["text"], fontsize=9)
    _style_ax(ax, f"Target Distribution — {task_name}")
    ax.set_ylabel("Count")
    return _fig_to_base64(fig)


def _chart_missingness(df: pd.DataFrame, top_n: int = 20) -> str | None:
    """Horizontal bar chart of top-N columns by missingness %."""
    miss = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    miss = miss[miss > 0].head(top_n)
    if miss.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(miss) * 0.35)))
    fig.patch.set_facecolor(_PAL["bg_card"])
    colors = [_PAL["accent_red"] if v > 50 else _PAL["accent_orange"] if v > 20 else _PAL["accent_blue"] for v in miss.values]
    ax.barh(miss.index[::-1], miss.values[::-1], color=colors[::-1], edgecolor=_PAL["border"], linewidth=0.5)
    ax.set_xlabel("Missing %")
    _style_ax(ax, "Missingness (pre-imputation)")
    ax.set_xlim(0, min(100, miss.max() * 1.15))
    return _fig_to_base64(fig)


def _chart_feature_decisions(fd: pd.DataFrame) -> str:
    """Stacked horizontal bar: keep vs drop breakdown by reason."""
    action_counts = fd["action"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [1, 2]})
    fig.patch.set_facecolor(_PAL["bg_card"])

    # pie: keep vs drop
    ax0 = axes[0]
    ax0.set_facecolor(_PAL["bg_card"])
    labels = action_counts.index.tolist()
    color_map = {"keep": _PAL["accent_green"], "drop": _PAL["accent_red"]}
    pie_colors = [color_map.get(l, _PAL["accent_orange"]) for l in labels]
    wedges, texts, autotexts = ax0.pie(
        action_counts.values, labels=labels, autopct="%1.0f%%",
        colors=pie_colors, textprops={"color": _PAL["text"], "fontsize": 9},
        wedgeprops={"edgecolor": _PAL["border"], "linewidth": 0.5},
    )
    for t in autotexts:
        t.set_color(_PAL["heading"])
        t.set_fontweight("bold")
    ax0.set_title("Action Split", color=_PAL["heading"], fontsize=11, fontweight="bold")

    # bar: drop reasons
    ax1 = axes[1]
    drop_reasons = fd[fd["action"] == "drop"]["drop_reason"].value_counts()
    if drop_reasons.empty:
        drop_reasons = pd.Series({"none": 0})
    ax1.barh(drop_reasons.index[::-1], drop_reasons.values[::-1],
             color=_PAL["accent_red"], edgecolor=_PAL["border"], linewidth=0.5, alpha=0.85)
    _style_ax(ax1, "Drop Reasons")
    ax1.set_xlabel("Count")
    fig.tight_layout(pad=2)
    return _fig_to_base64(fig)


def _chart_roc_curve(y_test, y_prob_test) -> str:
    """ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor(_PAL["bg_card"])
    ax.plot(fpr, tpr, color=_PAL["accent_blue"], linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color=_PAL["text_muted"], linewidth=0.8, alpha=0.6)
    ax.fill_between(fpr, tpr, alpha=0.08, color=_PAL["accent_blue"])
    _style_ax(ax, "ROC Curve (Held-Out Test)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", facecolor=_PAL["bg_card"], edgecolor=_PAL["border"],
              labelcolor=_PAL["text"], fontsize=9)
    return _fig_to_base64(fig)


def _chart_precision_recall(y_test, y_prob_test) -> str:
    """Precision-Recall curve."""
    prec, rec, _ = precision_recall_curve(y_test, y_prob_test)
    pr_auc = auc(rec, prec)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor(_PAL["bg_card"])
    ax.plot(rec, prec, color=_PAL["accent_green"], linewidth=2, label=f"PR-AUC = {pr_auc:.3f}")
    ax.fill_between(rec, prec, alpha=0.08, color=_PAL["accent_green"])
    baseline = y_test.mean() if hasattr(y_test, "mean") else np.mean(y_test)
    ax.axhline(baseline, linestyle="--", color=_PAL["text_muted"], linewidth=0.8, alpha=0.6, label=f"Baseline = {baseline:.3f}")
    _style_ax(ax, "Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", facecolor=_PAL["bg_card"], edgecolor=_PAL["border"],
              labelcolor=_PAL["text"], fontsize=9)
    return _fig_to_base64(fig)


def _chart_confusion_matrix(y_test, y_pred_test, labels=None) -> str:
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    fig.patch.set_facecolor(_PAL["bg_card"])

    # manual heatmap
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    tick_labels = labels if labels else [str(i) for i in range(cm.shape[0])]
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xticklabels(tick_labels, color=_PAL["text_muted"], fontsize=9)
    ax.set_yticklabels(tick_labels, color=_PAL["text_muted"], fontsize=9)
    ax.set_xlabel("Predicted", color=_PAL["text_muted"])
    ax.set_ylabel("Actual", color=_PAL["text_muted"])

    # annotate cells
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > thresh else _PAL["heading"])

    _style_ax(ax, "Confusion Matrix (Test)")
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    return _fig_to_base64(fig)


def _chart_feature_importance(fi_df: pd.DataFrame, top_n: int = 15) -> str | None:
    """Horizontal bar chart of feature importances."""
    if fi_df is None or fi_df.empty:
        return None
    # expect columns: feature, importance (or similar)
    col_feat = [c for c in fi_df.columns if "feat" in c.lower() or "column" in c.lower() or "name" in c.lower()]
    col_imp  = [c for c in fi_df.columns if "import" in c.lower() or "coef" in c.lower() or "weight" in c.lower()]
    if not col_feat or not col_imp:
        # fallback: first two columns
        col_feat = [fi_df.columns[0]]
        col_imp  = [fi_df.columns[1]]
    df_plot = fi_df.nlargest(top_n, col_imp[0])
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(df_plot) * 0.35)))
    fig.patch.set_facecolor(_PAL["bg_card"])
    vals = df_plot[col_imp[0]].values[::-1]
    names = df_plot[col_feat[0]].values[::-1]
    colors = [_PAL["accent_purple"] if v >= 0 else _PAL["accent_red"] for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, edgecolor=_PAL["border"], linewidth=0.5)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=8)
    _style_ax(ax, f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    return _fig_to_base64(fig)


def _chart_calibration(y_test, y_prob, n_bins: int = 10) -> str:
    """Calibration plot (reliability diagram)."""
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor(_PAL["bg_card"])
    ax.plot(prob_pred, prob_true, marker="o", color=_PAL["accent_orange"], linewidth=2, markersize=5, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color=_PAL["text_muted"], linewidth=0.8, alpha=0.6, label="Perfect")
    _style_ax(ax, "Calibration Plot")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend(loc="lower right", facecolor=_PAL["bg_card"], edgecolor=_PAL["border"],
              labelcolor=_PAL["text"], fontsize=9)
    return _fig_to_base64(fig)


def _chart_probability_histogram(y_test, y_prob) -> str:
    """Histogram of predicted probabilities, colored by true class."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor(_PAL["bg_card"])
    y_arr = np.asarray(y_test)
    p_arr = np.asarray(y_prob)
    ax.hist(p_arr[y_arr == 0], bins=30, alpha=0.6, color=_PAL["accent_blue"], label="Negative", edgecolor=_PAL["border"], linewidth=0.3)
    ax.hist(p_arr[y_arr == 1], bins=30, alpha=0.6, color=_PAL["accent_red"], label="Positive", edgecolor=_PAL["border"], linewidth=0.3)
    _style_ax(ax, "Predicted Probability Distribution")
    ax.set_xlabel("P(positive)")
    ax.set_ylabel("Count")
    ax.legend(facecolor=_PAL["bg_card"], edgecolor=_PAL["border"], labelcolor=_PAL["text"], fontsize=9)
    return _fig_to_base64(fig)


def _chart_model_comparison(mc_df: pd.DataFrame) -> str | None:
    """Bar chart comparing CV AUC across candidate models."""
    if mc_df is None or mc_df.empty:
        return None
    # detect column names
    name_col = [c for c in mc_df.columns if "model" in c.lower() or "name" in c.lower()]
    auc_col  = [c for c in mc_df.columns if "auc" in c.lower() and "mean" in c.lower()]
    std_col  = [c for c in mc_df.columns if "auc" in c.lower() and "std" in c.lower()]
    if not name_col or not auc_col:
        return None
    name_col, auc_col = name_col[0], auc_col[0]
    std_col = std_col[0] if std_col else None

    df_sorted = mc_df.sort_values(auc_col, ascending=True)
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(df_sorted) * 0.5)))
    fig.patch.set_facecolor(_PAL["bg_card"])
    y_pos = range(len(df_sorted))
    vals = df_sorted[auc_col].values
    errs = df_sorted[std_col].values if std_col else None
    colors = [_PAL["accent_green"] if v == vals.max() else _PAL["accent_blue"] for v in vals]
    ax.barh(y_pos, vals, xerr=errs, color=colors, edgecolor=_PAL["border"], linewidth=0.5, capsize=3,
            error_kw={"ecolor": _PAL["text_muted"], "linewidth": 1})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted[name_col].values, fontsize=9)
    _style_ax(ax, "Model Comparison (CV AUC)")
    ax.set_xlabel("Mean CV AUC")
    # annotate values
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", color=_PAL["text"], fontsize=8)
    return _fig_to_base64(fig)


# ============================================================
#                   HELPER: DataFrame → HTML
# ============================================================
def _df_to_html(df: pd.DataFrame, max_rows: int = 50, highlight_col: str | None = None) -> str:
    """Convert a DataFrame to a styled HTML table string (dark-themed)."""
    df_show = df.head(max_rows).copy()
    # round floats
    for c in df_show.select_dtypes(include="number").columns:
        df_show[c] = df_show[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) and isinstance(x, float) else x)

    html = '<div class="table-wrap"><table>\n<thead><tr>'
    for col in df_show.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead>\n<tbody>\n"
    for idx, row in df_show.iterrows():
        html += "<tr>"
        for col in df_show.columns:
            val = row[col]
            cell_class = ""
            if highlight_col and col == highlight_col:
                if str(val).lower() == "drop":
                    cell_class = ' class="cell-drop"'
                elif str(val).lower() == "keep":
                    cell_class = ' class="cell-keep"'
            html += f"<td{cell_class}>{val}</td>"
        html += "</tr>\n"
    html += "</tbody></table></div>\n"
    if len(df) > max_rows:
        html += f'<p class="muted">Showing {max_rows} of {len(df)} rows</p>\n'
    return html


# ============================================================
#                   JINJA2 HTML TEMPLATE
# ============================================================
_HTML_TEMPLATE = Template(textwrap.dedent(r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>
/* ===== RESET & BASE ===== */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', monospace;
    background: {{ pal.bg }};
    color: {{ pal.text }};
    line-height: 1.6;
    display: flex;
    min-height: 100vh;
}

/* ===== SIDEBAR ===== */
.sidebar {
    position: fixed;
    left: 0; top: 0;
    width: 260px;
    height: 100vh;
    background: {{ pal.bg_sidebar }};
    border-right: 1px solid {{ pal.border }};
    padding: 24px 16px;
    overflow-y: auto;
    z-index: 100;
}
.sidebar h1 {
    font-size: 14px;
    color: {{ pal.accent_green }};
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 6px;
}
.sidebar .task-badge {
    font-size: 11px;
    color: {{ pal.accent_orange }};
    padding: 3px 8px;
    border: 1px solid {{ pal.accent_orange }}44;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 20px;
}
.sidebar nav a {
    display: block;
    padding: 7px 12px;
    margin: 2px 0;
    color: {{ pal.text_muted }};
    text-decoration: none;
    font-size: 12px;
    border-radius: 6px;
    transition: all 0.15s ease;
}
.sidebar nav a:hover,
.sidebar nav a.active {
    color: {{ pal.heading }};
    background: {{ pal.highlight }};
}
.sidebar nav a .nav-icon {
    margin-right: 8px;
    opacity: 0.7;
}
.sidebar .meta-block {
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid {{ pal.border }};
    font-size: 11px;
    color: {{ pal.text_muted }};
    line-height: 1.8;
}

/* ===== MAIN ===== */
.main {
    margin-left: 260px;
    padding: 40px 48px 80px 48px;
    max-width: 1100px;
    width: 100%;
}

/* ===== SECTIONS ===== */
section {
    margin-bottom: 48px;
    padding-bottom: 32px;
    border-bottom: 1px solid {{ pal.border }}44;
}
h2 {
    font-size: 20px;
    color: {{ pal.heading }};
    margin-bottom: 20px;
    padding-bottom: 8px;
    border-bottom: 2px solid {{ pal.accent_blue }}55;
    display: flex;
    align-items: center;
    gap: 10px;
}
h2 .sec-icon { font-size: 22px; }
h3 {
    font-size: 15px;
    color: {{ pal.accent_blue }};
    margin: 20px 0 10px 0;
}

/* ===== CARDS / STATS ===== */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.stat-card {
    background: {{ pal.bg_card }};
    border: 1px solid {{ pal.border }};
    border-radius: 8px;
    padding: 16px 18px;
}
.stat-card .label {
    font-size: 11px;
    color: {{ pal.text_muted }};
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.stat-card .value {
    font-size: 24px;
    font-weight: 700;
    color: {{ pal.heading }};
}
.stat-card .value.green  { color: {{ pal.accent_green }}; }
.stat-card .value.blue   { color: {{ pal.accent_blue }}; }
.stat-card .value.orange { color: {{ pal.accent_orange }}; }
.stat-card .value.red    { color: {{ pal.accent_red }}; }
.stat-card .value.purple { color: {{ pal.accent_purple }}; }
.stat-card .sub {
    font-size: 11px;
    color: {{ pal.text_muted }};
    margin-top: 2px;
}

/* ===== TABLES ===== */
.table-wrap {
    overflow-x: auto;
    margin: 12px 0;
    border-radius: 8px;
    border: 1px solid {{ pal.border }};
}
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}
thead th {
    background: {{ pal.table_header }};
    color: {{ pal.heading }};
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
    position: sticky;
    top: 0;
    border-bottom: 2px solid {{ pal.border }};
    white-space: nowrap;
}
tbody td {
    padding: 8px 12px;
    border-bottom: 1px solid {{ pal.border }}55;
    white-space: nowrap;
}
tbody tr:nth-child(even) { background: {{ pal.table_row_even }}; }
tbody tr:nth-child(odd)  { background: {{ pal.table_row_odd }}; }
tbody tr:hover { background: {{ pal.highlight }}; }
.cell-drop { color: {{ pal.accent_red }}; font-weight: 600; }
.cell-keep { color: {{ pal.accent_green }}; font-weight: 600; }

/* ===== CHARTS ===== */
.chart-container {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin: 16px 0;
}
.chart-container img {
    border-radius: 8px;
    border: 1px solid {{ pal.border }};
    max-width: 100%;
    height: auto;
}
.chart-single img {
    border-radius: 8px;
    border: 1px solid {{ pal.border }};
    max-width: 100%;
    margin: 12px 0;
}

/* ===== METRIC HIGHLIGHT BAR ===== */
.metric-bar {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    padding: 16px 20px;
    background: {{ pal.bg_card }};
    border: 1px solid {{ pal.accent_green }}33;
    border-left: 4px solid {{ pal.accent_green }};
    border-radius: 8px;
    margin: 16px 0;
}
.metric-bar .item {
    display: flex;
    flex-direction: column;
}
.metric-bar .item .metric-label {
    font-size: 10px;
    color: {{ pal.text_muted }};
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-bar .item .metric-value {
    font-size: 18px;
    font-weight: 700;
    color: {{ pal.accent_green }};
}

/* ===== INFO BOXES ===== */
.info-box {
    padding: 14px 18px;
    border-radius: 8px;
    font-size: 12px;
    margin: 12px 0;
    border: 1px solid;
}
.info-box.warn {
    background: {{ pal.accent_orange }}11;
    border-color: {{ pal.accent_orange }}44;
    color: {{ pal.accent_orange }};
}
.info-box.danger {
    background: {{ pal.accent_red }}11;
    border-color: {{ pal.accent_red }}44;
    color: {{ pal.accent_red }};
}
.info-box.info {
    background: {{ pal.accent_blue }}11;
    border-color: {{ pal.accent_blue }}44;
    color: {{ pal.accent_blue }};
}
.info-box.success {
    background: {{ pal.accent_green }}11;
    border-color: {{ pal.accent_green }}44;
    color: {{ pal.accent_green }};
}
p { margin: 8px 0; font-size: 13px; }
.muted { color: {{ pal.text_muted }}; font-size: 11px; }
code {
    background: {{ pal.table_header }};
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 12px;
}
.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
}
.tag-green  { background: {{ pal.accent_green }}22; color: {{ pal.accent_green }}; }
.tag-red    { background: {{ pal.accent_red }}22;   color: {{ pal.accent_red }}; }
.tag-orange { background: {{ pal.accent_orange }}22; color: {{ pal.accent_orange }}; }
.tag-blue   { background: {{ pal.accent_blue }}22;   color: {{ pal.accent_blue }}; }
.tag-purple { background: {{ pal.accent_purple }}22; color: {{ pal.accent_purple }}; }

/* ===== RESPONSIVE ===== */
@media (max-width: 900px) {
    .sidebar { display: none; }
    .main { margin-left: 0; padding: 20px; }
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: {{ pal.bg }}; }
::-webkit-scrollbar-thumb { background: {{ pal.border }}; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: {{ pal.text_muted }}; }
</style>
</head>
<body>

<!-- ===== SIDEBAR ===== -->
<aside class="sidebar">
    <h1>Ylivertainen</h1>
    <span class="task-badge">{{ task_name }}</span>
    <nav>
        <a href="#overview"><span class="nav-icon">📋</span>Overview</a>
        <a href="#cohort"><span class="nav-icon">🚧</span>Cohort</a>
        <a href="#dda"><span class="nav-icon">🔍</span>DDA Summary</a>
        <a href="#eda"><span class="nav-icon">📊</span>EDA & Associations</a>
        <a href="#feature-decisions"><span class="nav-icon">⚙️</span>Feature Decisions</a>
        <a href="#model-frame"><span class="nav-icon">🧠</span>Model Frame</a>
        <a href="#predictive"><span class="nav-icon">🔮</span>Predictive Results</a>
        <a href="#diagnostics"><span class="nav-icon">🩺</span>Model Diagnostics</a>
        <a href="#environment"><span class="nav-icon">💻</span>Environment</a>
    </nav>
    <div class="meta-block">
        Generated: {{ timestamp }}<br>
        Task: {{ task_name }}<br>
        Target: {{ target_col }}<br>
        Report by: <code>the_report.py</code>
    </div>
</aside>

<!-- ===== MAIN CONTENT ===== -->
<div class="main">

<!-- ====== OVERVIEW ====== -->
<section id="overview">
    <h2><span class="sec-icon">📋</span> Pipeline Overview</h2>
    <div class="stat-grid">
        <div class="stat-card">
            <div class="label">Task</div>
            <div class="value blue">{{ task_name_short }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Target</div>
            <div class="value orange">{{ target_col }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Task Type</div>
            <div class="value purple">{{ task_type }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Generated</div>
            <div class="value" style="font-size:14px;">{{ timestamp }}</div>
        </div>
    </div>
    <div class="info-box info">
        Pipeline: <strong>Data Cleaning → Cohort → DDA → EDA → Feature Decisions → Model Frame → Predictive</strong>
    </div>
</section>

<!-- ====== COHORT ====== -->
<section id="cohort">
    <h2><span class="sec-icon">🚧</span> Cohort Summary</h2>
    <div class="stat-grid">
        <div class="stat-card">
            <div class="label">Cohort Rows</div>
            <div class="value green">{{ cohort_n_rows }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Cohort Columns</div>
            <div class="value blue">{{ cohort_n_cols }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Prevalence</div>
            <div class="value orange">{{ prevalence }}</div>
            <div class="sub">target mean</div>
        </div>
        <div class="stat-card">
            <div class="label">Positive Class</div>
            <div class="value green">{{ positive_n }}</div>
            <div class="sub">{{ positive_pct }} of cohort</div>
        </div>
        <div class="stat-card">
            <div class="label">Negative Class</div>
            <div class="value red">{{ negative_n }}</div>
            <div class="sub">{{ negative_pct }} of cohort</div>
        </div>
    </div>
    {% if chart_target_dist %}
    <div class="chart-single">
        <img src="{{ chart_target_dist }}" alt="Target Distribution">
    </div>
    {% endif %}
    {% if chart_missingness %}
    <h3>Pre-Imputation Missingness</h3>
    <div class="chart-single">
        <img src="{{ chart_missingness }}" alt="Missingness">
    </div>
    {% endif %}
</section>

<!-- ====== DDA ====== -->
<section id="dda">
    <h2><span class="sec-icon">🔍</span> DDA Summary</h2>
    {% if dda_numerical is not none %}
    <h3>Numerical Features</h3>
    {{ dda_numerical_html }}
    {% endif %}
    {% if dda_categorical is not none %}
    <h3>Categorical Features</h3>
    {{ dda_categorical_html }}
    {% endif %}
    {% if dda_binary is not none %}
    <h3>Binary Features</h3>
    {{ dda_binary_html }}
    {% endif %}
    {% if dda_numerical is none and dda_categorical is none and dda_binary is none %}
    <div class="info-box warn">DDA tables not provided. Pass <code>dda_tables</code> to include them.</div>
    {% endif %}
</section>

<!-- ====== EDA / ASSOCIATIONS ====== -->
<section id="eda">
    <h2><span class="sec-icon">📊</span> EDA & Associations</h2>
    {% if associations_html %}
    <h3>Associations Table</h3>
    <p>Predictor-target association tests (leakage-flagged in red).</p>
    {{ associations_html }}
    {% else %}
    <div class="info-box warn">Associations table not provided. Pass <code>associations_table</code> to include.</div>
    {% endif %}
</section>

<!-- ====== FEATURE DECISIONS ====== -->
<section id="feature-decisions">
    <h2><span class="sec-icon">⚙️</span> Feature Decisions</h2>
    <div class="stat-grid">
        <div class="stat-card">
            <div class="label">Total Features</div>
            <div class="value blue">{{ fd_total }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Kept</div>
            <div class="value green">{{ fd_kept }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Dropped</div>
            <div class="value red">{{ fd_dropped }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Leakage-Flagged</div>
            <div class="value orange">{{ fd_leakage }}</div>
        </div>
    </div>
    {% if chart_feature_decisions %}
    <div class="chart-single">
        <img src="{{ chart_feature_decisions }}" alt="Feature Decisions">
    </div>
    {% endif %}
    <h3>Full Decisions Table</h3>
    {{ feature_decisions_html }}
</section>

<!-- ====== MODEL FRAME ====== -->
<section id="model-frame">
    <h2><span class="sec-icon">🧠</span> Model Frame</h2>
    <div class="stat-grid">
        <div class="stat-card">
            <div class="label">Rows</div>
            <div class="value green">{{ mf_rows }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Columns</div>
            <div class="value blue">{{ mf_cols }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Predictors</div>
            <div class="value purple">{{ mf_predictors }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Remaining NaNs</div>
            <div class="value {{ 'green' if mf_nans == 0 else 'red' }}">{{ mf_nans }}</div>
            <div class="sub">{{ mf_nan_pct }} of cells</div>
        </div>
    </div>
    <h3>Sample (first 10 rows)</h3>
    {{ model_frame_sample_html }}
</section>

<!-- ====== PREDICTIVE RESULTS ====== -->
<section id="predictive">
    <h2><span class="sec-icon">🔮</span> Predictive Results</h2>

    <h3>Best Model</h3>
    <div class="metric-bar">
        <div class="item">
            <span class="metric-label">Winner</span>
            <span class="metric-value">{{ best_model_name }}</span>
        </div>
        {% for key, val in final_metrics_highlights.items() %}
        <div class="item">
            <span class="metric-label">{{ key }}</span>
            <span class="metric-value">{{ val }}</span>
        </div>
        {% endfor %}
    </div>

    {% if chart_model_comparison %}
    <h3>Model Comparison (CV)</h3>
    <div class="chart-single">
        <img src="{{ chart_model_comparison }}" alt="Model Comparison">
    </div>
    {% endif %}

    {% if model_comparison_html %}
    {{ model_comparison_html }}
    {% endif %}

    <h3>Final Metrics Table</h3>
    {{ final_metrics_html }}
</section>

<!-- ====== DIAGNOSTICS ====== -->
<section id="diagnostics">
    <h2><span class="sec-icon">🩺</span> Model Diagnostics</h2>

    <div class="chart-container">
        {% if chart_roc %}
        <img src="{{ chart_roc }}" alt="ROC Curve">
        {% endif %}
        {% if chart_pr %}
        <img src="{{ chart_pr }}" alt="Precision-Recall Curve">
        {% endif %}
    </div>

    <div class="chart-container">
        {% if chart_confusion %}
        <img src="{{ chart_confusion }}" alt="Confusion Matrix">
        {% endif %}
        {% if chart_calibration %}
        <img src="{{ chart_calibration }}" alt="Calibration">
        {% endif %}
    </div>

    <div class="chart-container">
        {% if chart_prob_hist %}
        <img src="{{ chart_prob_hist }}" alt="Probability Histogram">
        {% endif %}
        {% if chart_feature_importance %}
        <img src="{{ chart_feature_importance }}" alt="Feature Importance">
        {% endif %}
    </div>
</section>

<!-- ====== ENVIRONMENT ====== -->
<section id="environment">
    <h2><span class="sec-icon">💻</span> Environment & Run Metadata</h2>
    {% if run_metadata %}
    <div class="stat-grid">
        {% for key, val in run_metadata_display.items() %}
        <div class="stat-card">
            <div class="label">{{ key }}</div>
            <div class="value" style="font-size: 13px;">{{ val }}</div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <div class="info-box warn">Run metadata not available.</div>
    {% endif %}
</section>

</div><!-- /.main -->

<!-- ===== SIDEBAR ACTIVE HIGHLIGHT ===== -->
<script>
document.addEventListener('DOMContentLoaded', () => {
    const links = document.querySelectorAll('.sidebar nav a');
    const sections = document.querySelectorAll('section[id]');
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                links.forEach(l => l.classList.remove('active'));
                const active = document.querySelector(`.sidebar nav a[href="#${entry.target.id}"]`);
                if (active) active.classList.add('active');
            }
        });
    }, { rootMargin: '-20% 0px -80% 0px' });
    sections.forEach(s => observer.observe(s));
});
</script>
</body>
</html>
"""))


# ============================================================
#                      MAIN CLASS
# ============================================================
class YlivertainenTheReport:
    """
    Generates a self-contained HTML report from all pipeline artifacts.

    Parameters
    ----------
    root : Path
        Project root (for output path).
    metadata : dict
        Pipeline metadata with at least 'task_name', 'target_col'.
    cohorted_df : pd.DataFrame
        Post-cohort dataframe (for cohort stats + missingness chart).
    feature_decisions_df : pd.DataFrame
        Feature governance table from EDA stage.
    df_model : pd.DataFrame
        ML-ready table after feature filters + imputation.
    predictive_outputs : dict
        Output from run_predictive() — must contain:
        'model_comparison', 'final_metrics', 'best_model_name',
        'y_test', 'y_prob_test', 'y_pred_test', 'run_metadata'.
    dda_tables : dict, optional
        {"numerical": df, "categorical": df, "binary": df} from DDA stage.
    associations_table : pd.DataFrame, optional
        Predictor-target association table from EDA.
    feature_importance_df : pd.DataFrame, optional
        Feature importances from the predictive stage object.
    """

    def __init__(
        self,
        root: Path,
        metadata: dict,
        cohorted_df: pd.DataFrame,
        feature_decisions_df: pd.DataFrame,
        df_model: pd.DataFrame,
        predictive_outputs: dict,
        *,
        dda_tables: dict[str, pd.DataFrame | None] | None = None,
        associations_table: pd.DataFrame | None = None,
        feature_importance_df: pd.DataFrame | None = None,
    ):
        self.root = Path(root)
        self.metadata = metadata
        self.cohorted_df = cohorted_df
        self.feature_decisions_df = feature_decisions_df
        self.df_model = df_model
        self.pred = predictive_outputs
        self.dda_tables = dda_tables or {}
        self.associations_table = associations_table
        self.feature_importance_df = feature_importance_df

    # ----------------------------------------------------------
    def _build_context(self) -> dict[str, Any]:
        """Assemble all template variables."""
        md = self.metadata
        fd = self.feature_decisions_df
        pred = self.pred
        cohort = self.cohorted_df

        task_name = md.get("task_name", "unknown")
        target_col = md.get("target_col", "target")

        # ---- cohort stats ----
        cohort_n_rows, cohort_n_cols = cohort.shape
        target_series = cohort["target"] if "target" in cohort.columns else self.df_model.iloc[:, -1]
        prevalence = target_series.mean()
        vc = target_series.value_counts()
        positive_n = int(vc.get(True, vc.get(1, 0)))
        negative_n = int(vc.get(False, vc.get(0, 0)))

        # ---- feature decisions stats ----
        fd_total = len(fd)
        fd_kept = int((fd["action"] == "keep").sum()) if "action" in fd.columns else 0
        fd_dropped = int((fd["action"] == "drop").sum()) if "action" in fd.columns else 0
        fd_leakage = 0
        if "drop_reason" in fd.columns:
            fd_leakage = int((fd["drop_reason"] == "leakage").sum())

        # ---- model frame stats ----
        mf_rows, mf_cols = self.df_model.shape
        mf_nans = int(self.df_model.isna().sum().sum())
        mf_nan_pct = f"{(mf_nans / (mf_rows * mf_cols) * 100):.2f}%" if mf_rows and mf_cols else "0%"
        mf_predictors = mf_cols - 1  # subtract target

        # ---- predictive highlights ----
        best_model_name = pred.get("best_model_name", "?")
        fm = pred.get("final_metrics", pd.DataFrame())

        # extract key metrics from final_metrics (single-row df)
        final_metrics_highlights = {}
        highlight_keys = ["test_auc", "sensitivity", "specificity", "brier_score",
                          "auc_ci_lower", "auc_ci_upper"]
        if isinstance(fm, pd.DataFrame) and len(fm) > 0:
            row = fm.iloc[0]
            for k in fm.columns:
                kl = k.lower().replace(" ", "_")
                if any(hk in kl for hk in ["auc", "sensitiv", "specific", "brier", "threshold",
                                            "precision", "recall", "f1", "accuracy"]):
                    val = row[k]
                    if isinstance(val, (float, np.floating)):
                        final_metrics_highlights[k] = f"{val:.3f}"
                    else:
                        final_metrics_highlights[k] = str(val)

        # ---- run_metadata ----
        run_meta = pred.get("run_metadata", {})
        run_metadata_display = {}
        if run_meta:
            for k, v in run_meta.items():
                if isinstance(v, dict):
                    # flatten nested dicts (like env versions)
                    for kk, vv in v.items():
                        run_metadata_display[kk] = str(vv)
                elif isinstance(v, (list, tuple)):
                    run_metadata_display[k] = ", ".join(str(x) for x in v[:10])
                    if len(v) > 10:
                        run_metadata_display[k] += f" (+{len(v)-10} more)"
                else:
                    run_metadata_display[k] = str(v)

        # ---- charts ----
        y_test = np.asarray(pred.get("y_test", []))
        y_prob = np.asarray(pred.get("y_prob_test", []))
        y_pred = np.asarray(pred.get("y_pred_test", []))
        has_test = len(y_test) > 0 and len(y_prob) > 0

        chart_target_dist = _chart_target_distribution(target_series, task_name)
        chart_missingness = _chart_missingness(cohort)
        chart_feature_decisions = _chart_feature_decisions(fd) if fd_total > 0 else None

        chart_roc = _chart_roc_curve(y_test, y_prob) if has_test else None
        chart_pr = _chart_precision_recall(y_test, y_prob) if has_test else None
        chart_confusion = _chart_confusion_matrix(y_test, y_pred) if has_test and len(y_pred) > 0 else None
        chart_calibration = _chart_calibration(y_test, y_prob) if has_test else None
        chart_prob_hist = _chart_probability_histogram(y_test, y_prob) if has_test else None
        chart_feature_importance = _chart_feature_importance(self.feature_importance_df) if self.feature_importance_df is not None else None

        mc = pred.get("model_comparison", None)
        chart_model_comparison = _chart_model_comparison(mc) if mc is not None else None

        # ---- DDA tables ----
        dda_num = self.dda_tables.get("numerical")
        dda_cat = self.dda_tables.get("categorical")
        dda_bin = self.dda_tables.get("binary")

        # ---- associations ----
        assoc = self.associations_table

        return {
            "pal": _PAL,
            "title": f"Ylivertainen Report — {task_name}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "task_name": task_name,
            "task_name_short": task_name.replace("_match_binary", "").replace("_", " ").title(),
            "target_col": target_col,
            "task_type": md.get("task_type", "binary"),

            # cohort
            "cohort_n_rows": f"{cohort_n_rows:,}",
            "cohort_n_cols": cohort_n_cols,
            "prevalence": f"{prevalence:.3f}",
            "positive_n": f"{positive_n:,}",
            "positive_pct": f"{positive_n / cohort_n_rows * 100:.1f}%",
            "negative_n": f"{negative_n:,}",
            "negative_pct": f"{negative_n / cohort_n_rows * 100:.1f}%",
            "chart_target_dist": chart_target_dist,
            "chart_missingness": chart_missingness,

            # DDA
            "dda_numerical": dda_num,
            "dda_categorical": dda_cat,
            "dda_binary": dda_bin,
            "dda_numerical_html": _df_to_html(dda_num) if dda_num is not None else "",
            "dda_categorical_html": _df_to_html(dda_cat) if dda_cat is not None else "",
            "dda_binary_html": _df_to_html(dda_bin) if dda_bin is not None else "",

            # EDA / associations
            "associations_html": _df_to_html(assoc, max_rows=60) if assoc is not None else None,

            # feature decisions
            "fd_total": fd_total,
            "fd_kept": fd_kept,
            "fd_dropped": fd_dropped,
            "fd_leakage": fd_leakage,
            "chart_feature_decisions": chart_feature_decisions,
            "feature_decisions_html": _df_to_html(fd, max_rows=100, highlight_col="action"),

            # model frame
            "mf_rows": f"{mf_rows:,}",
            "mf_cols": mf_cols,
            "mf_predictors": mf_predictors,
            "mf_nans": mf_nans,
            "mf_nan_pct": mf_nan_pct,
            "model_frame_sample_html": _df_to_html(self.df_model.head(10)),

            # predictive
            "best_model_name": best_model_name,
            "final_metrics_highlights": final_metrics_highlights,
            "final_metrics_html": _df_to_html(fm.T.reset_index().rename(columns={"index": "metric", 0: "value"})) if isinstance(fm, pd.DataFrame) and len(fm) > 0 else "<p>No final metrics available.</p>",
            "model_comparison_html": _df_to_html(mc) if mc is not None else None,
            "chart_model_comparison": chart_model_comparison,

            # diagnostics
            "chart_roc": chart_roc,
            "chart_pr": chart_pr,
            "chart_confusion": chart_confusion,
            "chart_calibration": chart_calibration,
            "chart_prob_hist": chart_prob_hist,
            "chart_feature_importance": chart_feature_importance,

            # environment
            "run_metadata": bool(run_meta),
            "run_metadata_display": run_metadata_display,
        }

    # ----------------------------------------------------------
    def generate(self, output_path: str | Path | None = None) -> Path:
        """
        Render the HTML report and write it to disk.

        Parameters
        ----------
        output_path : str or Path, optional
            Where to save the file. Defaults to:
            root / 'reports' / '{task_name}_report.html'

        Returns
        -------
        Path to the saved HTML file.
        """
        ctx = self._build_context()
        html = _HTML_TEMPLATE.render(**ctx)

        if output_path is None:
            out_dir = self.root / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"{self.metadata.get('task_name', 'report')}_report.html"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(html, encoding="utf-8")
        print(f"✅ Report saved → {output_path}")
        print(f"   Size: {output_path.stat().st_size / 1024:.0f} KB")
        return output_path
