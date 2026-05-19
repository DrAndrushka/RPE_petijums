"""Universal HTML report builder for the schema-driven analysis pipeline.

``report.py`` is a *renderer and explainer*, not an analyst. It reads CSV /
SVG artifacts already produced by the cleaning / schema / DDA / missingness /
EDA / inferential stages and assembles a single readable, emoji-rich HTML
document aimed at a researcher (typically a clinician with limited stats
background).

Design rules
------------
* No statistics are recomputed. No data is cleaned. No models are fit.
* No project-specific column names are hardcoded. The report adapts to
  whatever ``output/`` contains.
* Pure Python: f-string HTML + ``pandas.DataFrame.to_html`` + inline CSS.
  No Jinja, no AI, no network calls.
* Missing artifacts produce yellow / red warning boxes; rendering continues
  for everything that *is* available.
* All SVG figures are embedded inline (base64 data URIs) so a single
  ``report.html`` opens anywhere with no sibling ``figures/`` folders.

CLI
---
::

    python report.py \\
        --output-root output \\
        --schema schema.json \\
        --targets upgrade upstage downgrade \\
        --title "Research Data Analysis Report" \\
        --author "Andy" \\
        --out output/report/report.html

If ``--schema`` is omitted the report falls back to ``output/schema/schema_summary.csv``
(if present) or skips the schema section entirely.
"""

from __future__ import annotations

import argparse
import base64
import html as _html
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Thresholds & constants (Cohen-style defaults, configurable via dataclass)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EffectThresholds:
    """Tier cutoffs used to translate raw effect magnitudes into badges.

    ``corr_*`` apply to correlation-like effects (Spearman rho, rank-biserial
    r, Cramer's V). ``or_*`` apply to odds ratios via ``abs(log(OR))`` so the
    same scale is used for protective (OR<1) and risk (OR>1) directions.
    """
    corr_weak: float = 0.10
    corr_moderate: float = 0.30
    corr_strong: float = 0.50
    # OR thresholds expressed on the log scale; the chosen anchors map to
    # OR ~ 1.11 / 1.35 / 1.65 (and their reciprocals 0.90 / 0.74 / 0.61).
    or_weak: float = math.log(1.11)
    or_moderate: float = math.log(1.35)
    or_strong: float = math.log(1.65)


@dataclass(frozen=True)
class MissingThresholds:
    low: float = 5.0
    moderate: float = 20.0
    high: float = 40.0


@dataclass
class ReportConfig:
    output_root: Path
    title: str = "Research Data Analysis Report"
    author: str = ""
    targets: Sequence[str] = field(default_factory=tuple)
    schema_path: Path | None = None
    fdr_alpha: float = 0.05
    nominal_alpha: float = 0.05
    effect: EffectThresholds = field(default_factory=EffectThresholds)
    missing: MissingThresholds = field(default_factory=MissingThresholds)
    # Primary exposure / predictor of interest — gets a dedicated synthesis section.
    focus_predictor: str | None = None
    # For one-hot nominal focus vars: which level is reference (e.g. transperineāla).
    focus_reference_level: str | None = None
    # Column used for multi-year focus figure (``<col>__bar_by_year.svg``).
    year_column: str | None = None


# ---------------------------------------------------------------------------
# Inline CSS (medical-academic, emoji-friendly, color-coded rows + badges)
# ---------------------------------------------------------------------------

_CSS = """
:root {
    --fg: #1f2937;
    --muted: #6b7280;
    --bg: #ffffff;
    --card: #f9fafb;
    --border: #e5e7eb;
    --accent: #3b7ddd;
    --green: #16a34a;
    --green-bg: #dcfce7;
    --yellow: #ca8a04;
    --yellow-bg: #fef9c3;
    --orange: #ea580c;
    --orange-bg: #ffedd5;
    --red: #dc2626;
    --red-bg: #fee2e2;
    --blue: #2563eb;
    --blue-bg: #dbeafe;
    --grey-bg: #f3f4f6;
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: var(--fg); background: var(--bg);
    line-height: 1.55; font-size: 15px;
}
.container { max-width: 1180px; margin: 0 auto; padding: 32px 28px 80px; }

h1 { font-size: 30px; margin: 0 0 4px; }
h2 { font-size: 22px; margin: 38px 0 12px; padding-bottom: 6px;
     border-bottom: 2px solid var(--border); }
h3 { font-size: 17px; margin: 22px 0 8px; color: #111827; }
h4 { font-size: 15px; margin: 14px 0 6px; color: var(--muted); font-weight: 600; }
p  { margin: 8px 0 12px; }
small, .muted { color: var(--muted); }
code { background: var(--grey-bg); padding: 1px 5px; border-radius: 4px;
       font-size: 13px; }

.report-section { margin-bottom: 28px; }

/* Header dashboard cards */
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px,1fr));
         gap: 12px; margin: 14px 0 4px; }
.card { background: var(--card); border: 1px solid var(--border);
        border-radius: 10px; padding: 12px 14px; }
.card .label { font-size: 12px; color: var(--muted); text-transform: uppercase;
               letter-spacing: 0.04em; }
.card .value { font-size: 22px; font-weight: 600; margin-top: 3px; }

/* Tables */
table.report { border-collapse: collapse; width: 100%; font-size: 13.5px;
               margin: 8px 0 14px; }
table.report th, table.report td { padding: 7px 10px; text-align: left;
                                   border-bottom: 1px solid var(--border);
                                   vertical-align: top; }
table.report thead th { background: var(--grey-bg); position: sticky; top: 0;
                        font-weight: 600; }
table.report tbody tr:hover { background: #fafafa; }

/* Row color coding */
tr.sig-fdr      { background: var(--green-bg) !important; }
tr.sig-nominal  { background: var(--yellow-bg) !important; }
tr.sig-none     { background: transparent; }
tr.or-risk      { background: #fde8e8 !important; }
tr.or-protective{ background: #dceaff !important; }
tr.or-neutral   { background: transparent; }
tr.missing-low      { background: var(--green-bg) !important; }
tr.missing-medium   { background: var(--yellow-bg) !important; }
tr.missing-high     { background: var(--orange-bg) !important; }
tr.missing-severe   { background: var(--red-bg) !important; }

/* Badges */
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px;
         font-size: 12px; font-weight: 600; }
.badge.effect-strong   { background: var(--green-bg);  color: var(--green); }
.badge.effect-moderate { background: var(--yellow-bg); color: var(--yellow); }
.badge.effect-weak     { background: var(--red-bg);    color: var(--red); }
.badge.effect-none     { background: var(--grey-bg);   color: var(--muted); }
.badge.kind            { background: var(--grey-bg);   color: var(--fg); }
.badge.target          { background: var(--blue-bg);   color: var(--blue); }

/* Warning / info boxes */
.warning-box, .info-box {
    border-left: 4px solid var(--yellow); background: var(--yellow-bg);
    padding: 10px 14px; border-radius: 6px; margin: 12px 0;
    font-size: 14px;
}
.warning-box.severe { border-left-color: var(--red); background: var(--red-bg); }
.info-box { border-left-color: var(--accent); background: #eff6ff; }

/* Figure grid */
.figure-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px,1fr));
               gap: 14px; margin: 12px 0 18px; }
.figure-card { border: 1px solid var(--border); border-radius: 8px;
               padding: 8px; background: #fff; }
.figure-card img, .figure-card object {
    width: 100%; height: auto; display: block;
}
.figure-card .caption { font-size: 12px; color: var(--muted);
                        margin-top: 4px; text-align: center;
                        word-break: break-word; }

/* Collapsible details */
details.collapsible { margin: 8px 0 14px; }
details.collapsible > summary {
    cursor: pointer; font-weight: 600; padding: 6px 0;
    color: var(--accent);
}

/* TL;DR list */
.tldr-list { padding-left: 22px; }
.tldr-list li { margin: 4px 0; }

/* Focus predictor spotlight */
.focus-hero {
    background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px;
    margin: 16px 0 20px;
}
.focus-hero h3 { margin: 0 0 6px; font-size: 20px; color: var(--accent); }
.focus-stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
    margin: 14px 0 6px;
}
.focus-stat-card {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
}
.focus-stat-card .label {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.focus-stat-card .value {
    font-size: 18px;
    font-weight: 700;
    color: var(--accent);
    margin-top: 4px;
    line-height: 1.2;
}
.focus-target-block {
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 14px 16px;
    margin: 16px 0;
    background: #fafbfc;
}
.focus-figure-hero {
    max-width: 520px;
    margin: 12px auto 18px;
}
.focus-figure-hero img { width: 100%; height: auto; display: block; }

/* Variable of interest — compact tables + capped figure width */
.focus-section { font-size: 15px; margin-top: 20px; }
.focus-section > h3 { font-size: 19px; margin-bottom: 10px; }
.focus-section h4 { font-size: 15px; margin: 10px 0 6px; }
.focus-section p { font-size: 15px; margin: 4px 0 8px; }
.focus-section table.report {
    font-size: 14px;
    margin: 6px 0 10px;
    max-width: 100%;
}
.focus-section table.report th,
.focus-section table.report td {
    padding: 3px 5px;
    line-height: 1.25;
}
.focus-section .focus-target-block {
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 10px 12px;
    margin: 10px 0;
    background: #fafbfc;
}
.focus-section .focus-hero {
    padding: 12px 14px;
    margin: 10px 0 14px;
}
.focus-section .focus-stat-grid {
    grid-template-columns: repeat(auto-fit, minmax(72px, 1fr));
    gap: 6px;
}
.focus-section .focus-stat-card { padding: 6px 8px; }
.focus-section .focus-stat-card .label { font-size: 9px; }
.focus-section .focus-stat-card .value { font-size: 13px; }
/* DDA distribution plot in Variable of interest (e.g. biopsy_type__bar.svg).
   Sized via .focus-figure-hero below — not the full-width .figure-grid. */
.focus-section .focus-figure-hero {
    max-width: 600px;
    margin: 8px auto 12px;
}
/* EDA plots per target — fixed width (avoids 1fr grid stretching one image full-page) */
.focus-section .focus-eda-figure {
    display: inline-block;
    max-width: 500px;
    width: 100%;
    margin: 6px 0 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px;
    background: #fff;
}
.focus-section .focus-eda-figure img {
    width: 100%;
    height: auto;
    display: block;
}
.focus-section .focus-eda-figure .caption {
    font-size: 12px;
    color: var(--muted);
    text-align: center;
    margin-top: 4px;
    word-break: break-word;
}
.focus-section ul { font-size: 11.5px; padding-left: 18px; }
.focus-section ul li { margin: 2px 0; }
.focus-route-note { font-size: 12px; color: #374151; margin: 6px 0 0; }
.figure-note { font-size: 11px; color: #4b5563; margin: 8px 0 0; max-width: 52rem; }
.focus-route-card {
    margin: 8px 0 12px;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: #fff;
}
.focus-route-card .focus-route-title {
    font-size: 12px;
    font-weight: 600;
    margin: 0 0 8px;
    color: var(--muted);
}
.focus-route-table { width: 100%; font-size: 13px; border-collapse: collapse; }
.focus-route-table th {
    text-align: left;
    font-size: 10px;
    font-weight: 600;
    color: var(--muted);
    padding: 4px 8px;
    border-bottom: 1px solid var(--border);
}
.focus-route-table td { padding: 7px 8px; border-bottom: 1px solid #f3f4f6; }
.focus-route-table tr.focus-route-highlight td {
    background: #eff6ff;
    border-left: 3px solid var(--accent);
}
.focus-route-table .mono {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-variant-numeric: tabular-nums;
}
.focus-dda-routes { margin: 6px 0 10px; }

/* Stats decoder */
.stat-decoder dt { font-weight: 600; margin-top: 12px; }
.stat-decoder dd { margin-left: 0; color: #374151; }

/* Footer */
.footer { color: var(--muted); font-size: 12px; margin-top: 60px;
          border-top: 1px solid var(--border); padding-top: 12px; }
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed_svg_src(path: Path) -> str | None:
    """Return a ``data:image/svg+xml;base64,...`` URI for embedding in ``<img src>``."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data.strip():
        return None
    encoded = base64.standard_b64encode(data).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _figure_img_html(path: Path) -> str:
    """``<img>`` tag with the SVG inlined as a data URI (self-contained HTML)."""
    src = _embed_svg_src(path)
    if src is None:
        return ""
    return f'<img src="{src}" alt="{_esc(path.stem)}" loading="lazy"/>'


def _esc(x: Any) -> str:
    """HTML-escape an arbitrary value (None / NaN -> empty string)."""
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    return _html.escape(str(x))


def human_pool_df(val: Any) -> str:
    """Pooled Rubin df for display (large / non-finite → ∞)."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    try:
        x = float(val)
    except (TypeError, ValueError):
        return str(val)
    if not math.isfinite(x) or x >= 9999:
        return "∞"
    if x == int(x):
        return str(int(x))
    return f"{x:.1f}"


def human_p(p: Any) -> str:
    """Format a p-value for display.

    Already-formatted strings (e.g. ``"<0.001"`` from ``cleaning.format_table_for_csv``)
    are passed through unchanged. Numeric values follow the same rule:
    ``p < 0.001`` -> ``"<0.001"``; otherwise 3 decimal places.
    """
    if p is None:
        return ""
    if isinstance(p, str):
        return p
    try:
        v = float(p)
    except (TypeError, ValueError):
        return _esc(p)
    if not math.isfinite(v):
        return ""
    if v < 0.001:
        return "<0.001"
    return f"{v:.3f}"


def _coerce_p(p: Any) -> float | None:
    """Best-effort numeric p-value parser; handles ``"<0.001"`` strings."""
    if p is None:
        return None
    if isinstance(p, (int, float)):
        v = float(p)
        return v if math.isfinite(v) else None
    s = str(p).strip()
    if not s:
        return None
    if s.startswith("<"):
        # Treat "<0.001" as 0.0005 (well below alpha); good enough for tiering.
        try:
            return float(s[1:]) / 2
        except ValueError:
            return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def _coerce_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    try:
        v = float(str(x).strip())
        return v if math.isfinite(v) else None
    except ValueError:
        return None


# Strength tier wording -------------------------------------------------------

_STRENGTH_WORDING = {
    "strong":   ("🟢", "strong",   "Large statistical signal; worth serious attention, but still needs clinical context."),
    "moderate": ("🟡", "moderate", "Visible statistical signal; worth attention and hypothesis generation."),
    "weak":     ("🔴", "weak",     "Small statistical signal; probably not enough alone to guide clinical decisions."),
    "none":     ("⚪", "none",     "No useful statistical pattern detected."),
}


def effect_badge(effect: Any, kind: str = "corr",
                 thr: EffectThresholds = EffectThresholds()) -> str:
    """Return inline HTML badge for an effect magnitude.

    Parameters
    ----------
    effect : numeric
        Spearman rho / rank-biserial r / Cramer's V / odds ratio.
    kind : {"corr", "or"}
        ``"corr"`` thresholds use ``|effect|`` directly; ``"or"`` thresholds
        use ``|log(OR)|`` so risk and protective directions share one scale.
    """
    tier = _strength_tier(effect, kind, thr)
    emoji, label, _ = _STRENGTH_WORDING[tier]
    return f'<span class="badge effect-{tier}">{emoji} {label}</span>'


def _strength_tier(effect: Any, kind: str, thr: EffectThresholds) -> str:
    e = _coerce_float(effect)
    if e is None:
        return "none"
    if kind == "or":
        if e <= 0:
            return "none"
        mag = abs(math.log(e))
        if mag >= thr.or_strong:   return "strong"
        if mag >= thr.or_moderate: return "moderate"
        if mag >= thr.or_weak:     return "weak"
        return "none"
    # corr-like
    mag = abs(e)
    if mag >= thr.corr_strong:   return "strong"
    if mag >= thr.corr_moderate: return "moderate"
    if mag >= thr.corr_weak:     return "weak"
    return "none"


def classify_significance(p: Any, p_fdr: Any, *,
                          fdr_alpha: float = 0.05,
                          nominal_alpha: float = 0.05) -> str:
    """Return one of ``"sig-fdr"`` / ``"sig-nominal"`` / ``"sig-none"``."""
    p_num = _coerce_p(p)
    fdr_num = _coerce_p(p_fdr)
    if fdr_num is not None and fdr_num < fdr_alpha:
        return "sig-fdr"
    if p_num is not None and p_num < nominal_alpha:
        return "sig-nominal"
    return "sig-none"


def classify_or_direction(or_val: Any, ci_lo: Any, ci_hi: Any) -> str:
    """Return ``"or-risk"`` / ``"or-protective"`` / ``"or-neutral"``."""
    o = _coerce_float(or_val); lo = _coerce_float(ci_lo); hi = _coerce_float(ci_hi)
    if o is None or lo is None or hi is None:
        return "or-neutral"
    if lo > 1.0 and o > 1.0:
        return "or-risk"
    if hi < 1.0 and o < 1.0:
        return "or-protective"
    return "or-neutral"


def classify_missing(pct: Any, thr: MissingThresholds = MissingThresholds()) -> str:
    v = _coerce_float(pct)
    if v is None:
        return "missing-low"
    if v >= thr.high:     return "missing-severe"
    if v >= thr.moderate: return "missing-high"
    if v >= thr.low:      return "missing-medium"
    return "missing-low"


def warning_box(msg: str, severe: bool = False) -> str:
    cls = "warning-box severe" if severe else "warning-box"
    emoji = "🚨" if severe else "⚠️"
    return f'<div class="{cls}">{emoji} {_esc(msg)}</div>'


def info_box(msg: str) -> str:
    return f'<div class="info-box">ℹ️ {_esc(msg)}</div>'


def table_to_html(df: pd.DataFrame, *, row_class_fn=None,
                  max_rows: int | None = None,
                  index: bool = False,
                  safe_html_cols: Iterable[str] = ()) -> str:
    """Render a DataFrame to HTML with optional per-row CSS class function.

    Parameters
    ----------
    row_class_fn : callable, optional
        Receives the row's Series and returns a CSS class string
        (or ``""``). Applied AFTER any truncation by ``max_rows``.
    safe_html_cols : iterable of column names
        Cells in these columns are emitted verbatim (NOT HTML-escaped).
        Use this for pre-built ``<span class='badge ...'>`` snippets.
        Any column not listed is still escaped — default-safe.
    """
    if df is None or df.empty:
        return '<p class="muted"><em>(empty table)</em></p>'
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).copy()
    cols = list(df.columns)
    safe_set = set(safe_html_cols)
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    if index:
        head = f"<th>{_esc(df.index.name or '')}</th>" + head

    body_rows = []
    for idx, row in df.iterrows():
        cls = row_class_fn(row) if row_class_fn else ""
        cls_attr = f' class="{cls}"' if cls else ""
        cells = "".join(
            # Pre-built HTML (badges) passes through verbatim; everything else
            # is escaped to keep the document safe even with weird data.
            f"<td>{row[c] if c in safe_set else _esc(row[c])}</td>"
            for c in cols
        )
        if index:
            cells = f"<td><strong>{_esc(idx)}</strong></td>" + cells
        body_rows.append(f"<tr{cls_attr}>{cells}</tr>")
    body = "".join(body_rows)
    return (f'<table class="report"><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table>')


def svg_grid(svg_paths: Iterable[Path], max_n: int | None = None) -> str:
    """Render an HTML grid of SVG figures embedded as base64 data URIs."""
    paths = [p for p in svg_paths if p.exists()]
    if max_n is not None:
        paths = paths[:max_n]
    if not paths:
        return '<p class="muted"><em>(no figures available)</em></p>'
    cards = []
    for p in paths:
        img = _figure_img_html(p)
        if not img:
            continue
        cards.append(
            f'<div class="figure-card">'
            f'{img}'
            f'<div class="caption">{_esc(p.stem)}</div>'
            f'</div>'
        )
    if not cards:
        return '<p class="muted"><em>(no figures available)</em></p>'
    return f'<div class="figure-grid">{"".join(cards)}</div>'


def _focus_eda_figure(svg_path: Path) -> str:
    """Single compact EDA plot for the Variable-of-interest section."""
    if not svg_path.exists():
        return '<p class="muted"><em>(figure not found)</em></p>'
    img = _figure_img_html(svg_path)
    if not img:
        return '<p class="muted"><em>(figure not found)</em></p>'
    return (
        '<div class="focus-eda-figure">'
        f'{img}'
        f'<div class="caption">{_esc(svg_path.stem)}</div>'
        '</div>'
    )


def details_block(summary: str, inner_html: str, *, open: bool = False) -> str:
    open_attr = " open" if open else ""
    return (f'<details class="collapsible"{open_attr}>'
            f'<summary>{_esc(summary)}</summary>{inner_html}</details>')


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------

@dataclass
class Artifacts:
    """All artifacts discovered under ``output_root``.

    Each table is loaded lazily as a DataFrame (or ``None`` if missing).
    Figures are stored as lists of ``Path`` objects so render functions can
    decide which subset to embed.
    """
    output_root: Path

    # Cleaning / schema
    cleaning_summary: pd.DataFrame | None = None
    cleaning_log: pd.DataFrame | None = None
    schema_summary: pd.DataFrame | None = None

    # DDA
    dda_overall: pd.DataFrame | None = None
    dda_continuous: pd.DataFrame | None = None
    dda_categorical: pd.DataFrame | None = None
    dda_binary: pd.DataFrame | None = None
    dda_datetime: pd.DataFrame | None = None
    dda_id_text: pd.DataFrame | None = None
    dda_figures: list[Path] = field(default_factory=list)

    # Missingness
    missingness_summary: pd.DataFrame | None = None
    top_missing: pd.DataFrame | None = None
    missingness_figures: list[Path] = field(default_factory=list)

    # EDA
    associations: pd.DataFrame | None = None
    eda_figures: list[Path] = field(default_factory=list)

    # Inferential
    inferential_summary: pd.DataFrame | None = None
    inferential_multivariable: dict[str, pd.DataFrame] = field(default_factory=dict)
    inferential_vif: dict[str, pd.DataFrame] = field(default_factory=dict)
    inferential_figures: list[Path] = field(default_factory=list)

    # Warnings accumulated during load (rendered in appendix)
    warnings: list[str] = field(default_factory=list)


def _maybe_read_csv(p: Path, warnings: list[str]) -> pd.DataFrame | None:
    """Read a CSV if it exists; record (non-fatal) warning otherwise."""
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:  # pragma: no cover - defensive
        warnings.append(f"Failed to read {p.name}: {e}")
        return None


def load_artifacts(cfg: ReportConfig) -> Artifacts:
    """Discover every CSV / SVG under ``cfg.output_root``.

    Missing files are recorded as warnings but never raise. The function is
    pipeline-agnostic: directories that don't exist are simply skipped.
    """
    root = cfg.output_root
    art = Artifacts(output_root=root)

    if not root.exists():
        art.warnings.append(f"output_root '{root}' does not exist.")
        return art

    # Cleaning
    cleaning_dir = root / "cleaning"
    art.cleaning_summary = _maybe_read_csv(cleaning_dir / "cleaning_summary.csv", art.warnings)
    art.cleaning_log     = _maybe_read_csv(cleaning_dir / "cleaning_log.csv", art.warnings)

    # Schema
    if cfg.schema_path and cfg.schema_path.exists():
        art.schema_summary = _load_schema_any(cfg.schema_path, art.warnings)
    else:
        art.schema_summary = _maybe_read_csv(root / "schema" / "schema_summary.csv", art.warnings)

    # DDA
    dda_tab = root / "dda" / "tables"
    art.dda_overall     = _maybe_read_csv(dda_tab / "dda_overall.csv", art.warnings)
    art.dda_continuous  = _maybe_read_csv(dda_tab / "dda_continuous.csv", art.warnings)
    art.dda_categorical = _maybe_read_csv(dda_tab / "dda_categorical.csv", art.warnings)
    art.dda_binary      = _maybe_read_csv(dda_tab / "dda_binary.csv", art.warnings)
    art.dda_datetime    = _maybe_read_csv(dda_tab / "dda_datetime.csv", art.warnings)
    art.dda_id_text     = _maybe_read_csv(dda_tab / "dda_id_text.csv", art.warnings)
    dda_fig = root / "dda" / "figures"
    if dda_fig.exists():
        art.dda_figures = sorted(dda_fig.glob("*.svg"))

    # Missingness
    miss_tab = root / "missingness" / "tables"
    art.missingness_summary = _maybe_read_csv(miss_tab / "missingness_summary.csv", art.warnings)
    art.top_missing         = _maybe_read_csv(miss_tab / "top_missing.csv", art.warnings)
    # Fall back to flat layout (older runs)
    if art.missingness_summary is None:
        art.missingness_summary = _maybe_read_csv(root / "missingness" / "missing_per_column.csv", art.warnings)
    miss_fig = root / "missingness" / "figures"
    if miss_fig.exists():
        art.missingness_figures = sorted(miss_fig.glob("*.svg"))
    else:
        # Older layout: figures dropped directly in missingness/
        flat = root / "missingness"
        if flat.exists():
            art.missingness_figures = sorted(flat.glob("*.svg"))

    # EDA
    art.associations = _maybe_read_csv(root / "eda" / "tables" / "associations.csv", art.warnings)
    eda_fig = root / "eda" / "figures"
    if eda_fig.exists():
        art.eda_figures = sorted(eda_fig.glob("*.svg"))

    # Inferential
    inf_tab = root / "inferential" / "tables"
    art.inferential_summary = _maybe_read_csv(inf_tab / "inferential_summary.csv", art.warnings)
    if inf_tab.exists():
        for f in sorted(inf_tab.glob("*__multivariable.csv")):
            target = f.stem.replace("__multivariable", "")
            art.inferential_multivariable[target] = pd.read_csv(f)
        for f in sorted(inf_tab.glob("*__vif.csv")):
            target = f.stem.replace("__vif", "")
            art.inferential_vif[target] = pd.read_csv(f)
    inf_fig = root / "inferential" / "figures"
    if inf_fig.exists():
        art.inferential_figures = sorted(inf_fig.glob("*.svg"))

    return art


def _load_schema_any(path: Path, warnings: list[str]) -> pd.DataFrame | None:
    """Accept JSON (dict of ColSpec-like) or CSV with one row per column."""
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = []
            for name, spec in data.items():
                if isinstance(spec, dict):
                    rows.append({"column": name, **spec})
                else:
                    rows.append({"column": name, "kind": str(spec)})
            return pd.DataFrame(rows)
        return pd.read_csv(path)
    except Exception as e:
        warnings.append(f"Failed to load schema from {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def render_header(cfg: ReportConfig, art: Artifacts) -> str:
    """🧾 Top-of-report dashboard with headline counts."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Dataset shape from DDA overall if present (coerce to int for display —
    # the CSV may store these as floats due to pandas type inference).
    n_rows = n_cols = None
    if art.dda_overall is not None and not art.dda_overall.empty:
        row = art.dda_overall.iloc[0]
        n_rows = _to_int_or_none(row.get("n_rows"))
        n_cols = _to_int_or_none(row.get("n_cols"))

    n_preds_screened = (len(art.associations.drop_duplicates("predictor"))
                        if art.associations is not None and not art.associations.empty
                        and "predictor" in art.associations.columns else 0)
    n_tests = (len(art.associations) if art.associations is not None else 0)
    n_models = len(art.inferential_multivariable)

    # Stages completed (presence-based)
    stages = []
    if art.cleaning_summary is not None: stages.append("cleaning")
    if art.schema_summary is not None:   stages.append("schema")
    if any(t is not None for t in [art.dda_continuous, art.dda_categorical,
                                    art.dda_binary, art.dda_datetime]): stages.append("DDA")
    if art.missingness_summary is not None: stages.append("missingness")
    if art.associations is not None:        stages.append("EDA")
    if n_models > 0:                        stages.append("inferential")

    def card(label: str, value: Any) -> str:
        return (f'<div class="card"><div class="label">{_esc(label)}</div>'
                f'<div class="value">{_esc(value)}</div></div>')

    cards = [
        card("Generated", now),
        card("Author", cfg.author or "—"),
        card("Rows", n_rows if n_rows is not None else "—"),
        card("Columns", n_cols if n_cols is not None else "—"),
        card("Targets", len(cfg.targets) or "—"),
        card("Predictors screened", n_preds_screened or "—"),
        card("EDA tests", n_tests or "—"),
        card("Inferential models", n_models or "—"),
    ]
    targets_html = ", ".join(f"<span class='badge target'>🎯 {_esc(t)}</span>"
                             for t in cfg.targets) or "<em>(none specified)</em>"
    stages_html = ", ".join(f"<code>{_esc(s)}</code>" for s in stages) or "<em>(none detected)</em>"

    blurb = ("This report summarizes automated data cleaning, schema profiling, "
             "descriptive data analysis, missingness assessment, exploratory "
             "association screening, and multivariable modelling.")

    return (
        f'<section class="report-section">'
        f'<h1>🧾 {_esc(cfg.title)}</h1>'
        f'<p class="muted">{blurb}</p>'
        f'<div class="cards">{"".join(cards)}</div>'
        f'<p><strong>Targets:</strong> {targets_html}</p>'
        f'<p><strong>Stages detected:</strong> {stages_html}</p>'
        f'</section>'
    )


def render_cleaning(cfg: ReportConfig, art: Artifacts) -> str:
    """🧹 Cleaning story."""
    blurb = ("The dataset was cleaned using a schema-driven process: declared "
             "null markers were applied, replacements were performed, data "
             "types were coerced, and skipped variables were excluded where "
             "appropriate.")
    body = [f'<h2>🧹 Cleaning story</h2><p>{blurb}</p>']

    if art.cleaning_summary is None and art.cleaning_log is None:
        body.append(warning_box(
            "No saved cleaning summary was found. Cleaning may have been "
            "performed, but no cleaning audit table was exported."))
        return f'<section class="report-section">{"".join(body)}</section>'

    if art.cleaning_summary is not None and not art.cleaning_summary.empty:
        body.append("<h3>Summary</h3>")
        body.append(table_to_html(art.cleaning_summary))
    if art.cleaning_log is not None and not art.cleaning_log.empty:
        body.append(details_block("📜 Full cleaning log",
                                  table_to_html(art.cleaning_log, max_rows=200)))
    return f'<section class="report-section">{"".join(body)}</section>'


def render_schema(cfg: ReportConfig, art: Artifacts) -> str:
    """🧬 Schema story with kind badges."""
    blurb = ("Variables were classified by analytical role. Continuous/count "
             "variables were treated numerically, ordinal variables preserved "
             "ordering, nominal variables were treated as unordered categories, "
             "and ID/text/skip variables were excluded from statistical "
             "screening where appropriate.")
    body = [f'<h2>🧬 Schema story</h2><p>{blurb}</p>']

    if art.schema_summary is None or art.schema_summary.empty:
        body.append(warning_box("No schema artifact was found."))
        return f'<section class="report-section">{"".join(body)}</section>'

    sch = art.schema_summary.copy()

    # Mark targets visually
    target_set = set(cfg.targets)
    if "column" in sch.columns:
        sch["role"] = sch["column"].apply(
            lambda c: "🎯 target" if c in target_set else "")

    # Kind -> emoji
    kind_emoji = {
        "continuous": "🔵 continuous", "count": "🔵 count",
        "ordinal": "🟣 ordinal", "binary": "🟢 binary",
        "nominal": "🟡 nominal", "datetime": "🕒 datetime",
        "id": "⚪ id", "text": "⚪ text", "skip": "⚪ skip",
    }
    if "kind" in sch.columns:
        sch["kind"] = sch["kind"].map(lambda k: kind_emoji.get(str(k), str(k)))

    body.append(table_to_html(sch, max_rows=400))
    return f'<section class="report-section">{"".join(body)}</section>'


def render_dda(cfg: ReportConfig, art: Artifacts) -> str:
    """📊 DDA story with per-kind subsections."""
    body = [
        '<h2>📊 Descriptive Data Analysis (DDA)</h2>',
        '<p>This section summarizes each variable on its own, before any '
        'association testing. Tables describe distribution shape and balance; '
        'figures show the same information visually.</p>',
    ]

    # Glossary so a clinician knows what each column means
    body.append(details_block("📖 What do these metrics mean?", _dda_glossary()))

    # Dataset overview
    if art.dda_overall is not None and not art.dda_overall.empty:
        body.append("<h3>📦 Dataset overview</h3>")
        body.append(table_to_html(art.dda_overall))

    sections = [
        ("📏 Continuous / count variables",
         "Summarized using median, mean, trimmed mean, spread, skewness, "
         "kurtosis, outlier-sensitive quantiles, and missingness.",
         art.dda_continuous,
         lambda r: classify_missing(r.get("missing_pct"), cfg.missing)),
        ("🏷️ Categorical / ordinal variables",
         "Summarized using dominant class, rarest class, class imbalance, "
         "Shannon entropy, and normalized balance.",
         art.dda_categorical,
         lambda r: classify_missing(r.get("missing_pct"), cfg.missing)),
        ("✅ Binary variables",
         "Same schema as categorical: dominant class, balance, missingness.",
         art.dda_binary,
         lambda r: classify_missing(r.get("missing_pct"), cfg.missing)),
        ("🕒 Datetime variables",
         "Range, span in days, and missingness.",
         art.dda_datetime,
         lambda r: classify_missing(r.get("missing_pct"), cfg.missing)),
        ("🪪 ID / text variables",
         "Listed for completeness; excluded from statistical screening.",
         art.dda_id_text,
         None),
    ]
    for heading, blurb, tbl, row_fn in sections:
        body.append(f"<h3>{heading}</h3>")
        body.append(f"<p>{blurb}</p>")
        if tbl is None or tbl.empty:
            body.append('<p class="muted"><em>(no variables of this kind)</em></p>')
        else:
            body.append(table_to_html(tbl, row_class_fn=row_fn))

    # Figures (collapsed by default; usually many)
    if art.dda_figures:
        grid_html = svg_grid(art.dda_figures)
        body.append(details_block(f"🖼️ DDA figures ({len(art.dda_figures)})",
                                  grid_html))

    return f'<section class="report-section">{"".join(body)}</section>'


def _dda_glossary() -> str:
    items = [
        ("missing_pct", "Percentage of missing values for this variable."),
        ("first_mode", "Most common value."),
        ("first_mode_pct", "How dominant the most common value is."),
        ("rarest", "Least common value."),
        ("max_class_imbalance", "first_mode_count / rarest_count. Higher = more imbalanced."),
        ("balance", "Normalized Shannon entropy (0–1). Closer to 1 = more evenly distributed."),
        ("entropy_bin", "Raw Shannon entropy in bits."),
        ("skewness", "Asymmetry of a numeric distribution. 0 = symmetric."),
        ("kurtosis", "Tail heaviness / outlier tendency. 0 = normal-like."),
        ("cv", "Relative spread (std / |mean|)."),
        ("iqr", "Middle 50% spread (Q3 − Q1)."),
    ]
    dt = "".join(f"<dt><code>{_esc(k)}</code></dt><dd>{_esc(v)}</dd>"
                 for k, v in items)
    return f'<dl class="stat-decoder">{dt}</dl>'


def render_missingness(cfg: ReportConfig, art: Artifacts) -> str:
    """🕳️ Missingness story."""
    body = [
        '<h2>🕳️ Missingness story</h2>',
        '<p>Missingness was assessed per variable and globally. Variables with '
        'high missingness should be interpreted cautiously, especially if used '
        'in association screening or models.</p>',
    ]
    if art.missingness_summary is None and not art.missingness_figures:
        body.append(warning_box("No saved missingness artifacts were found."))
        return f'<section class="report-section">{"".join(body)}</section>'

    if art.missingness_summary is not None and not art.missingness_summary.empty:
        body.append("<h3>Missingness per variable</h3>")
        body.append(table_to_html(
            art.missingness_summary,
            row_class_fn=lambda r: classify_missing(
                r.get("missing_pct", r.get("pct_missing")), cfg.missing),
            max_rows=200,
        ))

    if art.top_missing is not None and not art.top_missing.empty:
        body.append("<h3>Top missing</h3>")
        body.append(table_to_html(
            art.top_missing,
            row_class_fn=lambda r: classify_missing(
                r.get("missing_pct", r.get("pct_missing")), cfg.missing),
        ))

    if art.missingness_figures:
        body.append("<h3>Patterns</h3>")
        body.append(svg_grid(art.missingness_figures))

    return f'<section class="report-section">{"".join(body)}</section>'


def render_eda(cfg: ReportConfig, art: Artifacts) -> str:
    """🔍 EDA story — per-target, color-coded, with badges."""
    body = ['<h2>🔍 Exploratory association screening (EDA)</h2>']
    if art.associations is None or art.associations.empty:
        body.append(warning_box("No EDA associations table was found."))
        return f'<section class="report-section">{"".join(body)}</section>'

    body.append(
        '<p>Each predictor was screened against each target using a '
        'test matched to both outcome and predictor types (binary, continuous, '
        'ordinal, or nominal). '
        'p-values are corrected per target using Benjamini–Hochberg FDR.</p>'
    )

    df = art.associations.copy()
    # Ensure expected columns exist
    for col in ["target", "predictor", "kind", "test", "effect_label",
                "effect", "p", "p_fdr", "n_used"]:
        if col not in df.columns:
            df[col] = np.nan

    targets_in_data = list(df["target"].dropna().unique())
    # Render in the order user listed, then any extras
    order = [t for t in cfg.targets if t in targets_in_data]
    order += [t for t in targets_in_data if t not in order]

    for target in order:
        sub = df[df["target"] == target].copy()
        if sub.empty:
            continue

        body.append(f"<h3>🎯 Target: <code>{_esc(target)}</code></h3>")

        # Sort by FDR ascending, then |effect| descending
        sub["_p_num"] = sub["p_fdr"].apply(_coerce_p)
        sub["_eff_abs"] = sub["effect"].apply(
            lambda v: abs(_coerce_float(v)) if _coerce_float(v) is not None else -1)
        sub = sub.sort_values(["_p_num", "_eff_abs"],
                              ascending=[True, False], na_position="last")

        # Strength badge + significance row class
        def _row_class(r):
            tier = _strength_tier(r.get("effect"), "corr", cfg.effect)
            sig = classify_significance(
                r.get("p"), r.get("p_fdr"),
                fdr_alpha=cfg.fdr_alpha, nominal_alpha=cfg.nominal_alpha)
            return sig  # row tint via significance only; strength via badge column

        sub["strength"] = sub["effect"].apply(
            lambda v: effect_badge(v, "corr", cfg.effect))
        sub["significance"] = sub.apply(
            lambda r: {"sig-fdr": "🟢 FDR-sig",
                       "sig-nominal": "🟡 nominal",
                       "sig-none": "⚪ ns"}[
                classify_significance(
                    r.get("p"), r.get("p_fdr"),
                    fdr_alpha=cfg.fdr_alpha, nominal_alpha=cfg.nominal_alpha)],
            axis=1)

        # Mini summary line
        n_fdr = (sub["significance"] == "🟢 FDR-sig").sum()
        if n_fdr > 0:
            top = sub.iloc[0]
            line = (f"For target <code>{_esc(target)}</code>, "
                    f"<strong>{n_fdr}</strong> predictor"
                    f"{'s' if n_fdr > 1 else ''} survived FDR correction. "
                    f"Strongest exploratory association: "
                    f"<code>{_esc(top['predictor'])}</code> "
                    f"({_esc(top['effect_label'])} = {_esc(top['effect'])}, "
                    f"FDR p = {_esc(top['p_fdr'])}).")
        else:
            line = (f"No predictors survived FDR correction for "
                    f"<code>{_esc(target)}</code>. Any nominal findings below "
                    f"are exploratory only.")
        body.append(f"<p>{line}</p>")

        display_cols = ["predictor", "kind", "test", "effect_label", "effect",
                        "p", "p_fdr", "significance", "strength", "n_used"]
        display_cols = [c for c in display_cols if c in sub.columns]
        interp_df = sub.copy()
        sub["p"] = sub["p"].apply(human_p)
        sub["p_fdr"] = sub["p_fdr"].apply(human_p)
        body.append(table_to_html(
            sub[display_cols], row_class_fn=_row_class,
            # 'strength' is a pre-built <span> badge — don't HTML-escape it.
            safe_html_cols=("strength",),
        ))
        body.append(_render_eda_interpretation(target, interp_df, cfg))

        # Figures for this target
        figs = [p for p in art.eda_figures if p.stem.startswith(f"{target}__")]
        if figs:
            body.append(details_block(
                f"🖼️ EDA figures for {target} ({len(figs)})",
                svg_grid(figs)))

    return f'<section class="report-section">{"".join(body)}</section>'


def render_inferential(cfg: ReportConfig, art: Artifacts) -> str:
    """🧮 Multivariable / inferential modelling."""
    body = ['<h2>🧮 Multivariable modelling</h2>']
    if not art.inferential_multivariable and (art.inferential_summary is None
                                               or art.inferential_summary.empty):
        body.append(warning_box("No multivariable model artifacts were found."))
        return f'<section class="report-section">{"".join(body)}</section>'

    body.append(
        '<p>A multivariable logistic regression model was fitted for each '
        'target. Predictors were encoded according to schema type, '
        'continuous/count variables were standardized, nominal variables '
        'were one-hot encoded, and high-VIF predictors were pruned. '
        'Multiple imputation was pooled with Rubin\u2019s rules.</p>'
    )

    targets = list(art.inferential_multivariable.keys())
    # Reorder per user list
    targets = ([t for t in cfg.targets if t in targets]
               + [t for t in targets if t not in cfg.targets])

    for target in targets:
        tbl = art.inferential_multivariable[target].copy()
        body.append(f"<h3>🎯 Target: <code>{_esc(target)}</code></h3>")

        # Forest plot
        forest = [p for p in art.inferential_figures
                  if p.stem == f"{target}__forest"
                  or p.stem.startswith(f"{target}__forest")]
        if forest:
            body.append(svg_grid(forest))

        # VIF (collapsed)
        if target in art.inferential_vif:
            body.append(details_block(
                "🔢 VIF diagnostics",
                table_to_html(art.inferential_vif[target])))

        # Multivariable table
        # Normalise column names that may vary across pipeline versions
        col_or  = _first_present(tbl, ["or", "OR", "odds_ratio"])
        col_lo  = _first_present(tbl, ["or_ci_lo", "ci_lo", "lower"])
        col_hi  = _first_present(tbl, ["or_ci_hi", "ci_hi", "upper"])
        col_p   = _first_present(tbl, ["p", "pvalue", "p_value"])
        col_pred = _first_present(tbl, ["predictor_col", "predictor", "term"])

        def _row_cls(r):
            if col_or and col_lo and col_hi:
                return classify_or_direction(r.get(col_or), r.get(col_lo), r.get(col_hi))
            return ""

        # Sort by p-value, then effect strength (|log OR|)
        if col_p:
            tbl["_p_num"] = tbl[col_p].apply(_coerce_p)
        if col_or:
            tbl["_eff_abs"] = tbl[col_or].apply(
                lambda v: abs(math.log(v)) if (o := _coerce_float(v)) and o > 0 else -1)
        sort_cols = [c for c in ("_p_num", "_eff_abs", col_pred) if c and c in tbl.columns]
        if sort_cols:
            tbl = tbl.sort_values(
                sort_cols,
                ascending=[True, False, True][: len(sort_cols)],
                na_position="last",
            )
        tbl = tbl.drop(columns=[c for c in ("_p_num", "_eff_abs") if c in tbl.columns])

        # Pre-format p / df for display only
        if col_p and col_p in tbl.columns:
            tbl[col_p] = tbl[col_p].apply(human_p)
        if "df" in tbl.columns:
            tbl["df"] = tbl["df"].apply(human_pool_df)

        body.append(table_to_html(tbl, row_class_fn=_row_cls))

        # Plain-English interpretation
        body.append(_render_inferential_interpretation(
            target, tbl, col_pred, col_or, col_lo, col_hi, col_p))

    return f'<section class="report-section">{"".join(body)}</section>'


def _to_int_or_none(x: Any) -> int | None:
    v = _coerce_float(x)
    return int(v) if v is not None else None


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _render_inferential_interpretation(target: str, tbl: pd.DataFrame,
                                       col_pred: str | None, col_or: str | None,
                                       col_lo: str | None, col_hi: str | None,
                                       col_p: str | None) -> str:
    if not all([col_pred, col_or, col_lo, col_hi]):
        return ""
    lines = []
    for _, r in tbl.iterrows():
        o  = _coerce_float(r.get(col_or))
        lo = _coerce_float(r.get(col_lo))
        hi = _coerce_float(r.get(col_hi))
        if o is None or lo is None or hi is None:
            continue
        pred = _esc(r.get(col_pred))
        p_str = _esc(r.get(col_p)) if col_p else ""
        if lo > 1.0:
            lines.append(f"<li>🔴 <code>{pred}</code> was associated with "
                         f"<strong>higher</strong> odds of <code>{_esc(target)}</code> "
                         f"(OR={o:.2f}, 95% CI {lo:.2f}–{hi:.2f}"
                         + (f", p={p_str}" if p_str else "") + ").</li>")
        elif hi < 1.0:
            lines.append(f"<li>🔵 <code>{pred}</code> was associated with "
                         f"<strong>lower</strong> odds of <code>{_esc(target)}</code> "
                         f"(OR={o:.2f}, 95% CI {lo:.2f}–{hi:.2f}"
                         + (f", p={p_str}" if p_str else "") + ").</li>")
        else:
            lines.append(f"<li>⚪ <code>{pred}</code> did not show a stable "
                         f"independent association with <code>{_esc(target)}</code> "
                         f"(OR={o:.2f}, 95% CI {lo:.2f}–{hi:.2f}; CI crosses 1).</li>")
    if not lines:
        return ""
    return "<h4>Interpretation</h4><ul>" + "".join(lines) + "</ul>"


def _eda_direction_phrase(r: pd.Series, target: str) -> str:
    """Plain-language wording for one EDA association row (Spearman sign from rho)."""
    pred = _esc(r.get("predictor"))
    test = str(r.get("test") or "")
    eff = _coerce_float(r.get("effect"))
    if test == "spearman" and eff is not None:
        if eff > 0:
            return (f"Higher <code>{pred}</code> is associated with a higher rate of "
                    f"<code>{_esc(target)}</code>")
        if eff < 0:
            return (f"Higher <code>{pred}</code> is associated with a lower rate of "
                    f"<code>{_esc(target)}</code>")
    return (f"<code>{pred}</code> is associated with "
            f"<code>{_esc(target)}</code> (see figure for group differences)")


def _render_eda_interpretation(target: str, sub: pd.DataFrame,
                               cfg: ReportConfig) -> str:
    """Plain-English bullets for univariate EDA rows (one target)."""
    lines: list[str] = []
    for _, r in sub.iterrows():
        test = str(r.get("test") or "")
        if test == "skip":
            continue
        pred = _esc(r.get("predictor"))
        eff_label = _esc(r.get("effect_label"))
        eff = _esc(r.get("effect"))
        sig = classify_significance(
            r.get("p"), r.get("p_fdr"),
            fdr_alpha=cfg.fdr_alpha, nominal_alpha=cfg.nominal_alpha)
        tier = _strength_tier(r.get("effect"), "corr", cfg.effect)
        _, strength_word, strength_hint = _STRENGTH_WORDING[tier]
        p_fdr_str = _esc(human_p(r.get("p_fdr")))
        p_str = _esc(human_p(r.get("p")))
        phrase = _eda_direction_phrase(r, target)

        if sig == "sig-fdr":
            sig_note = f"FDR p = {p_fdr_str}"
            bullet = "🟢"
        elif sig == "sig-nominal":
            sig_note = f"nominal p = {p_str}; FDR p = {p_fdr_str} (exploratory only)"
            bullet = "🟡"
        else:
            sig_note = f"FDR p = {p_fdr_str}"
            bullet = "⚪"

        if sig == "sig-none":
            lines.append(
                f"<li>{bullet} <code>{pred}</code>: no clear marginal association "
                f"({sig_note}; {eff_label} = {eff}).</li>")
            continue

        lines.append(
            f"<li>{bullet} {phrase} "
            f"({eff_label} = {eff}, {sig_note}; <em>{strength_word}</em> effect). "
            f"{strength_hint}</li>")

    if not lines:
        return ""
    caveat = (
        "<li><em>Univariate screening only — effects are not adjusted for other "
        "predictors. Use the multivariable section to judge independent "
        "associations.</em></li>"
    )
    return "<h4>Interpretation</h4><ul>" + "".join(lines) + caveat + "</ul>"


def render_stats_decoder() -> str:
    """🧠 Plain-language statistics primer for clinicians."""
    items = [
        ("p-value",
         "How surprising this result would be if there were truly no "
         "association. Small p-value = the observed pattern is unlikely under "
         "the no-association assumption. It does <strong>not</strong> measure "
         "clinical importance."),
        ("FDR p-value",
         "p-value corrected for multiple testing. Use this for EDA "
         "conclusions when many predictors were screened — prefer FDR p over "
         "raw p."),
        ("Effect size",
         "How <em>large</em> the association is. Separate from p-value: a "
         "tiny effect can have a tiny p in a huge sample, and a huge effect "
         "can have a large p in a small sample."),
        ("Spearman ρ",
         "Whether higher values of one variable move with higher (positive) "
         "or lower (negative) values of another. −1 = strong inverse, 0 = no "
         "monotonic relationship, +1 = strong positive."),
        ("Rank-biserial r",
         "Difference in ranks between two groups. Useful when comparing a "
         "numeric predictor between a binary outcome's two groups."),
        ("Cramér's V",
         "Strength of association between categorical variables. 0 = none, "
         "1 = perfect association."),
        ("Kruskal–Wallis",
         "Whether a numeric predictor differs across multiple outcome "
         "groups (non-parametric alternative to one-way ANOVA)."),
        ("ε² (epsilon-squared)",
         "Effect size for Kruskal–Wallis; proportion of variance explained "
         "by group membership."),
        ("Odds ratio (OR)",
         "How many times higher or lower the odds of the outcome are. "
         "OR = 1: no difference. OR > 1: higher odds. OR < 1: lower odds."),
        ("95% confidence interval (CI)",
         "Range of plausible values for the estimate. Narrow CI = precise. "
         "Wide CI = imprecise. For OR, if CI crosses 1, do <strong>not</strong> "
         "call the result a stable independent association."),
        ("n_used",
         "How many patients actually contributed to a specific test. Low "
         "n_used means the result can wobble — interpret cautiously."),
        ("Unstable estimate",
         "The data are not strong enough to pin down the true effect. The "
         "observed association may change substantially if a few patients "
         "were added, removed, recoded, or if missing values were handled "
         "differently."),
    ]
    dt = "".join(f"<dt>{_esc(k)}</dt><dd>{v}</dd>" for k, v in items)

    warnings = [
        "very wide 95% CI",
        "OR CI crosses 1",
        "p-value not significant after FDR correction",
        "very small n_used",
        "very few outcome events",
        "rare predictor category",
        "high missingness in predictor or target",
        "heavy multiple imputation",
        "model convergence warnings",
        "extreme OR with huge CI (e.g. OR = 12, CI 0.8–180)",
    ]
    warn_list = "".join(f"<li>{_esc(w)}</li>" for w in warnings)

    plain = (
        "<h4>Plain wording you can re-use</h4>"
        "<ul>"
        "<li>“The direction may be real, but the data are too thin to be confident.”</li>"
        "<li>“The estimate suggests higher odds, but the confidence interval is "
        "wide, so the true effect could be much smaller, absent, or much larger.”</li>"
        "<li>“Because the CI crosses 1, this result should not be treated as a "
        "stable independent association.”</li>"
        "</ul>"
    )

    return (
        '<section class="report-section">'
        '<h2>🧠 Stats decoder for clinicians</h2>'
        '<p>Quick reference for interpreting numbers in the tables above. '
        'Designed for a clinician with minimal stats background.</p>'
        f'<dl class="stat-decoder">{dt}</dl>'
        '<h4>⚠️ Warning signs that an estimate is unstable</h4>'
        f'<ul>{warn_list}</ul>'
        f'{plain}'
        '</section>'
    )


def _dda_row_for_column(art: Artifacts, col: str) -> tuple[pd.Series | None, str]:
    """Return (row, table_label) for a column from any DDA summary table."""
    tables = (
        ("continuous / count", art.dda_continuous),
        ("categorical / ordinal", art.dda_categorical),
        ("binary", art.dda_binary),
        ("datetime", art.dda_datetime),
    )
    for label, tbl in tables:
        if tbl is None or tbl.empty or "column" not in tbl.columns:
            continue
        hit = tbl[tbl["column"].astype(str) == col]
        if not hit.empty:
            return hit.iloc[0], label
    return None, ""


def _figures_for_column(paths: Iterable[Path], col: str) -> list[Path]:
    return sorted(
        p for p in paths
        if p.stem == col or p.stem.startswith(f"{col}__")
    )


def _inferential_matches(term: Any, col: str) -> bool:
    t = str(term or "")
    return t == col or t.startswith(f"{col}_")


def _onehot_modeled_level(term: str, base: str) -> str | None:
    """Category name encoded by a one-hot column ``base_<level>``."""
    prefix = f"{base}_"
    if term.startswith(prefix):
        return term[len(prefix):]
    return None


def _invert_or_ci(o: float, lo: float, hi: float) -> tuple[float, float, float]:
    """OR and CI for the reference level when the model reports the other level."""
    if o <= 0 or lo <= 0 or hi <= 0:
        return (np.nan, np.nan, np.nan)
    return (1.0 / o, 1.0 / hi, 1.0 / lo)


def _or_ci_phrase(o: float, lo: float, hi: float) -> str:
    if not all(np.isfinite([o, lo, hi])):
        return "—"
    return f"{o:.2f} ({lo:.2f}–{hi:.2f})"


def _infer_focus_reference(
    col: str,
    modeled_level: str,
    cfg_ref: str | None,
    dda_row: pd.Series | None,
) -> str:
    if cfg_ref:
        return cfg_ref
    if dda_row is not None:
        for key in ("first_mode", "second_mode"):
            lv = dda_row.get(key)
            if lv is not None and not pd.isna(lv) and str(lv) != modeled_level:
                return str(lv)
    return "reference"


def _render_focus_dda_routes(dda_row: pd.Series, highlight: str | None) -> str:
    """Cohort mix for a binary/categorical focus predictor (two rows, one highlighted)."""
    rows: list[tuple[str, Any]] = []
    for lk, pk in (("first_mode", "first_mode_pct"), ("second_mode", "second_mode_pct")):
        if lk not in dda_row.index:
            continue
        lv = dda_row.get(lk)
        if lv is None or (isinstance(lv, float) and not np.isfinite(lv)) or pd.isna(lv):
            continue
        pct = dda_row.get(pk)
        pct_s = f"{float(pct):.1f}%" if _coerce_float(pct) is not None else "—"
        rows.append((str(lv), pct_s))
    if len(rows) < 2:
        return ""
    body_rows = []
    for lv, pct_s in rows:
        hl = ' class="focus-route-highlight"' if highlight and lv == highlight else ""
        body_rows.append(
            f"<tr{hl}><td><strong>{_esc(lv)}</strong></td>"
            f'<td class="mono">{_esc(pct_s)} of cohort</td></tr>')
    return (
        '<div class="focus-dda-routes">'
        '<table class="focus-route-table">'
        "<thead><tr><th>Route</th><th>Share in data</th></tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _render_focus_route_or_card(
    target: str,
    predictor: str,
    modeled_level: str,
    reference: str,
    o: float,
    lo: float,
    hi: float,
    p: Any,
) -> str:
    """Two-row adjusted OR table: reference (highlight) + modeled level (from regression)."""
    o_ref, lo_ref, hi_ref = _invert_or_ci(o, lo, hi)
    p_str = _esc(human_p(p))

    def _row(level: str, or_txt: str, note: str, p_cell: str, highlight: bool) -> str:
        hl = ' class="focus-route-highlight"' if highlight else ""
        return (
            f"<tr{hl}><td><strong>{_esc(level)}</strong></td>"
            f'<td class="mono">{or_txt}</td><td>{note}</td>'
            f"<td>{p_cell}</td></tr>"
        )

    rows_html = [
        _row(
            reference,
            _or_ci_phrase(o_ref, lo_ref, hi_ref),
            f"vs <strong>{_esc(modeled_level)}</strong>",
            p_str,
            highlight=True,
        ),
        _row(
            modeled_level,
            _or_ci_phrase(o, lo, hi),
            f"vs <strong>{_esc(reference)}</strong>",
            p_str,
            highlight=False,
        ),
    ]

    return (
        '<div class="focus-route-card">'
        f'<p class="focus-route-title">Adjusted OR · <code>{_esc(target)}</code> '
        f"(copy-ready)</p>"
        '<table class="focus-route-table">'
        "<thead><tr><th>Biopsy route</th><th>Adj. OR (95% CI)</th>"
        "<th>Comparison</th><th>p</th></tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
        '<p class="focus-route-note">'
        f"Regression reference: <strong>{_esc(reference)}</strong>. "
        f"Highlighted row is the OR for your primary route; the other row matches "
        f"<code>{_esc(predictor)}_{_esc(modeled_level)}</code> in the model output."
        "</p></div>"
    )


def _focus_stat_cards(row: pd.Series, kind: str = "") -> str:
    """Compact metric cards from one DDA summary row."""
    specs: list[tuple[str, str]] = [
        ("N", "n"),
        ("Missing %", "missing_pct"),
        ("Unique levels", "n_unique"),
        ("Dominant value", "first_mode"),
        ("Dominant %", "first_mode_pct"),
        ("Second value", "second_mode"),
        ("Second %", "second_mode_pct"),
        ("Median", "median"),
        ("Mean", "mean"),
        ("IQR", "iqr"),
        ("Balance", "balance"),
    ]
    cards = []
    for label, key in specs:
        if key not in row.index:
            continue
        val = row.get(key)
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            continue
        if pd.isna(val):
            continue
        disp = f"{float(val):.3g}" if isinstance(val, (int, float, np.floating)) else str(val)
        cards.append(
            '<div class="focus-stat-card">'
            f'<div class="label">{_esc(label)}</div>'
            f'<div class="value">{_esc(disp)}</div>'
            '</div>'
        )
    if not cards:
        return ""
    kind_badge = f' <span class="badge kind">{_esc(kind)}</span>' if kind else ""
    return f'<div class="focus-stat-grid">{"".join(cards)}</div>{kind_badge}'


def render_focus_predictor(cfg: ReportConfig, art: Artifacts) -> str:
    """Spotlight one predictor: DDA, EDA, and multivariable stats + figures."""
    col = (cfg.focus_predictor or "").strip()
    if not col:
        return ""

    ref_level = (cfg.focus_reference_level or "").strip() or None
    hero_extra = ""
    if ref_level:
        hero_extra = (
            f'<p class="focus-route-note">Primary route: '
            f"<strong>{_esc(ref_level)}</strong> "
            f"(highlighted below; adjusted ORs for both routes).</p>"
        )
    body: list[str] = [
        '<div class="focus-section">',
        '<h3>🔬 Variable of interest</h3>',
        '<div class="focus-hero">'
        f'<h4>Primary focus: <code>{_esc(col)}</code></h4>'
        '<p>Descriptive profile, univariate screening (EDA), and adjusted '
        'multivariable results for this variable — all in one place.</p>',
        hero_extra,
        '</div>',
    ]

    if art.schema_summary is not None and not art.schema_summary.empty:
        sc = art.schema_summary
        name_col = "column" if "column" in sc.columns else "name" if "name" in sc.columns else None
        if name_col:
            hit = sc[sc[name_col].astype(str) == col]
            if not hit.empty:
                show = [c for c in ("kind", "keep", "ordered_levels", "note") if c in hit.columns]
                if show:
                    body.append("<h4>Schema</h4>")
                    body.append(table_to_html(hit[show].head(1)))

    dda_row, dda_label = _dda_row_for_column(art, col)
    kind = str(dda_row.get("kind", "")).strip() if dda_row is not None else ""
    if dda_row is not None:
        dist_label = kind or dda_label
        body.append(f"<h4>📊 Distribution ({_esc(dist_label)})</h4>")
        cards_html = _focus_stat_cards(dda_row, kind)
        if cards_html:
            body.append(cards_html)
        routes_html = _render_focus_dda_routes(dda_row, ref_level)
        if routes_html:
            body.append(routes_html)
        body.append(table_to_html(pd.DataFrame([dda_row])))

    # DDA figure: first SVG from output/dda/figures/ matching focus_predictor
    # (stem == col or col__*, e.g. biopsy_type__bar.svg from dda.py). Display
    # size is .focus-section .focus-figure-hero in _CSS above.
    dda_figs = _figures_for_column(art.dda_figures, col)
    by_year = next((p for p in dda_figs if p.stem == f"{col}__bar_by_year"), None)
    hero = by_year if by_year is not None else (dda_figs[0] if dda_figs else None)
    if hero is not None:
        body.append("<h4>Distribution figure</h4>")
        hero_img = _figure_img_html(hero)
        if hero_img:
            body.append(
                '<div class="focus-figure-hero figure-card">'
                f'{hero_img}'
                f'<div class="caption">{_esc(hero.stem)}</div>'
                '</div>'
            )
        if by_year is not None:
            yr = (cfg.year_column or "year").strip()
            body.append(
                '<p class="figure-note">Top: cohort-wide counts. Bottom: category shares '
                f"within each {yr} (row-normalised). Descriptive only — not adjusted for "
                "confounding. Optional χ² p-value tests marginal association with calendar "
                f"year when expected counts ≥ 5.</p>"
            )

    if art.missingness_summary is not None and not art.missingness_summary.empty:
        ms = art.missingness_summary
        ncol = "column" if "column" in ms.columns else "variable" if "variable" in ms.columns else None
        if ncol:
            mhit = ms[ms[ncol].astype(str) == col]
            if not mhit.empty:
                body.append("<h4>🕳️ Missingness</h4>")
                body.append(table_to_html(mhit))

    targets = [t for t in cfg.targets if t]
    if not targets and art.associations is not None and "target" in art.associations.columns:
        targets = list(art.associations["target"].dropna().unique())

    eda_all = (
        art.associations[art.associations["predictor"].astype(str) == col].copy()
        if art.associations is not None and not art.associations.empty
        and "predictor" in art.associations.columns
        else None
    )
    if eda_all is not None and not eda_all.empty:
        eda_all["_p_num"] = eda_all["p_fdr"].apply(_coerce_p)
        eda_all["_eff_abs"] = eda_all["effect"].apply(
            lambda v: abs(_coerce_float(v)) if _coerce_float(v) is not None else -1)
        eda_all = eda_all.sort_values(
            ["_p_num", "_eff_abs"], ascending=[True, False], na_position="last")

    for target in targets:
        body.append(f'<div class="focus-target-block">')
        body.append(f'<h4>🎯 Outcome: <code>{_esc(target)}</code></h4>')

        if eda_all is not None:
            sub = eda_all[eda_all["target"].astype(str) == target]
            if not sub.empty:
                body.append("<p><strong>Univariate (EDA)</strong></p>")
                show = [c for c in (
                    "test", "effect_label", "effect", "p", "p_fdr",
                    "fdr_significant", "n_used",
                ) if c in sub.columns]
                disp = sub[show].copy()
                if "p" in disp.columns:
                    disp["p"] = disp["p"].apply(human_p)
                if "p_fdr" in disp.columns:
                    disp["p_fdr"] = disp["p_fdr"].apply(human_p)
                body.append(table_to_html(disp))
                body.append(_render_eda_interpretation(
                    target, sub, cfg))

        eda_fig = next(
            (p for p in art.eda_figures if p.stem == f"{target}__{col}"), None)
        if eda_fig is not None:
            body.append("<p><strong>EDA figure</strong></p>")
            body.append(_focus_eda_figure(eda_fig))

        if target in art.inferential_multivariable:
            tbl = art.inferential_multivariable[target].copy()
            col_pred = _first_present(tbl, ["predictor_col", "predictor", "term"])
            if col_pred:
                hit = tbl[tbl[col_pred].astype(str).apply(
                    lambda t: _inferential_matches(t, col))]
                if not hit.empty:
                    body.append("<p><strong>Multivariable (adjusted)</strong></p>")
                    col_or = _first_present(hit, ["or", "OR"])
                    col_lo = _first_present(hit, ["or_ci_lo", "ci_lo"])
                    col_hi = _first_present(hit, ["or_ci_hi", "ci_hi"])
                    col_p = _first_present(hit, ["p", "pvalue"])
                    showed_route_card = False
                    if (
                        len(hit) == 1
                        and col_pred
                        and col_or and col_lo and col_hi
                    ):
                        r0 = hit.iloc[0]
                        term = str(r0.get(col_pred))
                        modeled = _onehot_modeled_level(term, col)
                        o = _coerce_float(r0.get(col_or))
                        lo = _coerce_float(r0.get(col_lo))
                        hi = _coerce_float(r0.get(col_hi))
                        if modeled and o is not None and lo is not None and hi is not None:
                            reference = _infer_focus_reference(
                                col, modeled, ref_level, dda_row)
                            body.append(_render_focus_route_or_card(
                                target, col, modeled, reference, o, lo, hi,
                                r0.get(col_p) if col_p else None,
                            ))
                            showed_route_card = True
                    if not showed_route_card:
                        show = [c for c in (
                            col_pred, "or", "or_ci_lo", "or_ci_hi", "p", "coef", "se",
                        ) if c and c in hit.columns]
                        show = list(dict.fromkeys(show))
                        disp = hit[show].copy()
                        if "p" in disp.columns:
                            disp["p"] = disp["p"].apply(human_p)
                        body.append(table_to_html(disp))
                        body.append(_render_inferential_interpretation(
                            target, disp, col_pred, col_or, col_lo, col_hi, col_p))

        body.append("</div>")

    if not (
        (eda_all is not None and not eda_all.empty)
        or dda_row is not None
        or dda_figs
    ):
        body.append(warning_box(
            f"No artifacts were found for predictor '{col}'. "
            "Check the name matches the schema and that EDA / DDA were run.",
        ))

    body.append('</div>')
    return "".join(body)


def render_final_conclusion(cfg: ReportConfig, art: Artifacts) -> str:
    """🎯 Tiny TL;DR — top hits only, 5–8 bullets max."""
    bullets: list[str] = []

    if art.associations is not None and not art.associations.empty:
        df = art.associations.copy()
        df["_p_num"] = df.get("p_fdr").apply(_coerce_p) if "p_fdr" in df.columns else None
        df["_eff_abs"] = df.get("effect").apply(
            lambda v: abs(_coerce_float(v)) if _coerce_float(v) is not None else -1
        ) if "effect" in df.columns else -1

        fdr_hits = df[(df["_p_num"].notna()) & (df["_p_num"] < cfg.fdr_alpha)] \
            .sort_values(["_p_num", "_eff_abs"], ascending=[True, False])
        for _, r in fdr_hits.head(4).iterrows():
            tier = _strength_tier(r.get("effect"), "corr", cfg.effect)
            emoji, label, _ = _STRENGTH_WORDING[tier]
            bullets.append(
                f"{emoji} <code>{_esc(r['predictor'])}</code> showed a "
                f"<strong>{label}</strong> association with "
                f"<code>{_esc(r['target'])}</code> "
                f"({_esc(r.get('effect_label'))}={_esc(r.get('effect'))}, "
                f"FDR p={human_p(r.get('p_fdr'))})."
            )

    for target, tbl in art.inferential_multivariable.items():
        col_or = _first_present(tbl, ["or", "OR", "odds_ratio"])
        col_lo = _first_present(tbl, ["or_ci_lo", "ci_lo", "lower"])
        col_hi = _first_present(tbl, ["or_ci_hi", "ci_hi", "upper"])
        col_pred = _first_present(tbl, ["predictor_col", "predictor", "term"])
        if not all([col_or, col_lo, col_hi, col_pred]):
            continue
        for _, r in tbl.iterrows():
            o = _coerce_float(r.get(col_or))
            lo = _coerce_float(r.get(col_lo))
            hi = _coerce_float(r.get(col_hi))
            if None in (o, lo, hi):
                continue
            if lo > 1.0:
                bullets.append(
                    f"🔴 In multivariable analysis, <code>{_esc(r[col_pred])}</code> "
                    f"was associated with <strong>higher</strong> odds of "
                    f"<code>{_esc(target)}</code> (OR={o:.2f}, 95% CI {lo:.2f}–{hi:.2f}).")
            elif hi < 1.0:
                bullets.append(
                    f"🔵 In multivariable analysis, <code>{_esc(r[col_pred])}</code> "
                    f"was associated with <strong>lower</strong> odds of "
                    f"<code>{_esc(target)}</code> (OR={o:.2f}, 95% CI {lo:.2f}–{hi:.2f}).")

    if art.associations is not None and not art.associations.empty:
        for target in cfg.targets:
            sub = art.associations[art.associations.get("target") == target]
            if sub.empty:
                continue
            ps = sub.get("p_fdr").apply(_coerce_p)
            if not ((ps.notna()) & (ps < cfg.fdr_alpha)).any():
                bullets.append(
                    f"⚪ No FDR-significant association was found for "
                    f"<code>{_esc(target)}</code>.")

    cautions = []
    if art.missingness_summary is not None and not art.missingness_summary.empty:
        miss_col = ("missing_pct" if "missing_pct" in art.missingness_summary.columns
                    else "pct_missing" if "pct_missing" in art.missingness_summary.columns
                    else None)
        if miss_col:
            high = art.missingness_summary[
                art.missingness_summary[miss_col].apply(_coerce_float)
                .apply(lambda v: v is not None and v >= cfg.missing.high)]
            if not high.empty:
                cautions.append(
                    f"🚨 {len(high)} variable(s) had &gt;{cfg.missing.high:.0f}% "
                    f"missingness — interpret any analyses using them with caution.")
    bullets.extend(cautions)

    bullets = bullets[:8]
    if not bullets:
        bullets = ["<em>(No findings were detected from the supplied artifacts.)</em>"]

    lis = "".join(f"<li>{b}</li>" for b in bullets)
    focus = render_focus_predictor(cfg, art)
    return (
        '<section class="report-section">'
        '<h2>🎯 Final conclusion</h2>'
        '<p>The bottom line, distilled. For full details refer to the EDA and '
        'multivariable tables above.</p>'
        f'<ul class="tldr-list">{lis}</ul>'
        f'{focus}'
        '</section>'
    )


def render_appendix(cfg: ReportConfig, art: Artifacts) -> str:
    """📎 Appendix — warnings, artifact paths, anything not embedded earlier."""
    body = ['<h2>📎 Appendix</h2>']

    if art.warnings:
        body.append("<h3>Warnings during artifact load</h3>")
        body.append("<ul>" + "".join(f"<li>{_esc(w)}</li>" for w in art.warnings)
                    + "</ul>")

    # Full inferential summary (if not already shown)
    if art.inferential_summary is not None and not art.inferential_summary.empty:
        body.append(details_block(
            "🧾 Full inferential summary",
            table_to_html(art.inferential_summary)))

    # Full VIF tables collapsed
    if art.inferential_vif:
        for target, vif in art.inferential_vif.items():
            body.append(details_block(
                f"🔢 VIF — {target}", table_to_html(vif)))

    # Artifact path listing
    paths = sorted(p.relative_to(cfg.output_root)
                   for p in cfg.output_root.rglob("*")
                   if p.is_file() and p.suffix.lower() in {".csv", ".svg"})
    if paths:
        lst = "".join(f"<li><code>{_esc(p)}</code></li>" for p in paths)
        body.append(details_block(
            f"📂 Artifact files used ({len(paths)})", f"<ul>{lst}</ul>"))

    return f'<section class="report-section">{"".join(body)}</section>'


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def build_report(cfg: ReportConfig) -> str:
    """Assemble the full HTML document and return it as a string."""
    art = load_artifacts(cfg)

    sections = [
        render_header(cfg, art),
        render_cleaning(cfg, art),
        render_schema(cfg, art),
        render_dda(cfg, art),
        render_missingness(cfg, art),
        render_eda(cfg, art),
        render_inferential(cfg, art),
        render_stats_decoder(),
        render_final_conclusion(cfg, art),
        render_appendix(cfg, art),
    ]

    footer = (
        f'<div class="footer">Generated {datetime.now().isoformat(timespec="seconds")} '
        f'· output root: <code>{_esc(cfg.output_root)}</code></div>'
    )

    body = "".join(sections) + footer
    return _wrap_html(cfg.title, body)


def _wrap_html(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head>'
        '<meta charset="utf-8">'
        f'<title>{_esc(title)}</title>'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<style>{_CSS}</style>'
        '</head><body><div class="container">'
        f'{body}'
        '</div></body></html>'
    )


def write_html(html: str, out_path: Path) -> Path:
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", required=True, type=Path,
                   help="Root directory containing dda/, eda/, inferential/, etc.")
    p.add_argument("--targets", nargs="*", default=[],
                   help="Names of target columns (used for ordering / role tags).")
    p.add_argument("--schema", type=Path, default=None,
                   help="Optional schema CSV or JSON to render the schema section.")
    p.add_argument("--title", default="Research Data Analysis Report")
    p.add_argument("--author", default="")
    p.add_argument("--focus-predictor", default=None,
                   help="Column name to spotlight in the final conclusion section.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output HTML path. Defaults to <output-root>/report/report.html")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = ReportConfig(
        output_root=args.output_root.expanduser().resolve(),
        title=args.title,
        author=args.author,
        targets=tuple(args.targets),
        schema_path=args.schema.expanduser().resolve() if args.schema else None,
        focus_predictor=args.focus_predictor,
    )
    out_path = args.out or (cfg.output_root / "report" / "report.html")
    html = build_report(cfg)
    written = write_html(html, out_path)
    print(f"Report written: {written} (self-contained; figures embedded inline)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
