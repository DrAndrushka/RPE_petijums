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
* Folder layout (relative ``figures/`` links) is the default. A single
  self-contained HTML with base64-embedded SVGs can be added later.

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

def _esc(x: Any) -> str:
    """HTML-escape an arbitrary value (None / NaN -> empty string)."""
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    return _html.escape(str(x))


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


def svg_grid(svg_paths: Iterable[Path], rel_base: Path,
             max_n: int | None = None) -> str:
    """Render an HTML grid of SVG figures.

    ``rel_base`` is the directory containing the final ``report.html``; figure
    paths are written relative to it so the report stays portable.
    """
    paths = [p for p in svg_paths if p.exists()]
    if max_n is not None:
        paths = paths[:max_n]
    if not paths:
        return '<p class="muted"><em>(no figures available)</em></p>'
    cards = []
    for p in paths:
        try:
            rel = p.relative_to(rel_base)
        except ValueError:
            # Fall back to absolute file:// URI if not under rel_base
            rel = p.resolve()
        cards.append(
            f'<div class="figure-card">'
            f'<img src="{_esc(str(rel))}" alt="{_esc(p.stem)}" loading="lazy"/>'
            f'<div class="caption">{_esc(p.stem)}</div>'
            f'</div>'
        )
    return f'<div class="figure-grid">{"".join(cards)}</div>'


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
        rel_base = (cfg.output_root / "report").resolve()
        grid_html = svg_grid(art.dda_figures, rel_base)
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
        rel_base = (cfg.output_root / "report").resolve()
        body.append("<h3>Patterns</h3>")
        body.append(svg_grid(art.missingness_figures, rel_base))

    return f'<section class="report-section">{"".join(body)}</section>'


def render_eda(cfg: ReportConfig, art: Artifacts) -> str:
    """🔍 EDA story — per-target, color-coded, with badges."""
    body = ['<h2>🔍 Exploratory association screening (EDA)</h2>']
    if art.associations is None or art.associations.empty:
        body.append(warning_box("No EDA associations table was found."))
        return f'<section class="report-section">{"".join(body)}</section>'

    body.append(
        '<p>Each predictor was screened against each target using a '
        'statistical test chosen from the schema-defined variable type. '
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
        sub["p"] = sub["p"].apply(human_p)
        sub["p_fdr"] = sub["p_fdr"].apply(human_p)
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
        body.append(table_to_html(
            sub[display_cols], row_class_fn=_row_class,
            # 'strength' is a pre-built <span> badge — don't HTML-escape it.
            safe_html_cols=("strength",),
        ))

        # Figures for this target
        figs = [p for p in art.eda_figures if p.stem.startswith(f"{target}__")]
        if figs:
            rel_base = (cfg.output_root / "report").resolve()
            body.append(details_block(
                f"🖼️ EDA figures for {target} ({len(figs)})",
                svg_grid(figs, rel_base)))

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
            rel_base = (cfg.output_root / "report").resolve()
            body.append(svg_grid(forest, rel_base))

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

        # Pre-format p for display only
        if col_p and col_p in tbl.columns:
            tbl[col_p] = tbl[col_p].apply(human_p)

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


def render_final_conclusion(cfg: ReportConfig, art: Artifacts) -> str:
    """🎯 Tiny TL;DR — top hits only, 5–8 bullets max."""
    bullets: list[str] = []

    # 1. FDR-significant EDA findings (top by effect)
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

    # 2. Stable inferential findings (CI excludes 1)
    for target, tbl in art.inferential_multivariable.items():
        col_or  = _first_present(tbl, ["or", "OR", "odds_ratio"])
        col_lo  = _first_present(tbl, ["or_ci_lo", "ci_lo", "lower"])
        col_hi  = _first_present(tbl, ["or_ci_hi", "ci_hi", "upper"])
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

    # 3. Targets with no convincing signal
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

    # 4. Caution about data quality
    cautions = []
    if art.missingness_summary is not None and not art.missingness_summary.empty:
        col = ("missing_pct" if "missing_pct" in art.missingness_summary.columns
               else "pct_missing" if "pct_missing" in art.missingness_summary.columns
               else None)
        if col:
            high = art.missingness_summary[
                art.missingness_summary[col].apply(_coerce_float)
                .apply(lambda v: v is not None and v >= cfg.missing.high)]
            if not high.empty:
                cautions.append(
                    f"🚨 {len(high)} variable(s) had &gt;{cfg.missing.high:.0f}% "
                    f"missingness — interpret any analyses using them with caution."
                )
    bullets.extend(cautions)

    # Cap at 8
    bullets = bullets[:8]
    if not bullets:
        bullets = ["<em>(No findings were detected from the supplied artifacts.)</em>"]

    lis = "".join(f"<li>{b}</li>" for b in bullets)
    return (
        '<section class="report-section">'
        '<h2>🎯 Final tiny conclusion</h2>'
        '<p>The bottom line, distilled. For full details refer to the EDA and '
        'multivariable tables above.</p>'
        f'<ul class="tldr-list">{lis}</ul>'
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
    )
    out_path = args.out or (cfg.output_root / "report" / "report.html")
    html = build_report(cfg)
    written = write_html(html, out_path)
    print(f"Report written: {written}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
