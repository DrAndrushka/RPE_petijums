#============ imports ============
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import sys

from IPython.display import display
import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, pointbiserialr

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

#============ ANSI Escape Codes for formating ============
BOLD = "\033[1m"
BLUE = "\033[38;5;39m"
GREEN = "\033[38;5;28m"
YELLOW = "\033[38;5;214m"
ORANGE = "\033[38;5;202m"
RED = "\033[38;5;196m"
RESET = "\033[0m"


INFERENTIAL_COLS = [
    "predictor",
    "test_name",
    "test_stat",
    "effect_metric",
    "effect_value",
    "ci_low",
    "ci_high",
    "p_value",
    "p_value_fdr",
    "expected_cell_min",
    "expected_cell_lt5_pct",
    "expected_cell_warning",
    "n",
    "missing_pct",
    "status",
    "skipped_reason",
    "fdr_significant",
    "clinically_relevant",
]

_EFFECT_SIZE_THRESHOLDS = {
    "point_biserial_r": {"small": 0.10, "moderate": 0.30, "large": 0.50},
    "rank_biserial_r": {"small": 0.10, "moderate": 0.30, "large": 0.50},
    "cramers_v": {"small": 0.10, "moderate": 0.30, "large": 0.50},
}

_STRONG_SIGNAL_TEXT_COLOR = "#1b5e20"
_SIGNIFICANT_ONLY_TEXT_COLOR = "#c49000"
_CLINICAL_ONLY_TEXT_COLOR = "#d4605c"
_SKIPPED_TEXT_COLOR = "#757575"

INFERENTIAL_SCOPE_STATEMENT = (
    "Inferential scope is predictor -> target association only; "
    "no predictor-to-predictor claims and no causal claims."
)


# Keep the row-color logic outside the class so Styler does not drag the whole project object into deepcopy land.
def _inferential_row_text_style(row: pd.Series) -> list[str]:
    if row["status"] != "ok":
        color = _SKIPPED_TEXT_COLOR
    elif bool(row["fdr_significant"]) and bool(row["clinically_relevant"]):
        color = _STRONG_SIGNAL_TEXT_COLOR
    elif bool(row["fdr_significant"]):
        color = _SIGNIFICANT_ONLY_TEXT_COLOR
    elif bool(row["clinically_relevant"]):
        color = _CLINICAL_ONLY_TEXT_COLOR
    else:
        return [""] * len(row)

    return [f"color: {color};"] * len(row)


#================================================
#=============== Class definition ===============
#================================================
class YlivertainenInferential:

    # Register the model frame, the feature decisions, and the target we are going to test against.
    def __init__(
        self,
        root,
        df_model: pd.DataFrame,
        feature_decisions_df: pd.DataFrame,
        metadata: dict | None = None,
        target_col: str | None = None,
        default_positive_class=None,
        effect_size_thresholds: dict[str, dict[str, float]] | None = None,
    ):
        self.root = Path(root)
        self.INFERENTIAL = df_model.copy()
        self.feature_decisions_df = feature_decisions_df.copy()
        self.task_name = "inferential" if metadata is None else metadata.get("task_name", "inferential")
        self.target = self._resolve_target_col(target_col)
        self.scope_statement = INFERENTIAL_SCOPE_STATEMENT
        self.predictor_type_map = self._build_predictor_type_map()
        self.approved_predictors = self.approved_predictor_columns()
        self.effect_size_thresholds = self._build_effect_size_thresholds(effect_size_thresholds)
        self.default_positive_class = default_positive_class
        (
            self.target_binary,
            self.target_positive_class,
            self.target_negative_class,
        ) = self._encode_binary_series(
            self.INFERENTIAL[self.target],
            positive_class=default_positive_class,
            series_name=self.target,
        )

        self.inferential_results = pd.DataFrame(columns=INFERENTIAL_COLS)
        self.inferential_table = None
        self.run_metadata = {}

    # Give a plain-English overview so `print(project)` works like a briefing before the cut.
    def __str__(self) -> str:
        model_frame_predictors = [col for col in self.INFERENTIAL.columns if col != self.target]
        inferential_ready = len(self.inferential_results) > 0

        status_color = GREEN if inferential_ready else YELLOW
        status_text = "READY ✅" if inferential_ready else "NOT RUN YET ⏳"

        lines = [
            f"{BOLD}{BLUE}{'═' * 64}{RESET}",
            f"{BOLD}{BLUE}🧠  YlivertainenInferential Briefing{RESET}",
            f"{BOLD}{BLUE}{'═' * 64}{RESET}",
            f"{GRAY}📌 Task{RESET}                 {BOLD}{self.task_name}{RESET}",
            f"{GRAY}🎯 Target{RESET}               {ORANGE}{BOLD}{self.target}{RESET}",
            f"{GRAY}✅ Positive class{RESET}       {GREEN}{self.target_positive_class!r}{RESET}",
            f"{GRAY}👥 Rows (n){RESET}             {BOLD}{len(self.INFERENTIAL):,}{RESET}",
            f"{GRAY}🧰 Predictors (model){RESET}   {BOLD}{len(model_frame_predictors)}{RESET}",
            f"{GRAY}🧪 Predictors (approved){RESET} {BOLD}{len(self.approved_predictors)}{RESET}",
            f"{GRAY}📊 Inferential table{RESET}    {status_color}{BOLD}{status_text}{RESET}",
            f"{GRAY}🧭 Scope{RESET}                {self.scope_statement}",
            "",
            f"{BOLD}{YELLOW}Methods{RESET}",
            f"{GRAY}•{RESET} overview() / print(project) -> show this briefing",
            f"{GRAY}•{RESET} approved_predictor_columns() -> list approved predictors",
            f"{GRAY}•{RESET} run_inferential(...) -> build styled inferential table",
            "",
            f"{BOLD}{YELLOW}Stat map{RESET}",
            f"{GRAY}•{RESET} continuous/discrete/time_to_event -> point-biserial r + bootstrap CI",
            f"{GRAY}•{RESET} binary/categorical -> Fisher/chi-square + Cramer's V + bootstrap CI",
            f"{GRAY}•{RESET} valid p-values -> Benjamini-Hochberg FDR",
            f"{GRAY}•{RESET} clinical flag -> moderate-or-larger effect size (default)",
            "",
            f"{BOLD}{YELLOW}Workflow{RESET}",
            f"{GRAY}•{RESET} test predictor -> binary target only",
            f"{GRAY}•{RESET} skip unsupported, sparse, or degenerate predictors",
            f"{GRAY}•{RESET} build one tidy table with one row per predictor",
            f"{GRAY}•{RESET} color row text by FDR signal and clinical effect",
            f"{GRAY}•{RESET} requires explicit positive class lock by default",
            "",
            f"{BOLD}{YELLOW}Output columns{RESET}",
            f"{GRAY}•{RESET} {', '.join(INFERENTIAL_COLS)}",
            f"{BOLD}{BLUE}{'═' * 64}{RESET}",
        ]
        return "\n".join(lines)

    # Handy notebook alias when you want a quick spoken briefing instead of reading the source.
    def overview(self):
        print(self)
        return None

    # Keep only predictors that survived the earlier feature-decision craniotomy.
    def approved_predictor_columns(self, covariates: Sequence[str] | str | None = None) -> list[str]:
        required_cols = {"column_name", "role", "action"}
        if not required_cols.issubset(self.feature_decisions_df.columns):
            raise ValueError("feature_decisions_df must contain column_name, role, and action.")

        approved = self.feature_decisions_df.loc[
            (self.feature_decisions_df["role"] == "predictor")
            & (self.feature_decisions_df["action"] != "drop"),
            "column_name",
        ].dropna()
        approved = [str(col) for col in approved.tolist()]
        approved = [col for col in dict.fromkeys(approved) if col in self.INFERENTIAL.columns and col != self.target]

        if len(approved) == 0:
            raise ValueError("No approved predictors found in df_model.")

        if covariates is None:
            return approved

        requested = [covariates] if isinstance(covariates, str) else list(dict.fromkeys(covariates))
        missing_cols = [col for col in requested if col not in self.INFERENTIAL.columns]
        if missing_cols:
            raise ValueError(f"Requested covariates not found in df_model: {missing_cols}")

        unapproved = [col for col in requested if col not in approved]
        if unapproved:
            raise ValueError(f"Requested covariates are not approved predictors: {unapproved}")

        return requested

    # Run the full inferential screen and return the styled table for the notebook.
    def run_inferential(
        self,
        covariates: Sequence[str] | str | None = None,
        positive_class=None,
        alpha: float = 0.05,
        min_non_missing: int = 10,
        ci_level: float = 0.95,
        n_boot: int = 1000,
        random_state: int = 42,
        clinical_relevance_floor: str = "moderate",
        robust_numeric_fallback: bool = True,
        outlier_iqr_multiplier: float = 1.5,
        enforce_positive_class_lock: bool = True,
        show_table: bool = False,
        export: bool = False,
    ):
        if clinical_relevance_floor not in {"small", "moderate", "large"}:
            raise ValueError("clinical_relevance_floor must be one of: small, moderate, large.")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")
        if not 0 < ci_level < 1:
            raise ValueError("ci_level must be between 0 and 1.")
        if min_non_missing < 3:
            raise ValueError("min_non_missing must be at least 3.")
        if n_boot < 100:
            raise ValueError("n_boot must be at least 100 for usable bootstrap intervals.")
        if outlier_iqr_multiplier <= 0:
            raise ValueError("outlier_iqr_multiplier must be > 0.")

        predictors = self.approved_predictor_columns(covariates)
        resolved_positive_class = self.default_positive_class if positive_class is None else positive_class
        if enforce_positive_class_lock and resolved_positive_class is None:
            raise ValueError(
                "Explicit target polarity required. Pass positive_class in run_inferential() "
                "or set default_positive_class in YlivertainenInferential(...)."
            )
        y_binary, positive_class, negative_class = self._encode_binary_series(
            self.INFERENTIAL[self.target],
            positive_class=resolved_positive_class,
            series_name=self.target,
        )

        rows = []
        for idx, predictor in enumerate(predictors):
            predictor_kind = self._predictor_kind(predictor)
            seed = random_state + idx

            if predictor_kind == "numeric":
                row = self._run_numeric_row(
                    predictor=predictor,
                    y_binary=y_binary,
                    min_non_missing=min_non_missing,
                    n_boot=n_boot,
                    ci_level=ci_level,
                    random_state=seed,
                    robust_numeric_fallback=robust_numeric_fallback,
                    outlier_iqr_multiplier=outlier_iqr_multiplier,
                )
            elif predictor_kind == "categorical":
                row = self._run_categorical_row(
                    predictor=predictor,
                    y_binary=y_binary,
                    min_non_missing=min_non_missing,
                    n_boot=n_boot,
                    ci_level=ci_level,
                    random_state=seed,
                )
            else:
                missing_pct = float(self.INFERENTIAL[predictor].isna().mean() * 100.0)
                non_missing_n = int((self.INFERENTIAL[predictor].notna() & y_binary.notna()).sum())
                row = self._skip_row(
                    predictor=predictor,
                    effect_metric=pd.NA,
                    test_name=pd.NA,
                    test_stat=np.nan,
                    expected_cell_min=np.nan,
                    expected_cell_lt5_pct=np.nan,
                    expected_cell_warning=False,
                    n=non_missing_n,
                    missing_pct=missing_pct,
                    skipped_reason="unsupported_predictor_type",
                )

            rows.append(row)

        inferential_df = pd.DataFrame(rows, columns=INFERENTIAL_COLS)
        inferential_df = self._apply_fdr_flags(
            inferential_df=inferential_df,
            alpha=alpha,
            clinical_relevance_floor=clinical_relevance_floor,
        )
        inferential_df = self._sort_inferential_df(inferential_df)
        self._qa_gate(inferential_df, expected_predictor_count=len(predictors))

        self.inferential_results = inferential_df
        self.inferential_table = None
        self.run_metadata = {
            "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_name": self.task_name,
            "target": self.target,
            "positive_class": positive_class,
            "negative_class": negative_class,
            "sample_size": int(len(self.INFERENTIAL)),
            "target_prevalence": float(y_binary.mean()),
            "predictors_tested": predictors,
            "predictors_approved_count": len(self.approved_predictors),
            "alpha": alpha,
            "ci_level": ci_level,
            "min_non_missing": min_non_missing,
            "n_boot": n_boot,
            "random_state": random_state,
            "clinical_relevance_floor": clinical_relevance_floor,
            "robust_numeric_fallback": robust_numeric_fallback,
            "outlier_iqr_multiplier": outlier_iqr_multiplier,
            "multiplicity_method": "benjamini_hochberg_valid_p_only",
            "analysis_scope": self.scope_statement,
            "enforce_positive_class_lock": enforce_positive_class_lock,
            "module_sha1": self._module_sha1(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
        }

        if export:
            self._export_results(inferential_df)

        styled_table = self._style_inferential_table(inferential_df)

        if show_table:
            display(styled_table)

        return styled_table

    # Work out where the target column lives, because sawing into the wrong vessel is frowned upon.
    def _resolve_target_col(self, target_col: str | None) -> str:
        if target_col is not None:
            if target_col not in self.INFERENTIAL.columns:
                raise ValueError(f"Target column not found in df_model: {target_col}")
            return target_col

        if {"column_name", "role"}.issubset(self.feature_decisions_df.columns):
            target_rows = self.feature_decisions_df.loc[
                self.feature_decisions_df["role"] == "target",
                "column_name",
            ].dropna()
            target_rows = [str(col) for col in target_rows.tolist() if str(col) in self.INFERENTIAL.columns]
            if len(target_rows) == 1:
                return target_rows[0]

        if "target" in self.INFERENTIAL.columns:
            return "target"

        raise ValueError("Could not resolve target column from feature_decisions_df or df_model.")

    # Build a quick lookup so each predictor gets routed to the right test without guessing every time.
    def _build_predictor_type_map(self) -> dict[str, str]:
        required_cols = {"column_name", "inferred_type"}
        if not required_cols.issubset(self.feature_decisions_df.columns):
            return {}

        predictor_rows = self.feature_decisions_df.loc[
            self.feature_decisions_df["column_name"].isin(self.INFERENTIAL.columns),
            ["column_name", "inferred_type"],
        ].dropna(subset=["column_name", "inferred_type"]).drop_duplicates(subset="column_name")

        return {
            str(row["column_name"]): str(row["inferred_type"])
            for _, row in predictor_rows.iterrows()
        }

    # Allow custom clinical thresholds without forcing everyone into one-size-fits-none defaults.
    def _build_effect_size_thresholds(
        self,
        effect_size_thresholds: dict[str, dict[str, float]] | None,
    ) -> dict[str, dict[str, float]]:
        merged = {metric: values.copy() for metric, values in _EFFECT_SIZE_THRESHOLDS.items()}
        if effect_size_thresholds is None:
            return merged

        for metric, levels in effect_size_thresholds.items():
            if metric not in merged:
                raise ValueError(f"Unknown effect metric for thresholds: {metric}")
            if not {"small", "moderate", "large"}.issubset(levels):
                raise ValueError(f"Thresholds for {metric} must include small, moderate, and large.")

            small = float(levels["small"])
            moderate = float(levels["moderate"])
            large = float(levels["large"])
            if not (0 <= small <= moderate <= large):
                raise ValueError(f"Threshold order must be small <= moderate <= large for {metric}.")

            merged[metric] = {"small": small, "moderate": moderate, "large": large}
        return merged

    # Encode any binary column into 0/1 so the maths does not depend on whatever label naming whim was used upstream.
    def _encode_binary_series(self, s: pd.Series, positive_class=None, series_name: str = "series"):
        non_null = s.dropna()
        unique_values = non_null.drop_duplicates().tolist()

        if len(unique_values) != 2:
            raise ValueError(f"{series_name} must be binary. Found values: {unique_values}")

        if positive_class is None:
            unique_set = set(unique_values)
            if unique_set == {0, 1}:
                positive_class = 1
            elif unique_set == {False, True}:
                positive_class = True
            else:
                positive_class = sorted(unique_values, key=lambda value: str(value))[-1]

        if positive_class not in unique_values:
            raise ValueError(f"positive_class={positive_class!r} is not present in {series_name}.")

        negative_values = [value for value in unique_values if value != positive_class]
        if len(negative_values) != 1:
            raise ValueError(f"Could not resolve negative class for {series_name}.")
        negative_class = negative_values[0]

        encoded = s.map({negative_class: 0, positive_class: 1})
        encoded = pd.to_numeric(encoded, errors="coerce").astype("Float64")
        return encoded, positive_class, negative_class

    # Route each predictor to numeric, categorical, or unsupported land.
    def _predictor_kind(self, predictor: str) -> str:
        inferred_type = self.predictor_type_map.get(predictor)

        if inferred_type in {"continuous", "discrete", "time_to_event"}:
            return "numeric"
        if inferred_type in {"binary", "categorical_nominal", "categorical_ordinal"}:
            return "categorical"
        if inferred_type in {"datetime", "text"}:
            return "unsupported"

        series = self.INFERENTIAL[predictor]
        non_null = series.dropna()
        n_unique = int(non_null.nunique(dropna=True)) if len(non_null) > 0 else 0

        if pd.api.types.is_numeric_dtype(series) and n_unique > 2:
            return "numeric"
        if n_unique >= 2:
            return "categorical"
        return "unsupported"

    # Build a skipped row so the final table still has one entry per approved predictor.
    def _skip_row(
        self,
        predictor: str,
        effect_metric,
        test_name,
        test_stat: float,
        expected_cell_min: float,
        expected_cell_lt5_pct: float,
        expected_cell_warning: bool,
        n: int,
        missing_pct: float,
        skipped_reason: str,
    ) -> dict:
        return {
            "predictor": predictor,
            "test_name": test_name,
            "test_stat": test_stat,
            "effect_metric": effect_metric,
            "effect_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "p_value_fdr": np.nan,
            "expected_cell_min": expected_cell_min,
            "expected_cell_lt5_pct": expected_cell_lt5_pct,
            "expected_cell_warning": bool(expected_cell_warning),
            "n": int(n),
            "missing_pct": float(missing_pct),
            "status": "skipped",
            "skipped_reason": skipped_reason,
            "fdr_significant": False,
            "clinically_relevant": False,
        }

    # Build a successful row with the effect, interval, and p-value already slotted into the contract.
    def _ok_row(
        self,
        predictor: str,
        test_name: str,
        test_stat: float,
        effect_metric: str,
        effect_value: float,
        ci_low: float,
        ci_high: float,
        p_value: float,
        expected_cell_min: float,
        expected_cell_lt5_pct: float,
        expected_cell_warning: bool,
        n: int,
        missing_pct: float,
    ) -> dict:
        return {
            "predictor": predictor,
            "test_name": test_name,
            "test_stat": float(test_stat),
            "effect_metric": effect_metric,
            "effect_value": float(effect_value),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "p_value": float(p_value),
            "p_value_fdr": np.nan,
            "expected_cell_min": float(expected_cell_min) if pd.notna(expected_cell_min) else np.nan,
            "expected_cell_lt5_pct": float(expected_cell_lt5_pct) if pd.notna(expected_cell_lt5_pct) else np.nan,
            "expected_cell_warning": bool(expected_cell_warning),
            "n": int(n),
            "missing_pct": float(missing_pct),
            "status": "ok",
            "skipped_reason": pd.NA,
            "fdr_significant": False,
            "clinically_relevant": False,
        }

    # Numeric predictor vs binary target: use point-biserial r as a quick "does this drift with the target?" gauge.
    def _run_numeric_row(
        self,
        predictor: str,
        y_binary: pd.Series,
        min_non_missing: int,
        n_boot: int,
        ci_level: float,
        random_state: int,
        robust_numeric_fallback: bool,
        outlier_iqr_multiplier: float,
    ) -> dict:
        test_name = "point_biserial"
        test_stat = np.nan
        effect_metric = "point_biserial_r"
        expected_cell_min = np.nan
        expected_cell_lt5_pct = np.nan
        expected_cell_warning = False
        x = pd.to_numeric(self.INFERENTIAL[predictor], errors="coerce")
        pair = pd.DataFrame({"x": x, "y": y_binary}).dropna()
        n_used = int(len(pair))
        missing_pct = float(self.INFERENTIAL[predictor].isna().mean() * 100.0)
        x0 = pair.loc[pair["y"] == 0, "x"]
        x1 = pair.loc[pair["y"] == 1, "x"]

        if n_used < min_non_missing:
            return self._skip_row(
                predictor,
                effect_metric,
                test_name=test_name,
                test_stat=test_stat,
                expected_cell_min=expected_cell_min,
                expected_cell_lt5_pct=expected_cell_lt5_pct,
                expected_cell_warning=expected_cell_warning,
                n=n_used,
                missing_pct=missing_pct,
                skipped_reason="too_few_non_missing",
            )
        if pair["x"].nunique(dropna=True) < 2 or pair["y"].nunique(dropna=True) < 2:
            return self._skip_row(
                predictor,
                effect_metric,
                test_name=test_name,
                test_stat=test_stat,
                expected_cell_min=expected_cell_min,
                expected_cell_lt5_pct=expected_cell_lt5_pct,
                expected_cell_warning=expected_cell_warning,
                n=n_used,
                missing_pct=missing_pct,
                skipped_reason="too_few_unique",
            )

        use_robust = False
        if robust_numeric_fallback and len(x0) >= 3 and len(x1) >= 3:
            use_robust = self._should_use_robust_numeric_test(pair["x"], outlier_iqr_multiplier)

        if use_robust:
            test_name = "mann_whitney_u"
            effect_metric = "rank_biserial_r"
            test_stat, p_value = mannwhitneyu(x1, x0, alternative="two-sided", method="asymptotic")
            denom = len(x1) * len(x0)
            if denom == 0:
                return self._skip_row(
                    predictor,
                    effect_metric,
                    test_name=test_name,
                    test_stat=test_stat,
                    expected_cell_min=expected_cell_min,
                    expected_cell_lt5_pct=expected_cell_lt5_pct,
                    expected_cell_warning=expected_cell_warning,
                    n=n_used,
                    missing_pct=missing_pct,
                    skipped_reason="too_few_groups",
                )
            effect_value = (2.0 * float(test_stat) / float(denom)) - 1.0
            ci_low, ci_high = self._bootstrap_ci(
                x=pair["x"],
                y=pair["y"],
                effect_function=self._rank_biserial_effect,
                n_boot=n_boot,
                ci_level=ci_level,
                random_state=random_state,
            )
        else:
            effect_value, p_value = pointbiserialr(pair["y"], pair["x"])
            test_stat = effect_value
            ci_low, ci_high = self._bootstrap_ci(
                x=pair["x"],
                y=pair["y"],
                effect_function=self._point_biserial_effect,
                n_boot=n_boot,
                ci_level=ci_level,
                random_state=random_state,
            )

        if not np.isfinite(effect_value) or not np.isfinite(p_value) or not np.isfinite(test_stat):
            return self._skip_row(
                predictor,
                effect_metric,
                test_name=test_name,
                test_stat=test_stat,
                expected_cell_min=expected_cell_min,
                expected_cell_lt5_pct=expected_cell_lt5_pct,
                expected_cell_warning=expected_cell_warning,
                n=n_used,
                missing_pct=missing_pct,
                skipped_reason="test_failed",
            )
        if not np.isfinite(ci_low) or not np.isfinite(ci_high):
            return self._skip_row(
                predictor,
                effect_metric,
                test_name=test_name,
                test_stat=test_stat,
                expected_cell_min=expected_cell_min,
                expected_cell_lt5_pct=expected_cell_lt5_pct,
                expected_cell_warning=expected_cell_warning,
                n=n_used,
                missing_pct=missing_pct,
                skipped_reason="ci_failed",
            )

        return self._ok_row(
            predictor=predictor,
            test_name=test_name,
            test_stat=test_stat,
            effect_metric=effect_metric,
            effect_value=effect_value,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            expected_cell_min=expected_cell_min,
            expected_cell_lt5_pct=expected_cell_lt5_pct,
            expected_cell_warning=expected_cell_warning,
            n=n_used,
            missing_pct=missing_pct,
        )

    # Binary/categorical predictor vs binary target: use a contingency table and measure its strength with Cramer's V.
    def _run_categorical_row(
        self,
        predictor: str,
        y_binary: pd.Series,
        min_non_missing: int,
        n_boot: int,
        ci_level: float,
        random_state: int,
    ) -> dict:
        effect_metric = "cramers_v"
        test_name = "chi_square"
        test_stat = np.nan
        x = self.INFERENTIAL[predictor]
        pair = pd.DataFrame({"x": x, "y": y_binary}).dropna()
        n_used = int(len(pair))
        missing_pct = float(self.INFERENTIAL[predictor].isna().mean() * 100.0)
        expected_cell_min = np.nan
        expected_cell_lt5_pct = np.nan
        expected_cell_warning = False

        if n_used < min_non_missing:
            return self._skip_row(
                predictor,
                effect_metric,
                test_name=test_name,
                test_stat=test_stat,
                expected_cell_min=expected_cell_min,
                expected_cell_lt5_pct=expected_cell_lt5_pct,
                expected_cell_warning=expected_cell_warning,
                n=n_used,
                missing_pct=missing_pct,
                skipped_reason="too_few_non_missing",
            )

        table = pd.crosstab(pair["y"], pair["x"])
        if table.shape[0] < 2 or table.shape[1] < 2:
            return self._skip_row(
                predictor,
                effect_metric,
                test_name=test_name,
                test_stat=test_stat,
                expected_cell_min=expected_cell_min,
                expected_cell_lt5_pct=expected_cell_lt5_pct,
                expected_cell_warning=expected_cell_warning,
                n=n_used,
                missing_pct=missing_pct,
                skipped_reason="degenerate_contingency",
            )

        chi2_stat, chi2_p_value, _, expected = chi2_contingency(table, correction=False)
        expected_cell_min = float(np.min(expected))
        expected_cell_lt5_pct = float((expected < 5).mean() * 100.0)
        expected_cell_warning = bool((expected_cell_min < 1.0) or (expected_cell_lt5_pct > 20.0))

        if table.shape == (2, 2):
            test_name = "fisher_exact"
            odds_ratio, p_value = fisher_exact(table.to_numpy())
            test_stat = float(odds_ratio) if np.isfinite(odds_ratio) else np.nan
        else:
            test_name = "chi_square"
            test_stat = chi2_stat
            p_value = float(chi2_p_value)

        effect_value = self._cramers_v_from_table(table)
        if not np.isfinite(effect_value) or not np.isfinite(p_value) or not np.isfinite(test_stat):
            return self._skip_row(
                predictor,
                effect_metric,
                test_name=test_name,
                test_stat=test_stat,
                expected_cell_min=expected_cell_min,
                expected_cell_lt5_pct=expected_cell_lt5_pct,
                expected_cell_warning=expected_cell_warning,
                n=n_used,
                missing_pct=missing_pct,
                skipped_reason="test_failed",
            )

        ci_low, ci_high = self._bootstrap_ci(
            x=pair["x"],
            y=pair["y"],
            effect_function=self._cramers_v_effect,
            n_boot=n_boot,
            ci_level=ci_level,
            random_state=random_state,
        )
        if not np.isfinite(ci_low) or not np.isfinite(ci_high):
            return self._skip_row(
                predictor,
                effect_metric,
                test_name=test_name,
                test_stat=test_stat,
                expected_cell_min=expected_cell_min,
                expected_cell_lt5_pct=expected_cell_lt5_pct,
                expected_cell_warning=expected_cell_warning,
                n=n_used,
                missing_pct=missing_pct,
                skipped_reason="ci_failed",
            )

        return self._ok_row(
            predictor=predictor,
            test_name=test_name,
            test_stat=test_stat,
            effect_metric=effect_metric,
            effect_value=effect_value,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            expected_cell_min=expected_cell_min,
            expected_cell_lt5_pct=expected_cell_lt5_pct,
            expected_cell_warning=expected_cell_warning,
            n=n_used,
            missing_pct=missing_pct,
        )

    # Recompute point-biserial r on a resampled slice when building the bootstrap CI.
    def _point_biserial_effect(self, x: pd.Series, y: pd.Series) -> float:
        pair = pd.DataFrame(
            {
                "x": pd.to_numeric(pd.Series(x).reset_index(drop=True), errors="coerce"),
                "y": pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce"),
            }
        ).dropna()
        if len(pair) < 3:
            return np.nan
        if pair["x"].nunique(dropna=True) < 2 or pair["y"].nunique(dropna=True) < 2:
            return np.nan

        value, _ = pointbiserialr(pair["y"], pair["x"])
        return float(value) if np.isfinite(value) else np.nan

    # Recompute rank-biserial effect for robust numeric fallback during bootstrap.
    def _rank_biserial_effect(self, x: pd.Series, y: pd.Series) -> float:
        pair = pd.DataFrame(
            {
                "x": pd.to_numeric(pd.Series(x).reset_index(drop=True), errors="coerce"),
                "y": pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce"),
            }
        ).dropna()
        if len(pair) < 3:
            return np.nan

        x0 = pair.loc[pair["y"] == 0, "x"]
        x1 = pair.loc[pair["y"] == 1, "x"]
        if len(x0) < 2 or len(x1) < 2:
            return np.nan

        u_stat, _ = mannwhitneyu(x1, x0, alternative="two-sided", method="asymptotic")
        denom = len(x1) * len(x0)
        if denom == 0:
            return np.nan
        return float((2.0 * float(u_stat) / float(denom)) - 1.0)

    # Decide when robust numeric test is safer: heavy skew or many IQR outliers means use rank-based test.
    def _should_use_robust_numeric_test(self, x: pd.Series, outlier_iqr_multiplier: float) -> bool:
        x_num = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
        if len(x_num) < 10:
            return False

        skew_abs = abs(float(x_num.skew()))
        q1, q3 = np.percentile(x_num, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            outlier_rate = 0.0
        else:
            lower = q1 - outlier_iqr_multiplier * iqr
            upper = q3 + outlier_iqr_multiplier * iqr
            outlier_rate = float(((x_num < lower) | (x_num > upper)).mean())

        return (skew_abs > 1.0) or (outlier_rate > 0.05)

    # Recompute Cramer's V on a resampled slice when building the bootstrap CI.
    def _cramers_v_effect(self, x: pd.Series, y: pd.Series) -> float:
        pair = pd.DataFrame(
            {
                "x": pd.Series(x).reset_index(drop=True),
                "y": pd.Series(y).reset_index(drop=True),
            }
        ).dropna()
        table = pd.crosstab(pair["y"], pair["x"])
        return self._cramers_v_from_table(table)

    # Turn chi-square into a 0-to-1 effect size so "statistically significant" is not mistaken for "actually matters".
    def _cramers_v_from_table(self, table: pd.DataFrame) -> float:
        if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
            return np.nan

        n = float(table.to_numpy().sum())
        if n == 0:
            return np.nan

        chi2, _, _, _ = chi2_contingency(table, correction=False)
        min_dim = min(table.shape) - 1
        if min_dim <= 0:
            return np.nan

        # Cramer's V = sqrt(chi2 / (n * smallest free dimension)).
        return float(np.sqrt(chi2 / (n * min_dim)))

    # Bootstrap the effect size so we get a CI even when the analytic formula would be ugly or brittle.
    def _bootstrap_ci(
        self,
        x: pd.Series,
        y: pd.Series,
        effect_function,
        n_boot: int,
        ci_level: float,
        random_state: int,
    ) -> tuple[float, float]:
        pair = pd.DataFrame(
            {
                "x": pd.Series(x).reset_index(drop=True),
                "y": pd.Series(y).reset_index(drop=True),
            }
        ).dropna()
        if len(pair) < 3:
            return np.nan, np.nan

        rng = np.random.default_rng(random_state)
        boot_values = []
        n = len(pair)

        for _ in range(int(n_boot)):
            sample_idx = rng.integers(0, n, size=n)
            sample = pair.iloc[sample_idx].reset_index(drop=True)
            boot_value = effect_function(sample["x"], sample["y"])
            if pd.notna(boot_value) and np.isfinite(boot_value):
                boot_values.append(float(boot_value))

        min_valid_boot = max(10, int(n_boot * 0.10))
        if len(boot_values) < min_valid_boot:
            return np.nan, np.nan

        # Percentile bootstrap CI = keep the middle chunk and ignore the tails.
        alpha_tail = (1.0 - ci_level) / 2.0
        ci_low, ci_high = np.quantile(boot_values, [alpha_tail, 1.0 - alpha_tail])
        return float(ci_low), float(ci_high)

    # Benjamini-Hochberg keeps the false-discovery leak under control after testing a whole flotilla of predictors.
    def _benjamini_hochberg(self, p_values: pd.Series) -> pd.Series:
        ordered_index = np.argsort(p_values.to_numpy(dtype=float))
        ordered_p = p_values.to_numpy(dtype=float)[ordered_index]
        n_tests = len(ordered_p)
        ranks = np.arange(1, n_tests + 1, dtype=float)

        # BH adjusted p = p_i * m / rank_i, then forced to stay monotonic from the tail backward.
        adjusted = ordered_p * n_tests / ranks
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)

        result = np.empty(n_tests, dtype=float)
        result[ordered_index] = adjusted
        return pd.Series(result, index=p_values.index, dtype="Float64")

    # Translate a raw effect number into a practical band, so tiny but significant effects do not seduce you.
    def _effect_band(self, effect_metric: str, effect_value: float) -> str:
        if effect_metric not in self.effect_size_thresholds or pd.isna(effect_value):
            return "unknown"

        thresholds = self.effect_size_thresholds[effect_metric]
        effect_abs = abs(float(effect_value))

        if effect_abs >= thresholds["large"]:
            return "large"
        if effect_abs >= thresholds["moderate"]:
            return "moderate"
        if effect_abs >= thresholds["small"]:
            return "small"
        return "negligible"

    # Decide whether an effect is clinically worth noticing, not just mathematically able to make noise.
    def _is_clinically_relevant(
        self,
        effect_metric: str,
        effect_value: float,
        clinical_relevance_floor: str,
    ) -> bool:
        band = self._effect_band(effect_metric, effect_value)
        band_rank = {"unknown": -1, "negligible": 0, "small": 1, "moderate": 2, "large": 3}
        required_rank = {"small": 1, "moderate": 2, "large": 3}[clinical_relevance_floor]
        return band_rank.get(band, -1) >= required_rank

    # Build a quick hash fingerprint so exported metadata can be traced back to exact inferential code.
    def _module_sha1(self) -> str:
        try:
            with open(__file__, "rb") as handle:
                return hashlib.sha1(handle.read()).hexdigest()
        except OSError:
            return "unavailable"

    # Add BH-FDR and the clinical flag after all rows are assembled.
    def _apply_fdr_flags(
        self,
        inferential_df: pd.DataFrame,
        alpha: float,
        clinical_relevance_floor: str,
    ) -> pd.DataFrame:
        inferential_df = inferential_df.copy()
        valid_mask = inferential_df["status"].eq("ok") & inferential_df["p_value"].notna()

        inferential_df["p_value_fdr"] = pd.Series(np.nan, index=inferential_df.index, dtype="Float64")
        inferential_df["fdr_significant"] = False
        inferential_df["clinically_relevant"] = False

        if valid_mask.any():
            inferential_df.loc[valid_mask, "p_value_fdr"] = self._benjamini_hochberg(
                inferential_df.loc[valid_mask, "p_value"]
            )
            inferential_df.loc[valid_mask, "fdr_significant"] = (
                inferential_df.loc[valid_mask, "p_value_fdr"] < alpha
            )

        inferential_df["clinically_relevant"] = inferential_df.apply(
            lambda row: bool(
                row["status"] == "ok"
                and self._is_clinically_relevant(
                    effect_metric=row["effect_metric"],
                    effect_value=row["effect_value"],
                    clinical_relevance_floor=clinical_relevance_floor,
                )
            ),
            axis=1,
        )
        return inferential_df

    # Sort the table so the strongest clean signals rise to the top instead of drowning in administrative sludge.
    def _sort_inferential_df(self, inferential_df: pd.DataFrame) -> pd.DataFrame:
        inferential_df = inferential_df.copy()
        inferential_df["_effect_abs"] = inferential_df["effect_value"].abs()
        inferential_df = inferential_df.sort_values(
            by=["fdr_significant", "clinically_relevant", "_effect_abs", "p_value_fdr", "predictor"],
            ascending=[False, False, False, True, True],
            na_position="last",
        ).drop(columns="_effect_abs")
        return inferential_df.reset_index(drop=True)

    # Format the final notebook artifact so the numbers are readable without needing a jeweller's loupe.
    def _style_inferential_table(self, inferential_df: pd.DataFrame):
        return (
            inferential_df.style
            .apply(_inferential_row_text_style, axis=1)
            .format(
                {
                    "test_stat": "{:.4g}",
                    "effect_value": "{:.3f}",
                    "ci_low": "{:.3f}",
                    "ci_high": "{:.3f}",
                    "p_value": "{:.4g}",
                    "p_value_fdr": "{:.4g}",
                    "expected_cell_min": "{:.3f}",
                    "expected_cell_lt5_pct": "{:.1f}",
                    "missing_pct": "{:.1f}",
                },
                na_rep="NA",
            )
        )

    # Enforce the checklist contract before you trust the table.
    def _qa_gate(self, inferential_df: pd.DataFrame, expected_predictor_count: int):
        if len(inferential_df) != expected_predictor_count:
            raise ValueError("Inferential QA failed: row count does not match tested predictors.")
        if inferential_df["predictor"].duplicated().any():
            raise ValueError("Inferential QA failed: duplicate predictors found.")
        if not inferential_df["predictor"].isin(self.approved_predictors).all():
            raise ValueError("Inferential QA failed: non-approved predictor found in inferential table.")

        ok_mask = inferential_df["status"].eq("ok")
        skipped_mask = inferential_df["status"].eq("skipped")

        ok_required_cols = [
            "test_name",
            "test_stat",
            "effect_value",
            "ci_low",
            "ci_high",
            "p_value",
            "p_value_fdr",
        ]
        if inferential_df.loc[ok_mask, ok_required_cols].isna().any().any():
            raise ValueError("Inferential QA failed: some ok rows are missing effect, CI, p-value, or FDR.")
        if inferential_df.loc[skipped_mask, "skipped_reason"].isna().any():
            raise ValueError("Inferential QA failed: some skipped rows are missing skipped_reason.")

    # Save both the table and its run metadata with deterministic filenames.
    def _export_results(self, inferential_df: pd.DataFrame):
        export_dir = self.root / "reports" / "tables"
        export_dir.mkdir(parents=True, exist_ok=True)

        inferential_path = export_dir / f"(INFERENTIAL_table){self.task_name}.pickle"
        metadata_path = export_dir / f"(INFERENTIAL_metadata){self.task_name}.pickle"

        inferential_df.to_pickle(inferential_path)
        pd.Series(self.run_metadata, dtype="object").to_pickle(metadata_path)

        print(f"{GREEN}Saved inferential table:{RESET} {BOLD}{inferential_path}{RESET}")
        print(f"{GREEN}Saved inferential metadata:{RESET} {BOLD}{metadata_path}{RESET}")
