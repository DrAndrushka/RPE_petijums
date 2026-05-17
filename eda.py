"""
EDA helper for target-vs-predictor association screening.

Design goals:
- Keep notebook API simple.
- Accept one or many targets in the same `build_associations_table(...)`.
- Keep predictor selection whitelist-driven.
- Produce a stable long-format output that is easy to reuse in future studies.

TODO:
- Add optional exact contingency table export per (target, predictor) pair.
- Add bootstrap CI for effect sizes.
- Add optional robust numeric test fallback (rank-based).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import chi2_contingency, pearsonr

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# ANSI escape codes for notebook console output.
BOLD = "\033[1m"
BLUE = "\033[38;5;39m"
GREEN = "\033[38;5;28m"
RESET = "\033[0m"


ASSOCIATIONS_COLS = [
    "target",
    "predictor",
    "test_name",
    "test_stat",
    "effect_metric",
    "effect_value",
    "ci_low",
    "ci_high",
    "p_value",
    "p_value_fdr",
    "n",
    "missing_pct",
    "status",
    "skipped_reason",
    "fdr_significant",
    "clinically_relevant",]


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Return BH-adjusted p-values for a numeric p-value series."""
    ordered_idx = np.argsort(p_values.to_numpy(dtype=float))
    ordered_p = p_values.to_numpy(dtype=float)[ordered_idx]
    n = len(ordered_p)
    ranks = np.arange(1, n + 1, dtype=float)

    adjusted = ordered_p * n / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    out = np.empty(n, dtype=float)
    out[ordered_idx] = adjusted
    return pd.Series(out, index=p_values.index, dtype="Float64")


class YlivertainenEDA:
    """Whitelist predictors, profile features, and build association table(s)."""

    def __init__(self, df: pd.DataFrame, task_name: str):
        """
        Store a working copy and initialize classification buckets.

        The notebook can set:
        - `self.target = "upgrade"` for legacy single-target flow.
        - `self.target_cols = ["upgrade", "upstage"]` for multi-target flow.
        """
        self.EDA = df.copy()
        self.task_name = task_name

        # Legacy + new target API are both supported.
        self.target: str | None = None
        self.target_cols: list[str] = []

        self.predictor_continuous: list[str] = []
        self.predictor_discrete: list[str] = []
        self.predictor_binary: list[str] = []
        self.predictor_categorical_nominal: list[str] = []
        self.predictor_categorical_ordinal: list[str] = []
        self.predictor_time_to_event: list[str] = []
        self.predictor_datetime: list[str] = []
        self.predictor_text: list[str] = []

        self.unclassified_predictors: dict[str, str] = {}
        self.all_predictors: list[str] = []

        self.associations_table = pd.DataFrame(columns=ASSOCIATIONS_COLS)
        self.feature_decisions_table = pd.DataFrame()

    def _effective_unique(self, s: pd.Series, min_freq: float = 0.01) -> int:
        """Count effective unique values after dropping ultra-rare levels."""
        vc = s.value_counts(dropna=True, normalize=True)
        vc_lean = vc[vc >= min_freq]
        return int(vc_lean.index.size)

    def _resolve_targets(self, target_cols: Sequence[str] | str | None = None) -> list[str]:
        """
        Resolve targets from explicit arg or object attributes.

        Priority:
        1) `target_cols` argument
        2) `self.target_cols`
        3) `self.target`
        """
        if target_cols is None:
            if self.target_cols:
                targets = list(dict.fromkeys(self.target_cols))
            elif self.target:
                targets = [self.target]
            else:
                raise ValueError("No targets set. Pass target_cols or set project.target/project.target_cols.")
        elif isinstance(target_cols, str):
            targets = [target_cols]
        else:
            targets = list(dict.fromkeys(target_cols))

        missing = [col for col in targets if col not in self.EDA.columns]
        if missing:
            raise ValueError(f"Target columns not found in EDA DataFrame: {missing}")
        if not targets:
            raise ValueError("At least one target column is required.")
        return targets

    def whitelist_columns(
        self,
        predictors: Sequence[str],
        numerical_continuous: Sequence[str],):
        """
        Classify predictor whitelist into predictor-type buckets.

        Notes:
        - This method does not infer targets; targets are resolved separately.
        - Predictor whitelist remains the default source of predictors.
        """
        if not predictors:
            raise ValueError("predictors not set")

        requested_cols = list(dict.fromkeys([*predictors, *numerical_continuous]))
        missing_cols = [col for col in requested_cols if col not in self.EDA.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Reset buckets in case whitelist is run again.
        self.predictor_continuous = []
        self.predictor_discrete = []
        self.predictor_binary = []
        self.predictor_categorical_nominal = []
        self.predictor_categorical_ordinal = []
        self.predictor_time_to_event = []
        self.predictor_datetime = []
        self.predictor_text = []
        self.unclassified_predictors = {}

        for col in predictors:
            s = self.EDA[col].dropna()
            if "_timedelta_" in col:
                self.predictor_time_to_event.append(col)
            elif pd.api.types.is_numeric_dtype(s):
                eff_unique = self._effective_unique(s)
                if eff_unique == 2:
                    self.predictor_binary.append(col)
                elif col in numerical_continuous:
                    self.predictor_continuous.append(col)
                else:
                    self.predictor_discrete.append(col)
            elif isinstance(s.dtype, pd.api.types.CategoricalDtype) and s.dtype.ordered:
                self.predictor_categorical_ordinal.append(col)
            elif isinstance(s.dtype, pd.api.types.CategoricalDtype) and not s.dtype.ordered:
                self.predictor_categorical_nominal.append(col)
            elif pd.api.types.is_datetime64_any_dtype(s):
                self.predictor_datetime.append(col)
            elif pd.api.types.is_string_dtype(s):
                self.predictor_text.append(col)
            else:
                self.unclassified_predictors[col] = str(s.dtype)
                self.predictor_discrete.append(col)

        self.all_predictors = (
            self.predictor_continuous
            + self.predictor_discrete
            + self.predictor_binary
            + self.predictor_categorical_nominal
            + self.predictor_categorical_ordinal
            + self.predictor_time_to_event
            + self.predictor_datetime
            + self.predictor_text
        )

        # Keep only currently known targets + predictors for a cleaner working frame.
        configured_targets = []
        if self.target_cols:
            configured_targets.extend([t for t in self.target_cols if t in self.EDA.columns])
        if self.target and self.target in self.EDA.columns:
            configured_targets.append(self.target)
        configured_targets = list(dict.fromkeys(configured_targets))
        ordered_cols = list(dict.fromkeys([*configured_targets, *self.all_predictors]))
        if ordered_cols:
            self.EDA = self.EDA[ordered_cols].copy()

        if self.unclassified_predictors:
            print("Unclassified predictors:")
            for col, dtype in self.unclassified_predictors.items():
                print(f"- {col} | {dtype}")
        print(f"{BOLD}✅ Predictors classified{RESET}: {len(self.all_predictors)}")
        return self

    def _analysis_type(self, series: pd.Series) -> str:
        """Infer broad analysis type for target series."""
        non_null = series.dropna()
        n_unique = int(non_null.nunique(dropna=True)) if len(non_null) > 0 else 0
        if pd.api.types.is_bool_dtype(series) or n_unique == 2:
            return "binary"
        if pd.api.types.is_numeric_dtype(series) and n_unique > 2:
            return "continuous"
        return "categorical"

    def _predictor_type_map(self) -> dict[str, str]:
        """Return predictor->type mapping from current buckets."""
        out: dict[str, str] = {}
        for predictor_type, cols in [
            ("continuous", self.predictor_continuous),
            ("discrete", self.predictor_discrete),
            ("binary", self.predictor_binary),
            ("categorical_nominal", self.predictor_categorical_nominal),
            ("categorical_ordinal", self.predictor_categorical_ordinal),
            ("time_to_event", self.predictor_time_to_event),
            ("datetime", self.predictor_datetime),
            ("text", self.predictor_text),
        ]:
            for col in cols:
                out[col] = predictor_type
        return out

    def _run_association_test(
        self,
        x: pd.Series,
        y: pd.Series,
        predictor_type: str,
        target_type: str,) -> dict[str, object]:
        """Route one predictor-target pair to the appropriate test."""
        if predictor_type in {"datetime", "text"}:
            return {"status": "skipped", "skipped_reason": "unsupported_predictor_type"}

        if target_type == "continuous" and predictor_type in {"continuous", "discrete", "time_to_event"}:
            x_num = pd.to_numeric(x, errors="coerce")
            y_num = pd.to_numeric(y, errors="coerce")
            if x_num.nunique(dropna=True) < 2 or y_num.nunique(dropna=True) < 2:
                return {"status": "skipped", "skipped_reason": "too_few_unique"}
            stat, p_value = pearsonr(x_num, y_num)
            return {
                "status": "ok",
                "test_name": "pearsonr",
                "test_stat": float(stat),
                "effect_metric": "r",
                "effect_value": float(stat),
                "p_value": float(p_value),
            }

        if target_type == "binary" and predictor_type in {"continuous", "discrete", "time_to_event"}:
            x_num = pd.to_numeric(x, errors="coerce")
            if x_num.nunique(dropna=True) < 2:
                return {"status": "skipped", "skipped_reason": "too_few_unique"}
            if pd.api.types.is_bool_dtype(y):
                y_num = y.astype(int)
            else:
                y_num = pd.Series(pd.Categorical(y).codes, index=y.index)
            if y_num.nunique(dropna=True) < 2:
                return {"status": "skipped", "skipped_reason": "too_few_unique"}
            stat, p_value = pearsonr(x_num, y_num.astype(float))
            return {
                "status": "ok",
                "test_name": "point_biserial",
                "test_stat": float(stat),
                "effect_metric": "r",
                "effect_value": float(stat),
                "p_value": float(p_value),
            }

        if target_type in {"binary", "categorical"} and predictor_type in {
            "binary",
            "categorical_nominal",
            "categorical_ordinal",
            "discrete",
        }:
            table = pd.crosstab(y, x)
            if table.shape[0] < 2 or table.shape[1] < 2:
                return {"status": "skipped", "skipped_reason": "degenerate_contingency"}
            chi2, p_value, _, _ = chi2_contingency(table)
            n = table.to_numpy().sum()
            denom = n * (min(table.shape) - 1)
            cramers_v = float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan
            return {
                "status": "ok",
                "test_name": "chi2",
                "test_stat": float(chi2),
                "effect_metric": "cramers_v",
                "effect_value": cramers_v,
                "p_value": float(p_value),
            }

        return {"status": "skipped", "skipped_reason": "unsupported_pair"}

    def build_associations_table(
        self,
        target_cols: Sequence[str] | str | None = None,
        leakage_vars: Collection[str] | None = None,
        fdr_alpha: float = 0.05,
        clinical_effect_floor: float = 0.30,) -> pd.DataFrame:
        """
        Build one long association table for each (target, predictor) pair.

        Backward compatibility:
        - If `target_cols` is omitted, this uses `self.target_cols` then `self.target`.
        
        - `p_value` is the raw per-test value.
        - `p_value_fdr` is the multiple-testing-corrected value, needed because
          this method runs many tests (many predictors x many targets), and raw
          p-values alone would overstate significance.
        """
        if not 0 < fdr_alpha < 1:
            raise ValueError("fdr_alpha must be between 0 and 1.")
        if clinical_effect_floor < 0:
            raise ValueError("clinical_effect_floor must be >= 0.")

        leakage_vars = set() if leakage_vars is None else set(leakage_vars)
        targets = self._resolve_targets(target_cols)
        self.target_cols = targets
        self.target = targets[0]  # preserve legacy behavior for downstream code.

        predictor_type_map = self._predictor_type_map()
        predictors = [col for col in self.all_predictors if col in self.EDA.columns and col not in targets]

        rows: list[dict[str, object]] = []
        for target in targets:
            target_type = self._analysis_type(self.EDA[target])
            for predictor in predictors:
                analysis_series = self.EDA[target]
                predictor_series = self.EDA[predictor]
                mask_non_missing = analysis_series.notna() & predictor_series.notna()
                y = analysis_series.loc[mask_non_missing]
                x = predictor_series.loc[mask_non_missing]
                n_used = int(mask_non_missing.sum())

                base_row = {
                    "target": target,
                    "predictor": predictor,
                    "test_name": pd.NA,
                    "test_stat": np.nan,
                    "effect_metric": pd.NA,
                    "effect_value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "p_value_fdr": np.nan,
                    "n": n_used,
                    "missing_pct": float((~mask_non_missing).mean() * 100.0),
                    "status": "skipped",
                    "skipped_reason": "too_few_non_missing",
                    "fdr_significant": False,
                    "clinically_relevant": False,
                }

                if predictor in leakage_vars or target in leakage_vars:
                    base_row["skipped_reason"] = "leakage_flagged"
                    rows.append(base_row)
                    continue

                if n_used < 10:
                    rows.append(base_row)
                    continue

                result = self._run_association_test(
                    x=x,
                    y=y,
                    predictor_type=predictor_type_map.get(predictor, "unknown"),
                    target_type=target_type,
                )
                if result["status"] == "ok":
                    base_row.update(result)
                    base_row["skipped_reason"] = pd.NA
                    base_row["clinically_relevant"] = bool(
                        pd.notna(base_row["effect_value"])
                        and abs(float(base_row["effect_value"])) >= clinical_effect_floor
                    )
                else:
                    base_row["skipped_reason"] = result.get("skipped_reason", "test_failed")
                rows.append(base_row)

        associations_table = pd.DataFrame(rows, columns=ASSOCIATIONS_COLS)

        valid_mask = associations_table["status"].eq("ok") & associations_table["p_value"].notna()
        if valid_mask.any():
            for target in targets:
                target_mask = valid_mask & associations_table["target"].eq(target)
                if target_mask.any():
                    associations_table.loc[target_mask, "p_value_fdr"] = _benjamini_hochberg(
                        associations_table.loc[target_mask, "p_value"]
                    )
            associations_table.loc[valid_mask, "fdr_significant"] = (
                associations_table.loc[valid_mask, "p_value_fdr"] < fdr_alpha
            )

        # Stable dtypes.
        for col in ["target", "predictor", "test_name", "effect_metric", "status", "skipped_reason"]:
            associations_table[col] = associations_table[col].astype("string[python]")
        for col in ["test_stat", "effect_value", "ci_low", "ci_high", "p_value", "p_value_fdr", "missing_pct"]:
            associations_table[col] = pd.to_numeric(associations_table[col], errors="coerce").astype("Float64")
        associations_table["n"] = pd.to_numeric(associations_table["n"], errors="coerce").astype("Int64")
        associations_table["fdr_significant"] = associations_table["fdr_significant"].astype("boolean")
        associations_table["clinically_relevant"] = associations_table["clinically_relevant"].astype("boolean")

        self.associations_table = associations_table
        return associations_table

    def build_feature_decisions_table(
        self,
        missingness_dict: dict[str, str] | None = None,
        leakage_vars: Collection[str] | None = None,
        redundancy_pairs: pd.DataFrame | None = None,) -> pd.DataFrame:
        """
        Build feature governance table used for downstream model-frame prep.

        TODO:
        - Add stricter rule sets per specialty/project type.
        - Add confidence score for each decision row.
        """
        df = self.EDA
        missingness_dict = {} if missingness_dict is None else dict(missingness_dict)
        leakage_vars = set() if leakage_vars is None else set(leakage_vars)
        targets = self._resolve_targets(None) if (self.target or self.target_cols) else []
        target_set = set(targets)

        feature_columns = [
            "column_name",
            "role",
            "dtype",
            "inferred_type",
            "missing_pct",
            "drop_reason",
            "action",
            "missing_action",
            "notes",
        ]

        inferred_type_map: dict[str, str] = {}
        for inferred_type, cols in [
            ("continuous", self.predictor_continuous),
            ("discrete", self.predictor_discrete),
            ("binary", self.predictor_binary),
            ("categorical_nominal", self.predictor_categorical_nominal),
            ("categorical_ordinal", self.predictor_categorical_ordinal),
            ("time_to_event", self.predictor_time_to_event),
            ("datetime", self.predictor_datetime),
            ("text", self.predictor_text),
        ]:
            for col in cols:
                inferred_type_map[col] = inferred_type

        redundancy_drop_map: dict[str, str] = {}
        if redundancy_pairs is not None and len(redundancy_pairs) > 0:
            required_cols = {"var_a", "var_b", "preferred_keep"}
            missing_cols = required_cols.difference(redundancy_pairs.columns)
            if missing_cols:
                raise ValueError(f"redundancy_pairs missing required columns: {sorted(missing_cols)}")
            for _, pair_row in redundancy_pairs.iterrows():
                preferred_keep = pair_row["preferred_keep"]
                for candidate_col in [pair_row["var_a"], pair_row["var_b"]]:
                    if pd.isna(candidate_col) or candidate_col == preferred_keep:
                        continue
                    redundancy_drop_map[str(candidate_col)] = str(preferred_keep)

        def _missing_action(missing_pct: float, missingness_type: str | None, inferred_type: str) -> str:
            if missing_pct == 0:
                return "no_missing"
            if missingness_type == "STRUCTURAL pattern":
                return "do_not_impute"
            if missingness_type == "MNAR":
                return "flag_only"
            if missing_pct < 20:
                return "simple_impute"
            if missing_pct <= 60:
                if inferred_type in {"categorical_nominal", "categorical_ordinal"}:
                    return "simple_impute"
                return "advanced_impute"
            return "flag_only"

        rows: list[dict[str, object]] = []
        base_cols = [col for col in df.columns if not col.endswith("_missing")]
        for col in base_cols:
            dtype = str(df[col].dtype)
            inferred_type = inferred_type_map.get(col, "unknown")
            missing_pct = float(df[col].isna().mean() * 100.0)
            missingness_type = missingness_dict.get(f"{col}_missing")

            if col in target_set:
                role = "target"
            elif col in leakage_vars:
                role = "leakage"
            elif col in inferred_type_map:
                role = "predictor"
            else:
                role = "helper"

            action = "keep"
            drop_reason = ""
            notes: list[str] = []
            if missingness_type:
                notes.append(f"missingness_type={missingness_type}")
            if missingness_type == "STRUCTURAL pattern":
                notes.append("Structural; encode Not Applicable / keep missing")
            elif missingness_type == "MNAR":
                notes.append("MNAR; keep missing flag and avoid heavy imputation")
            if 40.0 < missing_pct <= 60.0:
                notes.append("high_missingness_flag_important")
            if role == "leakage":
                action = "drop"
                drop_reason = "leakage"
                notes.append("leakage_feature")
            elif missing_pct > 60.0 and missingness_type in {"MNAR", "Unclassifiable"}:
                action = "drop"
                drop_reason = "extreme_missing"
                notes.append("extreme_missingness")
            preferred_keep = redundancy_drop_map.get(col)
            if preferred_keep is not None:
                action = "drop"
                drop_reason = f"redundant_with_{preferred_keep}"
                notes.append(f"redundant_with={preferred_keep}")
            if role == "helper":
                notes.append("non_modeling_helper")

            rows.append(
                {
                    "column_name": col,
                    "role": role,
                    "dtype": dtype,
                    "inferred_type": inferred_type,
                    "missing_pct": missing_pct,
                    "drop_reason": drop_reason,
                    "action": action,
                    "missing_action": _missing_action(missing_pct, missingness_type, inferred_type),
                    "notes": "; ".join(dict.fromkeys(notes)) if notes else pd.NA,
                }
            )

        feature_decisions_df = pd.DataFrame(rows, columns=feature_columns)
        for col in ["column_name", "role", "dtype", "inferred_type", "drop_reason", "action", "missing_action", "notes"]:
            feature_decisions_df[col] = feature_decisions_df[col].astype("string[python]")
        feature_decisions_df["missing_pct"] = pd.to_numeric(feature_decisions_df["missing_pct"], errors="coerce").astype(
            "Float64"
        )

        display(feature_decisions_df)
        self.feature_decisions_table = feature_decisions_df
        return feature_decisions_df

    def export_both_tables(self, export: bool = False):
        """
        Export associations and feature decisions to Excel.

        TODO:
        - Add CSV/Parquet export options.
        - Add timestamp/versioning in filenames.
        """
        export_dir = Path("reports") / "tables"
        associations_path = export_dir / f"(EDA_associations_table){self.task_name}.xlsx"
        feature_decisions_path = export_dir / f"(EDA_feature_decisions_table){self.task_name}.xlsx"

        if export:
            export_dir.mkdir(parents=True, exist_ok=True)
            self.associations_table.to_excel(associations_path, index=False)
            self.feature_decisions_table.to_excel(feature_decisions_path, index=False)
            print(f"{GREEN}Saved:{RESET} {BOLD}{associations_path}{RESET}")
            print(f"{GREEN}Saved:{RESET} {BOLD}{feature_decisions_path}{RESET}")
        else:
            print(f"Prepared export path: {BLUE}{BOLD}{export_dir}{RESET}")
            print(f"Set export={BLUE}{BOLD}True{RESET} to save files.")
        return self
