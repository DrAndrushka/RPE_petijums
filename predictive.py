from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import sys

from IPython.display import display, Markdown
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV  # noqa: F401
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from joblib import dump as joblib_dump

BOLD = "\033[1m"
BLUE = "\033[38;5;39m"
GREEN = "\033[38;5;28m"
YELLOW = "\033[38;5;214m"
ORANGE = "\033[38;5;202m"
RED = "\033[38;5;196m"
RESET = "\033[0m"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

PREDICTIVE_COMPARISON_COLS = [
    "model_name",
    "best_cv_auc_mean",
    "best_cv_auc_std",
    "best_params",
]

PREDICTIVE_METRICS_COLS = [
    "selected_model",
    "threshold_strategy",
    "threshold",
    "test_auc",
    "test_auc_ci_low",
    "test_auc_ci_high",
    "test_brier",
    "test_sensitivity",
    "test_specificity",
    "test_tp",
    "test_fp",
    "test_tn",
    "test_fn",
    "n_train",
    "n_test",
]


class YlivertainenPredictive:
    def __init__(
        self,
        root,
        df_model: pd.DataFrame,
        feature_decisions_df: pd.DataFrame,
        metadata: dict | None = None,
        target_col: str | None = None,
        default_positive_class=None,
        random_state: int = 42,
    ):
        self.root = Path(root)
        self.PREDICTIVE = df_model.copy()
        self.feature_decisions_df = feature_decisions_df.copy()
        self.task_name = "predictive" if metadata is None else metadata.get("task_name", "predictive")
        self.target = self._resolve_target_col(target_col)
        self.approved_predictors = self.approved_predictor_columns()
        self.default_positive_class = default_positive_class
        self.random_state = int(random_state)

        (
            self.target_binary,
            self.target_positive_class,
            self.target_negative_class,
        ) = self._encode_binary_series(
            self.PREDICTIVE[self.target],
            positive_class=default_positive_class,
            series_name=self.target,
        )

        self.model_comparison = pd.DataFrame(columns=PREDICTIVE_COMPARISON_COLS)
        self.final_metrics = pd.DataFrame(columns=PREDICTIVE_METRICS_COLS)
        self.feature_importance_df = pd.DataFrame(columns=["feature", "importance"])
        self.best_model = None
        self.best_model_name = None
        self.y_test = None
        self.y_prob_test = None
        self.y_pred_test = None
        self.run_metadata: dict[str, object] = {}

    def __str__(self) -> str:
        model_frame_predictors = [col for col in self.PREDICTIVE.columns if col != self.target]
        predictive_ready = len(self.final_metrics) > 0

        status_color = GREEN if predictive_ready else YELLOW
        status_text = "READY ✅" if predictive_ready else "NOT RUN YET ⏳"

        lines = [
            f"{BOLD}{BLUE}{'═' * 64}{RESET}",
            f"{BOLD}{BLUE}🧠  YlivertainenPredictive Briefing{RESET}",
            f"{BOLD}{BLUE}{'═' * 64}{RESET}",
            f"{GRAY}📌 Task{RESET}                 {BOLD}{self.task_name}{RESET}",
            f"{GRAY}🎯 Target{RESET}               {ORANGE}{BOLD}{self.target}{RESET}",
            f"{GRAY}✅ Positive class{RESET}       {GREEN}{self.target_positive_class!r}{RESET}",
            f"{GRAY}👥 Rows (n){RESET}             {BOLD}{len(self.PREDICTIVE):,}{RESET}",
            f"{GRAY}🧰 Predictors (model){RESET}   {BOLD}{len(model_frame_predictors)}{RESET}",
            f"{GRAY}🧪 Predictors (approved){RESET} {BOLD}{len(self.approved_predictors)}{RESET}",
            f"{GRAY}📊 Predictive artifacts{RESET} {status_color}{BOLD}{status_text}{RESET}",
            "",
            f"{BOLD}{YELLOW}Methods{RESET}",
            f"{GRAY}•{RESET} overview() / print(project) -> show this briefing",
            f"{GRAY}•{RESET} approved_predictor_columns() -> list approved predictors",
            f"{GRAY}•{RESET} run_predictive(...) -> run model selection + final test evaluation",
            "",
            f"{BOLD}{YELLOW}Model strategy{RESET}",
            f"{GRAY}•{RESET} leakage-safe preprocessing via ColumnTransformer inside each CV fold",
            f"{GRAY}•{RESET} candidate models: logistic regression, random forest, gradient boosting",
            f"{GRAY}•{RESET} hyperparameter search with stratified CV on train split only",
            f"{GRAY}•{RESET} untouched final test split for unbiased report metrics",
            f"{GRAY}•{RESET} test AUC CI by bootstrap + threshold sensitivity/specificity",
            "",
            f"{BOLD}{YELLOW}Output columns{RESET}",
            f"{GRAY}•{RESET} model comparison: {', '.join(PREDICTIVE_COMPARISON_COLS)}",
            f"{GRAY}•{RESET} final metrics: {', '.join(PREDICTIVE_METRICS_COLS)}",
            f"{BOLD}{BLUE}{'═' * 64}{RESET}",
        ]
        return "\n".join(lines)

    def overview(self):
        print(self)
        return None

    def approved_predictor_columns(self, covariates: list[str] | str | None = None) -> list[str]:
        required_cols = {"column_name", "role", "action"}
        if not required_cols.issubset(self.feature_decisions_df.columns):
            raise ValueError("feature_decisions_df must contain column_name, role, and action.")

        approved = self.feature_decisions_df.loc[
            (self.feature_decisions_df["role"] == "predictor")
            & (self.feature_decisions_df["action"] != "drop"),
            "column_name",
        ].dropna()
        approved = [str(col) for col in approved.tolist()]
        approved = [col for col in dict.fromkeys(approved) if col in self.PREDICTIVE.columns and col != self.target]

        if len(approved) == 0:
            raise ValueError("No approved predictors found in df_model.")

        if covariates is None:
            return approved

        requested = [covariates] if isinstance(covariates, str) else list(dict.fromkeys(covariates))
        missing_cols = [col for col in requested if col not in self.PREDICTIVE.columns]
        if missing_cols:
            raise ValueError(f"Requested covariates not found in df_model: {missing_cols}")

        unapproved = [col for col in requested if col not in approved]
        if unapproved:
            raise ValueError(f"Requested covariates are not approved predictors: {unapproved}")

        return requested

    def run_predictive(
        self,
        covariates: list[str] | str | None = None,
        positive_class=None,
        test_size: float = 0.20,
        cv_folds: int = 5,
        n_boot: int = 1000,
        ci_level: float = 0.95,
        threshold_strategy: str = "youden",
        min_sensitivity_target: float = 0.75,
        verbose: bool = False,
        show_tables: bool = False,
        export: bool = False,
    ) -> dict[str, object]:
        if threshold_strategy not in {"youden", "fixed_0_50", "min_sensitivity"}:
            raise ValueError("threshold_strategy must be one of: youden, fixed_0_50, min_sensitivity.")
        if not 0 < test_size < 0.5:
            raise ValueError("test_size must be between 0 and 0.5.")
        if cv_folds < 3:
            raise ValueError("cv_folds must be at least 3.")
        if n_boot < 200:
            raise ValueError("n_boot must be at least 200 for a stable AUC CI.")
        if not 0 < ci_level < 1:
            raise ValueError("ci_level must be between 0 and 1.")

        predictors = self.approved_predictor_columns(covariates)
        resolved_positive_class = self.default_positive_class if positive_class is None else positive_class
        y_binary, positive_class, negative_class = self._encode_binary_series(
            self.PREDICTIVE[self.target],
            positive_class=resolved_positive_class,
            series_name=self.target,
        )

        X = self.PREDICTIVE[predictors].copy()
        nan_mask = y_binary.isna()
        if nan_mask.any():
            n_dropped = int(nan_mask.sum())
            print(f"Warning: dropping {n_dropped} rows with missing target before split.")
            X = X.loc[~nan_mask].copy()
            y_binary = y_binary.loc[~nan_mask].copy()
        y = y_binary.astype(int)
        split = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )
        X_train, X_test, y_train, y_test = split

        preprocessor = self._build_preprocessor(X_train)
        searchers = self._build_candidate_searches(
            preprocessor=preprocessor,
            cv_folds=cv_folds,
            verbose=int(verbose),
        )
        gb_sample_weight = compute_sample_weight("balanced", y_train)

        comparison_rows = []
        fitted_searches: dict[str, GridSearchCV] = {}
        for model_name, search in searchers.items():
            if model_name == "gradient_boosting":
                fit_params = {"model__sample_weight": gb_sample_weight}
                search.fit(X_train, y_train, **fit_params)
            else:
                search.fit(X_train, y_train)
            fitted_searches[model_name] = search

            best_idx = int(search.best_index_)
            std_col = "std_test_score"
            cv_std = float(search.cv_results_[std_col][best_idx]) if std_col in search.cv_results_ else np.nan
            comparison_rows.append(
                {
                    "model_name": model_name,
                    "best_cv_auc_mean": float(search.best_score_),
                    "best_cv_auc_std": cv_std,
                    "best_params": str(search.best_params_),
                }
            )

        comparison_df = pd.DataFrame(comparison_rows, columns=PREDICTIVE_COMPARISON_COLS)
        comparison_df = comparison_df.sort_values("best_cv_auc_mean", ascending=False).reset_index(drop=True)
        best_model_name = str(comparison_df.iloc[0]["model_name"])
        if len(comparison_df) >= 2:
            gap = float(comparison_df.iloc[0]["best_cv_auc_mean"]) - float(comparison_df.iloc[1]["best_cv_auc_mean"])
            winner_std = float(comparison_df.iloc[0]["best_cv_auc_std"])
            if gap < winner_std / 2:
                print(
                    f"Warning: top-2 model AUC gap ({gap:.4f}) < half of winner std ({winner_std / 2:.4f}). "
                    f"Selection of '{best_model_name}' over '{comparison_df.iloc[1]['model_name']}' is not conclusive."
                )
        best_search = fitted_searches[best_model_name]
        best_model = best_search.best_estimator_

        if threshold_strategy == "youden":
            threshold = self._resolve_youden_threshold(
                estimator=best_model,
                X_train=X_train,
                y_train=y_train,
                cv_folds=cv_folds,
            )
        elif threshold_strategy == "min_sensitivity":
            threshold = self._resolve_min_sensitivity_threshold(
                estimator=best_model,
                X_train=X_train,
                y_train=y_train,
                cv_folds=cv_folds,
                min_sensitivity_target=min_sensitivity_target,
            )
        else:
            threshold = 0.50

        y_prob_test = best_model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_prob_test >= threshold).astype(int)
        self.y_test = y_test.copy()
        self.y_prob_test = y_prob_test.copy()
        self.y_pred_test = y_pred_test.copy()

        test_auc = float(roc_auc_score(y_test, y_prob_test))
        ci_low, ci_high = self._bootstrap_auc_ci(
            y_true=y_test.to_numpy(),
            y_prob=y_prob_test,
            n_boot=n_boot,
            ci_level=ci_level,
            random_state=self.random_state,
        )
        test_brier = float(brier_score_loss(y_test, y_prob_test))
        sens, spec, tp, fp, tn, fn = self._binary_operating_metrics(y_test.to_numpy(), y_pred_test)

        metrics_df = pd.DataFrame(
            [
                {
                    "selected_model": best_model_name,
                    "threshold_strategy": threshold_strategy,
                    "threshold": float(threshold),
                    "test_auc": test_auc,
                    "test_auc_ci_low": float(ci_low),
                    "test_auc_ci_high": float(ci_high),
                    "test_brier": test_brier,
                    "test_sensitivity": float(sens),
                    "test_specificity": float(spec),
                    "test_tp": int(tp),
                    "test_fp": int(fp),
                    "test_tn": int(tn),
                    "test_fn": int(fn),
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                }
            ],
            columns=PREDICTIVE_METRICS_COLS,
        )

        self.model_comparison = comparison_df
        self.final_metrics = metrics_df
        self.best_model = best_model
        self.best_model_name = best_model_name
        self.feature_importance_df = self._extract_feature_importance(best_model, predictors)
        self.run_metadata = {
            "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_name": self.task_name,
            "target": self.target,
            "positive_class": positive_class,
            "negative_class": negative_class,
            "sample_size": int(len(self.PREDICTIVE)),
            "predictors_used": predictors,
            "predictors_approved_count": len(self.approved_predictors),
            "test_size": float(test_size),
            "cv_folds": int(cv_folds),
            "n_boot": int(n_boot),
            "ci_level": float(ci_level),
            "threshold_strategy": threshold_strategy,
            "min_sensitivity_target": float(min_sensitivity_target),
            "selected_model": best_model_name,
            "module_sha1": self._module_sha1(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "sklearn_version": sklearn.__version__,
        }

        if export:
            self._export_results()

        if show_tables:
            display(self.model_comparison)
            display(self.final_metrics)

        return {
            "model_comparison": self.model_comparison,
            "final_metrics": self.final_metrics,
            "best_model_name": self.best_model_name,
            "best_model": self.best_model,
            "run_metadata": self.run_metadata,
            "y_test": self.y_test,
            "y_prob_test": self.y_prob_test,
            "y_pred_test": self.y_pred_test,
        }

    def _resolve_target_col(self, target_col: str | None) -> str:
        if target_col is not None:
            if target_col not in self.PREDICTIVE.columns:
                raise ValueError(f"Target column not found in df_model: {target_col}")
            return target_col

        if {"column_name", "role"}.issubset(self.feature_decisions_df.columns):
            target_rows = self.feature_decisions_df.loc[
                self.feature_decisions_df["role"] == "target",
                "column_name",
            ].dropna()
            target_rows = [str(col) for col in target_rows.tolist() if str(col) in self.PREDICTIVE.columns]
            if len(target_rows) == 1:
                return target_rows[0]

        if "target" in self.PREDICTIVE.columns:
            return "target"

        raise ValueError("Could not resolve target column from feature_decisions_df or df_model.")

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

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_features = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical_features = [col for col in X.columns if col not in numeric_features]

        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_features),
                ("cat", categorical_pipe, categorical_features),
            ],
            remainder="drop",
        )

    def _build_candidate_searches(
        self,
        preprocessor: ColumnTransformer,
        cv_folds: int,
        verbose: int = 0,
    ) -> dict[str, GridSearchCV]:
        from sklearn.base import clone as _clone

        lr_preprocessor = _clone(preprocessor)
        rf_preprocessor = _clone(preprocessor)
        gb_preprocessor = _clone(preprocessor)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        candidates: dict[str, tuple[Pipeline, dict[str, list]]] = {
            "logistic_regression": (
                Pipeline(
                    steps=[
                        ("preprocess", lr_preprocessor),
                        ("model", LogisticRegression(max_iter=3000, random_state=self.random_state)),
                    ]
                ),
                {
                    "model__solver": ["lbfgs", "liblinear"],
                    "model__C": [0.1, 1.0, 10.0],
                    "model__class_weight": [None, "balanced"],
                },
            ),
            "random_forest": (
                Pipeline(
                    steps=[
                        ("preprocess", rf_preprocessor),
                        (
                            "model",
                            RandomForestClassifier(
                                random_state=self.random_state,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
                {
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [None, 8, 16],
                    "model__min_samples_leaf": [1, 3, 5],
                    "model__class_weight": [None, "balanced"],
                },
            ),
            "gradient_boosting": (
                Pipeline(
                    steps=[
                        ("preprocess", gb_preprocessor),
                        ("model", GradientBoostingClassifier(random_state=self.random_state)),
                    ]
                ),
                {
                    "model__n_estimators": [100, 200],
                    "model__learning_rate": [0.03, 0.1],
                    "model__max_depth": [2, 3],
                },
            ),
        }

        searches = {}
        for model_name, (pipeline, param_grid) in candidates.items():
            searches[model_name] = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                refit=True,
                return_train_score=False,
                verbose=verbose,
            )
        return searches

    def _resolve_youden_threshold(
        self,
        estimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int,
    ) -> float:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        oof_prob = cross_val_predict(
            clone(estimator),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        thresholds = np.linspace(0.05, 0.95, 181)
        best_threshold = 0.5
        best_score = -np.inf
        y_true = y_train.to_numpy(dtype=int)

        for threshold in thresholds:
            y_pred = (oof_prob >= threshold).astype(int)
            sens, spec, _, _, _, _ = self._binary_operating_metrics(y_true, y_pred)
            youden_j = sens + spec - 1.0
            if youden_j > best_score:
                best_score = youden_j
                best_threshold = float(threshold)

        return best_threshold

    def _resolve_min_sensitivity_threshold(
        self,
        estimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int,
        min_sensitivity_target: float,
    ) -> float:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        oof_prob = cross_val_predict(
            clone(estimator),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        thresholds = np.linspace(0.05, 0.95, 181)
        best_threshold = 0.5
        best_youden = -np.inf
        y_true = y_train.to_numpy(dtype=int)
        for threshold in thresholds:
            y_pred = (oof_prob >= threshold).astype(int)
            sens, spec, _, _, _, _ = self._binary_operating_metrics(y_true, y_pred)
            if np.isnan(sens) or sens < min_sensitivity_target:
                continue
            youden_j = sens + spec - 1.0
            if youden_j > best_youden:
                best_youden = youden_j
                best_threshold = float(threshold)
        return best_threshold

    def _binary_operating_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[float, float, int, int, int, int]:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
        return sens, spec, int(tp), int(fp), int(tn), int(fn)

    def _bootstrap_auc_ci(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_boot: int,
        ci_level: float,
        random_state: int,
    ) -> tuple[float, float]:
        from scipy.stats import bootstrap as scipy_bootstrap

        def _auc_stat(y_t, y_p):
            if len(np.unique(y_t)) < 2:
                return np.nan
            return roc_auc_score(y_t, y_p)

        try:
            result = scipy_bootstrap(
                (y_true, y_prob),
                statistic=lambda yt, yp: _auc_stat(yt, yp),
                n_resamples=n_boot,
                confidence_level=ci_level,
                method="BCa",
                random_state=random_state,
                paired=True,
            )
            return float(result.confidence_interval.low), float(result.confidence_interval.high)
        except Exception:
            # Fallback to percentile if BCa fails (e.g., degenerate samples)
            rng = np.random.default_rng(random_state)
            n = len(y_true)
            auc_values = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                s_y, s_p = y_true[idx], y_prob[idx]
                if len(np.unique(s_y)) < 2:
                    continue
                auc_values.append(float(roc_auc_score(s_y, s_p)))
            if len(auc_values) < max(50, int(n_boot * 0.20)):
                return np.nan, np.nan
            alpha = (1.0 - ci_level) / 2.0
            return float(np.quantile(auc_values, alpha)), float(np.quantile(auc_values, 1 - alpha))

    def _extract_feature_importance(self, model, feature_names: list[str]) -> pd.DataFrame:
        try:
            step = model.named_steps["model"]
            if hasattr(step, "feature_importances_"):
                importances = step.feature_importances_
            elif hasattr(step, "coef_"):
                importances = np.abs(step.coef_[0])
            else:
                return pd.DataFrame(columns=["feature", "importance"])
            try:
                transformed_names = model.named_steps["preprocess"].get_feature_names_out()
            except Exception:
                transformed_names = [f"feature_{i}" for i in range(len(importances))]
            df = pd.DataFrame({"feature": transformed_names, "importance": importances})
            return df.sort_values("importance", ascending=False).reset_index(drop=True)
        except Exception:
            return pd.DataFrame(columns=["feature", "importance"])

    def _module_sha1(self) -> str:
        try:
            with open(__file__, "rb") as handle:
                return hashlib.sha1(handle.read()).hexdigest()
        except OSError:
            return "unavailable"

    def _export_results(self):
        if self.best_model is None:
            raise ValueError("No trained model to export. Run run_predictive() first.")

        tables_dir = self.root / "reports" / "tables"
        models_dir = self.root / "reports" / "models"
        tables_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        comparison_path = tables_dir / f"(PREDICTIVE_model_comparison){self.task_name}.pickle"
        metrics_path = tables_dir / f"(PREDICTIVE_final_metrics){self.task_name}.pickle"
        metadata_path = tables_dir / f"(PREDICTIVE_metadata){self.task_name}.pickle"
        model_path = models_dir / f"(PREDICTIVE_model){self.task_name}.joblib"

        self.model_comparison.to_pickle(comparison_path)
        self.final_metrics.to_pickle(metrics_path)
        pd.Series(self.run_metadata, dtype="object").to_pickle(metadata_path)

        joblib_dump(self.best_model, model_path)

        print(f"{GREEN}Saved predictive model comparison:{RESET} {BOLD}{comparison_path}{RESET}")
        print(f"{GREEN}Saved predictive final metrics:{RESET} {BOLD}{metrics_path}{RESET}")
        print(f"{GREEN}Saved predictive metadata:{RESET} {BOLD}{metadata_path}{RESET}")
        print(f"{GREEN}Saved predictive model artifact:{RESET} {BOLD}{model_path}{RESET}")


    def predictive_summary(self):  
        # ===== PREDICTIVE SUMMARY CODE =====
        model_comparison = self.model_comparison.copy()
        final_metrics = self.final_metrics.copy()
        if final_metrics.empty:
            display(
                Markdown(
                    """### 🤖 Predictive Summary

> ⚠️ No predictive results yet. Run `run_predictive(...)` first.
"""
                )
            )
            return None

        best = final_metrics.iloc[0]

        selected_model = best["selected_model"]
        auc = float(best["test_auc"])
        auc_lo = float(best["test_auc_ci_low"])
        auc_hi = float(best["test_auc_ci_high"])
        threshold_strategy = best["threshold_strategy"]
        threshold = float(best["threshold"])
        sens = float(best["test_sensitivity"])
        spec = float(best["test_specificity"])
        brier = float(best["test_brier"])
        tp = int(best["test_tp"])
        fp = int(best["test_fp"])
        tn = int(best["test_tn"])
        fn = int(best["test_fn"])
        n_train = int(best["n_train"])
        n_test = int(best["n_test"])

        display(
            Markdown(
                f"""## 🤖 Predictive Summary

- **Task:** `{self.task_name}`
- **Selected model:** `{selected_model}`

### 📈 Test Performance
- <span style="color:#2563eb;"><b>ROC-AUC:</b></span> **`{auc:.3f}`**
- <span style="color:#2563eb;"><b>95% AUC CI:</b></span> **`{auc_lo:.3f}`** to **`{auc_hi:.3f}`**
- <span style="color:#7c3aed;"><b>Threshold:</b></span> strategy **`{threshold_strategy}`**, value **`{threshold:.3f}`**
- <span style="color:#16a34a;"><b>Sensitivity / Specificity:</b></span> **`{sens:.3f}`** / **`{spec:.3f}`**
- <span style="color:#f59e0b;"><b>Brier score:</b></span> **`{brier:.3f}`**

### 🧪 Candidate Model Comparison (CV ROC-AUC)
"""
            )
        )

        display(model_comparison)

        display(
            Markdown(
                f"""### ✅ Final Test Confusion Counts

- **TP / FP / TN / FN:** `{tp}` / `{fp}` / `{tn}` / `{fn}`
- **Train/Test sizes:** `{n_train}` / `{n_test}`
"""
            )
        )