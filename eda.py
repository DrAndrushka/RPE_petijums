#============ imports ============
from collections.abc import Collection
from IPython.display import display

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
import numpy as np
from scipy.stats import pearsonr, chi2_contingency

#============ ANSI Escape Codes for formating ============
BOLD = "\033[1m"
BLUE = "\033[38;5;39m"
GREEN = "\033[38;5;28m"
YELLOW = "\033[38;5;214m"
ORANGE = "\033[38;5;202m"
RED = "\033[38;5;196m"
RESET = "\033[0m"
#================================================
#                Class definition 
#================================================
class YlivertainenEDA:

    #============ Ship's Log: Registering the Loot ============
    def __init__(self, df, task_name):
        
        self.EDA = df.copy()
        self.task_name = task_name

        self.predictor_continuous = []
        self.predictor_discrete = []
        self.predictor_binary = []
        self.predictor_categorical_nominal = []
        self.predictor_categorical_ordinal = []
        self.predictor_time_to_event = []
        self.predictor_datetime = []
        self.predictor_text = []

        self.unclassified_predictors = {}
        
        self.all_predictors = []

        self.associations_table = pd.DataFrame()
        self.feature_decisions_table = pd.DataFrame()
  
    #============================================================
    #                       Whitelist Data
    #============================================================
    #========== HELPERs ==========
    def _effective_unique(self, s, min_freq=0.01):
        vc = s.value_counts(dropna=True, normalize=True)
        vc_lean = vc[vc >= min_freq]
        return vc_lean.index.size
    #=============================
    def whitelist_columns(self, predictors, numerical_continuous):

        if not predictors or len(predictors) == 0:
            raise ValueError('❌ predictors not set')

        requested_cols = list(dict.fromkeys([*predictors, *numerical_continuous]))
        missing_cols = [col for col in requested_cols if col not in self.EDA.columns]
        if missing_cols:
            raise ValueError(f'❌ Missing columns: {missing_cols}')

        #========== HELPERs ==========
        def _col_classifier(col, numerical_continuous):
            s = self.EDA[col].dropna()
            if '_timedelta_' in col:
                self.predictor_time_to_event.append(col)
            elif pd.api.types.is_numeric_dtype(s):
                eff_unique = self._effective_unique(s)
                if eff_unique == 2:
                    self.predictor_binary.append(col)
                else:
                    # here you decide continuous vs discrete
                    # simplest: floats => continuous, ints => discrete
                    if col in numerical_continuous:
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
                self.unclassified_predictors[col] = s.dtype
                self.predictor_discrete.append(col)

            self.all_predictors = (
                self.predictor_continuous +
                self.predictor_discrete +
                self.predictor_binary +
                self.predictor_categorical_nominal +
                self.predictor_categorical_ordinal +
                self.predictor_time_to_event +
                self.predictor_datetime +
                self.predictor_text)
        
        def _predictor_printer():
            if self.unclassified_predictors:
                print(f'❌ Those predictors were not classified:')
                for col,dtype in self.unclassified_predictors.items():
                    print(f'- {col}|{dtype}')
            else:
                print(f'{BOLD}✅ All cols classified {RESET}')
            
            print(f'{BOLD}===== Main variables ====={RESET}')
            print(f'{BOLD}Target:{RESET} {self.task_name} AS "target" ({self.EDA[self.target].dtype})')
            print(f'{BOLD}Predictors:{RESET} {len(self.all_predictors)} total')
            if self.predictor_continuous:
                print(f'- Numerical Continuous: {", ".join(self.predictor_continuous)}')
            if self.predictor_discrete:
                print(f'- Numerical Discrete: {", ".join(self.predictor_discrete)}')
            if self.predictor_binary:
                print(f'- Binary: {", ".join(self.predictor_binary)}')
            if self.predictor_categorical_nominal:
                print(f'- Categorical Nominal: {", ".join(self.predictor_categorical_nominal)}')
            if self.predictor_categorical_ordinal:
                print(f'- Categorical Ordinal: {", ".join(self.predictor_categorical_ordinal)}')
            if self.predictor_time_to_event:
                print(f'- Time To Event: {", ".join(self.predictor_time_to_event)}')
            if self.predictor_datetime:
                print(f'- Datetime: {", ".join(self.predictor_datetime)}')
            if self.predictor_text:
                print(f'- Text: {", ".join(self.predictor_text)}')
        #=============================
        
        for col in predictors:
            _col_classifier(col, numerical_continuous)

        _predictor_printer()

        ordered_cols = list(dict.fromkeys([self.target, *self.all_predictors]))
        self.EDA = self.EDA[ordered_cols].copy()
        return self

    #============================================================
    #                    Association Summary
    #============================================================
    #========== HELPERs ==========
    def _infer_type(self, col):
        if col.endswith('_missing'):
            return 'missingness'

        s = self.EDA[col]
        non_null = s.dropna()
        n_unique = int(non_null.nunique(dropna=True)) if len(non_null) > 0 else 0

        if pd.api.types.is_bool_dtype(s):
            return 'binary'
        if n_unique == 2:
            return 'binary'
        if pd.api.types.is_numeric_dtype(s):
            return 'number'
        if isinstance(s.dtype, pd.CategoricalDtype):
            return 'category'
        if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
            return 'category'
        return pd.NA

    def _pair_type(self, left_kind, right_kind):
        left = 'binary' if left_kind == 'missingness' else left_kind
        right = 'binary' if right_kind == 'missingness' else right_kind

        if left == 'number' and right == 'number':
            return 'number_x_number'
        if {left, right} == {'number', 'binary'}:
            return 'number_x_binary'
        if {left, right} == {'number', 'category'}:
            return 'number_x_category'
        if left == 'binary' and right == 'binary':
            return 'binary_x_binary'
        if {left, right} == {'binary', 'category'}:
            return 'binary_x_category'
        if left == 'category' and right == 'category':
            return 'category_x_category'
        return pd.NA

    #=============================
    def build_associations_table(
        self,
        leakage_vars: Collection[str] | None = None,
    ) -> pd.DataFrame:

        df = self.EDA
        target = self.target

        if not isinstance(target, str) or not target:
            raise ValueError('Target must be a single column name.')
        if target not in df.columns:
            raise ValueError('Target column not found in EDA DataFrame.')

        leakage_vars = set() if leakage_vars is None else set(leakage_vars)

        summary_columns = [
            'analysis_col',
            'predictor_col',
            'analysis_type',
            'predictor_type',
            'pair_type',
            'n_used',
            'analysis_excluded_n',
            'predictor_missing_n',
            'status',
            'note',
            'test_name',
            'test_stat',
            'p_value',
            'effect_size_name',
            'effect_size',
            'p_value_note',
            'leakage_flag',
        ]

        predictor_type_map = {
            'continuous': self.predictor_continuous,
            'discrete': self.predictor_discrete,
            'binary': self.predictor_binary,
            'categorical_nominal': self.predictor_categorical_nominal,
            'categorical_ordinal': self.predictor_categorical_ordinal,
            'time_to_event': self.predictor_time_to_event,
            'datetime': self.predictor_datetime,
            'text': self.predictor_text,
        }

        def _get_analysis_type_for_target(series: pd.Series) -> str:
            non_null = series.dropna()
            if pd.api.types.is_bool_dtype(series):
                return 'binary'

            n_unique = int(non_null.nunique(dropna=True)) if len(non_null) > 0 else 0
            if n_unique == 2:
                return 'binary'
            if pd.api.types.is_numeric_dtype(series) and n_unique > 2:
                return 'continuous'
            return 'categorical'

        def _p_value_note(p_value, status: str) -> str:
            if status != 'ok' or p_value is None or pd.isna(p_value):
                return 'not_run'
            if p_value < 0.001:
                return 'sig_0.001'
            if p_value < 0.01:
                return 'sig_0.01'
            if p_value < 0.05:
                return 'sig_0.05'
            return 'ns'

        def _cramers_v(table: pd.DataFrame):
            if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
                return np.nan

            n = table.to_numpy().sum()
            if n == 0:
                return np.nan

            chi2, _, _, _ = chi2_contingency(table)
            denom = n * (min(table.shape) - 1)
            if denom <= 0:
                return np.nan
            return float(np.sqrt(chi2 / denom))

        def _skipped_result(note: str, test_name=None) -> dict:
            return {
                'status': 'skipped',
                'note': note,
                'test_name': test_name,
                'test_stat': np.nan,
                'p_value': np.nan,
                'effect_size_name': None,
                'effect_size': np.nan,
            }

        def _ok_result(test_name: str, test_stat: float, p_value: float, effect_size_name=None, effect_size=np.nan) -> dict:
            return {
                'status': 'ok',
                'note': pd.NA,
                'test_name': test_name,
                'test_stat': float(test_stat),
                'p_value': float(p_value),
                'effect_size_name': effect_size_name,
                'effect_size': float(effect_size) if pd.notna(effect_size) else np.nan,
            }

        def _run_association_test(
            x: pd.Series,
            y: pd.Series,
            predictor_type: str,
            target_type: str,
        ) -> dict:
            if predictor_type in {'datetime', 'text'}:
                return _skipped_result('unsupported_predictor_type')

            if target_type == 'continuous' and predictor_type in {'continuous', 'discrete', 'time_to_event'}:
                x_num = pd.to_numeric(x, errors='coerce')
                y_num = pd.to_numeric(y, errors='coerce')
                if x_num.nunique(dropna=True) < 2 or y_num.nunique(dropna=True) < 2:
                    return _skipped_result('too_few_unique')
                stat, p_value = pearsonr(x_num, y_num)
                return _ok_result('pearsonr', stat, p_value, effect_size_name='r', effect_size=stat)

            if target_type == 'binary' and predictor_type in {'continuous', 'discrete', 'time_to_event'}:
                x_num = pd.to_numeric(x, errors='coerce')
                if x_num.nunique(dropna=True) < 2:
                    return _skipped_result('too_few_unique')

                if pd.api.types.is_bool_dtype(y):
                    y_num = y.astype(int)
                else:
                    y_num = pd.Series(pd.Categorical(y).codes, index=y.index)

                if y_num.nunique(dropna=True) < 2:
                    return _skipped_result('too_few_unique')

                stat, p_value = pearsonr(x_num, y_num.astype(float))
                return _ok_result('point_biserial', stat, p_value, effect_size_name='r', effect_size=stat)

            if target_type in {'binary', 'categorical'} and predictor_type in {
                'binary',
                'categorical_nominal',
                'categorical_ordinal',
                'discrete',
            }:
                table = pd.crosstab(y, x)
                if table.shape[0] < 2 or table.shape[1] < 2:
                    return _skipped_result('degenerate_contingency')

                chi2, p_value, _, _ = chi2_contingency(table)
                cramers_v = _cramers_v(table)
                return _ok_result('chi2', chi2, p_value, effect_size_name='cramers_v', effect_size=cramers_v)

            return _skipped_result('unsupported_pair')

        predictor_specs = []
        seen_predictors = set()
        for predictor_type, cols in predictor_type_map.items():
            for col in cols:
                if col == target or col in seen_predictors or col not in df.columns:
                    continue
                predictor_specs.append((col, predictor_type))
                seen_predictors.add(col)

        analysis_type = _get_analysis_type_for_target(df[target])
        rows = []

        for predictor_col, predictor_type in predictor_specs:
            analysis_series = df[target]
            predictor_series = df[predictor_col]
            mask_non_missing = analysis_series.notna() & predictor_series.notna()
            analysis_clean = analysis_series.loc[mask_non_missing]
            predictor_clean = predictor_series.loc[mask_non_missing]
            n_used = int(mask_non_missing.sum())

            row = {
                'analysis_col': target,
                'predictor_col': predictor_col,
                'analysis_type': analysis_type,
                'predictor_type': predictor_type,
                'pair_type': 'predictor_target',
                'n_used': n_used,
                'analysis_excluded_n': int(analysis_series.isna().sum()),
                'predictor_missing_n': int(predictor_series.isna().sum()),
                'status': 'skipped',
                'note': pd.NA,
                'test_name': pd.NA,
                'test_stat': np.nan,
                'p_value': np.nan,
                'effect_size_name': pd.NA,
                'effect_size': np.nan,
                'p_value_note': 'not_run',
                'leakage_flag': (target in leakage_vars) or (predictor_col in leakage_vars),
            }

            if n_used < 10:
                result = _skipped_result('too_few_non_missing')
            else:
                result = _run_association_test(
                    x=predictor_clean,
                    y=analysis_clean,
                    predictor_type=predictor_type,
                    target_type=analysis_type,
                )

            row.update(result)
            if row['status'] != 'ok':
                row['test_stat'] = np.nan
                row['p_value'] = np.nan
                row['effect_size'] = np.nan
                row['effect_size_name'] = None
            row['p_value_note'] = _p_value_note(row['p_value'], row['status'])
            rows.append(row)

        associations_table = pd.DataFrame(rows, columns=summary_columns)
        if len(associations_table) > 0:
            for col in ['analysis_col', 'predictor_col', 'note', 'test_name', 'effect_size_name', 'p_value_note']:
                associations_table[col] = associations_table[col].astype('string[python]')
            for col in ['analysis_type', 'predictor_type', 'pair_type', 'status']:
                associations_table[col] = associations_table[col].astype('string[python]')
            for col in ['n_used', 'analysis_excluded_n', 'predictor_missing_n']:
                associations_table[col] = pd.to_numeric(associations_table[col], errors='coerce').astype('Int64')
            for col in ['test_stat', 'p_value', 'effect_size']:
                associations_table[col] = pd.to_numeric(associations_table[col], errors='coerce').astype('Float64')
            associations_table['leakage_flag'] = associations_table['leakage_flag'].astype('boolean')
        else:
            associations_table = pd.DataFrame(
                {
                    'analysis_col': pd.Series(dtype='string[python]'),
                    'predictor_col': pd.Series(dtype='string[python]'),
                    'analysis_type': pd.Series(dtype='string[python]'),
                    'predictor_type': pd.Series(dtype='string[python]'),
                    'pair_type': pd.Series(dtype='string[python]'),
                    'n_used': pd.Series(dtype='Int64'),
                    'analysis_excluded_n': pd.Series(dtype='Int64'),
                    'predictor_missing_n': pd.Series(dtype='Int64'),
                    'status': pd.Series(dtype='string[python]'),
                    'note': pd.Series(dtype='string[python]'),
                    'test_name': pd.Series(dtype='string[python]'),
                    'test_stat': pd.Series(dtype='Float64'),
                    'p_value': pd.Series(dtype='Float64'),
                    'effect_size_name': pd.Series(dtype='string[python]'),
                    'effect_size': pd.Series(dtype='Float64'),
                    'p_value_note': pd.Series(dtype='string[python]'),
                    'leakage_flag': pd.Series(dtype='boolean'),
                }
            )

        self.associations_table = associations_table

        return associations_table

    #=============================================================
    #                 Feature Selection Decisions 
    #=============================================================
    def build_feature_decisions_table(
        self,
        missingness_dict: dict[str, str] | None = None,
        leakage_vars: Collection[str] | None = None,
        redundancy_pairs: pd.DataFrame | None = None,
    ) -> pd.DataFrame:

        df = self.EDA
        missingness_dict = {} if missingness_dict is None else dict(missingness_dict)
        leakage_vars = set() if leakage_vars is None else set(leakage_vars)

        feature_columns = [
            'column_name',
            'role',
            'dtype',
            'inferred_type',
            'missing_pct',
            'drop_reason',
            'action',
            'missing_action',
            'notes',
        ]

        inferred_type_map = {}
        for inferred_type, cols in [
            ('continuous', self.predictor_continuous),
            ('discrete', self.predictor_discrete),
            ('binary', self.predictor_binary),
            ('categorical_nominal', self.predictor_categorical_nominal),
            ('categorical_ordinal', self.predictor_categorical_ordinal),
            ('time_to_event', self.predictor_time_to_event),
            ('datetime', self.predictor_datetime),
            ('text', self.predictor_text),
        ]:
            for col in cols:
                inferred_type_map[col] = inferred_type

        redundancy_drop_map = {}
        if redundancy_pairs is not None and len(redundancy_pairs) > 0:
            required_cols = {'var_a', 'var_b', 'preferred_keep'}
            missing_cols = required_cols.difference(redundancy_pairs.columns)
            if missing_cols:
                raise ValueError(f'redundancy_pairs missing required columns: {sorted(missing_cols)}')

            for _, pair_row in redundancy_pairs.iterrows():
                preferred_keep = pair_row['preferred_keep']
                for candidate_col in [pair_row['var_a'], pair_row['var_b']]:
                    if pd.isna(candidate_col) or candidate_col == preferred_keep:
                        continue
                    redundancy_drop_map[str(candidate_col)] = str(preferred_keep)

        def _missing_action(missing_pct: float, missingness_type: str | None, inferred_type: str) -> str:
            if missing_pct == 0:
                return 'no_missing'
            if missingness_type == 'STRUCTURAL pattern':
                return 'do_not_impute'
            if missingness_type == 'MNAR':
                return 'flag_only'
            if missing_pct < 20:
                return 'simple_impute'
            if missing_pct <= 60:
                if inferred_type in {'categorical_nominal', 'categorical_ordinal'}:
                    return 'simple_impute'
                return 'advanced_impute'
            return 'flag_only'

        def _role_for_column(col: str) -> str:
            if col == self.target:
                return 'target'
            if col in leakage_vars:
                return 'leakage'
            if col in inferred_type_map:
                return 'predictor'
            return 'helper'

        rows = []
        base_cols = [col for col in df.columns if not col.endswith('_missing')]

        for col in base_cols:
            dtype = str(df[col].dtype)
            inferred_type = inferred_type_map.get(col, 'unknown')
            missing_pct = float(df[col].isna().mean() * 100.0)
            role = _role_for_column(col)
            missingness_type = missingness_dict.get(f'{col}_missing')
            missing_action = _missing_action(missing_pct, missingness_type, inferred_type)

            action = 'keep'
            drop_reason = ''
            notes = []

            if missingness_type:
                notes.append(f'missingness_type={missingness_type}')
            if missingness_type == 'STRUCTURAL pattern':
                notes.append('Structural; encode Not Applicable / keep missing')
            elif missingness_type == 'MNAR':
                notes.append('MNAR; keep missing flag and avoid heavy imputation')

            if 40.0 < missing_pct <= 60.0:
                notes.append('high_missingness_flag_important')

            if role == 'leakage':
                action = 'drop'
                drop_reason = 'leakage'
                notes.append('leakage_feature')
            elif missing_pct > 60.0 and missingness_type in {'MNAR', 'Unclassifiable'}:
                action = 'drop'
                drop_reason = 'extreme_missing'
                notes.append('extreme_missingness')

            preferred_keep = redundancy_drop_map.get(col)
            if preferred_keep is not None:
                action = 'drop'
                drop_reason = f'redundant_with_{preferred_keep}'
                notes.append(f'redundant_with={preferred_keep}')

            if role == 'id':
                notes.append('identifier_column')
            elif role == 'helper':
                notes.append('non_modeling_helper')

            rows.append(
                {
                    'column_name': col,
                    'role': role,
                    'dtype': dtype,
                    'inferred_type': inferred_type,
                    'missing_pct': missing_pct,
                    'drop_reason': drop_reason,
                    'action': action,
                    'missing_action': missing_action,
                    'notes': '; '.join(dict.fromkeys(notes)) if notes else pd.NA,
                }
            )

        feature_decisions_df = pd.DataFrame(rows, columns=feature_columns)
        if len(feature_decisions_df) > 0:
            for col in ['column_name', 'role', 'dtype', 'inferred_type', 'drop_reason', 'action', 'missing_action', 'notes']:
                feature_decisions_df[col] = feature_decisions_df[col].astype('string[python]')
            feature_decisions_df['missing_pct'] = pd.to_numeric(feature_decisions_df['missing_pct'], errors='coerce').astype('Float64')
        else:
            feature_decisions_df = pd.DataFrame(
                {
                    'column_name': pd.Series(dtype='string[python]'),
                    'role': pd.Series(dtype='string[python]'),
                    'dtype': pd.Series(dtype='string[python]'),
                    'inferred_type': pd.Series(dtype='string[python]'),
                    'missing_pct': pd.Series(dtype='Float64'),
                    'drop_reason': pd.Series(dtype='string[python]'),
                    'action': pd.Series(dtype='string[python]'),
                    'missing_action': pd.Series(dtype='string[python]'),
                    'notes': pd.Series(dtype='string[python]'),
                }
            )
        display(feature_decisions_df)
        self.feature_decisions_table = feature_decisions_df
        return feature_decisions_df

    #==============================================================
    #                      EXPORT BOTH TABLES 
    #==============================================================
    
    def export_both_tables(self, export=False):
        associations_path = 'reports'/'tables'/f'(EDA_associations_table){self.task_name}.xlsx'
        feature_decisions_path = 'reports'/'tables'/f'(EDA_feature_decisions_table){self.task_name}.xlsx'
        if export:
            self.associations_table.to_excel(associations_path)
            self.feature_decisions_table.to_excel(feature_decisions_path)
            print(f'✅ Successful EXPORT of EDA_feature_decisions_table')
            print(f'✅ Folder: {BLUE}{BOLD}ylivertainen/reports/tables/{RESET}')
            print(f'✅ Feature Decisions Table: {BLUE}{BOLD}(EDA_feature_decisions_table){self.task_name}{RESET}')
            print(f'✅ Associations Table: {BLUE}{BOLD}(EDA_associations_table){self.task_name}{RESET}')
            print(f'✅ FORMAT: {BLUE}{BOLD}excel{RESET}\n')
        else:

            print(f'🔄 Gonna save in: {BLUE}{BOLD}ylivertainen/reports/tables/{RESET}')
            print(f'🔄 Feature Decisions Table: {BLUE}{BOLD}(EDA_feature_decisions_table){self.task_name}{RESET}')
            print(f'🔄 Associations Table: {BLUE}{BOLD}(EDA_associations_table){self.task_name}{RESET}')
            print(f'🔄 FORMAT: {BLUE}{BOLD}excel{RESET}')
            print(f'===== {BOLD}export={BLUE}True{RESET} to save =====\n')
    
    
        print("═" * 70)
        print(f"{BOLD}🧠 EDA COMPLETE — FULL BRAIN ANGIOGRAM ACQUIRED — {RED}{self.task_name.upper()}{RESET}")
        print("─" * 70)

        n_rows, n_cols = self.EDA.shape
        total_nans = self.EDA.isna().sum().sum()
        nan_pct = (total_nans / (n_rows * n_cols)) * 100 if n_rows and n_cols else 0

        print(f"{GREEN}✔ EDA FRAME ONLINE FOR DECISION-MAKING{RESET}")
        print(f"   Shape         : {n_rows} rows x {n_cols} cols")
        print(f"   Remaining NaNs: {total_nans} cells ({nan_pct:0.2f}%)")
        print("   Contents      : associations mapped, redundancy flagged, leakage suspects tagged.")
        print()
        print(f"{BOLD}{GREEN}→ Neurosurgical note:{RESET} this is the point where others stop and "
            f"\n                        you push through — "
            f"\n                          turn this map into hard feature decisions and a clean model frame.")

        display(self.EDA.head(1))
