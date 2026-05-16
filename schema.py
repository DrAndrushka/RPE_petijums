#=====================================
#               IMPORTS
#=====================================
from dataclasses import dataclass
from typing import Literal
import numpy as np

#================================================
#           SCHEMA function definition
#================================================
@dataclass(frozen=True)
class ColSpec:
    # ===== SCHEMA =====
    name: str
    kind: Literal["numeric", "timedelta", "datetime", "categorical", "text", "boolean", "match", "delta"]
    nulls: tuple[object, ...] | None = None                                                         # one-item tuple ==> trailing comma ("something",)
    replace: dict[object, object] | None = None
    keep: bool = True                                                                               # keep=False = OK to drop after we’re done with it.
    # ===== dtype specific =====
    # Category:
    ordered: tuple[object, ...] | None = None                                                       # one-item tuple ==> trailing comma ("something",)
    # ===== DERIVED =====
    derive_from: tuple[str, str] | None = None                                                   # one-item tuple ==> trailing comma ("something",)
    # Match:
    match_by: tuple[object, ...] | None = None
    # Timedelta:
    timedelta_units: Literal['seconds', 'minutes', 'hours', 'days', 'months'] | None = None
    # Datetime:
    datetime_units: Literal["hour", "dow", "workday_bool", "month_name", "year"] | None = None

    def __post_init__(self):
        #print("DEBUG:", self.kind, self.timedelta_from)
        if self.kind != "categorical" and self.ordered is not None:
                raise ValueError("❌ trying to order non-categorical data")
        if self.ordered is not None and len(self.ordered) <= 1:
                raise ValueError("❌ While ordering categoricals: input is <=1 value")
        if self.kind == 'timedelta' and not self.timedelta_units:
                raise ValueError("❌ Specify which units to put in timedelta")
        if self.kind == 'match' and not self.match_by:
                raise ValueError("❌ Specify how to create the match column")
        if self.ordered is not None and self.nulls is not None:
                raise ValueError("❌ Redundant 'nulls' added. Unused values in 'ordered' are automatically filtered out")

#================================================
#=============== SCHEMA & DERIVED ===============
#================================================
SCHEMA = [
    ColSpec(name="nmpd_diag",
            kind="categorical",
            replace={1: 'G45.9', 2: 'I64', 3: 'I63.9'}),
    ColSpec(name="izrakstisanas_diag",
            kind="categorical"),
    ColSpec(name="vecums",
            kind="numeric"),
    ColSpec(name="dzimums",
            kind="categorical",
            replace={1: 'sieviete', 2: 'vīrietis'}),
    ColSpec(name="GKS",
            kind="categorical",
            ordered=tuple(range(3, 16, 1))),
    ColSpec(name="FastTest",
            kind="categorical",
            ordered=(1, 3, 4, 2)),
    ColSpec(name="izsaukuma_laiks",
            kind="datetime",
            keep=False),
    ColSpec(name="nogadasana_PSKUS_laiks",
            kind="datetime", 
            keep=False),
    ColSpec(name="patient_card_no",
            kind="text",
            keep=True),
]

DERIVED = [
    ColSpec(name="lidzPSKUS_timedelta_minutes",
            kind="timedelta",
            timedelta_units='minutes',
            derive_from=("izsaukuma_laiks", "nogadasana_PSKUS_laiks"),
            keep=False),
    ColSpec(name='TIA_match',
            kind='match',
            derive_from=('nmpd_diag', 'izrakstisanas_diag'),
            match_by=('G45',)),
    ColSpec(name='ischemic_match',
            kind='match',
            derive_from=('nmpd_diag', 'izrakstisanas_diag'),
            match_by=('I63', 'I64',)),
    ColSpec(name='any_cerebrovascular_match',
            kind='match',
            derive_from=('nmpd_diag', 'izrakstisanas_diag'),
            match_by=('I6', 'G45',)),
    ColSpec(name='izsaukuma_laiks_hour',
            kind='datetime',
            datetime_units='hour',
            derive_from=("izsaukuma_laiks",)),
    ColSpec(name='izsaukuma_laiks_dow',
            kind='datetime',
            datetime_units='dow',
            derive_from=("izsaukuma_laiks",)),
    ColSpec(name='izsaukuma_laiks_workday_bool',
            kind='datetime',
            datetime_units='workday_bool',
            derive_from=("izsaukuma_laiks",)),
    ColSpec(name='izsaukuma_laiks_month_name',
            kind='datetime',
            datetime_units='month_name',
            derive_from=("izsaukuma_laiks",)),
]