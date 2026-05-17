import pandas as pd
from IPython.display import Markdown, display

def _resolve_duplicate_masks(id_cols, df) -> tuple[list[str], pd.Series, pd.Series]:
    if id_cols is None:
        id_cols = []
    elif isinstance(id_cols, str):
        id_cols = [id_cols]
    else:
        id_cols = list(id_cols)

    if len(id_cols) == 0:
        empty_mask = pd.Series(False, index=df.index, dtype='boolean')
        return id_cols, empty_mask, empty_mask

    missing_ids = [col for col in id_cols if col not in df.columns]
    if missing_ids:
        raise ValueError(f'❌ ID columns not found: {missing_ids}')

    key_df = df[id_cols].copy()

    for col in id_cols:
        col_data = key_df[col]
        if (
            pd.api.types.is_object_dtype(col_data)
            or pd.api.types.is_string_dtype(col_data)
            or isinstance(col_data.dtype, pd.CategoricalDtype)
        ):
            normalized = col_data.astype('string[python]').str.strip().str.lower()
            key_df[col] = normalized.replace('', pd.NA)

    complete_id_mask = key_df.notna().all(axis=1)
    skipsfirst_dupe_mask = complete_id_mask & key_df.duplicated(subset=id_cols, keep='first')
    includesfirst_dupe_mask = complete_id_mask & key_df.duplicated(subset=id_cols, keep=False)

    return id_cols, skipsfirst_dupe_mask, includesfirst_dupe_mask

def resolve_dupes(df, id_cols: list, include_first: bool = True, drop: bool = False):
    print("═" * 70)
    print(f"🔍 DUPE SEARCH ")
    print("─" * 70)

    id_cols, skipsfirst_dupe_mask, includesfirst_dupe_mask = _resolve_duplicate_masks(id_cols, df)

    if len(id_cols) == 0:
        print(f'There are no ID columns')
        display(df.iloc[0:0])
        return 

    dup_mask = includesfirst_dupe_mask if include_first else skipsfirst_dupe_mask
    dupe_count = int(dup_mask.sum())

    if dupe_count > 0:
        print(f"✔ Dupe search:")
        if include_first:
            print(f'There are {dupe_count} rows in duplicate groups based on complete normalized ID: {id_cols}')
            print('Note: includes the first row in each duplicate group. Rows with incomplete IDs were ignored')
        else:
            print(f'There are {dupe_count} later duplicate rows based on complete normalized ID: {id_cols}')
            print('Note: not including the first. Rows with incomplete IDs were ignored')
    else:
        print(f'✅ No duplicates found based on complete normalized ID: {id_cols}')
        print(f"✔ Dupe search: no duplicates found based on complete normalized ID: {id_cols}")

    dupe_df = df.loc[dup_mask]
    display(dupe_df)

    if drop:
        print("═" * 70)
        print(f"🪚 DUPE REMOVAL ")
        print("─" * 70)

        id_cols, skipsfirst_dupe_mask, _ = _resolve_duplicate_masks(id_cols, df)

        if len(id_cols) == 0:
            print(f"✔ Dupe removal: no ID columns provided — skipping duplicate removal")
            return 

        dupe_count = int(skipsfirst_dupe_mask.sum())

        print(f"✔ Dupe removal: removed {dupe_count} later duplicates based on complete normalized ID: {id_cols}")

        df = df[~skipsfirst_dupe_mask].reset_index(drop=True)
        return 
    else:        
        return df[includesfirst_dupe_mask]
