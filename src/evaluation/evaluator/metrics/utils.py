from typing import Any, Iterable, Tuple

import pandas as pd


def append_metric_rows(
    df_existing: pd.DataFrame | None,
    results: Any,
) -> pd.DataFrame:
    """
    Generic row-wise merge for metric outputs.

    - df_existing: existing metrics dataframe (may be None or empty)
    - results:
        * pd.DataFrame
        * dict with any of:
            - "scores_per_triple"
            - "scores_per_instance"
            - other list-of-dict entries
        * None (no-op)

    Returns a dataframe with rows appended (schema union).
    """

    if results is None:
        return df_existing if df_existing is not None else pd.DataFrame()

    dfs = []

    if df_existing is not None and not df_existing.empty:
        dfs.append(df_existing)

    if isinstance(results, pd.DataFrame):
        dfs.append(results)

    elif isinstance(results, dict):
        for value in results.values():
            if isinstance(value, list) and value:
                dfs.append(pd.DataFrame(value))
            elif isinstance(value, pd.DataFrame):
                dfs.append(value)

    else:
        raise TypeError(f"Unsupported metric result type: {type(results)}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True, sort=False)

def extract_done_keys(
    df: pd.DataFrame | None,
    key_columns: Iterable[str],
    *,
    filters: dict | None = None,
) -> set[Tuple]:
    """
    Extract a set of tuple-keys from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame | None
        Existing metrics dataframe.
    key_columns : iterable of str
        Columns that define the identity of a computed unit.
    filters : dict | None
        Optional column -> value equality filters
        (e.g. {"metric": "completeness", "model_alias": "mpnet"}).

    Returns
    -------
    Set[Tuple]
        Set of key tuples.
    """

    if df is None or df.empty:
        return set()

    for col in key_columns:
        if col not in df.columns:
            return set()  # cannot reliably detect done keys

    sub = df
    if filters:
        for col, val in filters.items():
            if col not in sub.columns:
                return set()
            sub = sub[sub[col] == val]

    if sub.empty:
        return set()

    return set(
        tuple(row)
        for row in sub[list(key_columns)].itertuples(index=False, name=None)
    )