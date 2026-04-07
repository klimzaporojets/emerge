import logging
import pickle
from pathlib import Path
from typing import Dict, TypeVar, Type, List

import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar('T')


def divide_list(lst, nr_parts_to_divide_into):
    """Split a list into approximately equal-sized partitions."""
    n = len(lst)
    result = []

    # Compute fair sizes
    base = n // nr_parts_to_divide_into
    extra = n % nr_parts_to_divide_into

    start = 0
    for i in range(nr_parts_to_divide_into):
        size = base + (1 if i < extra else 0)
        end = start + size
        result.append(lst[start:end])
        start = end

    return result


def batch_iterator(input_list, batch_size):
    """Yield successive batches of batch_size from input_list."""
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]


def normalize_triple_string(value: str) -> str:
    """Normalize triple string by replacing underscores with spaces."""
    return str(value).replace('_', ' ')

def make_agg_all(agg, agg_open, drop_open_metrics=None, sanity_groupby_cols=None):
    """Merge CIE and Open IE aggregated metrics after validating no overlapping entries."""
    if drop_open_metrics is None:
        drop_open_metrics = ['completeness']  # default matches your current behavior

    agg_open = agg_open[~agg_open['metric'].isin(drop_open_metrics)]

    metrics_agg = (
        agg[['metric', 'evaluator_model', 'model']]
        .drop_duplicates()
        .sort_values(['metric', 'evaluator_model', 'model'])
    )

    metrics_agg_open = (
        agg_open[['metric', 'evaluator_model', 'model']]
        .drop_duplicates()
        .sort_values(['metric', 'evaluator_model', 'model'])
    )

    overlap = metrics_agg.merge(
        metrics_agg_open,
        on=['metric', 'evaluator_model', 'model'],
        how='inner'
    )

    if overlap.empty:
        logger.info('No overlapping (metric, evaluator_model, model) between agg and agg_open.')
    else:
        logger.error(f'Overlapping metrics detected — resolve before concatenation:\n{overlap.to_string(index=False)}')

    assert overlap.empty, 'Overlapping metrics found between agg and agg_open'

    agg_all = pd.concat([agg, agg_open], ignore_index=True)

    if sanity_groupby_cols is None:
        sanity_groupby_cols = ['tkgu_type', 'model', 'metric', 'evaluator_model']

    sanity_val_counts = agg_all.groupby(sanity_groupby_cols).size().value_counts()

    logger.info(f'sanity check, all have to be in 1: {sanity_val_counts}')

    return agg_all


model_name_to_latex = {
    'relik-oie': 'ReLiK RE',
    'relik-cie': 'ReLiK cIE',
    'rebel': 'REBEL',

    ##### EDC
    'edc-azure_ai/Mistral-Large-2411': 'EDC M-Lg',
    'edc-open_ai/GPT-5_1': 'EDC GPT 5.1',

    ##### EDC+
    # EDC+ zero-shot:
    'edc-plus-zshot-open_ai/GPT-5_1': 'EDC+ ZS GPT 5.1',
    'edc-plus-zshot-azure_ai/Mistral-Large-2411': 'EDC+ ZS M-Lg',

    # EDC+ non-canonicalized
    'edc-plus-open-ai/gpt-5.1/non-canonicalized': 'EDC+ GPT 5.1',
    'edc-plus-azure_ai/Mistral-Large-2411': 'EDC+ M-Lg',
    'edc-plus-azure_ai/Mistral-small': 'EDC+ M-Sm',

    # EDC+ canonicalized:
    'edc-plus-open-ai/gpt-5.1/canonicalized': 'EDC+ CN GPT 5.1',
    'edc-plus-azure_ai/Mistral-Large-2411/canonicalized': 'EDC+ CN M-Lg',
    'edc-plus-azure_ai/Mistral-small/canonicalized': 'EDC+ CN M-Sm',

    ##### KG-GEN
    'kg-gen/azure/gpt5.1': 'KGGen GPT 5.1',
    'kg-gen/azure_ai/Mistral-Large-2411': 'KGGen M-Lg',
    'kg-gen/azure_ai/Mistral-small': 'KGGen M-Sm',

    ##### RAKG
    'rakg/azure_ai/Mistral-Large-2411': 'RAKG M-Lg',
    'rakg/azure_ai/Mistral-small': 'RAKG M-Sm'

    # 'edc-azure_ai/Mistral-Large-2411': 'EDC M-Lg',

    # # EDC+
    # 'edc-plus-zshot-azure_ai/Mistral-Large-2411': 'EDC+ ZS M-Lg',
    # 'edc-plus-no-canonicalize-azure_ai/Mistral-Large-2411': 'EDC+ M-Lg',
    # # "edc-plus-azure/gpt4o": 'EDC+ GPT4o',
    # "edc-plus-azure/gpt-5.1": 'EDC+ GPT5.1',
    # # 'edc-plus-azure_ai/Meta-Llama-31-8B-Instruct': 'EDC+ L3.1-8B',
    # # 'edc-plus-azure_ai/Mistral-Nemo': 'EDC+ M-Nemo',
    # 'edc-plus-azure_ai/Mistral-small': 'EDC+ M-Sm',
    # 'edc-plus/azure_ai/Llama-3.3-70B-Instruct': 'EDC+ L3.3-70B',
    # 'edc-plus/azure/gpt-4o-mini': 'EDC+ 4o-mini',

    # # KG-Gen
    # 'kg-gen/azure_ai/Meta-Llama-31-8B-Instruct': 'KG L3.1-8B',
    # 'kg-gen/azure_ai/Mistral-Nemo': 'KG M-Nemo',
    # 'kg-gen/azure_ai/Mistral-small': 'KG M-Sm',
    # 'kg-gen/azure_ai/Mistral-Large-2411': 'KG M-Lg',
    # 'kg-gen/azure_ai/Llama-3.3-70B-Instruct': 'KG L3.3-70B',
    # 'kg-gen/azure/gpt-4o-mini': 'KG 4o-mini',
    # "kg-gen/azure/gpt4o": 'KG GPT4o',
    # 'kg-gen/azure/gpt5.1': 'KG GPT5.1',

    # # RAKG
    # 'rakg/azure_ai/Meta-Llama-31-8B-Instruct': 'RAKG L3.1-8B',
    # 'rakg/azure_ai/Mistral-Nemo': 'RAKG M-Nemo',
    # 'rakg/azure_ai/Mistral-small': 'RAKG M-Sm',
    # 'rakg/azure_ai/Mistral-Large-2411': 'RAKG M-Lg',
    # 'rakg/azure_ai/Llama-3.3-70B-Instruct': 'RAKG L3.3-70B',
    # 'rakg/azure/gpt-4o-mini': 'RAKG 4o-mini',
}

metrics_to_report = [
    # {'metric': 'completeness', 'evaluator_model': 'all-mpnet-base-v2', 'alias': 'C', 'group': '', 'multiply_by_100': True},
    {'metric': 'completeness_score', 'evaluator_model': 'all-mpnet-base-v2', 'alias': 'C', 'group': '', 'multiply_by_100': True},
    #
    {'metric': 'gj-recall', 'evaluator_model': 'bert-base', 'alias': 'R', 'group': 'BERT', 'multiply_by_100': True},
    # {'metric': 'gj-precision', 'evaluator_model': 'bert-base', 'alias': 'P', 'group': 'BERT', 'multiply_by_100': True},
    # {'metric': 'gj-f1', 'evaluator_model': 'bert-base', 'alias': 'F1', 'group': 'BERT', 'multiply_by_100': True},
    #
    # {'metric': 'gj-recall', 'evaluator_model': 'st-all-mpnet-base-v2', 'alias': 'R', 'group': 'MPNet', 'multiply_by_100': True},
    # {'metric': 'gj-precision', 'evaluator_model': 'st-all-mpnet-base-v2', 'alias': 'P', 'group': 'MPNet', 'multiply_by_100': True},
    # {'metric': 'gj-f1', 'evaluator_model': 'st-all-mpnet-base-v2', 'alias': 'F1', 'group': 'MPNet', 'multiply_by_100': True},
    #
    # {'metric': 'ent-coverage-all-precision', 'evaluator_model': 'all-mpnet-base-v2', 'alias': 'P', 'group': 'Ent. Cov.', 'multiply_by_100': True},
    # {'metric': 'ent-coverage-all-recall', 'evaluator_model': 'all-mpnet-base-v2', 'alias': 'R', 'group': 'Ent. Cov.', 'multiply_by_100': True},
    # {'metric': 'ent-coverage-all-f1', 'evaluator_model': 'all-mpnet-base-v2', 'alias': 'F1', 'group': 'Ent. Cov.', 'multiply_by_100': True},
    # #
    # {'metric': 'ent-coverage-emerg-precision', 'evaluator_model': 'all-mpnet-base-v2', 'alias': 'P', 'group': 'Em. Ent. Cov.', 'multiply_by_100': True},
    # {'metric': 'ent-coverage-emerg-recall', 'evaluator_model': 'all-mpnet-base-v2', 'alias': 'R', 'group': 'Em. Ent. Cov.', 'multiply_by_100': True},
    # {'metric': 'ent-coverage-emerg-f1', 'evaluator_model': 'all-mpnet-base-v2', 'alias': 'F1', 'group': 'Em. Ent. Cov.', 'multiply_by_100': True},
    # #
    # {'metric': 'ent-coverage-all-precision', 'evaluator_model': 'bert-base', 'alias': 'P', 'group': 'Ent. Cov. B', 'multiply_by_100': True},
    # {'metric': 'ent-coverage-all-recall', 'evaluator_model': 'bert-base', 'alias': 'R', 'group': 'Ent. Cov. B', 'multiply_by_100': True},
    # {'metric': 'ent-coverage-all-f1', 'evaluator_model': 'bert-base', 'alias': 'F1', 'group': 'Ent. Cov. B', 'multiply_by_100': True},
    # #
    # {'metric': 'ent-coverage-emerg-precision', 'evaluator_model': 'bert-base', 'alias': 'P', 'group': 'Em. Ent. Cov. B', 'multiply_by_100': True},
    # {'metric': 'ent-coverage-emerg-recall', 'evaluator_model': 'bert-base', 'alias': 'R', 'group': 'Em. Ent. Cov. B', 'multiply_by_100': True},
    # {'metric': 'ent-coverage-emerg-f1', 'evaluator_model': 'bert-base', 'alias': 'F1', 'group': 'Em. Ent. Cov. B', 'multiply_by_100': True},
    # #
]

def make_metrics_cli_table(
    *,
    agg_all: pd.DataFrame,
    spec: pd.DataFrame,
    allowed_models,
    model_name_map: dict,
    show_aliases,
    exclude_group: str = 'LLM-as-a-judge',
    score_col: str = 'score_scaled',
    swap_pr: bool = True,
    tkgu_type_order=('x-triples', 'e-triples', 'ee-triples', 'ee-kg-triples', 'd-triples'),
    highlight_best: bool = True,
):
    """
    Prints a clean ASCII table to terminal.
    """

    show_aliases = list(show_aliases)
    allowed_models = list(allowed_models)
    allowed_models_set = set(allowed_models)

    # 1) Filter rows
    df = agg_all.copy()
    df = df[df['model'].isin(allowed_models_set)]
    df = df[df['group'] != exclude_group]
    df = df[df['alias'].isin(show_aliases)]

    if df.empty:
        logger.warning('No data after filtering.')
        return

    df['op_key'] = df['group'] + '|' + df['alias']

    # 2) Pivot
    table = df.pivot_table(
        index='model',
        columns=['tkgu_type', 'op_key'],
        values=score_col,
        aggfunc='mean',
    )

    # 3) Column order
    spec_show = spec[spec['group'] != exclude_group]
    spec_show = spec_show[spec_show['alias'].isin(show_aliases)]
    op_order = (spec_show['group'] + '|' + spec_show['alias']).tolist()

    if not op_order:
        op_order = sorted({c[1] for c in table.columns})

    tkgu_seen = list(dict.fromkeys(df['tkgu_type'].tolist()))
    tkgu_order = [t for t in tkgu_type_order if t in tkgu_seen] + [
        t for t in tkgu_seen if t not in tkgu_type_order
    ]

    desired_cols = [(t, op) for t in tkgu_order for op in op_order]
    table = table.reindex(columns=pd.MultiIndex.from_tuples(desired_cols))

    # Optional swap P/R
    if swap_pr:
        def alias_of(op_key):
            return op_key.split('|')[1] if '|' in op_key else op_key

        rank = {'C': 0, 'R': 2, 'P': 1, 'F1': 3}
        new_cols = []
        for t in tkgu_order:
            cols_t = [c for c in table.columns if c[0] == t]
            cols_t_sorted = sorted(cols_t, key=lambda c: rank.get(alias_of(c[1]), 99))
            new_cols.extend(cols_t_sorted)

        table = table.reindex(columns=new_cols)

    # 4) Rename models
    table.index = table.index.map(lambda m: model_name_map.get(m, m))
    allowed_models_mapped = [model_name_map.get(m, m) for m in allowed_models]
    table = table.reindex(allowed_models_mapped)

    # 5) Highlight best / second best
    if highlight_best:
        best_mask = table.eq(table.max(axis=0, skipna=True))
        second_mask = pd.DataFrame(False, index=table.index, columns=table.columns)

        for col in table.columns:
            s = table[col].dropna()
            uniq = sorted(set(s.tolist()), reverse=True)
            if len(uniq) >= 2:
                second_val = uniq[1]
                second_mask[col] = table[col].eq(second_val)

        second_mask &= ~best_mask

        def fmt(x, b, s):
            if pd.isna(x):
                return '--'
            val = f'{x:.1f}'
            if b:
                return f'*{val}*'      # best
            if s:
                return f'_{val}_'      # second best
            return val

        formatted = table.copy()
        for col in table.columns:
            formatted[col] = [
                fmt(v, b, s)
                for v, b, s in zip(
                    table[col].tolist(),
                    best_mask[col].tolist(),
                    second_mask[col].tolist(),
                )
            ]
        table = formatted
    else:
        table = table.applymap(lambda x: '--' if pd.isna(x) else f'{x:.1f}')

    # 6) Flatten multiindex columns
    table.columns = [
        f'{t}|{op.split("|")[1]}' for t, op in table.columns
    ]

    # 7) Log clean ASCII table
    logger.info('\n' + table.to_string())


def make_agg_and_agg_open(df_wiki_metrics_cie, df_metrics_open_ie, metrics_to_report, groupby_cols=None):
    """Aggregate per-instance metric scores by model/task/metric, returning CIE and Open IE DataFrames."""
    def scale_score(r):
        # LLM-as-a-judge scores: original scale 0–5 → normalize to 0–100
        if r['group'] == 'LLM-as-a-judge' and r['metric'] != 'factualness':
            return r['score'] * 20

        # All other metrics: respect existing flag
        if r['multiply_by_100']:
            return r['score'] * 100

        return r['score']


    # ------------------------------------------------------------------
    # 2. Prepare data
    # ------------------------------------------------------------------
    # df_wiki_metrics_cie_copy was df before in case it is not found, has to be renamed to df_wiki_metrics_cie_copy
    df_wiki_metrics_cie_copy = df_wiki_metrics_cie.copy()
    df_wiki_metrics_cie_copy['score'] = df_wiki_metrics_cie_copy['score'].astype(float)

    df_wiki_metrics_cie_copy = df_wiki_metrics_cie_copy.merge(
        metrics_to_report[['metric', 'evaluator_model']],
        on=['metric', 'evaluator_model'],
        how='inner'
    )

    df_open = df_metrics_open_ie.copy()
    df_open['score'] = df_open['score'].astype(float)

    df_open = df_open.merge(
        metrics_to_report[['metric', 'evaluator_model']],
        on=['metric', 'evaluator_model'],
        how='inner'
    )

    # ------------------------------------------------------------------
    # 3. Aggregate
    # ------------------------------------------------------------------
    if groupby_cols is None:
        groupby_cols = ['tkgu_type', 'model', 'metric', 'evaluator_model']

    agg = (
        df_wiki_metrics_cie_copy
        .groupby(groupby_cols, as_index=False)
        ['score']
        .mean()
    )

    agg = agg.merge(metrics_to_report, on=['metric', 'evaluator_model'], how='left')

    agg_open = (
        df_open
        .groupby(
            groupby_cols,
            as_index=False
        )['score']
        .mean()
    )

    agg_open = agg_open.merge(
        metrics_to_report,
        on=['metric', 'evaluator_model'],
        how='left'
    )

    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    #### END - visually checking that there are no duplicate records across tkgu tasks for a particular model and metric

    # ------------------------------------------------------------------
    # 4. Scale
    # ------------------------------------------------------------------

    agg['score_scaled'] = agg.apply(scale_score, axis=1)
    agg_open['score_scaled'] = agg_open.apply(scale_score, axis=1)

    # print(f'agg.columns: {agg.columns}')
    # print(f'agg_open.columns: {agg_open.columns}')
    #  ...

    return agg, agg_open


def add_columns(inst_cols: List, df_wiki_metrics_cie, df_metrics_open_ie, df_preds_gt_cie, df_preds_open_ie,
                df_instances_v13,
                df_additional_stats):
    """Merge instance metadata columns (delta_weeks, snapshot_year, etc.) into metric DataFrames."""

    inst = df_instances_v13[inst_cols]

    if not df_wiki_metrics_cie.empty:
        df_wiki_metrics_cie = df_wiki_metrics_cie.merge(
            inst, on='hash_id', how='inner'
        )

    if not df_metrics_open_ie.empty:
        df_metrics_open_ie = df_metrics_open_ie.merge(
            inst, on='hash_id', how='inner'
        )

    if not df_preds_gt_cie.empty:
        df_preds_gt_cie = df_preds_gt_cie.merge(
            inst, on='hash_id', how='inner'
        )

    if not df_preds_open_ie.empty:
        df_preds_open_ie = df_preds_open_ie.merge(
            inst, on='hash_id', how='inner'
        )
    if not df_additional_stats.empty:
        df_additional_stats = df_additional_stats.merge(
            inst, on='hash_id', how='inner'
        )

    return df_wiki_metrics_cie, df_metrics_open_ie, df_preds_gt_cie, df_preds_open_ie, df_additional_stats



def filter_by_assessor_evaluation(
        assessor_by_prompt_type: Dict,
        df_wiki_metrics_cie: pd.DataFrame,
        df_preds_gt_cie: pd.DataFrame,
):
    """Filter metrics and predictions to only include triples approved by the specified LLM assessor."""
    keys = ['hash_id', 'triple_head', 'triple_relation', 'triple_tail', 'tkgu_type']

    # ---- 1) Filter df_preds_gt_cie (always) ----
    df_gt_cie = df_preds_gt_cie[df_preds_gt_cie['triple_type'] == 'in-dataset']

    df_preds_ok = df_gt_cie.copy()
    df_preds_ok['required_assessor'] = df_preds_ok['prompt_type'].map(assessor_by_prompt_type)

    df_preds_ok = df_preds_ok[
        df_preds_ok['required_assessor'].notna()
        & (df_preds_ok['llm_assessor'] == df_preds_ok['required_assessor'])
        & (df_preds_ok['llm_assessor_result'] == True)
        ]

    df_not_in = df_preds_gt_cie[df_preds_gt_cie['triple_type'] != 'in-dataset']
    df_preds_gt_cie_filtered = pd.concat([df_preds_ok, df_not_in], ignore_index=True)

    approved = df_preds_ok[keys].drop_duplicates()

    # ---- 2) Filter df_wiki_metrics_cie (only if possible) ----
    if df_wiki_metrics_cie is None or df_wiki_metrics_cie.empty:
        df_wiki_metrics_cie_filtered = df_wiki_metrics_cie
    else:
        missing_keys = [c for c in keys if c not in df_wiki_metrics_cie.columns]
        if missing_keys:
            # Can't merge-filter without the join keys; return as-is (or raise if you prefer).
            df_wiki_metrics_cie_filtered = df_wiki_metrics_cie
        else:
            has_gran = 'granularity_level' in df_wiki_metrics_cie.columns
            if has_gran:
                df_triple_level = df_wiki_metrics_cie[df_wiki_metrics_cie['granularity_level'] != 'instance']
                df_instance = df_wiki_metrics_cie[df_wiki_metrics_cie['granularity_level'] == 'instance']

                df_non_instance_filtered = df_triple_level.merge(approved, on=keys, how='inner')

                df_wiki_metrics_cie_filtered = pd.concat(
                    [df_non_instance_filtered, df_instance],
                    ignore_index=True,
                    sort=False,
                )
            else:
                # No granularity split; just keep rows whose keys are approved
                df_wiki_metrics_cie_filtered = df_wiki_metrics_cie.merge(approved, on=keys, how='inner')

    logger.info(f'df_wiki_metrics_cie.shape: {None if df_wiki_metrics_cie is None else df_wiki_metrics_cie.shape}')
    logger.info(
        f'df_wiki_metrics_cie_filtered.shape: {None if df_wiki_metrics_cie_filtered is None else df_wiki_metrics_cie_filtered.shape}')
    logger.info(f'df_preds_gt_cie.shape: {df_preds_gt_cie.shape}')
    logger.info(f'df_preds_gt_cie_filtered.shape: {df_preds_gt_cie_filtered.shape}')

    return df_wiki_metrics_cie_filtered, df_preds_gt_cie_filtered


def load_from_cache(path: str | Path, cls: Type[T]) -> T:
    """Load a pickled WikiEvalResult from cache, or return a default-initialized instance if not found."""
    path = Path(path)

    if not path.exists():
        return cls()  # default-initialized dataclass

    with path.open('rb') as f:
        obj = pickle.load(f)

    if obj is None:
        return cls()

    if not isinstance(obj, cls):
        raise TypeError(f'Expected {cls}, got {type(obj)}')

    return obj


def save_to_cache(path: str | Path, obj: T) -> None:
    """Atomically save an object to a pickle cache file (writes to .tmp then renames)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        pickle.dump(obj, f)

    tmp.replace(path)
