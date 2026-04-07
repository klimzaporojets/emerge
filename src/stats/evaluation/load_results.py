"""
Load evaluation results from WikiEvalResult .pkl files.

Ported from wikidata-temp/wikipedia-temp/src/stats/s14_load_results_v13.py
Adapted to load from the refactored single-pkl WikiEvalResult format
instead of the old multi-pkl pipeline.
"""
from typing import Dict, List, Any, Optional
import json
import logging
import os
import pickle
import hashlib

import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 0. Metric specification
# ------------------------------------------------------------------

metrics_to_report_cie = [
    {'metric': 'completeness', 'evaluator_model': 'all-mpnet-base-v2', 'alias': 'C', 'group': '', 'multiply_by_100': True},
    {'metric': 'gj-recall', 'evaluator_model': 'bert-base', 'alias': 'R', 'group': 'BERT', 'multiply_by_100': True},
    {'metric': 'gj-precision', 'evaluator_model': 'bert-base', 'alias': 'P', 'group': 'BERT', 'multiply_by_100': True},
    {'metric': 'gj-f1', 'evaluator_model': 'bert-base', 'alias': 'F1', 'group': 'BERT', 'multiply_by_100': True},
]

metrics_to_report_cie_entities = [
    {'metric': 'ent-coverage-emerg-precision', 'evaluator_model': 'all-mpnet-base-v2', 'alias': 'P', 'group': 'Em. Ent. Cov.', 'multiply_by_100': True},
    {'metric': 'ent-coverage-emerg-recall', 'evaluator_model': 'all-mpnet-base-v2', 'alias': 'R', 'group': 'Em. Ent. Cov.', 'multiply_by_100': True},
    {'metric': 'ent-coverage-emerg-f1', 'evaluator_model': 'all-mpnet-base-v2', 'alias': 'F1', 'group': 'Em. Ent. Cov.', 'multiply_by_100': True},
    {'metric': 'ent-coverage-all-precision', 'evaluator_model': 'bert-base', 'alias': 'P', 'group': 'All Ent. Cov.', 'multiply_by_100': True},
    {'metric': 'ent-coverage-all-recall', 'evaluator_model': 'bert-base', 'alias': 'R', 'group': 'All Ent. Cov.', 'multiply_by_100': True},
    {'metric': 'ent-coverage-all-f1', 'evaluator_model': 'bert-base', 'alias': 'F1', 'group': 'All Ent. Cov.', 'multiply_by_100': True},
    {'metric': 'ent-coverage-exist-precision', 'evaluator_model': 'bert-base', 'alias': 'P', 'group': 'Ex. Ent. Cov.', 'multiply_by_100': True},
    {'metric': 'ent-coverage-exist-recall', 'evaluator_model': 'bert-base', 'alias': 'R', 'group': 'Ex. Ent. Cov.', 'multiply_by_100': True},
    {'metric': 'ent-coverage-exist-f1', 'evaluator_model': 'bert-base', 'alias': 'F1', 'group': 'Ex. Ent. Cov.', 'multiply_by_100': True},
]

prediction_categories = ['x-triples', 'e-triples', 'ee-triples', 'd-triples', 'ee-kg-triples']

threshold_completeness = 0.9

# ------------------------------------------------------------------
# 1. Models
# ------------------------------------------------------------------

models_to_load = {
    'rebel',
    'relik-oie',
    'relik-cie',
    "edc-plus-azure_ai/Mistral-small",
    "edc-plus-azure_ai/Mistral-Large-2411",
    "edc-plus-open-ai/gpt-5.1/non-canonicalized",
    'kg-gen/azure_ai/Mistral-small',
    'kg-gen/azure_ai/Mistral-Large-2411',
    'kg-gen/azure/gpt5.1',
    'rakg/azure_ai/Mistral-small',
    'rakg/azure_ai/Mistral-Large-2411',
}

model_name_to_latex = {
    'relik-oie': 'ReLiK RE',
    'relik-cie': 'ReLiK cIE',
    'rebel': 'REBEL',
    # EDC+
    'edc-plus-open-ai/gpt-5.1/non-canonicalized': 'EDC+ GPT-5.1',
    'edc-plus-azure_ai/Mistral-Large-2411': 'EDC+ M-Lg',
    'edc-plus-azure_ai/Mistral-small': 'EDC+ M-Sm',
    # EDC+ zero-shot
    'edc-plus-zshot-open_ai/GPT-5_1': 'EDC+ ZS GPT 5.1',
    'edc-plus-zshot-azure_ai/Mistral-Large-2411': 'EDC+ ZS M-Lg',
    # KG-GEN
    'kg-gen/azure/gpt5.1': 'KGGen GPT-5.1',
    'kg-gen/azure_ai/Mistral-Large-2411': 'KGGen M-Lg',
    'kg-gen/azure_ai/Mistral-small': 'KGGen M-Sm',
    # RAKG
    'rakg/azure_ai/Mistral-Large-2411': 'RAKG M-Lg',
    'rakg/azure_ai/Mistral-small': 'RAKG M-Sm',
}


# ------------------------------------------------------------------
# 2. Load from WikiEvalResult pkl
# ------------------------------------------------------------------

def load_from_wiki_eval_result(pkl_path: str):
    """
    Load a WikiEvalResult .pkl and return the DataFrames in the same
    format as the old load_stats() function.

    Returns:
        (df_wiki_metrics_cie, df_metrics_open_ie, df_preds_gt_cie,
         df_preds_open_ie, df_instances, df_additional_stats)
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)

    df_wiki_metrics_cie = result.df_metrics_cie
    df_metrics_open_ie = result.df_metrics_open_ie
    df_preds_gt_cie = result.df_predictions_cie_and_gt
    df_preds_open_ie = result.df_predictions_open_ie
    df_instances = result.df_instances
    df_additional_stats = result.df_metrics_additional_triple_stats

    return (df_wiki_metrics_cie, df_metrics_open_ie, df_preds_gt_cie,
            df_preds_open_ie, df_instances, df_additional_stats)


def load_results(
    pkl_path: str,
    assessor_by_prompt_type: Optional[Dict] = None,
    filter_models: Optional[set] = None,
):
    """
    Load evaluation results from a WikiEvalResult pkl, optionally filter
    by assessor and models, and apply completeness thresholding.

    Args:
        pkl_path: path to wiki_eval_result.pkl
        assessor_by_prompt_type: dict mapping prompt_type -> assessor name
            for filtering GT triples by LLM assessor result
        filter_models: set of model names to keep (None = keep all)

    Returns:
        (df_wiki_metrics_cie, df_metrics_open_ie, df_preds_gt_cie,
         df_preds_open_ie, df_instances, df_additional_stats)
    """
    (df_wiki_metrics_cie, df_metrics_open_ie, df_preds_gt_cie,
     df_preds_open_ie, df_instances, df_additional_stats) = \
        load_from_wiki_eval_result(pkl_path)

    # Filter by models
    if filter_models is not None:
        filter_set = set(filter_models)
        if 'model' in df_wiki_metrics_cie.columns:
            df_wiki_metrics_cie = df_wiki_metrics_cie[
                df_wiki_metrics_cie['model'].isin(filter_set)]
        if 'model' in df_metrics_open_ie.columns:
            df_metrics_open_ie = df_metrics_open_ie[
                df_metrics_open_ie['model'].isin(filter_set)]
        df_preds_gt_cie = df_preds_gt_cie[
            (df_preds_gt_cie['triple_type'] == 'in-dataset')
            | (df_preds_gt_cie['model'].isin(filter_set))]
        if 'model' in df_preds_open_ie.columns:
            df_preds_open_ie = df_preds_open_ie[
                df_preds_open_ie['model'].isin(filter_set)]
        if 'model' in df_additional_stats.columns:
            df_additional_stats = df_additional_stats[
                df_additional_stats['model'].isin(filter_set)]

    # Filter by assessor
    if assessor_by_prompt_type:
        df_wiki_metrics_cie, df_preds_gt_cie = filter_by_assessor_evaluation(
            assessor_by_prompt_type=assessor_by_prompt_type,
            df_wiki_metrics_cie=df_wiki_metrics_cie,
            df_preds_gt_cie=df_preds_gt_cie,
        )

    # Apply completeness thresholding
    if 'metric' in df_wiki_metrics_cie.columns:
        mask = df_wiki_metrics_cie['metric'] == 'completeness'
        df_extra = df_wiki_metrics_cie.loc[mask].copy()
        df_extra['metric'] = 'completeness_score'
        df_wiki_metrics_cie.loc[mask, 'score'] = (
            (df_wiki_metrics_cie.loc[mask, 'score'] > threshold_completeness)
            .astype(float)
        )
        df_wiki_metrics_cie = pd.concat(
            [df_wiki_metrics_cie, df_extra], ignore_index=True)

    # Join instance-level columns (delta_weeks, snapshot_year) onto metrics
    inst_cols = ['hash_id', 'delta_weeks', 'snapshot_year']
    inst = df_instances[inst_cols]
    if not df_wiki_metrics_cie.empty and 'hash_id' in df_wiki_metrics_cie.columns:
        df_wiki_metrics_cie = df_wiki_metrics_cie.merge(
            inst, on='hash_id', how='inner')
    if not df_metrics_open_ie.empty and 'hash_id' in df_metrics_open_ie.columns:
        df_metrics_open_ie = df_metrics_open_ie.merge(
            inst, on='hash_id', how='inner')
    if not df_preds_gt_cie.empty and 'hash_id' in df_preds_gt_cie.columns:
        if 'delta_weeks' not in df_preds_gt_cie.columns:
            df_preds_gt_cie = df_preds_gt_cie.merge(
                inst, on='hash_id', how='inner')
    if not df_additional_stats.empty and 'hash_id' in df_additional_stats.columns:
        df_additional_stats = df_additional_stats.merge(
            inst, on='hash_id', how='inner')

    return (df_wiki_metrics_cie, df_metrics_open_ie, df_preds_gt_cie,
            df_preds_open_ie, df_instances, df_additional_stats)


# ------------------------------------------------------------------
# 3. Assessor filtering
# ------------------------------------------------------------------

def filter_by_assessor_evaluation(
    assessor_by_prompt_type: Dict,
    df_wiki_metrics_cie: pd.DataFrame,
    df_preds_gt_cie: pd.DataFrame,
):
    keys = ["hash_id", "triple_head", "triple_relation", "triple_tail", "tkgu_type"]

    # Filter df_preds_gt_cie
    df_gt_cie = df_preds_gt_cie[df_preds_gt_cie["triple_type"] == "in-dataset"]
    df_preds_ok = df_gt_cie.copy()
    df_preds_ok["required_assessor"] = df_preds_ok["prompt_type"].map(
        assessor_by_prompt_type)
    df_preds_ok = df_preds_ok[
        df_preds_ok["required_assessor"].notna()
        & (df_preds_ok["llm_assessor"] == df_preds_ok["required_assessor"])
        & (df_preds_ok["llm_assessor_result"] == True)
    ]

    df_not_in = df_preds_gt_cie[df_preds_gt_cie["triple_type"] != "in-dataset"]
    df_preds_gt_cie_filtered = pd.concat(
        [df_preds_ok, df_not_in], ignore_index=True)
    approved = df_preds_ok[keys].drop_duplicates()

    # Filter df_wiki_metrics_cie
    if df_wiki_metrics_cie is None or df_wiki_metrics_cie.empty:
        df_wiki_metrics_cie_filtered = df_wiki_metrics_cie
    else:
        missing_keys = [c for c in keys if c not in df_wiki_metrics_cie.columns]
        if missing_keys:
            df_wiki_metrics_cie_filtered = df_wiki_metrics_cie
        else:
            has_gran = "granularity_level" in df_wiki_metrics_cie.columns
            if has_gran:
                df_triple_level = df_wiki_metrics_cie[
                    df_wiki_metrics_cie["granularity_level"] != "instance"]
                df_instance = df_wiki_metrics_cie[
                    df_wiki_metrics_cie["granularity_level"] == "instance"]
                df_non_instance_filtered = df_triple_level.merge(
                    approved, on=keys, how="inner")
                df_wiki_metrics_cie_filtered = pd.concat(
                    [df_non_instance_filtered, df_instance],
                    ignore_index=True, sort=False)
            else:
                df_wiki_metrics_cie_filtered = df_wiki_metrics_cie.merge(
                    approved, on=keys, how="inner")

    logger.info(f'df_wiki_metrics_cie.shape: '
                f'{None if df_wiki_metrics_cie is None else df_wiki_metrics_cie.shape}')
    logger.info(f'df_wiki_metrics_cie_filtered.shape: '
                f'{None if df_wiki_metrics_cie_filtered is None else df_wiki_metrics_cie_filtered.shape}')
    logger.info(f'df_preds_gt_cie.shape: {df_preds_gt_cie.shape}')
    logger.info(f'df_preds_gt_cie_filtered.shape: {df_preds_gt_cie_filtered.shape}')

    return df_wiki_metrics_cie_filtered, df_preds_gt_cie_filtered


# ------------------------------------------------------------------
# 4. Aggregation helpers
# ------------------------------------------------------------------

def make_agg_and_agg_open(df_wiki_metrics_cie, df_metrics_open_ie, spec,
                          groupby_cols=None):
    def scale_score(r):
        if r['group'] == 'LLM-as-a-judge' and r['metric'] != 'factualness':
            return r['score'] * 20
        if r['multiply_by_100']:
            return r['score'] * 100
        return r['score']

    df_cie = df_wiki_metrics_cie.copy()
    df_cie['score'] = df_cie['score'].astype(float)
    df_cie = df_cie.merge(
        spec[['metric', 'evaluator_model']],
        on=['metric', 'evaluator_model'], how='inner')

    df_open = df_metrics_open_ie.copy()
    df_open['score'] = df_open['score'].astype(float)
    df_open = df_open.merge(
        spec[['metric', 'evaluator_model']],
        on=['metric', 'evaluator_model'], how='inner')

    if groupby_cols is None:
        groupby_cols = ['tkgu_type', 'model', 'metric', 'evaluator_model']

    agg = (df_cie.groupby(groupby_cols, as_index=False)['score'].mean()
           .merge(spec, on=['metric', 'evaluator_model'], how='left'))

    agg_open = (df_open.groupby(groupby_cols, as_index=False)['score'].mean()
                .merge(spec, on=['metric', 'evaluator_model'], how='left'))

    agg['score_scaled'] = agg.apply(scale_score, axis=1)
    agg_open['score_scaled'] = agg_open.apply(scale_score, axis=1)

    return agg, agg_open


def make_agg_all(agg, agg_open, drop_open_metrics=None,
                 sanity_groupby_cols=None):
    if drop_open_metrics is None:
        drop_open_metrics = ['completeness']

    agg_open = agg_open[~agg_open['metric'].isin(drop_open_metrics)]

    metrics_agg = (agg[['metric', 'evaluator_model', 'model']]
                   .drop_duplicates()
                   .sort_values(['metric', 'evaluator_model', 'model']))
    metrics_agg_open = (agg_open[['metric', 'evaluator_model', 'model']]
                        .drop_duplicates()
                        .sort_values(['metric', 'evaluator_model', 'model']))

    overlap = metrics_agg.merge(
        metrics_agg_open, on=['metric', 'evaluator_model', 'model'],
        how='inner')

    if overlap.empty:
        print('No overlapping (metric, evaluator_model, model) '
              'between agg and agg_open.')
    else:
        print('Overlapping metrics detected:')
        print(overlap)

    assert overlap.empty, \
        'Overlapping metrics found between agg and agg_open'

    agg_all = pd.concat([agg, agg_open], ignore_index=True)

    if sanity_groupby_cols is None:
        sanity_groupby_cols = ['tkgu_type', 'model', 'metric',
                               'evaluator_model']

    sanity = agg_all.groupby(sanity_groupby_cols).size().value_counts()
    print(f'sanity check, all have to be in 1: {sanity}')

    return agg_all


# ------------------------------------------------------------------
# 5. LaTeX table generation
# ------------------------------------------------------------------

tkgu_type_to_latex = {
    "d-triples": r"\opdeprecate",
    "e-triples": r"\opadd",
    "ee-kg-triples": r"\opinfer",
    "ee-triples": r"\opmintadd",
    "x-triples": r"\opexists",
}

alias_label = {
    "C": "C",
    "R": "G-R",
    "P": "G-P",
    "F1": "G-F1",
}


def arch_group_for_model_latex(model_latex: str) -> str:
    """Group concrete model runs into architecture buckets for plotting."""
    s = (model_latex or "").strip()
    if s.startswith("EDC+"):
        return "EDC+"
    if s.startswith("EDC"):
        return "EDC"
    if s.startswith("RAKG"):
        return "RAKG"
    if s.startswith("KG"):
        return "KGGen"
    return s


def arch_color_for_model(model_latex: str) -> str:
    s = model_latex.strip()
    if s.startswith("EDC+"):
        return "archEDCp"
    if s.startswith("EDC"):
        return "archEDC"
    if s.startswith("KG"):
        return "archKG"
    if s.startswith("RAKG"):
        return "archRAKG"
    if s.startswith("ReLiK"):
        return "archRELIK"
    return "archOther"


def _alias_from_op(op_key: str) -> str:
    return op_key.split("|", 1)[1] if "|" in op_key else op_key


def _fmt_cell(x):
    if pd.isna(x):
        return "--"
    return f"{x:.1f}"


def make_metrics_latex_table(
    *,
    agg_all: pd.DataFrame,
    spec: pd.DataFrame,
    allowed_models,
    model_name_to_latex: dict,
    show_aliases,
    exclude_group: str = "LLM-as-a-judge",
    score_col: str = "score_scaled",
    swap_pr: bool = True,
    tkgu_type_order=("x-triples", "e-triples", "ee-triples",
                     "ee-kg-triples", "d-triples"),
    tabcolsep_pt: int = 3,
    highlight_best: bool = True,
    best_cmd: str = r"\textbf",
    second_cmd: str = r"\underline",
):
    """
    Build a LaTeX table from aggregated metrics.

    Args:
        agg_all: aggregated DataFrame (from make_agg_all)
        spec: metric specification DataFrame
        allowed_models: ordered list of model names to include
        model_name_to_latex: mapping from model name to LaTeX label
        show_aliases: list of aliases to show (e.g. ["C", "G-R"])
        highlight_best: if True, bold best and underline second-best per column
    """
    show_aliases = list(show_aliases)
    inv_alias_label = {v: k for k, v in alias_label.items()}
    show_raw_aliases = set()
    for a in show_aliases:
        show_raw_aliases.add(inv_alias_label.get(a, a))

    allowed_models = list(allowed_models)
    allowed_models_set = set(allowed_models)

    agg_show = agg_all[agg_all["model"].isin(allowed_models_set)].copy()
    agg_show = agg_show[agg_show["group"] != exclude_group].copy()
    agg_show = agg_show[agg_show["alias"].isin(show_raw_aliases)].copy()

    if agg_show.empty:
        return "% No data after filtering."

    agg_show["op_key"] = agg_show["group"] + "|" + agg_show["alias"]

    table = agg_show.pivot_table(
        index=["model"],
        columns=["tkgu_type", "op_key"],
        values=score_col,
        aggfunc="mean",
    )

    # Models that don't support a TKGU type produce no predictions, so all
    # their scores are exactly 0.0 (due to score_empty_predictions_as_zero).
    # Replace with NaN so they display as "--" instead of 0.0.
    for tkgu in {c[0] for c in table.columns}:
        cols_t = [c for c in table.columns if c[0] == tkgu]
        all_zero = table[cols_t].eq(0.0).all(axis=1)
        table.loc[all_zero, cols_t] = float("nan")

    # Column ordering
    spec_show = spec[spec["group"] != exclude_group].copy()
    spec_show = spec_show[spec_show["alias"].isin(show_raw_aliases)].copy()
    op_order = (spec_show["group"] + "|" + spec_show["alias"]).tolist()
    if not op_order:
        op_order = sorted({c[1] for c in table.columns})

    tkgu_seen = list(dict.fromkeys(agg_show["tkgu_type"].tolist()))
    tkgu_order = [t for t in tkgu_type_order if t in tkgu_seen] + [
        t for t in tkgu_seen if t not in tkgu_type_order]

    desired_cols = [(t, op) for t in tkgu_order for op in op_order]
    table = table.reindex(
        columns=pd.MultiIndex.from_tuples(
            desired_cols, names=["TKGU", "op_key"]))

    if swap_pr:
        rank = {"C": 0, "R": 2, "P": 1, "F1": 3,
                "G-R": 2, "G-P": 1, "G-F1": 3}
        new_cols = []
        for t in tkgu_order:
            cols_t = [c for c in table.columns if c[0] == t]
            cols_t_sorted = sorted(
                cols_t,
                key=lambda c: rank.get(_alias_from_op(c[1]), 99))
            new_cols.extend(cols_t_sorted)
        table = table.reindex(
            columns=pd.MultiIndex.from_tuples(
                new_cols, names=["TKGU", "op_key"]))
        op_order = [c[1] for c in new_cols if c[0] == tkgu_order[0]]

    # Model name mapping + row ordering
    table.index = table.index.map(
        lambda m: model_name_to_latex.get(m, m))
    allowed_latex = [model_name_to_latex.get(m, m) for m in allowed_models]
    table = table.reindex(allowed_latex)

    # Best/second-best highlighting
    best_mask = second_mask = None
    if highlight_best:
        best_mask = table.eq(table.max(axis=0, skipna=True))
        second_mask = pd.DataFrame(
            False, index=table.index, columns=table.columns)
        for col in table.columns:
            s = table[col].dropna()
            if s.empty:
                continue
            uniq = sorted(set(s.tolist()), reverse=True)
            if len(uniq) < 2:
                continue
            second_mask[col] = table[col].eq(uniq[1])
        second_mask = second_mask & (~best_mask)

    def _fmt_hl(x, is_best, is_second):
        if pd.isna(x):
            return "--"
        s = f"{x:.1f}"
        if highlight_best and is_best:
            return f"{best_cmd}{{{s}}}"
        if highlight_best and is_second:
            return f"{second_cmd}{{{s}}}"
        return s

    if highlight_best:
        formatted = table.copy()
        for col in table.columns:
            bcol = best_mask[col]
            scol = second_mask[col]
            formatted[col] = [
                _fmt_hl(v, bool(b), bool(s))
                for v, b, s in zip(
                    table[col].tolist(), bcol.tolist(), scol.tolist())]
        table = formatted
    else:
        table = table.apply(lambda col: col.map(_fmt_cell))

    # LaTeX body
    body = table.to_latex(index=True, header=False, escape=False)
    body_lines = body.splitlines()
    body_content = "\n".join(body_lines[2:-2])
    body_content = "\n".join(
        line for line in body_content.splitlines()
        if not line.lstrip().startswith("model "))

    # Remove any midrules that pandas added in the body
    body_content = "\n".join(
        ln for ln in body_content.splitlines()
        if ln.strip() != r"\midrule")

    colored_lines = []
    for line in body_content.splitlines():
        if " & " not in line:
            colored_lines.append(line)
            continue
        first, rest = line.split(" & ", 1)
        indent = line[:len(line) - len(line.lstrip(" "))]
        model_cell = first.strip()
        colored_first = (f"{indent}\\cellcolor"
                         f"{{{arch_color_for_model(model_cell)}}} "
                         f"{model_cell}")
        colored_lines.append(colored_first + " & " + rest)
    body_content = "\n".join(colored_lines)

    # Headers + cmidrules
    tkgu_order_hdr = [tkgu_type_to_latex.get(t, t) for t in tkgu_order]

    header_top_parts = []
    for i, t in enumerate(tkgu_order_hdr):
        bar = "|" if i < len(tkgu_order_hdr) - 1 else ""
        header_top_parts.append(
            f"\\multicolumn{{{len(op_order)}}}{{c{bar}}}{{{t}}}")
    header_top = " & ".join(header_top_parts)

    cmid_parts = []
    start = 2
    for _ in range(len(tkgu_order_hdr)):
        end = start + len(op_order) - 1
        cmid_parts.append(f"\\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    cmidrules = "".join(cmid_parts)

    header_bottom_parts = []
    for _t in tkgu_order_hdr:
        for op in op_order:
            a = _alias_from_op(op)
            label = alias_label.get(a, a)
            header_bottom_parts.append(f"{label}$\\uparrow$")
    header_bottom = " & ".join(header_bottom_parts)

    group_spec = "c" * len(op_order)
    tabular_spec = "l|" + "|".join(
        [group_spec] * len(tkgu_order_hdr))

    latex = f"""\\begin{{table*}}
\\begin{{center}}
\\begin{{small}}
  \\begin{{sc}}
    \\setlength{{\\tabcolsep}}{{{tabcolsep_pt}pt}}
    \\rowcolors{{4}}{{rowwhite}}{{rowgray}}
    \\begin{{tabular}}{{{tabular_spec}}}
      \\toprule
      & {header_top} \\\\
      {cmidrules}
      Model & {header_bottom} \\\\
\\midrule
{body_content}
      \\bottomrule
    \\end{{tabular}}
  \\end{{sc}}
\\end{{small}}
\\end{{center}}
\\vskip -0.1in
\\end{{table*}}"""

    return latex
