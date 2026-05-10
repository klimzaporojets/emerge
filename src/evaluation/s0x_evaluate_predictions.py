# Main entry point for the EMERGE evaluation pipeline.
import json
import logging
import os
from typing import List, Dict, Any, TypeVar

import pandas as pd
from tqdm import tqdm

from dataset.emerge.loader.dataset_loader import EMERGEDatasetLoader
from dataset.emerge.loader.loader_config import EMERGELoaderConfig
from evaluation.misc.args_evaluation import ArgsEvaluation
from evaluation.misc.utils import filter_by_assessor_evaluation, add_columns, make_agg_and_agg_open, metrics_to_report, \
    make_agg_all, make_metrics_cli_table, model_name_to_latex, load_from_cache
from evaluation.misc.wiki_eval_result import WikiEvalResult
from evaluation.orchestrator.evaluation_orchestrator import EvaluationOrchestrator
from pathlib import Path
import pickle
import lz4.frame

logger = logging.getLogger(__name__)

T = TypeVar('T')


def save_cache(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as raw_f:
        with lz4.frame.open(
                raw_f,
                mode='wb',
                compression_level=0,  # IMPORTANT: fastest + lowest memory
                block_size=lz4.frame.BLOCKSIZE_MAX1MB
        ) as f:
            pickler = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
            pickler.dump(obj)


def load_cache(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None

    with lz4.frame.open(path, 'rb') as f:
        return pickle.load(f)


def save_cache_fast(path: str | Path, obj: T) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_cache_fast(path: str | Path) -> T | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open('rb') as f:
        return pickle.load(f)


def main():
    """Run the EMERGE evaluation pipeline: load dataset, compute metrics, and print results table."""
    arguments_main = ArgsEvaluation().parse_args(known_only=True)
    logger.info(f'After parse_args(): {arguments_main.config_file}')

    # Step 2: Load JSON config if provided
    if arguments_main.config_file:
        logger.info(f'Loading JSON config from {arguments_main.config_file}')
        with open(arguments_main.config_file) as f:
            config_data = json.load(f)

        # Step 3: Merge JSON config into arguments (but keep CLI priority)
        # Only update fields not set via CLI
        for key, value in config_data.items():
            if (getattr(arguments_main, key, None) == ArgsEvaluation().get_default(key)
                    or getattr(arguments_main, key) is None):
                setattr(arguments_main, key, value)

    cached: WikiEvalResult = load_from_cache(arguments_main.cache_path, WikiEvalResult)

    from evaluation.evaluator.metrics.base import log_gpu_memory  # noqa: E402
    log_gpu_memory('start_of_evaluation')

    if not arguments_main.load_results_from_cache:

        should_add_predictions = len(arguments_main.metrics_to_calculate) > 0

        emerge_loader_config = EMERGELoaderConfig(input_dataset_path=arguments_main.input_dataset_path,
                                                  should_add_predictions=should_add_predictions)

        emerge_dataset_loader = EMERGEDatasetLoader(
            config=emerge_loader_config,
            wikipedia_page_id_to_wikidata_qid=None,
            page_title_changes=None,
            wikipedia_page_id_to_wikipedia_page_title=None,
            wikipedia_page_id_to_redirected_page_id=None,
            wikipedia_page_title_to_wikipedia_page_id=None
        )
        filemap = arguments_main.snapshot_year_to_kg_file
        base = arguments_main.kg_snapshots_path
        loaded_dataset: List[Dict[str, Any]] = emerge_dataset_loader.load()

        snapshot_to_triples = None
        # logger.info(f'begin loading cache no compression from {cache_path_pkl}')
        # logger.info(f'end loading cache no compression from {cache_path_pkl}')
        if snapshot_to_triples is None:

            snapshot_to_triples = {}

            for snap, fname in tqdm(filemap.items(), desc='Loading snapshots'):
                triples = set()
                if fname is None:
                    snapshot_to_triples[snap] = triples
                    continue
                curr_snapshot_path = os.path.join(base, fname)
                if not os.path.exists(curr_snapshot_path):
                    logger.warning(
                        f'KG snapshot file not found: {curr_snapshot_path}. '
                        f'Proceeding with empty snapshot for {snap}.'
                    )
                    snapshot_to_triples[snap] = triples
                    continue
                logger.info(f'loading {fname}')
                with open(curr_snapshot_path, 'r', encoding='utf-8') as f:
                    for line in tqdm(f):
                        # split only first 3 columns, ignoring labels
                        line = line.strip()
                        h, r, t = line.split('\t', 3)[:3]
                        triples.add((h, r, t))
                snapshot_to_triples[snap] = triples

            # If any KG snapshot is missing, relik-cie predictions cannot be
            # classified as Exists vs Add → relik-cie scores would be invalid.
            # Skip relik-cie entirely (don't compute, don't cache wrong values).
            num_loaded = sum(1 for t in snapshot_to_triples.values() if len(t) > 0)
            num_total = len(snapshot_to_triples)
            models_list = getattr(arguments_main, 'models_to_report', None) or []
            if num_loaded < num_total and 'relik-cie' in models_list:
                logger.warning('=' * 72)
                logger.warning(
                    f'KG snapshots loaded: {num_loaded} of {num_total}.'
                )
                logger.warning(
                    'Skipping relik-cie scoring — without all KG snapshots, '
                    'relik-cie predictions cannot be classified as Exists vs '
                    'Add, so scores would be invalid. The relik-cie row will '
                    'be absent from the output.'
                )
                logger.warning('')
                logger.warning('To include relik-cie:')
                logger.warning(
                    '  1. Download KG snapshots: ./scripts/download_data.sh --kg'
                )
                logger.warning(
                    f'  2. Clear stale cache: rm -rf {arguments_main.cache_path}'
                )
                logger.warning('  3. Re-run: ./scripts/run/evaluate.sh')
                logger.warning('=' * 72)
                arguments_main.models_to_report = [
                    m for m in models_list if m != 'relik-cie'
                ]
                arguments_main._kg_skipped_relik_cie = True

        evaluation_orchestrator: EvaluationOrchestrator = EvaluationOrchestrator(
            config=arguments_main,
            snapshot_to_triples=snapshot_to_triples,
            cached=cached
        )

        wiki_eval_result: WikiEvalResult = evaluation_orchestrator.evaluate(
            dataset_to_evaluate=loaded_dataset
        )
    else:
        wiki_eval_result: WikiEvalResult = cached
    ##### BEGIN - LOGIC TO ACTUALLY OBTAIN SCORES AND BUILD EVALUATION TABLE
    # filters by LLM assessor, so exclusively the ones assessed as valid by LLM are evaluated
    logger.info('begin_filter_by_assessor_evaluation')
    df_metrics_cie_filtered, df_preds_gt_cie_filtered = \
        filter_by_assessor_evaluation(assessor_by_prompt_type=arguments_main.llm_assessors,
                                      df_wiki_metrics_cie=wiki_eval_result.df_metrics_cie,
                                      df_preds_gt_cie=wiki_eval_result.df_predictions_cie_and_gt)
    logger.info('end_filter_by_assessor_evaluation')
    wiki_eval_result.df_metrics_cie = df_metrics_cie_filtered
    # wiki_eval_result.df_preds_gt_cie = df_preds_gt_cie_filtered
    wiki_eval_result.df_predictions_cie_and_gt = df_preds_gt_cie_filtered

    logger.info('begin_adding_completeness_metric')

    if 'completeness' in arguments_main.metrics_to_calculate:
        extras = []

        for curr_similarity_model in arguments_main.metrics_to_calculate['completeness']['similarity_models']:
            mask = (
                    (wiki_eval_result.df_metrics_cie['metric'] == 'completeness') &
                    (wiki_eval_result.df_metrics_cie['evaluator_model'] == curr_similarity_model['model_alias'])
            )

            if not mask.any():
                continue

            completeness_threshold = curr_similarity_model['completeness_threshold']

            # Create binary version
            df_extra = wiki_eval_result.df_metrics_cie.loc[mask].copy()
            df_extra['metric'] = 'completeness_score'
            df_extra['score'] = (df_extra['score'] > completeness_threshold).astype(float)

            extras.append(df_extra)

        if extras:
            wiki_eval_result.df_metrics_cie = pd.concat(
                [wiki_eval_result.df_metrics_cie] + extras,
                ignore_index=True
            )

    df_metrics_cie, df_metrics_open_ie, df_preds_gt_cie, df_preds_open_ie, df_additional_stats = \
        add_columns(
            inst_cols=['hash_id', 'delta_weeks', 'snapshot_year'],
            df_wiki_metrics_cie=wiki_eval_result.df_metrics_cie,
            df_metrics_open_ie=wiki_eval_result.df_metrics_open_ie,
            # df_preds_gt_cie=df_preds_gt_cie,
            df_preds_gt_cie=wiki_eval_result.df_predictions_cie_and_gt,
            df_preds_open_ie=wiki_eval_result.df_predictions_open_ie,
            df_instances_v13=wiki_eval_result.df_instances,
            df_additional_stats=wiki_eval_result.df_metrics_additional_triple_stats
        )

    wiki_eval_result.df_metrics_cie = df_metrics_cie
    wiki_eval_result.df_metrics_open_ie = df_metrics_open_ie
    wiki_eval_result.df_predictions_cie_and_gt = df_preds_gt_cie
    wiki_eval_result.df_predictions_open_ie = df_preds_open_ie
    wiki_eval_result.df_metrics_additional_triple_stats = df_additional_stats

    df_metrics_to_report = pd.DataFrame(metrics_to_report)
    agg, agg_open = make_agg_and_agg_open(
        df_wiki_metrics_cie=wiki_eval_result.df_metrics_cie,
        df_metrics_open_ie=wiki_eval_result.df_metrics_open_ie,
        metrics_to_report=df_metrics_to_report,
        groupby_cols=['tkgu_type', 'model', 'metric', 'evaluator_model']
    )

    agg_all = make_agg_all(agg, agg_open)

    allowed_models_table1 = arguments_main.models_to_report
    make_metrics_cli_table(
        agg_all=agg_all,
        spec=df_metrics_to_report,
        allowed_models=allowed_models_table1,
        model_name_map=model_name_to_latex,  # same dict, just different param name
        show_aliases=['C', 'R', 'P', 'F1'],
    )

    # QID Exact-Match table (separate, only for QID-capable models)
    if 'cie_exact_match' in arguments_main.metrics_to_calculate:
        cie_qid_spec = pd.DataFrame([
            {'metric': 'cie-precision', 'evaluator_model': 'exact_match', 'alias': 'P', 'group': 'QID', 'multiply_by_100': True},
            {'metric': 'cie-recall', 'evaluator_model': 'exact_match', 'alias': 'R', 'group': 'QID', 'multiply_by_100': True},
            {'metric': 'cie-f1', 'evaluator_model': 'exact_match', 'alias': 'F1', 'group': 'QID', 'multiply_by_100': True},
        ])

        agg_qid, _ = make_agg_and_agg_open(
            df_wiki_metrics_cie=wiki_eval_result.df_metrics_cie,
            df_metrics_open_ie=wiki_eval_result.df_metrics_open_ie,
            metrics_to_report=cie_qid_spec,
            groupby_cols=['tkgu_type', 'model', 'metric', 'evaluator_model']
        )
        empty_agg_open = pd.DataFrame(columns=agg_qid.columns)
        agg_qid_all = make_agg_all(agg_qid, empty_agg_open)

        cie_exact_match_config = arguments_main.metrics_to_calculate['cie_exact_match']
        if 'models_to_report' in cie_exact_match_config:
            cie_models = cie_exact_match_config['models_to_report']
        else:
            cie_models = allowed_models_table1

        logger.info('\n\n=== QID Exact-Match Metrics (P / R / F1) ===')
        make_metrics_cli_table(
            agg_all=agg_qid_all,
            spec=cie_qid_spec,
            allowed_models=cie_models,
            model_name_map=model_name_to_latex,
            show_aliases=['P', 'R', 'F1'],
        )

    logger.info('end_adding_completeness_metric')

    # Final post-table notice if relik-cie was skipped due to missing KG snapshots.
    if getattr(arguments_main, '_kg_skipped_relik_cie', False):
        logger.info('')
        logger.info('=' * 72)
        logger.info(
            f'NOTE: relik-cie not calculated — KG snapshots not found in '
            f'{arguments_main.kg_snapshots_path}'
        )
        logger.info('')
        logger.info(
            'To include relik-cie in the results table, download the KG '
            'snapshots and re-run:'
        )
        logger.info('  ./scripts/download_data.sh --kg')
        logger.info(
            f'  rm -rf {arguments_main.cache_path}    # clear stale cache'
        )
        logger.info('  ./scripts/run/evaluate.sh')
        logger.info('=' * 72)

    logger.info('END_MAIN_BYE_BYE')

    ##### END - LOGIC TO ACTUALLY OBTAIN SCORES AND BUILD EVALUATION TABLE


if __name__ == '__main__':
    main()
