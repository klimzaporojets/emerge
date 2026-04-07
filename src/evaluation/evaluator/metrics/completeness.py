import copy
import logging
import time
from typing import List, Tuple

import torch

from dataset.emerge.utils.constants import TKGU_TASKS
from evaluation.evaluator.metrics.base import Metric, log_gpu_memory
from evaluation.misc.wiki_eval_result import WikiEvalResult
from evaluation.scorers.completeness_scorer import calculate_batched_completeness

logger = logging.getLogger(__name__)


class CompletenessMetric(Metric):
    """Measures whether predicted triples cover the same information as ground-truth triples."""
    name = 'completeness'

    def compute(self, wiki_eval_result: WikiEvalResult) -> WikiEvalResult:
        """Compute completeness scores using sentence-transformer similarity."""
        model_task_pairs: List[Tuple[str, str]] = [
            (model_name, task_name)
            for model_name in self.models_to_evaluate
            for task_name in TKGU_TASKS
        ]
        for curr_similarity_model in self.config['similarity_models']:
            models_copy = None
            effective_max_workers = None
            logger.debug('invoking calculate_batched_completeness for batch_completeness_cie')

            arg_device = curr_similarity_model['device']
            arg_triples_per_prompt = curr_similarity_model['batch_size']
            arg_model_alias = curr_similarity_model['model_alias']
            start_time = time.time()
            nr_processed = 0
            for curr_model_name, curr_tkgu_type in model_task_pairs:
                logger.info('==================================================================')
                logger.info(f'calculating_completeness_for: {curr_model_name} -- {curr_tkgu_type}')
                assert curr_tkgu_type in {'d-triples', 'x-triples', 'e-triples', 'ee-triples', 'ee-kg-triples'}

                logger.debug(f'calculate_batched_completeness for curr_model_name {curr_model_name} '
                             f'and curr_tkgu_type {curr_tkgu_type} '
                             f'len(gt_pred_triples): '
                             f'{len(wiki_eval_result.batch_completeness_openie[curr_model_name][curr_tkgu_type])}')

                #### BEGIN added cache
                logger.info('starting_potentially_slow_part')

                t_total_start = time.perf_counter()

                # ---- done_hash_ids_cies ----
                t0 = time.perf_counter()
                done_hash_ids_cies = self._done_hash_ids(
                    df=wiki_eval_result.df_metrics_cie,
                    metric='completeness',
                    model=curr_model_name,
                    tkgu_type=curr_tkgu_type,
                    model_alias=arg_model_alias,
                )
                t_done_cies = time.perf_counter() - t0

                # ---- done_hash_ids_oies ----
                t0 = time.perf_counter()
                done_hash_ids_oies = self._done_hash_ids(
                    df=wiki_eval_result.df_metrics_open_ie,
                    metric='completeness',
                    model=curr_model_name,
                    tkgu_type=curr_tkgu_type,
                    model_alias=arg_model_alias,
                )
                t_done_oies = time.perf_counter() - t0

                # CIE may have more hash_ids than OIE when score_empty_predictions_as_zero
                # is True: instances with GT triples but no predictions get CIE entries
                # (GT scored as 0.0) but no OIE entries (nothing to score on pred side).
                assert done_hash_ids_oies <= done_hash_ids_cies, (
                    f'OIE has hash_ids not in CIE — this should never happen. '
                    f'Extra OIE: {done_hash_ids_oies - done_hash_ids_cies}'
                )

                # ---- filter_cached_rows ----
                t0 = time.perf_counter()
                gt_pred_triples = self._filter_cached_rows(
                    df_metrics=wiki_eval_result.df_metrics_cie,
                    metric='completeness',
                    model=curr_model_name,
                    tkgu_type=curr_tkgu_type,
                    model_alias=arg_model_alias,
                    rows=wiki_eval_result.batch_completeness_openie[curr_model_name][curr_tkgu_type],
                    hash_id_fn=lambda row: row[0],
                    log_prefix=f'completeness {curr_model_name} / {curr_tkgu_type}',
                )
                t_filter = time.perf_counter() - t0

                t_total = time.perf_counter() - t_total_start

                logger.info(
                    f'ending_potentially_slow_part | '
                    f'done_cie: {t_done_cies:.3f}s | '
                    f'done_openie: {t_done_oies:.3f}s | '
                    f'filter: {t_filter:.3f}s | '
                    f'total: {t_total:.3f}s'
                )

                logger.info(f'len(gt_pred_triples): {len(gt_pred_triples)}')
                if not gt_pred_triples:
                    logger.info('skipping_in_completeness, nothing to calculate or in cache')
                    continue

                #### END added cache
                #
                if models_copy is None:
                    models_copy = self._obtain_sentence_transformer_copies(curr_similarity_model)
                    effective_max_workers = len(models_copy)
                nr_processed += 1
                lst_metrics_cie, lst_metrics_open_ie = \
                    calculate_batched_completeness(
                        # gt_pred_triples=wiki_eval_result.batch_completeness_openie[curr_model_name][curr_tkgu_type],
                        gt_pred_triples=gt_pred_triples,
                        batch_size=arg_triples_per_prompt,
                        max_workers=effective_max_workers,
                        models=models_copy,
                        device=arg_device,
                        model=curr_model_name,
                        tkgu_type=curr_tkgu_type,
                        max_records_calculate_completeness_score=curr_similarity_model['max_records'],
                        model_alias=arg_model_alias
                    )

                wiki_eval_result.df_metrics_cie = self._merge_graph_judge_metrics(
                    df_wiki_metrics=wiki_eval_result.df_metrics_cie,
                    gj_results_model={
                        'scores_per_triple': lst_metrics_cie
                    }
                )

                wiki_eval_result.df_metrics_open_ie = self._merge_graph_judge_metrics(
                    df_wiki_metrics=wiki_eval_result.df_metrics_open_ie,
                    gj_results_model={
                        'scores_per_triple': lst_metrics_open_ie
                    }
                )
                logger.info('begin_saving_cache')
                self._save_cache(wiki_eval_result=wiki_eval_result)
                logger.info('end_saving_cache')

                logger.info('==================================================================')

            del models_copy
            log_gpu_memory(f'completeness_before_cleanup_nr_processed={nr_processed}')
            if arg_device != 'cpu' and nr_processed>0:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            log_gpu_memory(f'completeness_after_cleanup_nr_processed={nr_processed}')

            logger.debug(f'batched_completeness_cie calculated in '
                         f'{time.time() - start_time} secs')

            logger.info('begin_saving_the_dfs_completeness')
        return wiki_eval_result
