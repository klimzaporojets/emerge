import logging
import time
from typing import List, Tuple, Dict

import pandas as pd
import torch

from dataset.emerge.utils.constants import TKGU_TASKS
from evaluation.evaluator.metrics.base import Metric
from evaluation.misc.wiki_eval_result import WikiEvalResult
from evaluation.scorers.graph_scorers import calculate_gj_model, calculate_gj_bleu_rouge
from evaluation.scorers.misc.graph_matching import prepare_input_to_calculate_graph_scorers

logger = logging.getLogger(__name__)


class GraphJudgeMetric(Metric):
    """Graph-level triple matching using BLEU, ROUGE, BERTScore, and sentence-transformer similarities."""
    name = 'graph_judge'

    def compute(self, wiki_eval_result: WikiEvalResult) -> WikiEvalResult:
        """Compute G-BERTScore and related graph-judge metrics for all models."""
        model_task_pairs: List[Tuple[str, str]] = [
            (model_name, task_name)
            for model_name in self.models_to_evaluate
            for task_name in TKGU_TASKS
        ]

        logger.debug('invoking calculate_batched_completeness for batch_completeness_cie')
        start = time.time()

        logger.info(f'calculating_graph_judge_bleu_rouge')

        ### BEGIN - graph_judge BLEU AND ROUGE METRICS
        # if 'similarity_bleu_rouge' in self.arg_metrics_to_calculate['graph_judge']:
        if 'similarity_bleu_rouge' in self.config:
            model_alias = 'bleu_rouge'
            effective_max_workers = max(1,
                                        self.config['similarity_bleu_rouge']['workers'])
            for curr_model_name, curr_tkgu_type in model_task_pairs:
                logger.debug(f'calculating_graph_judge_bleu_rouge_for: {curr_model_name} -- {curr_tkgu_type}')
                assert curr_tkgu_type in {'d-triples', 'x-triples', 'e-triples', 'ee-triples', 'ee-kg-triples'}

                logger.debug('==================================================================')
                # ---- BEGIN cache filtering ----
                gt_pred_triples = self._filter_cached_rows(
                    df_metrics=wiki_eval_result.df_metrics_cie,
                    # metric='graph_judge',
                    metric='bleu-f1',
                    model=curr_model_name,
                    tkgu_type=curr_tkgu_type,
                    model_alias=model_alias,
                    rows=wiki_eval_result.batch_completeness_openie[curr_model_name][curr_tkgu_type],
                    hash_id_fn=lambda row: row[0],
                    log_prefix=f'graph_judge bleu/rouge {curr_model_name} / {curr_tkgu_type}',
                )

                logger.info(f'len(gt_pred_triples): {len(gt_pred_triples)}')

                if not gt_pred_triples:
                    logger.info('skipping_in_graph_judge_1, nothing to calculate or in cache')
                    continue

                # ---- END cache filtering ----

                gj_results_bleu_rouge = calculate_gj_bleu_rouge(
                    gt_pred_triples=gt_pred_triples,
                    # gt_pred_triples=wiki_eval_result.batch_completeness_openie[curr_model_name][curr_tkgu_type],
                    tkgu_type=curr_tkgu_type,
                    model=curr_model_name,
                    max_workers=effective_max_workers,
                    model_alias=model_alias
                )
                if not gj_results_bleu_rouge:
                    logger.info('skipping_in_graph_judge_2, nothing to calculate or in cache')
                    continue

                wiki_eval_result.df_metrics_cie = self._merge_graph_judge_metrics(
                    df_wiki_metrics=wiki_eval_result.df_metrics_cie,
                    gj_results_model=gj_results_bleu_rouge,
                )
                logger.info('begin_saving_cache')
                self._save_cache(wiki_eval_result=wiki_eval_result)
                logger.info('end_saving_cache')

            logger.info('end_saving_the_dfs_similarity_bleu_rouge')

        end = time.time()
        elapsed = end - start

        logger.info(f'elapsed_bleu_rouge: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')
        ### END - graph_judge BLEU AND ROUGE METRICS

        ### BEGIN - graph_judge metrics based on similarity models (e.g., sentence transformers)
        model_scorers: List
        curr_similarity_model: Dict
        for curr_similarity_model in self.config['similarity_models']:
            start = time.time()
            logger.info(f'calculating_graph_judge_{curr_similarity_model["model_alias"]}')

            model_scorers = []
            effective_max_workers = max(1, curr_similarity_model['workers'])
            nr_processed = 0
            for curr_model_name, curr_tkgu_type in model_task_pairs:
                logger.info('==================================================================')
                logger.info(f'calculating_graph_judge_{curr_similarity_model["model_alias"]}_'
                            f'for: {curr_model_name} -- {curr_tkgu_type}')
                assert curr_tkgu_type in {'d-triples', 'x-triples', 'e-triples', 'ee-triples', 'ee-kg-triples'}

                #
                # ---- BEGIN cache filtering ----
                gt_pred_triples = self._filter_cached_rows(
                    df_metrics=wiki_eval_result.df_metrics_cie,
                    # metric='graph_judge',
                    metric='gj-f1',
                    model=curr_model_name,
                    tkgu_type=curr_tkgu_type,
                    model_alias=curr_similarity_model['model_alias'],
                    rows=wiki_eval_result.batch_completeness_openie[curr_model_name][curr_tkgu_type],
                    hash_id_fn=lambda row: row[0],
                    log_prefix=(
                        f'graph_judge {curr_similarity_model["model_alias"]} '
                        f'{curr_model_name} / {curr_tkgu_type}'
                    ),
                )

                logger.info(f'graph_judge len(gt_pred_triples): {len(gt_pred_triples)}')

                if not gt_pred_triples:
                    logger.info('skipping_in_graph_judge_3, nothing to calculate or in cache')
                    continue

                # ---- END cache filtering ----
                batches_inter = prepare_input_to_calculate_graph_scorers(gt_pred_triples=gt_pred_triples)
                if not batches_inter:
                    logger.info('skipping_in_graph_judge_4, nothing to calculate or in cache')
                    continue

                if len(model_scorers) == 0:
                    model_scorers = self._obtain_scorers(similarity_model_config=curr_similarity_model)

                #
                nr_processed += 1
                gj_results_model = calculate_gj_model(
                    # gt_pred_triples=wiki_eval_result.batch_completeness_openie[curr_model_name][curr_tkgu_type],
                    # gt_pred_triples=gt_pred_triples,
                    tkgu_type=curr_tkgu_type,
                    model=curr_model_name,
                    model_scorers=model_scorers,
                    scorer_workers=effective_max_workers,
                    model_alias=curr_similarity_model['model_alias'],
                    model_batch_size=curr_similarity_model['scorer_batch_size'],
                    data_batch_size=curr_similarity_model['data_batch_size'],
                    shared_model_across_threads=curr_similarity_model['shared_model_across_threads'],
                    implementation_class=curr_similarity_model['implementation_class'],
                    batches_inter=batches_inter
                )
                if not gj_results_model:
                    logger.info('skipping_in_graph_judge_5, nothing to calculate or in cache')
                    continue
                #
                wiki_eval_result.df_metrics_cie = self._merge_graph_judge_metrics(
                    df_wiki_metrics=wiki_eval_result.df_metrics_cie,
                    gj_results_model=gj_results_model,
                )
                if gj_results_model is not None:
                    df_add = pd.DataFrame(gj_results_model['scores_per_triple_additional_stats'])
                    if not df_add.empty:
                        wiki_eval_result.df_metrics_additional_triple_stats = pd.concat(
                            [wiki_eval_result.df_metrics_additional_triple_stats, df_add],
                            ignore_index=True,
                            sort=False,
                        )
                else:
                    logger.info('skipping_in_graph_judge_6, nothing to calculate or in cache')
                    continue
                logger.info('begin_saving_cache')
                self._save_cache(wiki_eval_result=wiki_eval_result)
                logger.info('end_saving_cache')
                logger.info('==================================================================')

            ## BEGIN releases memory for potential next round with other similarity model
            for i in range(len(model_scorers)):
                model_scorers[i] = None  # drop references held by the list itself

            del model_scorers

            import gc
            gc.collect()

            if 'cuda' in curr_similarity_model['device'] and nr_processed > 0:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            # END memory release
            logger.info(f'end_saving_the_dfs_graph_judge_{curr_similarity_model["model_alias"]}')
            end = time.time()
            elapsed = end - start

            logger.info(
                f'elapsed_graph_judge_{curr_similarity_model["model_alias"]}: '
                f'{elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')
        ### END - graph_judge metrics based on similarity models (e.g., sentence transformers)
        return wiki_eval_result
