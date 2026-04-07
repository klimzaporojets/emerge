import logging
import time
from typing import List, Tuple, Dict

import torch

from dataset.emerge.utils.constants import TKGU_TASKS
from evaluation.evaluator.metrics.base import Metric, log_gpu_memory
from evaluation.misc.wiki_eval_result import WikiEvalResult
from evaluation.scorers.entity_coverage_scorer import calculate_entity_coverage_model
from evaluation.scorers.misc.graph_matching import prepare_input_to_calculate_ent_coverage

logger = logging.getLogger(__name__)


class EntityCoverage(Metric):
    """Measures how well predicted entities match ground-truth entities (BERTScore + sentence-transformers)."""
    name = 'entity_coverage'

    def compute(self, wiki_eval_result: WikiEvalResult) -> WikiEvalResult:
        """Compute entity coverage scores for all models and TKGU operations."""
        model_task_pairs: List[Tuple[str, str]] = [
            (model_name, task_name)
            for model_name in self.models_to_evaluate
            for task_name in TKGU_TASKS
        ]
        curr_similarity_model: Dict

        for curr_similarity_model in self.config['similarity_models']:
            start = time.time()
            # model_scorers: List = None
            nr_processed = 0
            logger.info(f'calculating_entity_coverage_{curr_similarity_model["model_alias"]}')

            model_scorers = []
            # model_scorers = self._obtain_scorers(similarity_model_config=curr_similarity_model)

            effective_max_workers = max(1, curr_similarity_model['workers'])
            logger.info(f'effective_max_workers: {effective_max_workers}')
            for curr_model_name, curr_tkgu_type in model_task_pairs:
                logger.info('==================================================================')
                logger.info(f'calculating_entity_coverage_{curr_similarity_model["model_alias"]}_'
                            f'for: {curr_model_name} -- {curr_tkgu_type}')
                assert curr_tkgu_type in {'d-triples', 'x-triples', 'e-triples', 'ee-triples', 'ee-kg-triples'}

                # ---- BEGIN cache filtering ----
                gt_pred_triples = self._filter_cached_rows(
                    df_metrics=wiki_eval_result.df_metrics_cie,
                    metric='ent-coverage-all-f1',
                    model=curr_model_name,
                    tkgu_type=curr_tkgu_type,
                    model_alias=curr_similarity_model['model_alias'],
                    rows=wiki_eval_result.batch_completeness_openie[curr_model_name][curr_tkgu_type],
                    hash_id_fn=lambda row: row[0],
                    log_prefix=(
                        f'entity_coverage {curr_similarity_model["model_alias"]} '
                        f'{curr_model_name} / {curr_tkgu_type}'
                    ),
                )
                logger.info(f'len(gt_pred_triples): {len(gt_pred_triples)}')
                if not gt_pred_triples:
                    logger.info('skipping_in_entity_coverage, nothing to calculate or in cache')
                    continue

                # # ------------------------
                # # Split into batches
                # # ------------------------
                batches_inter = prepare_input_to_calculate_ent_coverage(gt_pred_triples=gt_pred_triples)
                if not batches_inter:
                    logger.info('skipping_in_entity_coverage, nothing to calculate or in cache')
                    continue

                if len(model_scorers) == 0:
                    model_scorers = self._obtain_scorers(similarity_model_config=curr_similarity_model)
                nr_processed += 1
                ec_results_model = calculate_entity_coverage_model(
                    gt_pred_triples=gt_pred_triples,
                    tkgu_type=curr_tkgu_type,
                    model=curr_model_name,
                    model_scorers=model_scorers,
                    scorer_workers=effective_max_workers,
                    model_alias=curr_similarity_model['model_alias'],
                    model_batch_size=curr_similarity_model.get('scorer_batch_size', 64),
                    shared_model_across_threads=curr_similarity_model['shared_model_across_threads'],
                    implementation_class=curr_similarity_model['implementation_class'],
                    batches_inter=batches_inter
                )
                if not ec_results_model:
                    logger.info('skipping_in_entity_coverage, nothing to calculate or in cache')
                    continue
                #
                wiki_eval_result.df_metrics_cie = self._merge_graph_judge_metrics(
                    df_wiki_metrics=wiki_eval_result.df_metrics_cie,
                    gj_results_model=ec_results_model,
                )
                logger.info('begin_saving_cache')
                self._save_cache(wiki_eval_result=wiki_eval_result)
                logger.info('end_saving_cache')
                logger.info('==================================================================')

            ## BEGIN releases memory for potential next round with other similarity model
            log_gpu_memory(f'before_cleanup_{curr_similarity_model["model_alias"]}_nr_processed={nr_processed}')
            for i in range(len(model_scorers)):
                model_scorers[i] = None  # drop references held by the list itself

            del model_scorers

            import gc
            gc.collect()

            if 'cuda' in curr_similarity_model['device'] and nr_processed > 0:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            log_gpu_memory(f'after_cleanup_{curr_similarity_model["model_alias"]}')
            # END memory release
            #
            #
            end = time.time()
            elapsed = end - start

            logger.info('end_saving_the_dfs_entity_coverage')

            logger.info(
                f'elapsed_entity_coverage_{curr_similarity_model["model_alias"]}: '
                f'{elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)'
            )
        return wiki_eval_result
