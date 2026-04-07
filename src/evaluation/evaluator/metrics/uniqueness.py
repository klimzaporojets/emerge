import logging
import time
from typing import List, Tuple, Dict

import torch

from dataset.emerge.utils.constants import TKGU_TASKS
from evaluation.evaluator.metrics.base import Metric
from evaluation.misc.wiki_eval_result import WikiEvalResult
from evaluation.scorers.uniqueness_scorer import calculate_uniqueness_model

logger = logging.getLogger(__name__)

class Uniqueness(Metric):
    name = 'uniqueness'

    def compute(self, wiki_eval_result: WikiEvalResult) -> WikiEvalResult:
        model_task_pairs: List[Tuple[str, str]] = [
            (model_name, task_name)
            for model_name in self.models_to_evaluate
            for task_name in TKGU_TASKS
        ]
        model_scorers: List
        curr_similarity_model: Dict
        for curr_similarity_model in self.config['similarity_models']:

            start = time.time()
            logger.info(f'calculating_uniqueness_{curr_similarity_model["model_alias"]}')

            model_scorers = self._obtain_scorers(similarity_model_config=curr_similarity_model)
            effective_max_workers = max(1, curr_similarity_model['workers'])

            if curr_similarity_model['ground_truth']:
                for curr_tkgu_type in TKGU_TASKS:
                    ### extra caching level
                    # done_hash_ids = self._done_hash_ids(
                    #     df=wiki_eval_result.df_metrics_cie,
                    #     metric='uniqueness',
                    #     model='ground-truth',
                    #     tkgu_type=curr_tkgu_type,
                    #     model_alias=curr_similarity_model['model_alias'],
                    # )
                    #
                    # gt_pred_triples = wiki_eval_result.batch_completeness_gt[curr_tkgu_type]
                    #
                    # gt_pred_triples = [
                    #     row for row in gt_pred_triples
                    #     if str(row[0]) not in done_hash_ids
                    # ]
                    gt_pred_triples = self._filter_cached_rows(
                        df_metrics=wiki_eval_result.df_metrics_cie,
                        metric='uniqueness',
                        model='ground-truth',
                        tkgu_type=curr_tkgu_type,
                        model_alias=curr_similarity_model['model_alias'],
                        rows=wiki_eval_result.batch_completeness_gt[curr_tkgu_type],
                        hash_id_fn=lambda row: row[0],
                        log_prefix=(
                            f'uniqueness ground-truth {curr_similarity_model["model_alias"]} '
                            f'{curr_tkgu_type}'
                        ),
                    )

                    logger.info(f'len(gt_pred_triples) uniqueness ground-truth: {len(gt_pred_triples)}')

                    if not gt_pred_triples:
                        logger.info(
                            f'skipping uniqueness ground-truth / {curr_tkgu_type} (cached)'
                        )
                        continue
                    ###

                    uniqueness_results_model = calculate_uniqueness_model(
                        gt_pred_triples=gt_pred_triples,
                        # gt_pred_triples=wiki_eval_result.batch_completeness_gt[curr_tkgu_type],
                        tkgu_type=curr_tkgu_type,
                        model='ground-truth',
                        model_scorers=model_scorers,
                        scorer_workers=effective_max_workers,
                        model_alias=curr_similarity_model['model_alias'],
                        model_batch_size=curr_similarity_model.get('scorer_batch_size', 64),
                        shared_model_across_threads=curr_similarity_model['shared_model_across_threads'],
                        implementation_class=curr_similarity_model['implementation_class'],
                        phi=curr_similarity_model['phi'],
                        calculate_on_pred=False
                    )
                    #
                    wiki_eval_result.df_metrics_cie = self._merge_graph_judge_metrics(
                        df_wiki_metrics=wiki_eval_result.df_metrics_cie,
                        gj_results_model=uniqueness_results_model,
                    )
                    logger.info('begin_saving_cache')
                    self._save_cache(wiki_eval_result=wiki_eval_result)
                    logger.info('end_saving_cache')

            for curr_model_name, curr_tkgu_type in model_task_pairs:
                # ----> start cache
                # done_hash_ids = self._done_hash_ids(
                #     df=wiki_eval_result.df_metrics_open_ie,
                #     metric='uniqueness',
                #     model=curr_model_name,
                #     tkgu_type=curr_tkgu_type,
                #     model_alias=curr_similarity_model['model_alias'],
                # )
                #
                # gt_pred_triples = wiki_eval_result.batch_completeness_openie[curr_model_name][curr_tkgu_type]
                #
                # gt_pred_triples = [
                #     row for row in gt_pred_triples
                #     if str(row[0]) not in done_hash_ids
                # ]
                #
                # if not gt_pred_triples:
                #     logger.info(
                #         f'skipping uniqueness {curr_similarity_model["model_alias"]} '
                #         f'for {curr_model_name} / {curr_tkgu_type} (cached)'
                #     )
                #     continue
                logger.info('==================================================================')

                logger.info(f'calculating_uniqueness_{curr_model_name}_for_{curr_tkgu_type}')
                gt_pred_triples = self._filter_cached_rows(
                    df_metrics=wiki_eval_result.df_metrics_open_ie,
                    metric='uniqueness',
                    model=curr_model_name,
                    tkgu_type=curr_tkgu_type,
                    model_alias=curr_similarity_model['model_alias'],
                    rows=wiki_eval_result.batch_completeness_openie[curr_model_name][curr_tkgu_type],
                    hash_id_fn=lambda row: row[0],
                    log_prefix=(
                        f'uniqueness {curr_similarity_model["model_alias"]} '
                        f'{curr_model_name} / {curr_tkgu_type}'
                    ),
                )
                logger.info(f'len(gt_pred_triples) uniqueness: {len(gt_pred_triples)}')

                if not gt_pred_triples:
                    logger.info(
                        f'skipping uniqueness {curr_similarity_model["model_alias"]} '
                        f'for {curr_model_name} / {curr_tkgu_type} (cached)'
                    )
                    continue

                # ----> end cache
                ###
                uniqueness_results_model = calculate_uniqueness_model(
                    gt_pred_triples=gt_pred_triples,
                    # gt_pred_triples=wiki_eval_result.batch_completeness_openie[curr_model_name][curr_tkgu_type],
                    tkgu_type=curr_tkgu_type,
                    model=curr_model_name,
                    model_scorers=model_scorers,
                    scorer_workers=effective_max_workers,
                    model_alias=curr_similarity_model['model_alias'],
                    model_batch_size=curr_similarity_model.get('scorer_batch_size', 64),
                    shared_model_across_threads=curr_similarity_model['shared_model_across_threads'],
                    implementation_class=curr_similarity_model['implementation_class'],
                    phi=curr_similarity_model['phi'],
                    calculate_on_pred=True
                )

                wiki_eval_result.df_metrics_open_ie = self._merge_graph_judge_metrics(
                    df_wiki_metrics=wiki_eval_result.df_metrics_open_ie,
                    gj_results_model=uniqueness_results_model,
                )
                #
                logger.info('begin_saving_cache')
                self._save_cache(wiki_eval_result=wiki_eval_result)
                logger.info('end_saving_cache')
                logger.info('==================================================================')

            ###
            ## BEGIN releases memory for potential next round with other similarity model
            for i in range(len(model_scorers)):
                model_scorers[i] = None  # drop references held by the list itself

            del model_scorers

            import gc
            gc.collect()

            if 'cuda' in curr_similarity_model['device']:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            # END memory release
            #
            #
            logger.info(f'end_saving_the_dfs_uniqueness_{curr_similarity_model["model_alias"]}')

            end = time.time()
            elapsed = end - start

            logger.info(
                f'elapsed_uniqueness_{curr_similarity_model["model_alias"]}: '
                f'{elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')
        return wiki_eval_result