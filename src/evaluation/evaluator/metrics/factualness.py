import logging
import os

import pandas as pd
from unified_llm_client import UnifiedLLMClient

from evaluation.evaluator.metrics.base import Metric
from evaluation.misc.wiki_eval_result import WikiEvalResult
from evaluation.scorers.factualness_scorer import calculate_batched_factualness

logger = logging.getLogger(__name__)

class Factualness(Metric):
    name = 'factualness'

    def compute(self, wiki_eval_result: WikiEvalResult) -> WikiEvalResult:
        metric_conf = self.config

        arg_triples_per_prompt = metric_conf['triples_per_prompt']
        effective_max_workers = max(1, metric_conf['concurrency'])

        #
        env_var = metric_conf.get('pwd_env_variable')
        api_key = os.getenv(env_var) if env_var else None
        client = UnifiedLLMClient(
            base_url=metric_conf['base_url'],  # e.g. Azure / vLLM / OpenAI
            model=metric_conf['model'],  # e.g. meta-llama/Llama-3.1-70B-Instruct
            api_key=api_key,  # None for vLLM
            concurrency=metric_conf['concurrency'],
            timeout=metric_conf['timeout'],
            temperature=metric_conf['temperature'],
            max_tokens=metric_conf['max_tokens'],
            model_max_context=None
        )
        #
        ########### BEGIN gt
        if metric_conf['ground_truth']:
            triples_with_passages = self._filter_cached_rows(
                df_metrics=wiki_eval_result.df_metrics_cie,
                metric='factualness',
                model='ground-truth',
                tkgu_type='*',
                model_alias=metric_conf['model'],
                rows=wiki_eval_result.batch_factualness_gt,
                hash_id_fn=lambda row: row['field_hash_id'],
                log_prefix='factualness ground-truth',
            )
            logger.info(f'len(triples_with_passages) ground-truth: {len(triples_with_passages)}')

            if not triples_with_passages:
                logger.info('skipping_in_factualness ground-truth, nothing to calculate or in cache')
            else:
                factualness_gt = \
                    calculate_batched_factualness(
                        triples_with_passages=triples_with_passages,
                        # triples_with_passages=wiki_eval_result.batch_factualness_gt,
                        # batch_size=arg_triples_per_prompt,
                        triples_per_prompt=arg_triples_per_prompt,
                        max_workers=effective_max_workers,
                        max_records=metric_conf['max_records'],
                        # tokenizer=tokenizer,
                        prompt_paths=metric_conf['prompt_paths'],
                        llm_backend=metric_conf['llm_backend'],
                        model=metric_conf['model'],
                        client=client,
                        triples_with_qids=True
                    )

                df_wiki_factualness_metrics_gt = pd.DataFrame(factualness_gt['scores_per_triple'])
                keys = ['hash_id', 'prompt_type',
                        'triple_head', 'triple_relation', 'triple_tail',
                        'triple_head_label', 'triple_relation_label', 'triple_tail_label']

                df_wiki_gt_only = \
                    wiki_eval_result.df_predictions_cie_and_gt[ \
                        wiki_eval_result.df_predictions_cie_and_gt['triple_type'] == 'in-dataset'].copy()

                a_map = (
                    df_wiki_gt_only[keys + ['model', 'tkgu_type']]
                    .drop_duplicates()
                )

                df_wiki_metrics_gt_completed = (
                    df_wiki_factualness_metrics_gt
                    .drop(columns=['model', 'tkgu_type'], errors='ignore')
                    .merge(a_map, on=keys, how='left')
                )

                # strict validation: no missing model / tkgu_type allowed
                df_wiki_metrics_gt_completed['model'] = 'ground-truth'
                missing = df_wiki_metrics_gt_completed[
                    df_wiki_metrics_gt_completed[['model', 'tkgu_type']].isna().any(axis=1)
                ]
                if not missing.empty:
                    raise ValueError(
                        f'Unmatched rows found: {len(missing)} rows have NaN in model/tkgu_type'
                    )

                ###########

                wiki_eval_result.df_metrics_cie = self._merge_graph_judge_metrics(
                    df_wiki_metrics=wiki_eval_result.df_metrics_cie,
                    gj_results_model=df_wiki_metrics_gt_completed
                )
                logger.info('begin_saving_cache')
                self._save_cache(wiki_eval_result=wiki_eval_result)
                logger.info('end_saving_cache')

                logger.info(f'factualness df_wiki_metrics_open_ie.shape: '
                            f'{wiki_eval_result.df_metrics_open_ie.shape}')
                #
        ########### END gt

        ########### BEGIN pred
        logger.info('before_factualness_df_wiki_metrics_open_ie.shape: '
                    f'{wiki_eval_result.df_metrics_open_ie.shape}')
        triples_with_passages = self._filter_cached_rows(
            df_metrics=wiki_eval_result.df_metrics_open_ie,
            metric='factualness',
            model='*',
            tkgu_type='*',
            model_alias=metric_conf['model'],
            rows=wiki_eval_result.batch_factualness_openie,
            hash_id_fn=lambda row: row['field_hash_id'],
            log_prefix='factualness predictions',
        )
        logger.info(f'len(triples_with_passages): {len(triples_with_passages)}')
        if not triples_with_passages:
            logger.info('skipping_in_factualness, nothing to calculate or in cache')
            return wiki_eval_result

        factualness_pred = \
            calculate_batched_factualness(
                triples_with_passages=triples_with_passages,
                # triples_with_passages=wiki_eval_result.batch_factualness_openie,
                triples_per_prompt=arg_triples_per_prompt,
                max_workers=effective_max_workers,
                max_records=metric_conf['max_records'],
                # tokenizer=tokenizer,
                prompt_paths=metric_conf['prompt_paths'],
                llm_backend=metric_conf['llm_backend'],
                model=metric_conf['model'],
                client=client,
                triples_with_qids=False
            )

        df_wiki_factualness_metrics_open_ie = pd.DataFrame(factualness_pred['scores_per_triple'])
        keys = ['hash_id', 'prompt_type', 'triple_head_label', 'triple_relation_label', 'triple_tail_label']

        a_map = (
            wiki_eval_result.df_predictions_open_ie[keys + ['model', 'tkgu_type']]
            .drop_duplicates()
        )

        df_wiki_metrics_open_ie_completed = (
            df_wiki_factualness_metrics_open_ie
            .drop(columns=['model', 'tkgu_type'], errors='ignore')
            .merge(a_map, on=keys, how='left')
        )

        # strict validation: no missing model / tkgu_type allowed
        missing = df_wiki_metrics_open_ie_completed[
            df_wiki_metrics_open_ie_completed[['model', 'tkgu_type']].isna().any(axis=1)
        ]
        #
        if not missing.empty:
            raise ValueError(
                f'Unmatched rows found: {len(missing)} rows have NaN in model/tkgu_type'
            )

        ###########

        wiki_eval_result.df_metrics_open_ie = self._merge_graph_judge_metrics(
            df_wiki_metrics=wiki_eval_result.df_metrics_open_ie,
            gj_results_model=df_wiki_metrics_open_ie_completed
        )
        logger.info(f'factualness df_wiki_metrics_open_ie.shape: '
                    f'{wiki_eval_result.df_metrics_open_ie.shape}')
        logger.info('begin_saving_cache')
        self._save_cache(wiki_eval_result=wiki_eval_result)
        logger.info('end_saving_cache')

        #
        ########### END pred

        logger.info(f'end_saving_the_dfs_factualness')

        logger.debug('==================================================================')
        return wiki_eval_result