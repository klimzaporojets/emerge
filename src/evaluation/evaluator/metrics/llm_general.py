import logging
import os

import pandas as pd
from transformers import AutoTokenizer
from unified_llm_client import UnifiedLLMClient

from evaluation.evaluator.metrics.base import Metric
from evaluation.misc.wiki_eval_result import WikiEvalResult
from evaluation.scorers.llm_as_a_judge_scorer import calculate_batched_llm_general_v13

logger = logging.getLogger(__name__)

class LLMGeneral(Metric):
    name = 'llm_general'

    def compute(self, wiki_eval_result: WikiEvalResult) -> WikiEvalResult:
        metric_conf = self.config
        arg_triples_per_prompt = metric_conf['triples_per_prompt']
        effective_max_workers = max(1, metric_conf['concurrency'])

        arg_tokenizer = metric_conf.get('tokenizer', None)
        tokenizer = None
        if arg_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(arg_tokenizer)

        env_var = metric_conf.get('pwd_env_variable')
        api_key = os.getenv(env_var) if env_var else None
        client = UnifiedLLMClient(
            base_url=metric_conf['base_url'],
            model=metric_conf['model'],
            api_key=api_key,
            concurrency=metric_conf['concurrency'],
            timeout=metric_conf['timeout'],
            temperature=metric_conf['temperature'],
            max_tokens=metric_conf['max_tokens'],
            model_max_context=None
        )

        ########### BEGIN gt
        if metric_conf['ground_truth']:
            # done_hash_ids = self._done_hash_ids(
            #     df=wiki_eval_result.df_metrics_cie,
            #     metric='llm_general',
            #     model='ground-truth',
            #     tkgu_type='*',
            #     model_alias=metric_conf['model'],
            # )
            #
            # triples_with_passages = [
            #     row for row in wiki_eval_result.batch_factualness_gt
            #     if str(row['field_hash_id']) not in done_hash_ids
            # ]
            #
            # if not triples_with_passages:
            #     logger.info('skipping llm_general ground-truth (cached)')
            triples_with_passages = self._filter_cached_rows(
                df_metrics=wiki_eval_result.df_metrics_cie,
                metric='llm_general',
                model='ground-truth',
                tkgu_type='*',
                model_alias=metric_conf['model'],
                rows=wiki_eval_result.batch_factualness_gt,
                hash_id_fn=lambda row: row['field_hash_id'],
                log_prefix='llm_general ground-truth',
            )

            logger.info(f'len(triples_with_passages) llm_general ground-truth: {len(triples_with_passages)}')
            if not triples_with_passages:
                logger.info('skipping_in_graph_judge llm_general ground-truth (cached)')
            else:
                llm_general_gt = calculate_batched_llm_general_v13(
                    triples_with_passages=triples_with_passages,
                    triples_per_prompt=arg_triples_per_prompt,
                    max_workers=effective_max_workers,
                    max_records=metric_conf['max_records'],
                    tokenizer=tokenizer,
                    prompt_paths=metric_conf['prompt_paths'],
                    llm_backend=metric_conf['llm_backend'],
                    model=metric_conf['model'],
                    client=client,
                    triples_with_qids=True
                )

                df_wiki_llm_general_metrics_gt = pd.DataFrame(
                    llm_general_gt['scores_per_triple']
                )

                keys = [
                    'hash_id', 'prompt_type',
                    'triple_head', 'triple_relation', 'triple_tail',
                    'triple_head_label', 'triple_relation_label', 'triple_tail_label'
                ]

                df_wiki_gt_only = wiki_eval_result.df_predictions_cie_and_gt[
                    wiki_eval_result.df_predictions_cie_and_gt['triple_type'] == 'in-dataset'
                ].copy()

                a_map = (
                    df_wiki_gt_only[keys + ['model', 'tkgu_type']]
                    .drop_duplicates()
                )

                df_wiki_metrics_gt_completed = (
                    df_wiki_llm_general_metrics_gt
                    .drop(columns=['model', 'tkgu_type'], errors='ignore')
                    .merge(a_map, on=keys, how='left')
                )

                df_wiki_metrics_gt_completed['model'] = 'ground-truth'
                missing = df_wiki_metrics_gt_completed[
                    df_wiki_metrics_gt_completed[['model', 'tkgu_type']].isna().any(axis=1)
                ]
                if not missing.empty:
                    raise ValueError(
                        f'Unmatched rows found: {len(missing)} rows have NaN in model/tkgu_type'
                    )

                wiki_eval_result.df_metrics_cie = self._merge_graph_judge_metrics(
                    df_wiki_metrics=wiki_eval_result.df_metrics_cie,
                    gj_results_model=df_wiki_metrics_gt_completed
                )
                logger.info('begin_saving_cache')
                self._save_cache(wiki_eval_result=wiki_eval_result)
                logger.info('end_saving_cache')

        ########### END gt

        ########### BEGIN pred
        # done_hash_ids = self._done_hash_ids(
        #     df=wiki_eval_result.df_metrics_open_ie,
        #     metric='llm_general',
        #     model='*',
        #     tkgu_type='*',
        #     model_alias=metric_conf['model'],
        # )
        #
        # triples_with_passages = [
        #     row for row in wiki_eval_result.batch_factualness_openie
        #     if str(row['field_hash_id']) not in done_hash_ids
        # ]
        #
        # if not triples_with_passages:
        #     logger.info('skipping llm_general predictions (cached)')
        #     return wiki_eval_result
        triples_with_passages = self._filter_cached_rows(
            df_metrics=wiki_eval_result.df_metrics_open_ie,
            metric='llm_general',
            model='*',
            tkgu_type='*',
            model_alias=metric_conf['model'],
            rows=wiki_eval_result.batch_factualness_openie,
            hash_id_fn=lambda row: row['field_hash_id'],
            log_prefix='llm_general predictions',
        )

        logger.info(f'len(triples_with_passages) llm_general: {len(triples_with_passages)}')

        if not triples_with_passages:
            logger.info(f'skipping_in_llm_general')
            return wiki_eval_result

        llm_general_pred = calculate_batched_llm_general_v13(
            triples_with_passages=triples_with_passages,
            triples_per_prompt=arg_triples_per_prompt,
            max_workers=effective_max_workers,
            max_records=metric_conf['max_records'],
            tokenizer=tokenizer,
            prompt_paths=metric_conf['prompt_paths'],
            llm_backend=metric_conf['llm_backend'],
            model=metric_conf['model'],
            client=client,
            triples_with_qids=False
        )

        df_wiki_llm_general_metrics_open_ie = pd.DataFrame(
            llm_general_pred['scores_per_triple']
        )

        keys = [
            'hash_id', 'prompt_type',
            'triple_head_label', 'triple_relation_label', 'triple_tail_label'
        ]

        a_map = (
            wiki_eval_result.df_predictions_open_ie[keys + ['model', 'tkgu_type']]
            .drop_duplicates()
        )

        df_wiki_metrics_open_ie_completed = (
            df_wiki_llm_general_metrics_open_ie
            .drop(columns=['model', 'tkgu_type'], errors='ignore')
            .merge(a_map, on=keys, how='left')
        )

        missing = df_wiki_metrics_open_ie_completed[
            df_wiki_metrics_open_ie_completed[['model', 'tkgu_type']].isna().any(axis=1)
        ]
        if not missing.empty:
            raise ValueError(
                f'Unmatched rows found: {len(missing)} rows have NaN in model/tkgu_type'
            )

        wiki_eval_result.df_metrics_open_ie = self._merge_graph_judge_metrics(
            df_wiki_metrics=wiki_eval_result.df_metrics_open_ie,
            gj_results_model=df_wiki_metrics_open_ie_completed
        )
        logger.info('begin_saving_cache')
        self._save_cache(wiki_eval_result=wiki_eval_result)
        logger.info('end_saving_cache')

        logger.info('end_saving_the_dfs_llm_general')
        return wiki_eval_result
