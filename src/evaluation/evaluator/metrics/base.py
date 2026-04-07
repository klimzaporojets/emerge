# evaluation/metrics/base.py
import copy
import logging
from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
import torch
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer

from evaluation.misc.utils import save_to_cache
from evaluation.misc.wiki_eval_result import WikiEvalResult

logger = logging.getLogger(__name__)


def log_gpu_memory(label: str):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f'GPU_MEMORY [{label}] allocated={allocated:.2f}GB reserved={reserved:.2f}GB total={total:.2f}GB')
    else:
        logger.info(f'GPU_MEMORY [{label}] CUDA not available')


class Metric(ABC):
    """Base class for evaluation metrics. Subclasses implement compute() for specific metrics."""
    name: str  # e.g. "completeness"

    def __init__(self, config: dict, models: List, cache_path: str):
        self.config = config
        self.models_to_evaluate = models
        self.cache_path = cache_path

    def _save_cache(self, wiki_eval_result: WikiEvalResult) -> None:
        save_to_cache(self.cache_path, wiki_eval_result)

    def _obtain_sentence_transformer_copies(self, curr_similarity_model):
        cfg = dict(curr_similarity_model)
        cfg['implementation_class'] = 'sentence_transformer'
        cfg['shared_model_across_threads'] = False
        return self._obtain_scorers(cfg)

    def _filter_cached_rows(
            self,
            *,
            df_metrics,
            metric,
            model,
            tkgu_type,
            model_alias,
            rows,
            hash_id_fn,
            log_prefix,
    ):
        done_hash_ids = self._done_hash_ids(
            df=df_metrics,
            metric=metric,
            model=model,
            tkgu_type=tkgu_type,
            model_alias=model_alias,
        )
        if not done_hash_ids:
            return rows

        filtered = [
            row for row in rows
            if str(hash_id_fn(row)) not in done_hash_ids
        ]

        if not filtered:
            logger.info(f'skipping {log_prefix} (cached)')

        return filtered

    @staticmethod
    def _obtain_scorers(similarity_model_config: Dict):
        model_scorers = list()
        nr_models = 1
        if not similarity_model_config['shared_model_across_threads']:
            nr_models = similarity_model_config['workers']

        # model_instance = None
        log_gpu_memory(f'before_loading_{similarity_model_config["implementation_class"]}')
        if similarity_model_config['implementation_class'] == 'bert_scorer':
            model_instance = BERTScorer(
                # model_type='bert-base-uncased',
                model_type=similarity_model_config['model'],
                lang='en',
                idf=False,
                device=similarity_model_config['device'],
                batch_size=similarity_model_config['scorer_batch_size'],
                nthreads=similarity_model_config['scorer_nr_threads']
            )
        elif similarity_model_config['implementation_class'] == 'sentence_transformer':
            model_instance = SentenceTransformer(similarity_model_config['model']). \
                to(similarity_model_config['device'])
        else:
            raise RuntimeError(f'similarity model implementation_class not recognized: '
                               f'{similarity_model_config["implementation_class"]}')

        log_gpu_memory(f'after_loading_{similarity_model_config["implementation_class"]}')
        next_to_append = model_instance
        for idx_bs in range(nr_models):
            logger.debug(f'loading_model_{similarity_model_config["model_alias"]}_scorer_idx {idx_bs}')
            model_scorers.append(
                next_to_append
            )
            if idx_bs < nr_models - 1:
                if similarity_model_config['implementation_class'] == 'bert_scorer':
                    next_to_append = copy.deepcopy(model_instance)
                elif similarity_model_config['implementation_class'] == 'sentence_transformer':
                    next_to_append = copy.deepcopy(model_instance).to(similarity_model_config['device'])
                else:
                    raise RuntimeError('implementation_class not recognized')
        return model_scorers

    @staticmethod
    # def _done_hash_ids(df, metric, model, tkgu_type, model_alias):
    #     if df is None or df.empty:
    #         return set()
    #
    #     required_cols = {'hash_id', 'metric', 'model', 'tkgu_type', 'evaluator_model'}
    #     missing = required_cols - set(df.columns)
    #     assert not missing, f'missing required columns in metrics df: {missing}'
    #
    #     mask = (df['metric'] == metric)
    #
    #     if model != '*':
    #         mask &= (df['model'] == model)
    #
    #     if tkgu_type != '*':
    #         mask &= (df['tkgu_type'] == tkgu_type)
    #
    #     if model_alias != '*':
    #         mask &= (df['evaluator_model'] == model_alias)
    #
    #     return set(df.loc[mask, 'hash_id'].astype(str))

    def _done_hash_ids(df, metric, model, tkgu_type, model_alias):
        if df is None or df.empty:
            return set()

        # Fast path: assume schema already validated upstream
        query_parts = ["metric == @metric"]

        if model != "*":
            query_parts.append("model == @model")

        if tkgu_type != "*":
            query_parts.append("tkgu_type == @tkgu_type")

        if model_alias != "*":
            query_parts.append("evaluator_model == @model_alias")

        query_str = " & ".join(query_parts)

        result = df.query(query_str)["hash_id"]

        # Avoid repeated astype if already string
        if result.dtype != "object":
            result = result.astype(str)
        result = set(result)
        # assert all(isinstance(x, str) for x in result), (
        #     'hash_id set contains non-str values'
        # )
        return result

    @staticmethod
    def _merge_graph_judge_metrics(
            df_wiki_metrics: pd.DataFrame,
            gj_results_model: dict | pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        Append instance-level and (optionally) triple-level metric results
        to the main dataframe by row-wise concatenation.
        """

        if gj_results_model is None:
            return df_wiki_metrics

        # If results_dict is already a DataFrame, just append it
        if isinstance(gj_results_model, pd.DataFrame):
            return pd.concat(
                [df_wiki_metrics, gj_results_model],
                ignore_index=True,
                sort=False,
            )

        dfs = [df_wiki_metrics]

        # INSTANCE-LEVEL scores
        if 'scores_per_instance' in gj_results_model:
            df_instance = pd.DataFrame(gj_results_model['scores_per_instance'])
            dfs.append(df_instance)

        # TRIPLE-LEVEL scores
        if 'scores_per_triple' in gj_results_model:
            df_triple = pd.DataFrame(gj_results_model['scores_per_triple'])
            dfs.append(df_triple)

        # Concatenate rows (union of schemas)
        return pd.concat(dfs, ignore_index=True, sort=False)

    @abstractmethod
    def compute(self, wiki_eval_result: WikiEvalResult) -> WikiEvalResult:
        """
        Compute ONLY missing metric entries and append them
        """
        pass
