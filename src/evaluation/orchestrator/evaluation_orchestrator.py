# Orchestrates the evaluation pipeline: preloading, metric computation, and caching.
import logging
import pickle
from pathlib import Path
from typing import List, Dict, TypeVar, Type, Any

from evaluation.evaluator.evaluator import Evaluator
from evaluation.misc.args_evaluation import ArgsEvaluation
from evaluation.misc.utils import load_from_cache, save_to_cache
from evaluation.misc.wiki_eval_result import WikiEvalResult
from evaluation.preloader.preloader_emerge import EMERGEPreLoader


logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """Orchestrates the full evaluation pipeline: preloading, metric computation, and caching."""

    def __init__(self, config: ArgsEvaluation,
                 snapshot_to_triples,
                 cached: WikiEvalResult):
        self.config = config
        self.cache_path = config.cache_path
        self.metrics_to_calculate = config.metrics_to_calculate
        logger.info(f'begin_loading_from_cache: {self.cache_path}')
        self.cached: WikiEvalResult = cached
        logger.info(f'end_loading_from_cache: {self.cache_path}')
        self.snapshot_to_triples = snapshot_to_triples

    def evaluate(
            self,
            dataset_to_evaluate: List[Dict[str, Any]]
    ) -> WikiEvalResult:
        """Run preloading and all configured metrics on the dataset.

        Args:
            dataset_to_evaluate: List of dataset instances with model predictions.

        Returns:
            WikiEvalResult with computed metric scores.
        """
        import time
        start = time.perf_counter()

        preloader = EMERGEPreLoader(config=self.config,
                                    loaded_dataset=dataset_to_evaluate,
                                    snapshot_to_triples=self.snapshot_to_triples)
        preloaded: WikiEvalResult = preloader.preload(self.cached)

        elapsed_sec = time.perf_counter() - start
        logger.info(f'execution_time_preload: {elapsed_sec / 60:.2f} min ({elapsed_sec:.2f} s)')

        evaluator = Evaluator(config=self.config)
        wiki_eval_result = evaluator.evaluate(wiki_eval_result=preloaded)

        self.update_cache(wiki_eval_result=wiki_eval_result)
        return wiki_eval_result

    def update_cache(self, wiki_eval_result: WikiEvalResult):
        logger.info('updating cache')
        save_to_cache(self.cache_path, wiki_eval_result)
        self.cached = wiki_eval_result
