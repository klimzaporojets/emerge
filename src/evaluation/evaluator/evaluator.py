import logging

from evaluation.misc.args_evaluation import ArgsEvaluation
from evaluation.misc.utils import save_to_cache
from evaluation.misc.wiki_eval_result import WikiEvalResult

from evaluation.evaluator.metrics.completeness import CompletenessMetric
from evaluation.evaluator.metrics.entity_coverage import EntityCoverage
from evaluation.evaluator.metrics.graph_judge import GraphJudgeMetric
from evaluation.evaluator.metrics.uniqueness import Uniqueness
from evaluation.evaluator.metrics.factualness import Factualness
from evaluation.evaluator.metrics.llm_general import LLMGeneral
from evaluation.evaluator.metrics.cie_exact_match import CIEExactMatchMetric

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs all configured metrics (completeness, entity_coverage, graph_judge, etc.) on preloaded data."""

    def __init__(self, config: ArgsEvaluation):
        self.config = config
        self.metrics_to_calculate = config.metrics_to_calculate
        self.models_to_evaluate = config.models_to_evaluate
        self.cache_path = config.cache_path

        self.metrics = self._build_metrics()

    def _build_metrics(self):
        """
        Instantiate metric objects in execution order.
        """
        metrics = []

        if 'completeness' in self.metrics_to_calculate:
            metrics.append(
                CompletenessMetric(
                    config=self.metrics_to_calculate['completeness'],
                    models=self.models_to_evaluate,
                    cache_path=self.cache_path
                )
            )

        if 'entity_coverage' in self.metrics_to_calculate:
            metrics.append(
                EntityCoverage(
                    config=self.metrics_to_calculate['entity_coverage'],
                    models=self.models_to_evaluate,
                    cache_path=self.cache_path
                )
            )

        if 'graph_judge' in self.metrics_to_calculate:
            metrics.append(
                GraphJudgeMetric(
                    config=self.metrics_to_calculate['graph_judge'],
                    models=self.models_to_evaluate,
                    cache_path=self.cache_path
                )
            )

        if 'uniqueness' in self.metrics_to_calculate:
            metrics.append(
                Uniqueness(
                    config=self.metrics_to_calculate['uniqueness'],
                    models=self.models_to_evaluate,
                    cache_path=self.cache_path
                )
            )

        if 'factualness' in self.metrics_to_calculate:
            metrics.append(
                Factualness(
                    config=self.metrics_to_calculate['factualness'],
                    models=self.models_to_evaluate,
                    cache_path=self.cache_path
                )
            )

        if 'llm_general' in self.metrics_to_calculate:
            metrics.append(
                LLMGeneral(
                    config=self.metrics_to_calculate['llm_general'],
                    models=self.models_to_evaluate,
                    cache_path=self.cache_path
                )
            )

        if 'cie_exact_match' in self.metrics_to_calculate:
            metrics.append(
                CIEExactMatchMetric(
                    config=self.metrics_to_calculate['cie_exact_match'],
                    models=self.models_to_evaluate,
                    cache_path=self.cache_path
                )
            )

        return metrics

    def evaluate(self, wiki_eval_result: WikiEvalResult) -> WikiEvalResult:
        """
        Sequentially apply all metrics.
        Each metric is responsible for caching and skipping.
        """
        logger.info('starting evaluation pipeline')

        for metric in self.metrics:
            logger.info(f'running metric: {metric.name}')
            wiki_eval_result = metric.compute(wiki_eval_result)

        logger.info('evaluation pipeline finished')
        return wiki_eval_result
