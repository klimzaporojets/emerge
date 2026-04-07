from evaluation.metrics.metric import Metric
from evaluation.misc.wiki_eval_result import WikiEvalResult


class CompletenessMetric(Metric):
    def __init__(self, wiki_eval_result: WikiEvalResult):
        super().__init__()
        self.wiki_eval_result = wiki_eval_result
