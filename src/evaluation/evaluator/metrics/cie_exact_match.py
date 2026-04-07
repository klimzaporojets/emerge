import logging

from dataset.emerge.utils.constants import TKGU_TASKS
from evaluation.evaluator.metrics.base import Metric
from evaluation.misc.wiki_eval_result import WikiEvalResult
from evaluation.scorers.cie_exact_match_scorer import calculate_cie_exact_match

logger = logging.getLogger(__name__)


class CIEExactMatchMetric(Metric):
    """Exact-match evaluation of Wikidata QID predictions (Closed IE models only)."""
    name = 'cie_exact_match'

    def compute(self, wiki_eval_result: WikiEvalResult) -> WikiEvalResult:
        """Compute QID exact-match precision, recall, and F1 for CIE models."""
        if wiki_eval_result.batch_cie_qid_triples is None:
            logger.info('batch_cie_qid_triples is None, skipping cie_exact_match')
            return wiki_eval_result

        model_alias = self.config.get('model_alias', 'exact_match')

        for curr_model_name in self.models_to_evaluate:
            if curr_model_name not in wiki_eval_result.batch_cie_qid_triples:
                continue

            for curr_tkgu_type in TKGU_TASKS:
                rows = wiki_eval_result.batch_cie_qid_triples[curr_model_name].get(curr_tkgu_type, [])

                if not rows:
                    continue

                # Filter out already-cached hash_ids
                rows = self._filter_cached_rows(
                    df_metrics=wiki_eval_result.df_metrics_cie,
                    metric='cie-f1',
                    model=curr_model_name,
                    tkgu_type=curr_tkgu_type,
                    model_alias=model_alias,
                    rows=rows,
                    hash_id_fn=lambda row: row[0],
                    log_prefix=f'cie_exact_match {curr_model_name} / {curr_tkgu_type}',
                )

                if not rows:
                    continue

                logger.info(f'calculating cie_exact_match for {curr_model_name} / {curr_tkgu_type} '
                            f'({len(rows)} instances)')

                results = calculate_cie_exact_match(
                    batch_cie_qid_triples=rows,
                    model=curr_model_name,
                    tkgu_type=curr_tkgu_type,
                    model_alias=model_alias,
                )

                wiki_eval_result.df_metrics_cie = self._merge_graph_judge_metrics(
                    df_wiki_metrics=wiki_eval_result.df_metrics_cie,
                    gj_results_model=results,
                )

                self._save_cache(wiki_eval_result=wiki_eval_result)

        return wiki_eval_result
