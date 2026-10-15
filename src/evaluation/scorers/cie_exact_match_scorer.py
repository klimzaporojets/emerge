import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def _is_valid_qid_triple(triple) -> bool:
    """Check that all three components are valid Wikidata identifiers (Q/P/Q)."""
    if len(triple) != 3:
        return False
    h, r, t = triple
    if any(x is None for x in (h, r, t)):
        return False
    h, r, t = str(h), str(r), str(t)
    if any(x in ('', '--NME--', 'NME') for x in (h, r, t)):
        return False
    return h.startswith('Q') and r.startswith('P') and t.startswith('Q')


def calculate_cie_exact_match(
    batch_cie_qid_triples: list,
    model: str,
    tkgu_type: str,
    model_alias: str = 'exact_match',
    exclude_empty_gold: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Compute exact-match precision, recall, F1 on QID triples.

    Each row in batch_cie_qid_triples is:
        [hash_id, gt_qid_list, pred_qid_list, assessments]

    GT is filtered by LLM assessment flags. Predicted triples with invalid
    QIDs (null, --NME--, wrong prefix) are excluded.

    `exclude_empty_gold` (default False = the published Table-3 behavior): when True, an instance
    whose 405B/LLM-assessment-filtered `gt_set` is EMPTY (it has gold triples of this tkgu_type but
    none is assessed-true, so there is nothing valid to recall) is SKIPPED entirely — no P/R/F1 row.
    This makes cie-recall consistent with completeness / graph-judge, which aggregate per-triple and
    therefore already drop such instances (an instance with 0 valid gold triples contributes nothing).
    With False, an empty gold set yields recall `0/0 -> 0` and the instance is kept (dragging the mean).
    See rebuttal/neurips_2026/BUG_er_empty_gold_population.md. Toggle uniformly across ALL models.

    Returns dict with 'scores_per_instance' key containing metric rows.
    """
    scores = []

    for row in batch_cie_qid_triples:
        assert len(row) == 4, (
            f'batch_cie_qid_triples row must have 4 elements [hash_id, gt_qids, pred_qids, assessments], '
            f'got {len(row)}'
        )
        hash_id = row[0]
        gt_qid_list = row[1]
        pred_qid_list = row[2]
        assessments = row[3]

        assert len(gt_qid_list) == len(assessments), (
            f'GT QID list ({len(gt_qid_list)}) and assessments ({len(assessments)}) '
            f'length mismatch for hash_id={hash_id}'
        )

        # Filter GT by LLM assessment
        gt_set = set()
        for i, gt_triple in enumerate(gt_qid_list):
            if i < len(assessments) and assessments[i]:
                gt_set.add(tuple(gt_triple))

        # No valid (assessed-true) gold of this tkgu_type -> recall is undefined (0/0). Under
        # exclude_empty_gold, drop the instance so it does not contribute a recall-0 to the mean
        # (matches completeness/graph-judge, which are per-triple and already exclude it).
        if exclude_empty_gold and len(gt_set) == 0:
            continue

        # Filter pred: skip triples with invalid QID components
        pred_set = set()
        for pred_triple in pred_qid_list:
            if _is_valid_qid_triple(pred_triple):
                pred_set.add(tuple(pred_triple))

        # Compute P/R/F1
        if len(pred_set) == 0 and len(gt_set) == 0:
            precision = recall = f1 = 0.0
        else:
            tp = len(gt_set & pred_set)
            precision = tp / len(pred_set) if pred_set else 0.0
            recall = tp / len(gt_set) if gt_set else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        base = {
            'hash_id': hash_id,
            'tkgu_type': tkgu_type,
            'model': model,
            'evaluator_model': model_alias,
            'granularity_level': 'instance',
        }
        scores.append({**base, 'metric': 'cie-precision', 'score': precision})
        scores.append({**base, 'metric': 'cie-recall', 'score': recall})
        scores.append({**base, 'metric': 'cie-f1', 'score': f1})

    return {'scores_per_instance': scores}
