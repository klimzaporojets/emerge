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
) -> Dict[str, List[Dict]]:
    """
    Compute exact-match precision, recall, F1 on QID triples.

    Each row in batch_cie_qid_triples is:
        [hash_id, gt_qid_list, pred_qid_list, assessments]

    GT is filtered by LLM assessment flags. Predicted triples with invalid
    QIDs (null, --NME--, wrong prefix) are excluded.

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
