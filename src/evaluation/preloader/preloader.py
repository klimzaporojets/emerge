from abc import ABC
from typing import Dict

import logging

from dataset.emerge.utils.constants import TKGU_TASKS
from evaluation.misc.utils import normalize_triple_string

logger = logging.getLogger(__name__)


class EvaluationPreLoader(ABC):

    @staticmethod
    def process_other_oie_predictions(
            curr_model_name: str,
            curr_predictions: Dict,
            predicted_triples_open_ie: Dict
    ) -> None:
        """Process Relik model predictions and populate triple dictionaries."""
        if len(curr_predictions) >= 1:
            assert isinstance(curr_predictions, dict), \
                f'{curr_model_name} predictions should be in a dictionary'

            for curr_relik_predicted_triple in curr_predictions['predicted_triples']:
                extracted_rel = tuple(curr_relik_predicted_triple['extracted_relation'])
                for task in ['x-triples', 'e-triples', 'ee-triples', 'ee-kg-triples']:
                    predicted_triples_open_ie[curr_model_name][task].add(extracted_rel)

    @staticmethod
    def process_llm_tool_predictions(
            curr_model_name: str,
            curr_predictions: Dict,
            predicted_triples_open_ie: Dict,
            predicted_triples_cie: Dict,
            pred_triple_qid_to_labels: Dict
    ) -> None:
        """Process LLM-tool model predictions and populate triple dictionaries."""
        for curr_task in TKGU_TASKS:
            entry_curr_task = f'predicted_{curr_task.replace("-", "_")}'
            if entry_curr_task in curr_predictions:
                for curr_predicted_triple in curr_predictions[entry_curr_task]:
                    curr_labels = curr_predicted_triple['triple_labels']
                    normalized_triple = (
                        normalize_triple_string(curr_labels[0]),
                        curr_labels[1],
                        normalize_triple_string(curr_labels[2])
                    )
                    predicted_triples_open_ie[curr_model_name][curr_task].add(normalized_triple)
                    predicted_triples_cie[curr_model_name][curr_task].add(
                        tuple(curr_predicted_triple['triple_qids'])
                    )
                    pred_triple_qid_to_labels[tuple(curr_predicted_triple['triple_qids'])] = \
                        tuple(curr_labels)

    @staticmethod
    def process_edc_original_predictions(
            curr_model_name: str,
            curr_predictions: Dict,
            predicted_triples_open_ie: Dict
    ) -> None:
        """Process EDC model predictions and populate triple dictionaries."""
        if len(curr_predictions) >= 1:
            assert isinstance(curr_predictions, dict), \
                'edc predictions should be in a dictionary'

            oie_predictions = curr_predictions['predicted_triples_oie']

            for curr_edc_pred_triple in oie_predictions.get('oie_add', []):
                normalized_triple = (
                    normalize_triple_string(curr_edc_pred_triple[0]),
                    normalize_triple_string(curr_edc_pred_triple[1]),
                    normalize_triple_string(curr_edc_pred_triple[2])
                )
                for task in ['x-triples', 'e-triples', 'ee-triples', 'ee-kg-triples']:
                    predicted_triples_open_ie[curr_model_name][task].add(normalized_triple)

    @staticmethod
    def process_edc_plus_predictions(
            curr_model_name: str,
            curr_predictions: Dict,
            predicted_triples_open_ie: Dict
    ) -> None:
        """Process EDC model predictions and populate triple dictionaries."""
        if len(curr_predictions) >= 1:
            assert isinstance(curr_predictions, dict), \
                'edc predictions should be in a dictionary'

            oie_predictions = curr_predictions['predicted_triples_oie']

            for curr_edc_pred_triple in oie_predictions.get('oie_deprecate', []):
                normalized_triple = (
                    normalize_triple_string(curr_edc_pred_triple[0]),
                    normalize_triple_string(curr_edc_pred_triple[1]),
                    normalize_triple_string(curr_edc_pred_triple[2])
                )
                predicted_triples_open_ie[curr_model_name]['d-triples'].add(normalized_triple)

            for curr_edc_pred_triple in oie_predictions.get('oie_add', []):
                normalized_triple = (
                    normalize_triple_string(curr_edc_pred_triple[0]),
                    normalize_triple_string(curr_edc_pred_triple[1]),
                    normalize_triple_string(curr_edc_pred_triple[2])
                )
                for task in ['x-triples', 'e-triples', 'ee-triples']:
                    predicted_triples_open_ie[curr_model_name][task].add(normalized_triple)

            for curr_edc_pred_triple in oie_predictions.get('oie_not_in_text', []):
                normalized_triple = (
                    normalize_triple_string(curr_edc_pred_triple[0]),
                    normalize_triple_string(curr_edc_pred_triple[1]),
                    normalize_triple_string(curr_edc_pred_triple[2])
                )
                predicted_triples_open_ie[curr_model_name]['ee-kg-triples'].add(normalized_triple)

    @staticmethod
    def process_relik_predictions(
            curr_model_name: str,
            curr_predictions: Dict,
            predicted_triples_open_ie: Dict,
            predicted_triples_cie: Dict,
            pred_triple_qid_to_labels: Dict,
            snapshot_to_triples: Dict,
            str_snapshot_year: str,
            use_triple_labels_as_surface_forms: bool = False
    ) -> None:
        """Process Relik model predictions and populate triple dictionaries."""
        assert curr_model_name in {'relik-oie', 'relik-cie'}
        if len(curr_predictions) >= 1:
            assert isinstance(curr_predictions, dict), \
                'relik predictions should be in a dictionary'

            for curr_relik_predicted_triple in curr_predictions['predicted_triples']:
                if curr_model_name == 'relik-oie':
                    extracted_rel = tuple(curr_relik_predicted_triple['extracted_relation'])
                    for task in ['x-triples', 'e-triples', 'ee-triples', 'ee-kg-triples']:
                        predicted_triples_open_ie[curr_model_name][task].add(extracted_rel)
                elif (curr_model_name == 'relik-cie'
                      and curr_relik_predicted_triple['triple_qids'][0].startswith('Q')
                      and curr_relik_predicted_triple['triple_qids'][1].startswith('P')
                      and curr_relik_predicted_triple['triple_qids'][2].startswith('Q')):
                    triple_qids = tuple(curr_relik_predicted_triple['triple_qids'])
                    labels = curr_relik_predicted_triple['triple_labels']
                    if (use_triple_labels_as_surface_forms
                            and labels is not None
                            and all(l is not None and l != '--NME--' for l in labels)):
                        extracted_rel = tuple(labels)
                    else:
                        extracted_rel = tuple(curr_relik_predicted_triple['extracted_relation'])

                    if triple_qids in snapshot_to_triples[str_snapshot_year]:
                        logger.debug(f'adding_x_triples {curr_relik_predicted_triple["triple_qids"]}')
                        predicted_triples_open_ie[curr_model_name]['x-triples'].add(extracted_rel)
                        predicted_triples_cie[curr_model_name]['x-triples'].add(triple_qids)
                    else:
                        logger.debug(f'adding_e_triples {curr_relik_predicted_triple["triple_qids"]}')
                        predicted_triples_cie[curr_model_name]['e-triples'].add(triple_qids)
                        predicted_triples_open_ie[curr_model_name]['e-triples'].add(extracted_rel)

                    if triple_qids not in pred_triple_qid_to_labels:
                        pred_triple_qid_to_labels[triple_qids] = \
                            tuple(curr_relik_predicted_triple['triple_labels'])

    # kg_prompt DECLARES the TKGU operation per triple (the `operation` tag) -> route by
    # what the model SAYS (no snapshot/KG lookup needed; relik needs the snapshot only
    # because it can't tell EXISTS from ADD). This also holds the model accountable for
    # its own operation classification.
    KG_PROMPT_OP_TO_TASK = {
        'EXISTS': 'x-triples',
        'ADD': 'e-triples',
        'MINT_ADD': 'ee-triples',
        'MINT+ADD': 'ee-triples',
        'INFER': 'ee-kg-triples',
        'DEPRECATE': 'd-triples',
    }

    @staticmethod
    def process_kg_prompt_predictions(
            curr_model_name: str,
            curr_predictions: Dict,
            predicted_triples_open_ie: Dict,
            predicted_triples_cie: Dict,
            pred_triple_qid_to_labels: Dict
    ) -> None:
        """Process kg_prompt (KG-conditioned baseline) predictions.

        Routes each triple by the model's declared `operation` tag (EXISTS/ADD/MINT_ADD/
        INFER/DEPRECATE -> x/e/ee/ee-kg/d-triples) -- no KG verification, unlike relik.
        - open_ie (C / G-BERTScore-R): surface form from `triple_labels` (head, rel, tail);
          head/tail normalized, relation kept verbatim (mirrors process_llm_tool).
        - cie (Executable-R): the QID triple, ONLY when all three are valid Q/P/Q. Minted
          entities are None -> excluded, so Mint+Add / Infer are E-R=0 by construction.
        Requires the per-triple `operation` tag (present since the 2026-07-26 parser fix).
        """
        if len(curr_predictions) < 1:
            return
        assert isinstance(curr_predictions, dict), \
            f'{curr_model_name} predictions should be in a dictionary'

        for pred in curr_predictions['predicted_triples']:
            op = pred.get('operation')
            task = (EvaluationPreLoader.KG_PROMPT_OP_TO_TASK.get(str(op).upper())
                    if op is not None else None)
            if task is None:
                logger.warning(
                    f'{curr_model_name}: triple with missing/unroutable operation={op!r} '
                    f'skipped (re-run PREDICT with the operation-tag parser?): {pred}')
                continue

            labels = pred.get('triple_labels') or [None, None, None]
            # Build the semantic surface (C / G-BERTScore-R) from LABELS ONLY (no QIDs),
            # normalizing ALL THREE components exactly like process_edc_plus_predictions so
            # the surface convention matches the EDC+ baseline we compare against (no-op for
            # our KG-sourced labels, which have no underscores; safe if any label ever does).
            surface = (
                normalize_triple_string(labels[0]),
                normalize_triple_string(labels[1]),
                normalize_triple_string(labels[2])
            )
            predicted_triples_open_ie[curr_model_name][task].add(surface)

            qids = pred.get('triple_qids') or [None, None, None]
            if (isinstance(qids[0], str) and qids[0].startswith('Q')
                    and isinstance(qids[1], str) and qids[1].startswith('P')
                    and isinstance(qids[2], str) and qids[2].startswith('Q')):
                triple_qids = tuple(qids)
                predicted_triples_cie[curr_model_name][task].add(triple_qids)
                if triple_qids not in pred_triple_qid_to_labels:
                    pred_triple_qid_to_labels[triple_qids] = tuple(labels)
