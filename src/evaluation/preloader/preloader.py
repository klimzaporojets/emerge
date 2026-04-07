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
