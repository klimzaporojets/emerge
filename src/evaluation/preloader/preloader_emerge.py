import logging
from typing import List, Any, Dict

import pandas as pd
from tqdm import tqdm

from dataset.emerge.utils.constants import TKGU_TASKS, ACTION_CATEGORY_ASSERT, ACTION_CATEGORY_DEPRECATE
from dataset.emerge.utils.utils import get_llm_assessment
from datetime import datetime

from evaluation.misc.args_evaluation import ArgsEvaluation
from evaluation.misc.wiki_eval_result import WikiEvalResult
from evaluation.preloader.preloader import EvaluationPreLoader

logger = logging.getLogger(__name__)


class EMERGEPreLoader(EvaluationPreLoader):
    """Loads and preprocesses the EMERGE dataset for evaluation, extracting predictions and ground truth."""

    def __init__(self,
                 config: ArgsEvaluation,
                 loaded_dataset: List[Dict[str, Any]],
                 snapshot_to_triples
                 ):
        super().__init__()
        self.config = config
        self.arg_models_to_evaluate = self.config.models_to_evaluate
        self.arg_llm_assessors = self.config.llm_assessors
        self.arg_should_add_predictions = len(self.arg_models_to_evaluate) > 0
        self.score_empty_predictions_as_zero = getattr(self.config, 'score_empty_predictions_as_zero', True)
        # self.arg_llm_names_to_load = None
        self.loaded_dataset = loaded_dataset
        self.snapshot_to_triples = snapshot_to_triples

    def preload(self, existing: WikiEvalResult | None) -> WikiEvalResult:
        """Extract predictions and ground truth from the dataset, merging with existing cached results."""
        models_to_load = self.arg_models_to_evaluate
        models_to_load_set = set(models_to_load)

        llm_assessment_field_names: List[str] = [
            f'{llm_name}_{action}'
            for action, llm_name in self.arg_llm_assessors.items()
        ]
        #
        # line_records = list()
        tot_nr_read_lines = 0

        #
        def _empty_pred_container():
            return {task: set() for task in TKGU_TASKS}

        batch_completeness_openie = {
            curr_model: {k: list() for k in ['x-triples', 'e-triples', 'ee-triples', 'ee-kg-triples', 'd-triples']}
            for curr_model in models_to_load
        }

        batch_cie_qid_triples = {
            curr_model: {k: list() for k in TKGU_TASKS}
            for curr_model in models_to_load
        }

        batch_completeness_gt = {task: [] for task in TKGU_TASKS}

        batch_factualness_openie = list()

        tot_nr_tkgu_triples = 0
        processed_field_hash_ids = set()

        df_cie_triples_lst = list()
        df_instances_lst = list()

        df_field_predictions_open_ie_lst = list()

        tkgu_task_to_prompt_name = {
            task: (
                ACTION_CATEGORY_DEPRECATE
                if task == 'd-triples'
                else ACTION_CATEGORY_ASSERT
            )
            for task in TKGU_TASKS
        }

        batch_factualness_gt = list()

        ####### BEGIN ADDING CACHING

        seen_hash_ids = set()

        # ------------------------------------------------------------------
        # INCREMENTAL PRELOAD CHANGE:
        #   guard against existing is None (you had this commented out)
        # ------------------------------------------------------------------
        if existing is not None and existing.df_instances is not None:
            # ------------------------------------------------------------------
            # authoritative source: df_instances (1 row per hash_id)
            seen_hash_ids = set(existing.df_instances['hash_id'].unique())

            # ---- consistency assert (debug / safety) ----
            bc_hash_ids = set()
            for model in existing.batch_completeness_openie:
                for task in existing.batch_completeness_openie[model]:
                    bc_hash_ids |= {x[0] for x in existing.batch_completeness_openie[model][task]}

            cie_hash_ids = set(existing.df_predictions_cie_and_gt['hash_id'].unique()) \
                if not existing.df_predictions_cie_and_gt.empty else set()

            openie_hash_ids = set(existing.df_predictions_open_ie['hash_id'].unique()) \
                if not existing.df_predictions_open_ie.empty else set()

            assert seen_hash_ids == cie_hash_ids == openie_hash_ids, (
                'Preload invariant violated: hash_id partially present across containers'
            )
            assert bc_hash_ids <= seen_hash_ids <= cie_hash_ids <= openie_hash_ids, (
                'Preload invariant violated: bc_hash_ids <= seen_hash_ids <= cie_hash_ids <= openie_hash_ids'
            )
        ####### END ADDING CACHING

        #
        for parsed_line in tqdm(self.loaded_dataset,
                                desc='evaluation preloading EMERGE',
                                total=len(self.loaded_dataset)):
            field_hash_id = parsed_line['hash_id']
            if field_hash_id in seen_hash_ids:
                continue
            field_passage = parsed_line['passage']
            field_passage_timestamp = parsed_line['passage_timestamp']
            assert parsed_line['delta_timestamps'][0] < parsed_line['delta_timestamps'][1]
            field_delta_start_date = parsed_line['delta_dates'][0]
            # any predicted triple open_ie, e.g., relik re or edc non-canicolized
            predicted_triples_open_ie = {curr_model: _empty_pred_container() for curr_model in models_to_load}
            predicted_triples_cie = {curr_model: _empty_pred_container() for curr_model in models_to_load}

            qid_to_mention = dict()
            qid_to_mention_text = dict()
            for curr_mention in parsed_line['mentions']:
                if curr_mention['qid'] not in qid_to_mention:
                    qid_to_mention[curr_mention['qid']] = \
                        f'{curr_mention["mention_text"]}({curr_mention["target_entity"]})'
                    qid_to_mention_text[curr_mention['qid']] = f'{curr_mention["mention_text"]}'

            field_interval_start = parsed_line['delta_timestamps'][0]
            field_interval_end = parsed_line['delta_timestamps'][1]

            delta_dates = parsed_line['delta_dates']

            assert len(delta_dates) == 2
            d1 = datetime.fromisoformat(delta_dates[0])
            d2 = datetime.fromisoformat(delta_dates[1])
            snapshot_year = d1.year
            str_snapshot_year = str(snapshot_year)

            field_tot_nr_mentions = len(parsed_line['mentions'])
            field_tot_nr_entities_in_mentions = len(set([cm['qid'] for cm in parsed_line['mentions']]))

            if 'passage_timestamp' not in parsed_line:
                parsed_line['passage_timestamp'] = parsed_line.pop('revision_timestamp')

            if 'passage_revision_id' not in parsed_line:
                parsed_line['passage_revision_id'] = parsed_line.pop('revision_id')

            field_revision_timestamp = parsed_line['passage_timestamp']
            field_revision_id = parsed_line['passage_revision_id']
            field_anchor_page_qid = parsed_line['anchor_page_qid']

            tkgu_triples = parsed_line['tkgu_triples']
            tot_nr_tkgu_triples += len(tkgu_triples)

            ground_truth_task_to_triple_ids = {curr_task: set() for curr_task in TKGU_TASKS}
            ground_truth_task_to_triple_ids_lst = {curr_task: list() for curr_task in TKGU_TASKS}
            ground_truth_task_to_triple_labels_lst = {curr_task: list() for curr_task in TKGU_TASKS}
            ground_truth_task_to_triple_ent_types_lst = {curr_task: list() for curr_task in TKGU_TASKS}
            ground_truth_task_to_triple_assessments = {
                curr_task: {
                    f'{self.arg_llm_assessors[curr_llm_prompt_name]}_{curr_llm_prompt_name}': list()
                    for curr_llm_prompt_name in [ACTION_CATEGORY_ASSERT, ACTION_CATEGORY_DEPRECATE]
                }
                for curr_task in TKGU_TASKS
            }
            #
            gt_per_prompt_type = {
                'field_hash_id': field_hash_id,
                'field_passage_timestamp': field_passage_timestamp,
                'field_delta_start_date': field_delta_start_date,
                'passage': field_passage,
                ACTION_CATEGORY_ASSERT: set(),
                ACTION_CATEGORY_DEPRECATE: set()
            }

            for curr_ground_truth_triple in tkgu_triples:
                triple_operations = set(curr_ground_truth_triple['tkgu_operations'])
                for curr_task in TKGU_TASKS:
                    if curr_task in triple_operations:
                        llm_prompt_name = ACTION_CATEGORY_ASSERT
                        if curr_task.lower() == 'd-triples':
                            llm_prompt_name = ACTION_CATEGORY_DEPRECATE
                        gt_per_prompt_type[llm_prompt_name]. \
                            add(tuple(curr_ground_truth_triple['triple_labels'] +
                                      curr_ground_truth_triple['triple']))
                        llm_fp_assessor_name = self.arg_llm_assessors[llm_prompt_name]
                        llm_fp_complete_name = f'{llm_fp_assessor_name}_{llm_prompt_name}'
                        ground_truth_task_to_triple_ids[curr_task].add(
                            tuple(curr_ground_truth_triple['triple'])
                        )
                        #
                        ground_truth_task_to_triple_ids_lst[curr_task].append(
                            tuple(curr_ground_truth_triple['triple'])
                        )
                        ground_truth_task_to_triple_labels_lst[curr_task].append(
                            tuple(curr_ground_truth_triple['triple_labels'])
                        )
                        ground_truth_task_to_triple_ent_types_lst[curr_task].append(
                            tuple((curr_ground_truth_triple['emerging_head'],
                                   curr_ground_truth_triple['emerging_tail']))
                        )
                        curr_llm_fp_assessment = get_llm_assessment(triple=curr_ground_truth_triple,
                                                                    llm_assessor_name=llm_fp_assessor_name,
                                                                    llm_prompt_type=llm_prompt_name,
                                                                    hash_id=field_hash_id)

                        (ground_truth_task_to_triple_assessments[curr_task][llm_fp_complete_name]
                         .append(curr_llm_fp_assessment))

            for curr_prediction_tkgu_type in TKGU_TASKS:
                curr_prompt_type = ACTION_CATEGORY_ASSERT
                if curr_prediction_tkgu_type.lower() == 'd-triples':
                    curr_prompt_type = ACTION_CATEGORY_DEPRECATE
                llm_assessor = self.arg_llm_assessors[curr_prompt_type]
                assessment_name = f'{llm_assessor}_{curr_prompt_type}'

                if len(ground_truth_task_to_triple_ids.get(curr_prediction_tkgu_type, [])) > 0:
                    batch_completeness_gt[curr_prediction_tkgu_type]. \
                        append([
                        field_hash_id,
                        ground_truth_task_to_triple_ids_lst[curr_prediction_tkgu_type],
                        ground_truth_task_to_triple_labels_lst[curr_prediction_tkgu_type],
                        None,
                        ground_truth_task_to_triple_assessments[curr_prediction_tkgu_type][assessment_name],
                        ground_truth_task_to_triple_ent_types_lst[curr_prediction_tkgu_type],
                    ])
            #
            field_assessed_by = {curr_model: False for curr_model in models_to_load}
            batch_factualness_gt.append(gt_per_prompt_type)
            model_name_to_type = dict()
            pred_triple_qid_to_labels = dict()
            delta_weeks = round((d2 - d1).days / 7)

            logger.debug(f'interval_start_is: {field_interval_start} '
                         f'and interval_end_is: {field_interval_end}')
            #
            fields_instance = {
                'passage': field_passage,
                'hash_id': field_hash_id,
                'tot_nr_read_lines': tot_nr_read_lines,
                'anchor_page_qid': field_anchor_page_qid,
                'delta_weeks': delta_weeks,
                'snapshot_year': snapshot_year,
                'interval_start': field_interval_start,
                'interval_end': field_interval_end,
                'revision_id': field_revision_id,
                'revision_timestamp': field_revision_timestamp,
                'tot_nr_mentions': field_tot_nr_mentions,
                'tot_nr_entities_in_mentions': field_tot_nr_entities_in_mentions
            }
            df_instances_lst.append(fields_instance)
            #
            if self.arg_should_add_predictions:
                curr_batch_factualness_openai = dict()
                curr_batch_factualness_openai['field_hash_id'] = field_hash_id
                curr_batch_factualness_openai['field_passage_timestamp'] = field_passage_timestamp
                curr_batch_factualness_openai['field_delta_start_date'] = field_delta_start_date
                curr_batch_factualness_openai[ACTION_CATEGORY_ASSERT] = set()
                curr_batch_factualness_openai[ACTION_CATEGORY_DEPRECATE] = set()
                curr_batch_factualness_openai['passage'] = field_passage
                for curr_model_name, curr_predictions in parsed_line['predictions'].items():
                    if curr_model_name not in models_to_load_set:
                        continue
                    field_assessed_by[curr_model_name] = True
                    model_type = curr_predictions['model_type']
                    model_name_to_type[curr_model_name] = model_type
                    logger.debug(f'model_type_is: {model_type}')
                    if model_type == 'relik':
                        self.process_relik_predictions(
                            curr_model_name=curr_model_name,
                            curr_predictions=curr_predictions,
                            predicted_triples_open_ie=predicted_triples_open_ie,
                            predicted_triples_cie=predicted_triples_cie,
                            pred_triple_qid_to_labels=pred_triple_qid_to_labels,
                            snapshot_to_triples=self.snapshot_to_triples,
                            str_snapshot_year=str_snapshot_year,
                            use_triple_labels_as_surface_forms=self.config.use_triple_labels_as_surface_forms
                        )
                    elif model_type == 'edc':
                        self.process_edc_plus_predictions(
                            curr_model_name=curr_model_name,
                            curr_predictions=curr_predictions,
                            predicted_triples_open_ie=predicted_triples_open_ie
                        )
                    elif model_type == 'edc-original':
                        self.process_edc_original_predictions(
                            curr_model_name=curr_model_name,
                            curr_predictions=curr_predictions,
                            predicted_triples_open_ie=predicted_triples_open_ie
                        )
                    elif model_type == 'llm-tool':
                        self.process_llm_tool_predictions(
                            curr_model_name=curr_model_name,
                            curr_predictions=curr_predictions,
                            predicted_triples_open_ie=predicted_triples_open_ie,
                            predicted_triples_cie=predicted_triples_cie,
                            pred_triple_qid_to_labels=pred_triple_qid_to_labels
                        )
                    elif model_type in {'kg-gen', 'rebel', 'rakg'}:
                        self.process_other_oie_predictions(
                            curr_model_name=curr_model_name,
                            curr_predictions=curr_predictions,
                            predicted_triples_open_ie=predicted_triples_open_ie
                        )
                    else:
                        raise RuntimeError(f'model_type_not_recognized: {model_type}')

                    curr_model_per_prompt_predictions = {
                        ACTION_CATEGORY_ASSERT: set(),
                        ACTION_CATEGORY_DEPRECATE: set()
                    }
                    for curr_prediction_tkgu_type in TKGU_TASKS:
                        curr_prompt_type = ACTION_CATEGORY_ASSERT
                        if curr_prediction_tkgu_type.lower() == 'd-triples':
                            curr_prompt_type = ACTION_CATEGORY_DEPRECATE
                        llm_assessor = self.arg_llm_assessors[curr_prompt_type]
                        assessment_name = f'{llm_assessor}_{curr_prompt_type}'

                        has_open_ie_preds = len(
                            predicted_triples_open_ie[curr_model_name][curr_prediction_tkgu_type]) > 0
                        has_tkgu_gt_triples = len(
                            ground_truth_task_to_triple_ids_lst[curr_prediction_tkgu_type]) > 0

                        # Always add individual OIE predictions to the predictions DataFrame
                        if has_open_ie_preds:
                            assert len(ground_truth_task_to_triple_labels_lst[curr_prediction_tkgu_type]) == \
                                   len(ground_truth_task_to_triple_assessments[curr_prediction_tkgu_type][
                                           assessment_name])

                            for curr_pred_triple in predicted_triples_open_ie[curr_model_name][
                                curr_prediction_tkgu_type]:
                                assert len(curr_pred_triple) == 3
                                pred_field_to_add = {
                                    'hash_id': fields_instance['hash_id'],
                                    'prompt_type': curr_prompt_type,
                                    'tkgu_type': curr_prediction_tkgu_type,
                                    'model': curr_model_name,
                                    'triple_head_label': curr_pred_triple[0],
                                    'triple_relation_label': curr_pred_triple[1],
                                    'triple_tail_label': curr_pred_triple[2]
                                }
                                df_field_predictions_open_ie_lst.append(pred_field_to_add)

                        # Decide whether to create batch entries for scoring
                        # score_empty_predictions_as_zero=True: create entry whenever GT exists
                        #   (empty predictions → score=0, correctly included in average)
                        # score_empty_predictions_as_zero=False: legacy behavior, only when predictions exist
                        should_add_to_batch = has_tkgu_gt_triples and (
                            has_open_ie_preds or self.score_empty_predictions_as_zero
                        )

                        if should_add_to_batch:
                            batch_completeness_openie[curr_model_name][curr_prediction_tkgu_type]. \
                                append([
                                field_hash_id,
                                ground_truth_task_to_triple_ids_lst[curr_prediction_tkgu_type],
                                ground_truth_task_to_triple_labels_lst[curr_prediction_tkgu_type],
                                list(predicted_triples_open_ie[curr_model_name][curr_prediction_tkgu_type]),
                                ground_truth_task_to_triple_assessments[curr_prediction_tkgu_type][assessment_name],
                                ground_truth_task_to_triple_ent_types_lst[curr_prediction_tkgu_type],
                            ])

                            # QID exact-match: store predicted QID triples alongside GT QIDs
                            # (empty pred → P=0, R=0, F1=0, correctly included in average)
                            batch_cie_qid_triples[curr_model_name][curr_prediction_tkgu_type].append([
                                field_hash_id,
                                ground_truth_task_to_triple_ids_lst[curr_prediction_tkgu_type],
                                list(predicted_triples_cie[curr_model_name][curr_prediction_tkgu_type]),
                                ground_truth_task_to_triple_assessments[curr_prediction_tkgu_type][assessment_name],
                            ])

                        # Track for factualness (only actual predictions)
                        if has_open_ie_preds:
                            curr_model_per_prompt_predictions[curr_prompt_type] \
                                .update(predicted_triples_open_ie[curr_model_name]
                                        [curr_prediction_tkgu_type])

                    for curr_prompt_type in [ACTION_CATEGORY_ASSERT, ACTION_CATEGORY_DEPRECATE]:
                        curr_batch_factualness_openai[curr_prompt_type].update(
                            curr_model_per_prompt_predictions[curr_prompt_type])
                batch_factualness_openie.append(curr_batch_factualness_openai)
            assert field_hash_id not in processed_field_hash_ids
            processed_field_hash_ids.add(field_hash_id)

            for idx_tkgu_triple, curr_tkgu_triple in enumerate(tkgu_triples):
                set_tkgu_operations = set(curr_tkgu_triple['tkgu_operations'])

                field_triple_source_delta_type = None
                if 'source_delta_type' in curr_tkgu_triple:
                    field_triple_source_delta_type = curr_tkgu_triple['source_delta_type']

                # TODO: this assert should be somewhere else before in the pipeline when tkgu_operations is being set
                assert len(curr_tkgu_triple['tkgu_operations']) <= 2
                if len(curr_tkgu_triple['tkgu_operations']) == 2:
                    assert 'd-triples' in set_tkgu_operations

                field_triple_head_id = curr_tkgu_triple['triple'][0]
                field_triple_relation_id = curr_tkgu_triple['triple'][1]
                field_triple_tail_id = curr_tkgu_triple['triple'][2]
                field_relation_definition = ''
                field_triple_head_label = curr_tkgu_triple['triple_labels'][0]
                field_triple_relation_label = curr_tkgu_triple['triple_labels'][1]
                field_triple_tail_label = curr_tkgu_triple['triple_labels'][2]

                field_triple_emerging_head = curr_tkgu_triple['emerging_head']
                field_triple_emerging_tail = curr_tkgu_triple['emerging_tail']
                field_triple_timestamp_from = curr_tkgu_triple['triple_lifespan_timestamp'][0]
                field_triple_timestamp_to = curr_tkgu_triple['triple_lifespan_timestamp'][1]
                field_head_creation_timestamp = curr_tkgu_triple['head_creation_timestamp']
                field_tail_creation_timestamp = curr_tkgu_triple['tail_creation_timestamp']

                field_qualifier_timestamp = None
                field_qualifier_qid = None
                field_qualifier_label = None

                if len(curr_tkgu_triple['qualifier_info']) > 0:
                    field_qualifier_timestamp = curr_tkgu_triple['qualifier_info']['qualifier_timestamp']
                    field_qualifier_qid = curr_tkgu_triple['qualifier_info']['qualifier_qid']
                    field_qualifier_label = curr_tkgu_triple['qualifier_info']['qualifier_label']

                field_assessment_to_res = {field_name: None for field_name in llm_assessment_field_names}

                for curr_llm_assessment in curr_tkgu_triple['llm_assessment']:
                    curr_llm_name = curr_llm_assessment['llm_name']
                    field_assessment_to_res[
                        f'{curr_llm_name}_{curr_llm_assessment["llm_prompt_type"]}'
                    ] = curr_llm_assessment['llm_assessment']
                #
                for curr_tkgu_operation in set_tkgu_operations:
                    curr_tkgu_prompt = tkgu_task_to_prompt_name[curr_tkgu_operation]
                    # for curr_llm_assessor in self.arg_llm_assessors[curr_tkgu_operation]:
                    curr_llm_assessor = self.arg_llm_assessors[curr_tkgu_prompt]
                    key_assessor = f'{curr_llm_assessor}_{curr_tkgu_prompt}'

                    curr_llm_assessor_result = None
                    if key_assessor in field_assessment_to_res:
                        curr_llm_assessor_result = field_assessment_to_res[key_assessor]
                    #
                    curr_prompt_type = ACTION_CATEGORY_ASSERT
                    if curr_tkgu_operation == 'd-triples':
                        curr_prompt_type = ACTION_CATEGORY_DEPRECATE
                    df_cie_triples_lst.append(
                        {
                            'hash_id': field_hash_id,
                            'triple_source': field_triple_source_delta_type,
                            'triple_type': 'in-dataset',
                            'model': None,
                            'triple_emerging_head': field_triple_emerging_head,
                            'triple_emerging_tail': field_triple_emerging_tail,
                            'head_creation_timestamp': field_head_creation_timestamp,
                            'tail_creation_timestamp': field_tail_creation_timestamp,
                            'triple_timestamp_from': field_triple_timestamp_from,
                            'triple_timestamp_to': field_triple_timestamp_to,
                            'triple_head': field_triple_head_id,
                            'triple_relation': field_triple_relation_id,
                            'triple_tail': field_triple_tail_id,
                            'triple_head_label': field_triple_head_label,
                            'triple_relation_label': field_triple_relation_label,
                            'triple_tail_label': field_triple_tail_label,
                            'tkgu_type': curr_tkgu_operation,
                            'prompt_type': curr_prompt_type,
                            'qualifier_timestamp': field_qualifier_timestamp,
                            'qualifier_id': field_qualifier_qid,
                            'qualifier_label': field_qualifier_label,
                            # 'llm_assessor': key_assessor,
                            'llm_assessor': curr_llm_assessor,
                            'llm_assessor_result': curr_llm_assessor_result,
                            'relation_definition': field_relation_definition
                        }
                    )

            if self.arg_should_add_predictions:
                for model_name, task_dict in predicted_triples_cie.items():
                    for task, triples in task_dict.items():
                        if task in TKGU_TASKS:
                            for triple in triples:
                                triple_label = [None, None, None]
                                if triple in pred_triple_qid_to_labels:
                                    triple_label = pred_triple_qid_to_labels[triple]

                                df_cie_triples_lst.append(
                                    {
                                        'hash_id': field_hash_id,
                                        'triple_source': None,
                                        'triple_type': 'predicted',
                                        'model': model_name,
                                        'triple_emerging_head': None,
                                        'triple_emerging_tail': None,
                                        'head_creation_timestamp': None,
                                        'tail_creation_timestamp': None,
                                        'triple_timestamp_from': None,
                                        'triple_timestamp_to': None,
                                        'triple_head': triple[0],
                                        'triple_relation': triple[1],
                                        'triple_tail': triple[2],
                                        'triple_head_label': triple_label[0],
                                        'triple_relation_label': triple_label[1],
                                        'triple_tail_label': triple_label[2],
                                        'tkgu_type': task,
                                        'qualifier_timestamp': None,
                                        'qualifier_id': None,
                                        'qualifier_label': None,
                                        'llm_assessor': None,
                                        'llm_assessor_result': None,
                                        'relation_definition': ''  # TODO - complete this as well,
                                    }
                                )

            tot_nr_read_lines += 1

            all_triples = parsed_line['tkgu_triples']
            triple_qids = set([tuple(ct['triple']) for ct in all_triples])
            assert len(triple_qids) == len(all_triples)


        logger.info(
            f'tot_nr_read_lines: {tot_nr_read_lines}, '
            f'len(df_cie_triples_lst): {len(df_cie_triples_lst)} , '
            f'tot_nr_tkgu_triples: {tot_nr_tkgu_triples}'
        )
        # ------------------------------------------------------------
        # BEGIN CACHE GUARD:
        # If everything was skipped due to caching, avoid building empty
        # DFs and touching missing columns. Just return existing as-is.
        # ------------------------------------------------------------
        if tot_nr_read_lines == 0:
            logger.info(
                "Preload: 0 new lines (all hash_ids already cached) -> returning existing unchanged."
            )
            if existing is not None:
                return existing

            # If existing is None, return a well-formed empty result
            empty = WikiEvalResult()
            empty.batch_completeness_openie = batch_completeness_openie
            empty.batch_cie_qid_triples = batch_cie_qid_triples
            empty.batch_completeness_gt = batch_completeness_gt
            empty.batch_factualness_openie = batch_factualness_openie
            empty.batch_factualness_gt = batch_factualness_gt

            # Provide empty DFs with expected columns so downstream code won't KeyError
            empty.df_instances = pd.DataFrame(columns=[
                'passage', 'hash_id', 'tot_nr_read_lines', 'anchor_page_qid', 'delta_weeks',
                'snapshot_year', 'interval_start', 'interval_end', 'revision_id',
                'revision_timestamp', 'tot_nr_mentions', 'tot_nr_entities_in_mentions',
                'interval_delta'
            ])
            empty.df_predictions_cie_and_gt = pd.DataFrame(columns=[
                'hash_id', 'triple_source', 'triple_type', 'model', 'triple_emerging_head',
                'triple_emerging_tail', 'head_creation_timestamp', 'tail_creation_timestamp',
                'triple_timestamp_from', 'triple_timestamp_to', 'triple_head', 'triple_relation',
                'triple_tail', 'triple_head_label', 'triple_relation_label', 'triple_tail_label',
                'tkgu_type', 'prompt_type', 'qualifier_timestamp', 'qualifier_id',
                'qualifier_label', 'llm_assessor', 'llm_assessor_result',
                'relation_definition'
            ])
            empty.df_predictions_open_ie = pd.DataFrame(columns=[
                'hash_id', 'prompt_type', 'tkgu_type', 'model',
                'triple_head_label', 'triple_relation_label', 'triple_tail_label'
            ])
            return empty
        # ------------------------------------------------------------
        # END CACHE GUARD
        # ------------------------------------------------------------

        df_instances = pd.DataFrame(df_instances_lst)
        df_wiki_predictions_cie_and_gt = pd.DataFrame(df_cie_triples_lst)
        logger.debug(f'finished building df of the following shape: {df_wiki_predictions_cie_and_gt.shape}')
        df_instances['revision_timestamp'] = pd.to_datetime(
            df_instances['revision_timestamp'], unit='s')

        df_wiki_predictions_cie_and_gt['head_creation_timestamp'] = pd.to_datetime(
            df_wiki_predictions_cie_and_gt['head_creation_timestamp'],
            unit='s')
        df_wiki_predictions_cie_and_gt['tail_creation_timestamp'] = pd.to_datetime(
            df_wiki_predictions_cie_and_gt['tail_creation_timestamp'],
            unit='s')
        df_wiki_predictions_cie_and_gt['triple_timestamp_from'] = pd.to_datetime(
            df_wiki_predictions_cie_and_gt['triple_timestamp_from'], unit='s')

        df_wiki_predictions_cie_and_gt['triple_timestamp_to'] = pd.to_datetime(
            df_wiki_predictions_cie_and_gt['triple_timestamp_to'], unit='s')

        df_instances['interval_start'] = pd.to_datetime(
            df_instances['interval_start'], unit='s').dt.ceil('D')
        df_instances['interval_end'] = pd.to_datetime(df_instances['interval_end'],
                                                      unit='s').dt.ceil('D')

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        df_instances['interval_delta'] = (
                df_instances['interval_end'] - df_instances['interval_start']
        ).dt.days

        logger.debug('df_wiki_metrics_cie_pred_and_gt calculated, now calculating batched completeness')

        df_wiki_predictions_open_ie = pd.DataFrame(df_field_predictions_open_ie_lst)

        wiki_eval_results = WikiEvalResult()
        wiki_eval_results.batch_completeness_openie = batch_completeness_openie
        wiki_eval_results.batch_cie_qid_triples = batch_cie_qid_triples
        wiki_eval_results.batch_completeness_gt = batch_completeness_gt
        wiki_eval_results.batch_factualness_openie = batch_factualness_openie
        wiki_eval_results.batch_factualness_gt = batch_factualness_gt

        wiki_eval_results.df_predictions_cie_and_gt = df_wiki_predictions_cie_and_gt
        wiki_eval_results.df_predictions_open_ie = df_wiki_predictions_open_ie
        wiki_eval_results.df_instances = df_instances

        # ------------------------------------------------------------------
        # INCREMENTAL PRELOAD CHANGE:
        #   merge NEW results into existing instead of overwriting
        #   (only if existing is present; otherwise return new)
        # ------------------------------------------------------------------
        if existing is None or existing.df_instances is None:
            ###### BEGIN - just checking consistency that all hash_ids are in all the structures
            # authoritative source: df_instances (1 row per hash_id)
            seen_hash_ids = set(wiki_eval_results.df_instances['hash_id'].unique())

            # ---- consistency assert (debug / safety) ----
            bc_hash_ids = set()
            for model in wiki_eval_results.batch_completeness_openie:
                for task in wiki_eval_results.batch_completeness_openie[model]:
                    bc_hash_ids |= {x[0] for x in wiki_eval_results.batch_completeness_openie[model][task]}

            cie_hash_ids = set(wiki_eval_results.df_predictions_cie_and_gt['hash_id'].unique()) \
                if not wiki_eval_results.df_predictions_cie_and_gt.empty else set()

            openie_hash_ids = set(wiki_eval_results.df_predictions_open_ie['hash_id'].unique()) \
                if not wiki_eval_results.df_predictions_open_ie.empty else set()
            assert seen_hash_ids == cie_hash_ids == openie_hash_ids, (
                'Preload invariant violated: hash_id partially present across containers'
            )
            assert bc_hash_ids <= seen_hash_ids <= cie_hash_ids <= openie_hash_ids, (
                'Preload invariant violated: bc_hash_ids <= seen_hash_ids <= cie_hash_ids <= openie_hash_ids'
            )
            ###### END - just checking consistency that all hash_ids are in all the structures
            return wiki_eval_results

        # concatenate dataframes
        existing.df_instances = pd.concat([existing.df_instances, wiki_eval_results.df_instances],
                                          ignore_index=True)
        existing.df_predictions_cie_and_gt = pd.concat(
            [existing.df_predictions_cie_and_gt, wiki_eval_results.df_predictions_cie_and_gt],
            ignore_index=True
        )
        existing.df_predictions_open_ie = pd.concat(
            [existing.df_predictions_open_ie, wiki_eval_results.df_predictions_open_ie],
            ignore_index=True
        )

        # extend batch lists
        existing.batch_factualness_gt.extend(wiki_eval_results.batch_factualness_gt)
        existing.batch_factualness_openie.extend(wiki_eval_results.batch_factualness_openie)

        # merge dict-of-lists (gt)
        for task, rows in wiki_eval_results.batch_completeness_gt.items():
            existing.batch_completeness_gt.setdefault(task, []).extend(rows)

        # merge dict-of-dict-of-lists (openie)
        for model, task_map in wiki_eval_results.batch_completeness_openie.items():
            existing.batch_completeness_openie.setdefault(model, {})
            for task, rows in task_map.items():
                existing.batch_completeness_openie[model].setdefault(task, []).extend(rows)

        # merge dict-of-dict-of-lists (cie qid triples)
        if wiki_eval_results.batch_cie_qid_triples is not None:
            if existing.batch_cie_qid_triples is None:
                existing.batch_cie_qid_triples = {}
            for model, task_map in wiki_eval_results.batch_cie_qid_triples.items():
                existing.batch_cie_qid_triples.setdefault(model, {})
                for task, rows in task_map.items():
                    existing.batch_cie_qid_triples[model].setdefault(task, []).extend(rows)

        return existing
        # ------------------------------------------------------------------