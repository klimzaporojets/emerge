# 2025.09.01 - the _v4 is adapted to the final dataset format and also a single
# annotation round to be prepared for ICLR submission, there will be probably
# only a few changes wrt _v3 as this latter also works with new dataset format.
# However, here I also plan to add additional features such as showing the
# descriptions of the relations, etc.

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import List, Dict

import os
from dataset.emerge.utils.constants import ACTION_CATEGORY_DEPRECATE, ACTION_CATEGORY_ASSERT

import pandas as pd

from dataset.emerge.utils.s09_annotate_dataset_utils_v4 import obtain_property_ids_to_definitions, \
    get_instance_tkgu_types_llm_assessments, load_all_instances, stats_all_instances, get_annotation_statistics, \
    human_assessment_exists, clear_stdin, get_print_annotation_statistics, \
    show_discrepancies, exceeds_max_per_tkgu_type, update_count_annotated, update_count_annotated_per_triple

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/s09_annotate_dataset/20250901/'
                                's09_annotate_dataset.json',
                        help='The config file that contains all the parameters')
    parser.add_argument('--annotator_name', required=False, type=str,
                        default='',
                        help='The name of the annotator.')
    parser.add_argument('--show_discrepancies', action='store_true',
                        help='Show discrepancies in annotations')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.width', 200)

    args = parser.parse_args()
    config = json.load(open(args.config_file, 'rt'))
    if args.annotator_name != '':
        config['annotator_name'] = args.annotator_name

    config['show_discrepancies'] = args.show_discrepancies

    tkgu_operations_to_annotate = set(config['tkgu_operations_to_annotate'])
    tkgu_operations_to_max_annotations = dict(config['tkgu_operations_to_annotate'])

    nr_annotated_per_tkgu_operation = dict()

    input_dataset_path = config['input_dataset_path']
    output_annotation_path = config['output_annotation_path']
    os.makedirs(output_annotation_path, exist_ok=True)
    #

    annotator_name = config['annotator_name']
    already_annotated_hash_ids = set()
    files_to_annotate = os.listdir(input_dataset_path)

    input_relation_dictionaries = config['input_relation_dictionaries']
    root = Path(input_relation_dictionaries)
    dir_dates = [d.name for d in root.iterdir() if d.is_dir()]
    # date_object = datetime.fromtimestamp(curr_parsed_line['delta_timestamps'][0])
    # Format the date as yyyy-mm-dd
    # formatted_date = date_object.strftime('%Y-%m-%d')
    logger.info('BEGIN loading property ids to definitions')
    date_to_property_id_to_definition: Dict[str, Dict[str, str]] = dict()
    for formatted_date in dir_dates:
        if formatted_date not in date_to_property_id_to_definition:
            dictionary_path = os.path.join(config['input_relation_dictionaries'],
                                           formatted_date,
                                           'documents.jsonl')

            date_to_property_id_to_definition[formatted_date] = \
                obtain_property_ids_to_definitions(dictionary_path)
    logger.info('END loading property ids to definitions')

    logger.info('BEGIN loading already annotated content')
    lst_already_annotated_instances = list()
    continue_next = True
    for filename in files_to_annotate:
        # Check if the file has a .jsonl extension
        in_file_path = os.path.join(output_annotation_path, filename)
        if os.path.exists(in_file_path):
            for curr_instance in open(in_file_path, 'rt', encoding='utf-8'):
                parsed_line = json.loads(curr_instance)
                if config['show_discrepancies'] and continue_next:
                    logger.info('*****************************************************')
                    logger.info('showing discrepancies with new instance')
                    logger.info('*****************************************************')
                    continue_next = show_discrepancies(
                        parsed_line=parsed_line,
                        config=config
                    )
                nr_annotated_per_tkgu_operation = update_count_annotated(
                    p_instance=parsed_line,
                    p_annotator_name=config['annotator_name'],
                    p_nr_annotated_per_tkgu_operation=nr_annotated_per_tkgu_operation
                )
                already_annotated_hash_ids.add(parsed_line['hash_id'])
                lst_already_annotated_instances.append(parsed_line)
    logger.info('END loading already annotated content')
    logger.info(f'nr_annotated_per_tkgu_operation: '
                f'{nr_annotated_per_tkgu_operation}')

    filename_to_outfile = dict()

    filename_to_instances: Dict[str, List] = dict()
    logger.info('BEGIN loading content to annotate')
    # distribution_anno_stats = dict()

    for filename in files_to_annotate:
        # Check if the file has a .jsonl extension
        output_file_path = os.path.join(output_annotation_path, filename)
        output_file = open(output_file_path, 'a', encoding='utf-8')
        filename_to_outfile[filename] = output_file

        file_path = os.path.join(input_dataset_path, filename)

        for curr_instance in open(file_path, 'rt', encoding='utf-8'):
            parsed_line = json.loads(curr_instance)

            if parsed_line['hash_id'] in already_annotated_hash_ids:
                logger.info(f'continuing, hash_id {parsed_line["hash_id"]} '
                            f'already annotated')
                continue
            if filename not in filename_to_instances:
                filename_to_instances[filename] = list()
            filename_to_instances[filename].append(parsed_line)

    logger.info('END loading content to annotate')
    logger.info('BEGIN permuting content to annotate')
    nr_added_instances_per_filename: Dict[str, int] = dict()
    stop_traversing = False
    sorted_keys = sorted(filename_to_instances.keys())
    instances_to_annotate = list()

    while not stop_traversing:
        no_extracted = True
        for curr_filename in sorted_keys:
            if curr_filename not in nr_added_instances_per_filename:
                nr_added_instances_per_filename[curr_filename] = 0
            id_to_extract = nr_added_instances_per_filename[curr_filename]
            if id_to_extract < len(filename_to_instances[curr_filename]):
                instances_to_annotate.append(
                    {
                        'filename': curr_filename,
                        'instance': filename_to_instances[curr_filename][id_to_extract],
                    }
                )
                nr_added_instances_per_filename[curr_filename] += 1
                no_extracted = False
        if no_extracted:
            stop_traversing = True

    logger.info('END permuting content to annotate')

    nr_already_annotated = len(lst_already_annotated_instances)
    logger.info('BEGIN: obtaining stats for annotation')
    all_instances_lst = lst_already_annotated_instances + [i['instance'] for i in instances_to_annotate]
    logger.info('loading all instances')
    all_instances_df = load_all_instances(lst_all_instances=all_instances_lst)
    logger.info('calculating stats on all instances')
    stats_all_instances(
        df_all_instances=all_instances_df
    )
    logger.info('END: obtaining stats for annotation')

    logger.info('BEGIN annotating')

    for idx_instance, curr_inst in enumerate(instances_to_annotate):
        curr_instance = curr_inst['instance']
        curr_passage = curr_instance['passage']
        jump_next_line = False

        # p_nr_annotated_per_tkgu_operation=nr_annotated_per_tkgu_operation,
        # p_max_annotations_per_tkgu_operation=tkgu_operations_to_max_annotations

        all_in_max = True
        for tkgu_op, max_nr in tkgu_operations_to_max_annotations.items():
            if tkgu_op not in nr_annotated_per_tkgu_operation or \
                    nr_annotated_per_tkgu_operation[tkgu_op] < max_nr:
                all_in_max = False
        df_stats = None
        if len(lst_already_annotated_instances) > 0:
            df_stats = get_annotation_statistics(
                annotated_instances=lst_already_annotated_instances,
                annotator_names=[annotator_name]
            )

        if all_in_max:
            logger.info('ALREADY ANNOTATED MAXIMUM NR OF TRIPLES PER OPERATION TYPE, '
                        'FINISHING')
            logger.info(get_print_annotation_statistics(df_statistics=df_stats))
            break
        qid_to_mentions = dict()
        for curr_mention in curr_instance['mentions']:
            if curr_mention['qid'] not in qid_to_mentions:
                qid_to_mentions[curr_mention['qid']] = list()
            qid_to_mentions[curr_mention['qid']].append(curr_mention['mention_text'])


        field_passage_date = curr_instance['revision_date']
        nas = get_instance_tkgu_types_llm_assessments(
            instance=curr_instance,
            config=config
        )
        curr_annotating = False

        for idx_tkgu_triple, curr_tkgu_triple in \
                enumerate(curr_instance['tkgu_triples']):
            jump_next_triple = False
            jump_next_line = False
            curr_triple_labels = curr_tkgu_triple['triple_labels']
            triple_show = []
            curr_triple = curr_tkgu_triple['triple']
            curr_tkgu_operations = set(curr_tkgu_triple['tkgu_operations'])
            #
            if len(curr_tkgu_operations.intersection(tkgu_operations_to_annotate)) == 0:
                logger.info(f'ignoring triple {curr_tkgu_triple["triple_labels"]} '
                            f'as tkgu_operations ({curr_tkgu_operations}) are not in '
                            f'{tkgu_operations_to_annotate}')
                continue
            if curr_triple[0] not in qid_to_mentions:
                triple_show.append(f'{curr_triple_labels[0]} ({curr_triple[0]})')
            else:
                triple_show.append(
                    f'{" / ".join(qid_to_mentions[curr_triple[0]])} ({curr_triple[0]})'
                )
            triple_show.append(f'{curr_triple_labels[1]} ({curr_triple[1]})')
            if curr_triple[2] not in qid_to_mentions:
                triple_show.append(f'{curr_triple_labels[2]} ({curr_triple[2]})')
            else:
                triple_show.append(
                    f'{" / ".join(qid_to_mentions[curr_triple[2]])} ({curr_triple[2]})'
                )

            curr_action_categories = list()

            if len({'ee-triples', 'x-triples', 'ee-kg-triples', 'e-triples'} \
                           .intersection(curr_tkgu_operations)) > 0:
                curr_action_categories.append(ACTION_CATEGORY_ASSERT)
            if 'd-triples' in curr_tkgu_operations:
                curr_action_categories.append(ACTION_CATEGORY_DEPRECATE)

            field_qualifier_info = curr_tkgu_triple['qualifier_info']
            field_triple_lifespan_date = curr_tkgu_triple['triple_lifespan_date']
            if 'human_assessment' not in curr_tkgu_triple:
                curr_tkgu_triple['human_assessment'] = list()

            triple_rel_definition = '<NOT FOUND>'
            date_object = datetime.fromtimestamp(curr_instance['delta_timestamps'][0])

            # Format the date as yyyy-mm-dd
            formatted_date = date_object.strftime('%Y-%m-%d')
            # date_formatted = None
            curr_rel_id = curr_tkgu_triple['triple'][1]
            if curr_rel_id in \
                    date_to_property_id_to_definition[formatted_date]:
                triple_rel_definition = \
                    date_to_property_id_to_definition[formatted_date][curr_rel_id]

            for curr_action_category in curr_action_categories:
                if human_assessment_exists(
                        p_tkgu_triple=curr_tkgu_triple,
                        p_prompt_type=curr_action_category,
                        p_annotator_name=annotator_name
                ):
                    logger.info(f'already assessed {curr_tkgu_triple["triple_labels"]} '
                                f'tkgu operations: {curr_tkgu_operations} '
                                f'for prompt {curr_action_category}: '
                                f'{curr_tkgu_triple["human_assessment"]}')
                    continue
                if exceeds_max_per_tkgu_type(
                        p_tkgu_triple=curr_tkgu_triple,
                        p_prompt_type=curr_action_category,
                        # p_annotator_name=annotator_name,
                        p_nr_annotated_per_tkgu_operation=nr_annotated_per_tkgu_operation,
                        p_max_annotations_per_tkgu_operation=tkgu_operations_to_max_annotations
                ):
                    logger.info(f'exceeds count of '
                                f'already annotated for '
                                f'prompt_type {curr_action_category} '
                                f'triple: {curr_tkgu_triple["triple_labels"]} '
                                f'tkgu operations: {curr_tkgu_operations} '
                                f'nr_annotated_per_tkgu_operation: '
                                f'{nr_annotated_per_tkgu_operation} -- '
                                f'tkgu_operations_to_max_annotations: '
                                f'{tkgu_operations_to_max_annotations}')
                    continue

                curr_annotating = True
                clear_stdin()
                user_input = ''

                while user_input not in {'y', 'n', '1', '2'}:
                    user_input = input(
                        '==========================CURRENT AGREEMENT STATS===========================\n'
                        # nr of annotated triples of each tkgu type and annotator
                        # agreement humans (grouped by annotator name) by llm name 
                        # and triple type
                        f'{get_print_annotation_statistics(df_statistics=df_stats)}'
                        f'-------------- nr_annotated_per_tkgu_operation: \n'
                        f'{nr_annotated_per_tkgu_operation}: \n'
                        f'============CURRENT PASSAGE '
                        f'({idx_instance + 1 + nr_already_annotated}'
                        # f'============PASSAGE from instance nr {idx_instance + 1}'
                        f' of {len(instances_to_annotate) + nr_already_annotated} -- {curr_inst["filename"]})'
                        f' ==============================\n'
                        f'{curr_passage} \n'
                        f'=========================================================\n'
                        f'DISTRIBUTIONS OF TRIPLES: \n'
                        f'x-triples (llm t: {nas["nr_assessed_true_x_triples"]} - '
                        f'f: {nas["nr_assessed_false_x_triples"]}) ** '
                        f'e-triples (llm t: {nas["nr_assessed_true_e_triples"]} - '
                        f'f: {nas["nr_assessed_false_e_triples"]}) ** '
                        f'ee-triples (llm t: {nas["nr_assessed_true_ee_triples"]} - '
                        f'f: {nas["nr_assessed_false_ee_triples"]}) ** '
                        f'ee-kg-triples (llm t: {nas["nr_assessed_true_ee_kg_triples"]} - '
                        f'f: {nas["nr_assessed_false_ee_kg_triples"]}) ** '
                        f'd-triples (llm t: {nas["nr_assessed_true_d_triples"]} - '
                        f'f: {nas["nr_assessed_false_d_triples"]}) \n'
                        '==========================================================\n'
                        # f'instance: {idx_instance + 1} of {curr_inst["filename"]} '
                        # f'nr triple: {idx_tkgu_triple + 1} of {len(curr_instance["tkgu_triples"])}'
                        # f' -- TKGU: {curr_tkgu_operations} -- prompt: {curr_action_category.upper()} \n'
                        f'--------------- TRIPLE {idx_tkgu_triple + 1} of {len(curr_instance["tkgu_triples"])}'
                        f' -- TKGU: {curr_tkgu_operations} -- prompt: {curr_action_category.upper()} \n'
                        # f'Does the above passage contains explicit or implicit knowledge to '
                        # f'support the triple ({curr_triple_labels})?: '
                        # f'support the triple:\n({triple_show})? WHERE: \n'
                        f'({triple_show})?\n '
                        f'"{triple_show[1]}": "{triple_rel_definition}" ; \n'
                        f'----------- \n'
                        # f'TKGU OPERATIONS: ({curr_tkgu_operations}) ; \n'
                        # f'PROMPT TYPE: {curr_action_category.upper()} ; \n'
                        f'-------------- LLM ASSESSMENT: \n'
                        f'{curr_tkgu_triple["llm_assessment"]} \n'
                        f'----------- \n'
                        # f'qualifier_date ({field_qualifier_info}) \n'
                        # f'field_revision_date ({field_passage_date}) \n'
                        # f'field_triple_lifespan_date ({field_triple_lifespan_date}) \n'
                        # f'Definition of "{triple_show[1]}": "{triple_rel_definition}" \n'
                        f'enter Y/y for yes and N/n for no, 2 to skip '
                        f'the instance (saving already annotated triples)'
                        f' 1 to skip triple.')

                    user_input = user_input.lower().strip()

                curr_human_assessment = False
                if user_input == 'y':
                    curr_human_assessment = True
                elif user_input == 'n':
                    curr_human_assessment = False
                elif user_input == '1':
                    jump_next_triple = True
                    break
                elif user_input == '2':
                    jump_next_line = True
                    break
                else:
                    logger.warning(f'NOT RECOGNIZED TOKEN!: {user_input}')

                curr_tkgu_triple['human_assessment'].append(
                    {
                        'annotator_name': annotator_name,
                        'prompt_type': curr_action_category,
                        'assessment': curr_human_assessment,
                        'human_readable_triple': str(triple_show),
                        'definition_relation': triple_rel_definition
                    }
                )
                nr_annotated_per_tkgu_operation = update_count_annotated_per_triple(
                    p_tkgu_triple=curr_tkgu_triple,
                    p_prompt_type=curr_action_category,
                    p_nr_annotated_per_tkgu_operation=nr_annotated_per_tkgu_operation
                )
            if jump_next_triple:
                continue
            if jump_next_line:
                break

        lst_already_annotated_instances.append(curr_instance)
        if curr_annotating:
            filename_to_outfile[curr_inst['filename']].write(
                json.dumps(curr_instance, ensure_ascii=False) + '\n'
            )
            filename_to_outfile[curr_inst['filename']].flush()

    logger.info('END annotating')
