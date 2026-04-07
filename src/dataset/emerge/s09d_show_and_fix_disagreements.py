# Shows and fixes the annotation disagreements takes as input the output from
# s09b_annotate_dataset_v4.py
# first merges the annotations of different users, also including llms from
# two or more different folders. The output will always contain this merged version already,
# plus any changes to the annotation the reviewer wants to make.

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
    show_discrepancies, exceeds_max_per_tkgu_type, update_count_annotated, update_count_annotated_per_triple, \
    merge_annotations, show_discrepancies_and_ask_correct

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

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.width', 200)

    args = parser.parse_args()
    config = json.load(open(args.config_file, 'rt'))
    if args.annotator_name != '':
        config['annotator_name'] = args.annotator_name

    tkgu_operations_to_check = set(config['tkgu_operations_to_check'])

    nr_annotated_per_tkgu_operation = dict()

    input_annotation_paths = config['input_annotation_paths']

    output_solved_disagreements_path = config['output_solved_disagreements_path']

    os.makedirs(output_solved_disagreements_path, exist_ok=True)

    annotator_name = config['annotator_name']
    annotators_to_compare = config['annotators_to_compare']
    hash_id_to_instance: Dict[str, Dict] = dict()

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

    logger.info('BEGIN loading already annotated content and merging')

    continue_next = True
    for in_dir_path in input_annotation_paths:
        curr_annotated_file_names = os.listdir(in_dir_path)
        for curr_annotated_filename in curr_annotated_file_names:
            in_file_path = os.path.join(in_dir_path, curr_annotated_filename)
            for curr_instance in open(in_file_path, 'rt', encoding='utf-8'):
                parsed_line = json.loads(curr_instance)
                curr_hash_id = parsed_line['hash_id']
                if curr_hash_id not in hash_id_to_instance:
                    hash_id_to_instance[curr_hash_id] = parsed_line
                else:
                    merged_annotations = merge_annotations(
                        instance1=hash_id_to_instance[curr_hash_id],
                        instance2=parsed_line
                    )
                    hash_id_to_instance[curr_hash_id] = merged_annotations

    #
    logger.info('BEGIN replacing already fixed')
    output_file_path = os.path.join(output_solved_disagreements_path,
                                    'solved_disagreements.jsonl')
    already_fixed_hash_ids = set()
    if os.path.exists(output_file_path):
        mode_disagreements = 'at'
        for curr_line in open(output_file_path, 'rt', encoding='utf-8'):
            pars_line = json.loads(curr_line)
            hash_id_to_instance[pars_line['hash_id']] = pars_line
            already_fixed_hash_ids.add(pars_line['hash_id'])
    else:
        mode_disagreements = 'wt'
    logger.info('END replacing already fixed')
    #
    logger.info('BEGIN calculating statistics')
    df_stats = get_annotation_statistics(
        annotated_instances=list(hash_id_to_instance.values()),
        annotator_names=[annotator_name] + \
                        annotators_to_compare
    )
    #
    logger.info('END calculating statistics')

    logger.info('END loading already annotated content and merging')
    check_instance = True
    with open(output_file_path, mode_disagreements, encoding='utf-8') as out_file:
        for idx_passage, (curr_hash_id, curr_instance) in enumerate(hash_id_to_instance.items()):
            ################################
            if curr_hash_id in already_fixed_hash_ids:
                logger.info(f'curr_hash_id was already assessed and fixed '
                            f'{curr_hash_id}')
                continue
            if check_instance:
                logger.info('*****************************************************')
                logger.info('showing discrepancies with new instance')
                logger.info('*****************************************************')
                curr_instance, next_action = \
                    show_discrepancies_and_ask_correct(
                        instance=curr_instance,
                        config=config,
                        df_stats=df_stats,
                        idx_passage = idx_passage
                    )

                hash_id_to_instance[curr_hash_id] = curr_instance

                if next_action == 2:
                    check_instance = False
                    break
                if next_action == 3:
                    logger.info('BEGIN recalculating statistics')
                    df_stats = get_annotation_statistics(
                        annotated_instances=list(hash_id_to_instance.values()),
                        annotator_names=[annotator_name] + \
                                        annotators_to_compare
                    )
                    logger.info('END recalculating statistics')

                nr_annotated_per_tkgu_operation = update_count_annotated(
                    p_instance=curr_instance,
                    p_annotator_name=config['annotator_name'],
                    p_nr_annotated_per_tkgu_operation=nr_annotated_per_tkgu_operation
                )
                out_file.write(json.dumps(curr_instance, ensure_ascii=False) + '\n')
                out_file.flush()
        logger.info('END loading already annotated content')
        logger.info(f'nr_annotated_per_tkgu_operation: '
                    f'{nr_annotated_per_tkgu_operation}')

    ####################################

    logger.info('END annotating')
