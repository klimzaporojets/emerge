# 2025.09.03 - subsampling the dataset to annotate, making sure more or less
# equal number of tkgu triple types, greedily focus to subsample the lower one
# until the number of instances is reached

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
    get_instance_tkgu_types_llm_assessments, load_all_instances, stats_all_instances, \
    human_assessment_exists, clear_stdin, get_print_annotation_statistics, \
    show_discrepancies, get_nr_triples_per_tkgu, is_lowest_dominant_present, update_nr_subsampled_triples

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/s09_annotate_dataset/20250901/'
                                's09_annotate_dataset.json',
                        help='The config file that contains all the parameters')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.width', 200)

    args = parser.parse_args()
    config = json.load(open(args.config_file, 'rt'))

    # tkgu_operations_to_annotate = set(config['tkgu_operations_to_annotate'])
    input_dataset_path = config['input_dataset_path']
    output_annotation_path = config['output_annotation_path']
    output_subsampling_path = config['output_subsampling_path']
    os.makedirs(output_annotation_path, exist_ok=True)
    os.makedirs(output_subsampling_path, exist_ok=True)
    #

    # annotator_name = config['annotator_name']
    already_annotated_hash_ids = set()
    files_to_annotate = os.listdir(input_dataset_path)

    logger.info('BEGIN loading already annotated content')
    # lst_already_annotated_instances = list()
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
                already_annotated_hash_ids.add(parsed_line['hash_id'])
                # lst_already_annotated_instances.append(parsed_line)
    logger.info('END loading already annotated content')

    # filename_to_outfile = dict()

    filename_to_instances: Dict[str, List] = dict()
    logger.info('BEGIN loading content to annotate')
    # distribution_anno_stats = dict()

    for filename in files_to_annotate:
        # Check if the file has a .jsonl extension
        # output_file_path = os.path.join(output_annotation_path, filename)
        # output_file = open(output_file_path, 'a', encoding='utf-8')
        # filename_to_outfile[filename] = output_file

        file_path = os.path.join(input_dataset_path, filename)

        for curr_instance in open(file_path, 'rt', encoding='utf-8'):
            parsed_line = json.loads(curr_instance)

            if parsed_line['hash_id'] in already_annotated_hash_ids:
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

    total_subsampled_instances = config['total_subsampled_instances']

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

    nr_tkgu_subsampled_triples = {
        'x-triples': 0,
        'e-triples': 0,
        'ee-triples': 0,
        'ee-kg-triples': 0,
        'd-triples': 0
    }
    #
    subsampled_instances = list()
    for curr_instance in instances_to_annotate:
        c_inst = curr_instance['instance']
        curr_tkgu_subsampled_triples = {
            'x-triples': 0,
            'e-triples': 0,
            'ee-triples': 0,
            'ee-kg-triples': 0,
            'd-triples': 0
        }
        curr_tkgu_subsampled_triples = \
            get_nr_triples_per_tkgu(c_inst, curr_tkgu_subsampled_triples)
        ilw = is_lowest_dominant_present(nr_tkgu_subsampled_triples, curr_tkgu_subsampled_triples)
        if ilw:
            subsampled_instances.append(curr_instance)
            nr_tkgu_subsampled_triples = \
                update_nr_subsampled_triples(nr_tkgu_subsampled_triples,
                                             curr_tkgu_subsampled_triples)
        if len(subsampled_instances) >= total_subsampled_instances:
            break

    logger.info('END selecting instances to annotate that have both positive and negative '
                'llm assessments')


    subs_instc = [i['instance']for i in subsampled_instances]
    subsampled_instances_df = load_all_instances(lst_all_instances=subs_instc)
    logger.info('calculating stats on all instances')
    stats_all_instances(
        df_all_instances=subsampled_instances_df
    )

    outfile_path = os.path.join(output_subsampling_path, 'instances_to_annotate.jsonl')
    # output_file_path
    with open(outfile_path, 'wt', encoding='utf-8') as outfile:
        for curr_subsampled_instance in subs_instc:
            outfile.write(json.dumps(curr_subsampled_instance, ensure_ascii=False) +
                          '\n')

