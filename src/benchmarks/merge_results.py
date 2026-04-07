# 2025.06.06: the _v4 uses the new dataset format
import argparse
import base64
import hashlib
import json
import logging
import os
import pickle
import traceback
from datetime import datetime
from typing import Dict

from tqdm import tqdm



# from utils.load_wiki_sql_tables import load_wiki_page_title_to_wiki_page_id, load_wiki_page_id_to_redirected_page_id, \
#     load_wiki_page_id_to_wikidata_qid, load_wdata_qid_to_page_ids, get_page_title_changes

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def obtain_property_labels_to_ids_dict(dictionary_path):
    """Load property label -> Wikidata property ID mapping from a JSONL dictionary file."""
    to_ret_dict = dict()
    for curr_line_dict in open(dictionary_path, 'rt', encoding='utf-8'):
        curr_pars_line = json.loads(curr_line_dict)
        property_label = curr_pars_line['text'].strip().lower()
        property_id = curr_pars_line['metadata']['property']
        if property_label not in to_ret_dict:
            to_ret_dict[property_label] = list()
        to_ret_dict[property_label].append(property_id)
    return to_ret_dict


def obtain_property_ids_to_labels_dict(dictionary_path):
    """Load Wikidata property ID -> label mapping from a JSONL dictionary file."""
    to_ret_dict = dict()
    for curr_line_dict in open(dictionary_path, 'rt', encoding='utf-8'):
        curr_pars_line = json.loads(curr_line_dict)
        property_label = curr_pars_line['text'].strip().lower()
        property_id = curr_pars_line['metadata']['property']
        # if property_id not in to_ret_dict:
        to_ret_dict[property_id] = property_label
        # to_ret_dict[property_label].append(property_id)
    return to_ret_dict


def get_input_files_for_experiment(root_dir):
    """Recursively collect all file paths under root_dir, sorted alphabetically."""
    to_ret = list()
    for root, dirs, files in os.walk(root_dir):
        dirs.sort()
        files.sort()
        for file in files:
            to_ret.append(os.path.join(root, file))

        # The `dirs` list is sorted, so os.walk will traverse in alphabetical order
        # No need to modify the traversal logic further
    return to_ret


def count_lines_in_file(file_path):
    """Return the total number of lines in a file."""
    with open(file_path, 'r') as file:
        line_count = sum(1 for line in file)
    return line_count


def filter_out_new_relation_types(curr_parsed_line, properties_dictionary):
    filtered_triples = list()
    for curr_emerging_triple in curr_parsed_line['tkgu_triples']:
        if curr_emerging_triple['triple_qids'][1] not in properties_dictionary:
            logger.warning('following_curr_emerging_triple ignored because property id '
                           'not in dictionary: '
                           f'{curr_emerging_triple["triple_qids"]} ; '
                           f'{curr_emerging_triple["triple_labels"]}')
            continue
        filtered_triples.append(curr_emerging_triple)
    if len(filtered_triples) == 0:
        logger.error('PROBLEM: NO EMERGING TRIPLES LEFT AFTER FILTERING!!')
    curr_parsed_line['tkgu_triples'] = filtered_triples
    # filtered_emerging_triple_to_kg = list()
    # for curr_emerging_triple_to_kg in curr_parsed_line['matched_triples_entities_to_kg']:
    #     if curr_emerging_triple_to_kg['triple_qids'][1] not in properties_dictionary:
    #         logger.debug('following_curr_emerging_triple ignored because property id '
    #                      'not in dictionary: '
    #                      f'{curr_emerging_triple_to_kg["triple_qids"]} ; '
    #                      f'{curr_emerging_triple_to_kg["triple_labels"]}')
    #         continue
    #     filtered_emerging_triple_to_kg.append(curr_emerging_triple_to_kg)
    #
    # curr_parsed_line['matched_triples_entities_to_kg'] = filtered_emerging_triple_to_kg
    # filtered_existing_knowledge = list()
    # for curr_existing_knowledge in curr_parsed_line['existing_knowledge']:
    #     if curr_existing_knowledge['triple_qids'][1] not in properties_dictionary:
    #         logger.debug('following_curr_existing_knowledge ignored because property id '
    #                      'not in dictionary: '
    #                      f'{curr_existing_knowledge["triple_qids"]} ; '
    #                      f'{curr_existing_knowledge["triple_labels"]}')
    #         continue
    #     filtered_existing_knowledge.append(curr_existing_knowledge)
    # curr_parsed_line['existing_knowledge'] = filtered_existing_knowledge


def extend_predictions_with_mentions(parsed_line,
                                     label_to_property_ids,
                                     property_id_to_label):
    mentions = parsed_line['mentions']
    mention_to_qids = dict()
    for curr_mention in mentions:
        curr_mention_text = curr_mention['mention_text'].lower().strip()
        if curr_mention_text not in mention_to_qids:
            mention_to_qids[curr_mention_text] = set()
        mention_to_qids[curr_mention_text].add(curr_mention['qid'])

    for curr_predictions in parsed_line['predictions']:
        to_ret_extended_triples = set()
        predicted_triple_ids = set()
        predicted_relations = set()
        for curr_prediction in curr_predictions['predicted_triples']:
            if curr_prediction['triple_qids'][0].startswith('Q') and \
                    curr_prediction['triple_qids'][2].startswith('Q'):
                predicted_triple_ids.add(tuple(curr_prediction['triple_qids']))
            predicted_relations.add(tuple(curr_prediction['extracted_relation']))
        for curr_predicted_relation in predicted_relations:
            if curr_predictions['alias_name'].lower() == 'edc':
                curr_head_label = str(curr_predicted_relation[0]).replace('_', ' ').strip().lower()
                curr_tail_label = str(curr_predicted_relation[2]).replace('_', ' ').strip().lower()
            else:
                curr_head_label = str(curr_predicted_relation[0]).strip().lower()
                curr_tail_label = str(curr_predicted_relation[2]).strip().lower()
            curr_rel = curr_predicted_relation[1].lower().strip()
            if curr_head_label in mention_to_qids and curr_tail_label in mention_to_qids:
                for curr_head_qid in mention_to_qids[curr_head_label]:
                    for curr_tail_qid in mention_to_qids[curr_tail_label]:
                        curr_rel_property_ids = list()
                        if curr_rel not in label_to_property_ids and curr_predicted_relation[
                            1] not in property_id_to_label:
                            logger.error(f'{curr_predictions["alias_name"]} -- '
                                         f'should not happen that curr_rel_can_not_be_identified '
                                         f'curr_rel {curr_rel} , prediction_relation: '
                                         f'{curr_predicted_relation} '
                                         f'prediction alias: {curr_predictions["alias_name"]} '
                                         f'continuing')
                            continue
                        else:
                            if curr_rel in label_to_property_ids:
                                curr_rel_property_ids = label_to_property_ids[curr_rel]
                            elif curr_predicted_relation[1] in property_id_to_label:
                                curr_rel_property_ids = [curr_predicted_relation[1]]
                        # for curr_property_id in label_to_property_ids[curr_rel]:
                        for curr_property_id in curr_rel_property_ids:
                            potential_triple = (curr_head_qid, curr_property_id, curr_tail_qid)
                            if potential_triple not in predicted_triple_ids:
                                to_ret_extended_triples.add(potential_triple)
        curr_predictions['extended_triples'] = list(to_ret_extended_triples)
        # return to_ret_extended_triples


def generate_short_hash(input_string: str, hash_length: int):
    # Create a SHA256 hash of the input string
    hash_object = hashlib.sha256(input_string.encode())
    # Convert the hash to a byte array
    hash_bytes = hash_object.digest()
    # Encode the byte array to a base64 string and strip unwanted characters
    short_hash = base64.urlsafe_b64encode(hash_bytes).decode('utf-8').rstrip('=')
    # Return the first 8 characters for a shorter hash
    return short_hash[:hash_length]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/s02b_merge_results_relik/20250407_slurm/'
                                's02_relik_all.json',
                        help='The config file that contains all the parameters')
    # the merge has to have some characteristics:
    # 1. qid have to be ALWAYS assigned either it is open ie or cie , so for example if
    #  only one open ie file has to be "merged", this script should add the respective qids based on
    #  the exact match with the mentions.
    # 2. each of the predicted triples has to have source(s) which indicate the model used to predict it.
    #   the source is taken from "experiment_alias" attribute of the config json.

    args = parser.parse_args()
    config = json.load(open(args.config_file, 'rt'))
    all_dataset_path = config['input_dataset_dir']
    cache_dataset_dir = config['cache_dataset_dir']
    os.makedirs(cache_dataset_dir, exist_ok=True)
    cache_dataset_file = os.path.join(cache_dataset_dir, 'cache_dataset.pickle')
    if os.path.exists(cache_dataset_file):
        logger.info('loading complete dataset from cache')
        (parsed_instances_to_predictions, nr_instances_dataset) = \
            pickle.load(open(cache_dataset_file, 'rb'))
    else:
        #
        # loads the complete ground truth dataset
        logger.info('loading complete dataset')
        parsed_instances_to_predictions = dict()
        nr_instances_dataset = dict()
        for dirpath, dirnames, filenames in os.walk(all_dataset_path):
            for filename in filenames:
                if filename.endswith('.jsonl'):
                    file_path = os.path.join(dirpath, filename)
                    logger.info(f'Loading: {file_path}')
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            curr_parsed_line = json.loads(line)  # Or process the JSON line here
                            curr_hash_id = generate_short_hash(curr_parsed_line['passage'],
                                                               hash_length=256)
                            curr_entry = (curr_parsed_line['delta_timestamps'][0],
                                          curr_parsed_line['delta_timestamps'][1],
                                          curr_hash_id)
                            parsed_instances_to_predictions[curr_entry] = curr_parsed_line
                            parsed_instances_to_predictions[curr_entry]['predictions'] = []
                            nr_instance_entry = (curr_parsed_line['delta_timestamps'][0],
                                                 curr_parsed_line['delta_timestamps'][1])
                            if nr_instance_entry not in nr_instances_dataset:
                                nr_instances_dataset[nr_instance_entry] = 0
                            nr_instances_dataset[nr_instance_entry] += 1

        pickle.dump((parsed_instances_to_predictions,
                     nr_instances_dataset), open(cache_dataset_file, 'wb'))
        logger.info('end loading complete dataset')
    #
    # input_base_dir = config['input_base_dir']
    output_base_dir = config['output_base_dir']
    date_to_property_label_to_id = dict()
    date_to_property_id_to_label = dict()
    for curr_experiments_to_merge in config['experiments_to_merge']:
        experiment_alias_to_input_file_paths = dict()
        prev_file_names = []
        output_dir = curr_experiments_to_merge['output_dir']
        curr_output_dir = os.path.join(output_base_dir, output_dir)
        os.makedirs(curr_output_dir, exist_ok=True)
        # parsed_instances_to_predictions = dict()
        for curr_experiment in curr_experiments_to_merge['to_merge']:
            nr_instances_dataset_predicted = dict()
            # curr_experiment_path = os.path.join(input_base_dir, curr_experiment['experiment_path'])
            curr_experiment_path = os.path.join(curr_experiment['experiment_path'])
            curr_experiment_alias = curr_experiment['experiment_alias']
            logger.info(f'========== merging experimental results of {curr_experiment_alias} =========')
            experiment_alias_to_input_file_paths[curr_experiment_alias] = \
                get_input_files_for_experiment(curr_experiment_path)
            nr_not_found_instances = 0

            for curr_file_path in experiment_alias_to_input_file_paths[curr_experiment_alias]:
                content_passage = ''
                logger.info(f'parsing {os.path.basename(curr_file_path)}')
                for idx_parsed_line, curr_line in enumerate(open(curr_file_path, 'rt', encoding='utf-8')):
                    curr_parsed_line = json.loads(curr_line)
                    curr_hash_id = \
                        generate_short_hash(curr_parsed_line['passage'], hash_length=256)
                    curr_instance_entry = (curr_parsed_line['delta_timestamps'][0],
                                           curr_parsed_line['delta_timestamps'][1],
                                           curr_hash_id)
                    curr_instance_dataset = (curr_parsed_line['delta_timestamps'][0],
                                             curr_parsed_line['delta_timestamps'][1])

                    if 'predictions' not in curr_parsed_line:
                        curr_parsed_line['predictions'] = []
                        logger.error(f'no predictions for {curr_experiment_alias}')
                    else:
                        for curr_prediction in curr_parsed_line['predictions']:
                            curr_prediction['alias_name'] = curr_experiment_alias

                    if curr_instance_entry not in parsed_instances_to_predictions:
                        # parsed_instances_to_predictions[curr_instance_entry] = curr_parsed_line['predictions']
                        nr_not_found_instances += 1
                        # logger.error(f'**** {curr_experiment_alias} ({nr_not_found_instances}) instances not found '
                        #              f' \n{curr_parsed_line} \n\n')
                    else:
                        if curr_instance_dataset not in nr_instances_dataset_predicted:
                            nr_instances_dataset_predicted[curr_instance_dataset] = 0
                        nr_instances_dataset_predicted[curr_instance_dataset] += 1
                        parsed_instances_to_predictions[curr_instance_entry]['predictions'] += curr_parsed_line[
                            'predictions']

            if nr_not_found_instances > 0:
                logger.warning(
                    f'**** {curr_experiment_alias} ({nr_not_found_instances}) instances not found'
                    f' (There are still some records being removed by s15_final_filter_and_upload_to_huggingface,'
                    f' indicating that some entities for some reason (e.g., redirects) are not in the '
                    f'dictionary. TODO: further improve the entity resolution done in '
                    f's13_rename_relik_index_v3.py.)'
                )
            for curr_predicted_dataset_entry, curr_predicted_nr_instances in nr_instances_dataset_predicted.items():
                gt_nr_instances = nr_instances_dataset[curr_predicted_dataset_entry]
                if gt_nr_instances != curr_predicted_nr_instances:
                    logger.warning(f'for dataset entry {curr_predicted_dataset_entry} '
                                   f'the nr of instances differ: {gt_nr_instances} vs '
                                   f'{curr_predicted_nr_instances}')
                else:
                    logger.info(f'OK {curr_predicted_dataset_entry} '
                                f'the nr of instances matches: {gt_nr_instances} vs '
                                f'{curr_predicted_nr_instances}')
        ############### BEGIN: another loop to actually write all the predictions out
        logger.info('CONTINUING')
        # continue
        for dirpath, dirnames, filenames in os.walk(all_dataset_path):
            for filename in filenames:
                if filename.endswith('.jsonl'):
                    file_path = os.path.join(dirpath, filename)
                    logger.info(f'Loading for saving: {file_path}')
                    with open(file_path, 'r', encoding='utf-8') as f:
                        curr_file_name = os.path.basename(file_path)
                        output_file_path = os.path.join(curr_output_dir, curr_file_name)
                        with open(output_file_path, 'wt', encoding='utf-8') as outfile:
                            for line in f:
                                curr_parsed_line = json.loads(line)  # Or process the JSON line here
                                curr_hash_id = generate_short_hash(curr_parsed_line['passage'],
                                                                   hash_length=256)
                                curr_entry = (curr_parsed_line['delta_timestamps'][0],
                                              curr_parsed_line['delta_timestamps'][1],
                                              curr_hash_id)
                                # parsed_instances_to_predictions[curr_entry] = curr_parsed_line
                                # parsed_instances_to_predictions[curr_entry]['predictions'] = []

                                if config['calculate_genres_metrics']:
                                    pass

                                # date_object = datetime.fromtimestamp(curr_parsed_line['delta_timestamps'][0])
                                date_object = datetime.fromtimestamp(
                                    parsed_instances_to_predictions[curr_entry]['delta_timestamps'][0]
                                )
                                formatted_date = date_object.strftime('%Y-%m-%d')
                                if formatted_date not in date_to_property_label_to_id:
                                    dictionary_path = os.path.join(config['input_relation_dictionaries'],
                                                                   formatted_date,
                                                                   'documents.jsonl')

                                    date_to_property_label_to_id[formatted_date] = \
                                        obtain_property_labels_to_ids_dict(dictionary_path)
                                    date_to_property_id_to_label[formatted_date] = \
                                        obtain_property_ids_to_labels_dict(dictionary_path)

                                extend_predictions_with_mentions(parsed_instances_to_predictions[curr_entry],
                                                                 date_to_property_label_to_id[formatted_date],
                                                                 date_to_property_id_to_label[formatted_date]
                                                                 )
                                filter_out_new_relation_types(parsed_instances_to_predictions[curr_entry],
                                                              date_to_property_id_to_label[formatted_date])

                                # extends with current kg qids
                                # extend_predictions_with_kg_qids(parsed_instances_to_predictions[curr_entry])

                                # 2025.06.14 - commenting because hash_id has to be assigned in s15 when the dataset is created
                                # parsed_instances_to_predictions[curr_entry]['hash_id'] = \
                                #     generate_short_hash(str(parsed_instances_to_predictions[curr_entry]),
                                #                         hash_length=256)
                                # 2025.06.14 - end commenting

                                outfile.write(
                                    json.dumps(parsed_instances_to_predictions[curr_entry],
                                               ensure_ascii=False)
                                    + '\n'
                                )
                                outfile.flush()
