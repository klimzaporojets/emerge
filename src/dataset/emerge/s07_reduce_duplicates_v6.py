# _v6 also accounts for textual duplicates and leaves the text with the highest
# number of supported triples. Prioritizes
# Triples that have llm_assessment for all triples
#  - most d-triples supported, and if tie:
#  - most emerging entities triples supported, and if tie:
#  - most emerging relations triples supported, and if tie:
#  - most total number of triples supported, and if tie:
#  - most total number of triples, and if tie:
#  - the one with longest text length, and if tie:
#  - randomly select between two
# - Heuristic before calculating distances to detect duplicates:
#   - same anchor page
#   - similar length (e.g., above 0.85 of ratio for same anchor page and above 0.95 of ratio for
#   different anchors).

import argparse
import json
import logging
import os
import os.path
import random
import re
from datetime import datetime
from typing import Dict

from tqdm import tqdm

from dataset.emerge.utils.constants import ACTION_CATEGORY_ADD, ACTION_CATEGORY_DEPRECATE, ACTION_CATEGORY_ASSERT
from dataset.emerge.utils.text_utils import calculate_english_word_percentage

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)


def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import jaccard
import Levenshtein


def jaccard_similarity(pass1, pass2):
    # Convert the passages into sets of tokens
    vectorizer = CountVectorizer().fit_transform([pass1, pass2])
    vectors = vectorizer.toarray()

    # Calculate Jaccard similarity
    jaccard_sim = 1 - jaccard(vectors[0], vectors[1])
    return jaccard_sim


def normalized_edit_distance(pass1, pass2):
    # Calculate the edit distance using the Levenshtein library
    edit_dist = Levenshtein.distance(pass1, pass2)

    # Calculate the maximum possible distance
    max_length = len(pass1) + len(pass2)

    # Normalize the edit distance
    normalized_distance = edit_dist / max_length
    return normalized_distance


def count_lines(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for _ in file)


def obtain_property_ids_to_labels_dict(dictionary_path):
    to_ret_dict = dict()
    for curr_line_dict in open(dictionary_path, 'rt', encoding='utf-8'):
        curr_pars_line = json.loads(curr_line_dict)
        property_label = curr_pars_line['text'].strip().lower()
        property_id = curr_pars_line['metadata']['property']
        to_ret_dict[property_id] = property_label
    return to_ret_dict


def filter_out_new_relation_types(curr_parsed_line, properties_dictionary):
    filtered_emerging_triple = list()
    for curr_emerging_triple in curr_parsed_line['tkgu_triples']:
        if curr_emerging_triple['triple'][1] not in properties_dictionary:
            logger.warning('following_curr_emerging_triple ignored because property id '
                           'not in dictionary: '
                           f'{curr_emerging_triple["triple"]} ; '
                           f'{curr_emerging_triple["triple_labels"]}')
            continue
        filtered_emerging_triple.append(curr_emerging_triple)
    if len(filtered_emerging_triple) == 0:
        logger.error('*****PROBLEM: no_triples_left AFTER FILTERING!!*****')
    curr_parsed_line['tkgu_triples'] = filtered_emerging_triple


def get_supported_triples(triples, prompt_type, llm_names_with_qualifiers,
                          llm_names_wout_qualifiers):
    supported_triples = list()

    for curr_triple in triples:
        llm_prompt_to_assessment = dict()
        for curr_llm_assessment in curr_triple['llm_assessment']:
            llm_prompt_to_assessment[(curr_llm_assessment['llm_name'],
                                      curr_llm_assessment['llm_prompt_type'])] = \
                curr_llm_assessment['llm_assessment']

        if len(curr_triple['qualifier_info']) == 0:
            llm_names_to_check = llm_names_wout_qualifiers
        else:
            llm_names_to_check = llm_names_with_qualifiers

        assessed_positively = True
        for curr_llm_name in llm_names_to_check:
            if (curr_llm_name, prompt_type) not in llm_prompt_to_assessment \
                    or not llm_prompt_to_assessment[(curr_llm_name, prompt_type)]:
                assessed_positively = False
        if assessed_positively:
            supported_triples.append(curr_triple)
    return supported_triples


def deduplicate_instance(instance1, instance2, config):
    # Deduplication rules:
    #  - most d-triples supported, and if tie:
    #  - most emerging entities triples supported, and if tie:
    #  - most emerging relations triples supported, and if tie:
    #  - ee-kg triples supported, and if tie:
    #  - most total number of triples supported, and if tie:
    #  - most total number of triples, and if tie:
    #  - the one with longest text length, and if tie:
    #  - randomly select between two
    # llm_assessor_deprecation_triples = config['llm_assessor_deprecation_triples']
    prompt_types_to_assessors = config['prompt_type_to_assessors']
    wout_qualifiers_assertion_llms = prompt_types_to_assessors['without_qualifiers'] \
        [ACTION_CATEGORY_ASSERT]
    wout_qualifiers_deprecation_llms = prompt_types_to_assessors['without_qualifiers'] \
        [ACTION_CATEGORY_DEPRECATE]
    with_qualifiers_assertion_llms = prompt_types_to_assessors['with_qualifiers'] \
        [ACTION_CATEGORY_ASSERT]
    with_qualifiers_deprecation_llms = prompt_types_to_assessors['with_qualifiers'] \
        [ACTION_CATEGORY_DEPRECATE]

    d_triples_inst1 = list()
    d_triples_inst2 = list()

    ee_triples_inst1 = list()
    ee_triples_inst2 = list()

    e_triples_inst1 = list()
    e_triples_inst2 = list()

    ee_kg_triples_inst1 = list()
    ee_kg_triples_inst2 = list()

    x_triples_inst1 = list()
    x_triples_inst2 = list()

    for curr_triple1 in instance1['tkgu_triples']:
        tkgu_operations = set(curr_triple1['tkgu_operations'])
        if 'x-triples' in tkgu_operations:
            x_triples_inst1.append(curr_triple1)
        if 'd-triples' in tkgu_operations:
            d_triples_inst1.append(curr_triple1)
        if 'ee-triples' in tkgu_operations:
            ee_triples_inst1.append(curr_triple1)
        if 'e-triples' in tkgu_operations:
            e_triples_inst1.append(curr_triple1)
        if 'ee-kg-triples' in tkgu_operations:
            ee_kg_triples_inst1.append(curr_triple1)

    for curr_triple2 in instance2['tkgu_triples']:
        tkgu_operations = set(curr_triple2['tkgu_operations'])
        if 'x-triples' in tkgu_operations:
            x_triples_inst2.append(curr_triple2)
        if 'd-triples' in tkgu_operations:
            d_triples_inst2.append(curr_triple2)
        if 'ee-triples' in tkgu_operations:
            ee_triples_inst2.append(curr_triple2)
        if 'e-triples' in tkgu_operations:
            e_triples_inst2.append(curr_triple2)
        if 'ee-kg-triples' in tkgu_operations:
            ee_kg_triples_inst2.append(curr_triple2)

    supported_triples_inst1 = \
        get_supported_triples(d_triples_inst1, ACTION_CATEGORY_DEPRECATE,
                              llm_names_with_qualifiers=with_qualifiers_deprecation_llms,
                              llm_names_wout_qualifiers=wout_qualifiers_deprecation_llms)
    supported_triples_inst2 = \
        get_supported_triples(d_triples_inst2, ACTION_CATEGORY_DEPRECATE,
                              llm_names_with_qualifiers=with_qualifiers_deprecation_llms,
                              llm_names_wout_qualifiers=wout_qualifiers_deprecation_llms)

    if len(supported_triples_inst1) > len(supported_triples_inst2):
        logger.debug('*********************deprecation_resolution**************************')
        logger.debug(f'*****INSTANCE1: {instance1}')
        logger.debug(f'*****')
        logger.debug(f'*****INSTANCE2: {instance2}')
        logger.debug(f'*****')
        logger.debug(f'*****FAVOUR: instance1')
        logger.debug('*********************************************************************')
        return instance1
    elif len(supported_triples_inst2) > len(supported_triples_inst1):
        logger.debug('*********************deprecation_resolution**************************')
        logger.debug(f'*****INSTANCE1: {instance1}')
        logger.debug(f'*****')
        logger.debug(f'*****INSTANCE2: {instance2}')
        logger.debug(f'*****')
        logger.debug(f'*****FAVOUR: instance2')
        logger.debug('*********************************************************************')
        return instance2

    supported_triples_inst1 = \
        get_supported_triples(ee_triples_inst1, ACTION_CATEGORY_ASSERT,
                              llm_names_with_qualifiers=with_qualifiers_assertion_llms,
                              llm_names_wout_qualifiers=wout_qualifiers_assertion_llms)
    supported_triples_inst2 = \
        get_supported_triples(ee_triples_inst2, ACTION_CATEGORY_ASSERT,
                              llm_names_with_qualifiers=with_qualifiers_assertion_llms,
                              llm_names_wout_qualifiers=wout_qualifiers_assertion_llms)

    if len(supported_triples_inst1) > len(supported_triples_inst2):
        logger.debug('*********************ee_resolution**************************')
        logger.debug(f'*****INSTANCE1: {instance1}')
        logger.debug(f'*****')
        logger.debug(f'*****INSTANCE2: {instance2}')
        logger.debug(f'*****')
        logger.debug(f'*****FAVOUR: instance1')
        logger.debug('*********************************************************************')

        return instance1
    elif len(supported_triples_inst2) > len(supported_triples_inst1):
        logger.debug('*********************ee_resolution**************************')
        logger.debug(f'*****INSTANCE1: {instance1}')
        logger.debug(f'*****')
        logger.debug(f'*****INSTANCE2: {instance2}')
        logger.debug(f'*****')
        logger.debug(f'*****FAVOUR: instance2')
        logger.debug('*********************************************************************')

        return instance2

    supported_triples_inst1 = \
        get_supported_triples(e_triples_inst1, ACTION_CATEGORY_ASSERT,
                              llm_names_with_qualifiers=with_qualifiers_assertion_llms,
                              llm_names_wout_qualifiers=wout_qualifiers_assertion_llms)
    supported_triples_inst2 = \
        get_supported_triples(e_triples_inst2, ACTION_CATEGORY_ASSERT,
                              llm_names_with_qualifiers=with_qualifiers_assertion_llms,
                              llm_names_wout_qualifiers=wout_qualifiers_assertion_llms)

    if len(supported_triples_inst1) > len(supported_triples_inst2):
        logger.debug('*********************e_resolution**************************')
        logger.debug(f'*****INSTANCE1: {instance1}')
        logger.debug(f'*****')
        logger.debug(f'*****INSTANCE2: {instance2}')
        logger.debug(f'*****')
        logger.debug(f'*****FAVOUR: instance1')
        logger.debug('*********************************************************************')

        return instance1
    elif len(supported_triples_inst2) > len(supported_triples_inst1):
        logger.debug('*********************e_resolution**************************')
        logger.debug(f'*****INSTANCE1: {instance1}')
        logger.debug(f'*****')
        logger.debug(f'*****INSTANCE2: {instance2}')
        logger.debug(f'*****')
        logger.debug(f'*****FAVOUR: instance2')
        logger.debug('*********************************************************************')
        return instance2

    supported_triples_inst1 = \
        get_supported_triples(ee_kg_triples_inst1, ACTION_CATEGORY_ASSERT,
                              llm_names_with_qualifiers=with_qualifiers_assertion_llms,
                              llm_names_wout_qualifiers=wout_qualifiers_assertion_llms)
    supported_triples_inst2 = \
        get_supported_triples(ee_kg_triples_inst2, ACTION_CATEGORY_ASSERT,
                              llm_names_with_qualifiers=with_qualifiers_assertion_llms,
                              llm_names_wout_qualifiers=wout_qualifiers_assertion_llms)

    if len(supported_triples_inst1) > len(supported_triples_inst2):
        logger.debug('*********************ee_kg_resolution**************************')
        logger.debug(f'*****INSTANCE1: {instance1}')
        logger.debug(f'*****')
        logger.debug(f'*****INSTANCE2: {instance2}')
        logger.debug(f'*****')
        logger.debug(f'*****FAVOUR: instance1')
        logger.debug('*********************************************************************')
        return instance1
    elif len(supported_triples_inst2) > len(supported_triples_inst1):
        logger.debug('*********************ee_kg_resolution**************************')
        logger.debug(f'*****INSTANCE1: {instance1}')
        logger.debug(f'*****')
        logger.debug(f'*****INSTANCE2: {instance2}')
        logger.debug(f'*****')
        logger.debug(f'*****FAVOUR: instance2')
        logger.debug('*********************************************************************')
        return instance2

    supported_triples_inst1 = \
        get_supported_triples(x_triples_inst1, ACTION_CATEGORY_ASSERT,
                              llm_names_with_qualifiers=with_qualifiers_assertion_llms,
                              llm_names_wout_qualifiers=wout_qualifiers_assertion_llms)
    supported_triples_inst2 = \
        get_supported_triples(x_triples_inst2, ACTION_CATEGORY_ASSERT,
                              llm_names_with_qualifiers=with_qualifiers_assertion_llms,
                              llm_names_wout_qualifiers=wout_qualifiers_assertion_llms)

    if len(supported_triples_inst1) > len(supported_triples_inst2):
        logger.debug('*********************x_resolution**************************')
        logger.debug(f'*****INSTANCE1: {instance1}')
        logger.debug(f'*****')
        logger.debug(f'*****INSTANCE2: {instance2}')
        logger.debug(f'*****')
        logger.debug(f'*****FAVOUR: instance1')
        logger.debug('*********************************************************************')
        return instance1
    elif len(supported_triples_inst2) > len(supported_triples_inst1):
        logger.debug('*********************x_resolution**************************')
        logger.debug(f'*****INSTANCE1: {instance1}')
        logger.debug(f'*****')
        logger.debug(f'*****INSTANCE2: {instance2}')
        logger.debug(f'*****')
        logger.debug(f'*****FAVOUR: instance2')
        logger.debug('*********************************************************************')
        return instance2

    tot_triples_inst1_len = len(x_triples_inst1) + len(e_triples_inst1) + \
                            len(ee_triples_inst1) + len(ee_kg_triples_inst1) + \
                            len(d_triples_inst1)
    tot_triples_inst2_len = len(x_triples_inst2) + len(e_triples_inst2) + \
                            len(ee_triples_inst2) + len(ee_kg_triples_inst2) + \
                            len(d_triples_inst2)
    if tot_triples_inst1_len > tot_triples_inst2_len:
        logger.debug('*********************tot_triples_resolution**************************')
        logger.debug(f'*****INSTANCE1: {instance1}')
        logger.debug(f'*****')
        logger.debug(f'*****INSTANCE2: {instance2}')
        logger.debug(f'*****')
        logger.debug(f'*****FAVOUR: instance1')
        logger.debug('*********************************************************************')
        return instance1
    elif tot_triples_inst2_len > tot_triples_inst1_len:
        logger.debug('*********************tot_triples_resolution**************************')
        logger.debug(f'*****INSTANCE1: {instance1}')
        logger.debug(f'*****')
        logger.debug(f'*****INSTANCE2: {instance2}')
        logger.debug(f'*****')
        logger.debug(f'*****FAVOUR: instance2')
        logger.debug('*********************************************************************')
        return instance2

    if len(instance1['passage']) > len(instance2['passage']):
        return instance1
    elif len(instance2['passage']) > len(instance1['passage']):
        return instance2

    return random.choice([instance1, instance2])


def count_star_groups(text):
    return len(re.findall(r'\*+', text))


counter_ignored = 0


def should_be_added_final_check(instance: Dict):
    global counter_ignored
    nr_star_groups = count_star_groups(text=instance['passage'])
    passage_to_analyze = instance['passage'].split(':', 1)[1].strip()
    passage_title = instance['passage'].split(':', 1)[0].strip()
    # nr_tokens = len(instance['passage'].split(' '))
    tot_cnt = passage_to_analyze.count('|') + passage_to_analyze.count(':')  # + passage_to_analyze.count('=')

    if tot_cnt > 0 and len(passage_to_analyze.split(' ')) / tot_cnt < 3:
        logger.info(
            f'{counter_ignored} -- {len(passage_to_analyze.split(" ")) / tot_cnt:.4f} special_chars: {instance["passage"]}')
        counter_ignored += 1
        return False

    if nr_star_groups > 1:
        nr_tokens = len(passage_to_analyze.split(' '))
        frac_stars = nr_tokens / nr_star_groups
        if 'list of' in passage_title.lower():
            return True
        if '==cast==' in passage_to_analyze.lower() or '== cast ==' in passage_to_analyze.lower():
            return True
        # if frac_stars < 15 and '==see also==' in passage_to_analyze.lower() \
        #         or '== see also ==' in passage_to_analyze.lower():
        # if '==see also==' in passage_to_analyze.lower() \
        #         or '== see also ==' in passage_to_analyze.lower():
        # if frac_stars < 15 and (passage_to_analyze.lower().startswith('==see also==')
        #                         or passage_to_analyze.lower().startswith('== see also ==')):
        if frac_stars < 1 and (passage_to_analyze.lower().startswith('==see also==')
                               or passage_to_analyze.lower().startswith('== see also ==')):
            percentage_english_words = \
                calculate_english_word_percentage(passage_to_analyze.lower())
            logger.info(f'{counter_ignored} to_ignore_stars_see_also: {frac_stars:.4f} '
                        f'en_words: {percentage_english_words:.4f} '
                        f'passage: {instance["passage"]}')
            counter_ignored += 1
            return False
        if frac_stars < 1:
            percentage_english_words = \
                calculate_english_word_percentage(passage_to_analyze.lower())
            logger.info(f'{counter_ignored} to_ignore_only_stars: {frac_stars:.4f} '
                        f'en_words: {percentage_english_words:.4f} '
                        f'passage: {instance["passage"]}')
            counter_ignored += 1
            return False
    else:
        if (passage_to_analyze.lower().startswith('==see also==')
            or passage_to_analyze.lower().startswith('== see also ==')) and \
                len(passage_to_analyze.split(" ")) < 2:
            logger.info(f'{counter_ignored} to_ignore_only_see_also '
                        f'{len(passage_to_analyze.split(" "))} starts_with_see_also: '
                        f'{instance["passage"]}')
            counter_ignored += 1
            return False

    # nr_tkgu_triples = len(instance['tkgu_triples'])

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/s07_reduce_duplicates/20250826/'
                                's07_reduce_duplicates.json',
                        help='The config file that contains all the parameters')

    # removed_triple_actions = {'qualifier_removed_edge', 'removed_edge'}

    ####
    args = parser.parse_args()
    config = json.load(open(args.config_file, 'rt'))
    input_dir = config['input_dir']
    output_dir = config['output_dir']
    output_tmp_dir = config['output_tmp_dir']
    assessor_mappings = config['assessor_mappings']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_tmp_dir, exist_ok=True)
    max_duplicates_per_delta = config['max_duplicates_per_delta']
    ####

    # nr_triples_assessed = 0
    tot_nr_triples = dict()
    nr_triples_not_assessed = dict()
    # nr_instances_assessed_correctly = 0
    nr_instances_not_assessed = 0
    nr_lines_ignored = 0
    tot_nr_lines = 0
    date_to_property_id_to_label = dict()
    known_assessors = set([cka for cka in config['known_assessors']])
    diff_anchor_conf = config['preliminary_pass']['different_anchor']
    same_anchor_conf = config['preliminary_pass']['same_anchor']
    ##### PASS _v6 general deduplication
    logger.info('first_deduplication_pass_beginning')
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename not in tot_nr_triples:
                tot_nr_triples[filename] = 0
            if filename not in nr_triples_not_assessed:
                nr_triples_not_assessed[filename] = 0
            if not filename.endswith('.jsonl'):
                continue
            print(f'====================== {filename} =======================')
            passages_in_file = list()
            file_path = os.path.join(root, filename)
            print('loading instances')
            tmp_file_path = os.path.join(output_tmp_dir, filename)
            if os.path.exists(tmp_file_path):
                print(f'path_exists, ignoring: {tmp_file_path}')
                continue
            for curr_line in open(file_path, 'rt', encoding='utf-8'):
                curr_parsed_line = json.loads(curr_line)
                tot_nr_lines += 1

                #################
                if '{{' in curr_parsed_line['passage']:
                    logger.debug('ignoring_passage_brackets')
                    nr_lines_ignored += 1
                    continue
                if not should_be_added_final_check(instance=curr_parsed_line):
                    nr_lines_ignored += 1
                    continue
                # here we also ignore where there are too many special characters

                ### BEGIN sanity check based on s4b_sanity_check to check emerging and existing
                ### knowledge (triples) do not overlap
                for curr_mention in curr_parsed_line['mentions']:
                    m_start_char = curr_mention['start_char']
                    m_end_char = curr_mention['end_char']
                    assert curr_mention['mention_text'] == curr_parsed_line['passage'][m_start_char:m_end_char]
                #
                tkgu_triples_set = set()
                updated_tkgu_triples = list()
                date_object = datetime.fromtimestamp(curr_parsed_line['delta_timestamps'][0])

                # Format the date as yyyy-mm-dd
                formatted_date = date_object.strftime('%Y-%m-%d')
                if formatted_date not in date_to_property_id_to_label:
                    dictionary_path = os.path.join(config['input_relation_dictionaries'],
                                                   formatted_date,
                                                   'documents.jsonl')

                    date_to_property_id_to_label[formatted_date] = \
                        obtain_property_ids_to_labels_dict(dictionary_path)
                filter_out_new_relation_types(
                    curr_parsed_line,
                    date_to_property_id_to_label[formatted_date]
                )
                d_triples_insts = list()

                ee_triples_insts = list()

                e_triples_insts = list()

                ee_kg_triples_insts = list()

                x_triples_insts = list()
                for curr_tkgu_triple in curr_parsed_line['tkgu_triples']:
                    if curr_tkgu_triple['triple'][0] == curr_tkgu_triple['triple'][2]:
                        logger.debug(f'emerging ignoring {curr_tkgu_triple["triple"]}')
                        continue

                    tot_nr_triples[filename] += 1

                    if len(curr_tkgu_triple['llm_assessment']) == 0:
                        logger.debug('found_not_assessed_triple, ignoring it')
                        nr_triples_not_assessed[filename] += 1
                        logger.debug(f'not_assessed_1: {curr_tkgu_triple}')
                        continue

                    for curr_assessment in curr_tkgu_triple['llm_assessment']:
                        if curr_assessment['llm_name'] in assessor_mappings:
                            curr_assessment['llm_name'] = assessor_mappings[curr_assessment['llm_name']]
                        if curr_assessment['llm_name'] not in known_assessors:
                            logger.error(f'curr_assessment["llm_name"] not recognized: '
                                         f'{curr_assessment["llm_name"]} not in {known_assessors}')
                        assert curr_assessment['llm_name'] in known_assessors

                    tkgu_triples_set.add((
                        curr_tkgu_triple['triple'][0],
                        curr_tkgu_triple['triple'][1],
                        curr_tkgu_triple['triple'][2]
                    ))

                    tkgu_operations = set(curr_tkgu_triple['tkgu_operations'])
                    if 'x-triples' in tkgu_operations:
                        x_triples_insts.append(curr_tkgu_triple)
                    if 'd-triples' in tkgu_operations:
                        d_triples_insts.append(curr_tkgu_triple)
                    if 'ee-triples' in tkgu_operations:
                        ee_triples_insts.append(curr_tkgu_triple)
                    if 'e-triples' in tkgu_operations:
                        e_triples_insts.append(curr_tkgu_triple)
                    if 'ee-kg-triples' in tkgu_operations:
                        ee_kg_triples_insts.append(curr_tkgu_triple)

                    updated_tkgu_triples.append(curr_tkgu_triple)

                # checks that there are not duplicate triples
                assert len(tkgu_triples_set) == len(updated_tkgu_triples)

                curr_parsed_line['tkgu_triples'] = updated_tkgu_triples

                if len(updated_tkgu_triples) == 0:
                    nr_lines_ignored += 1
                    logger.warning(f'{nr_lines_ignored} nr_lines_ignored ignoring the line as no '
                                   f'emerging triples')
                    continue
                emerging_knowledge_triples = [
                    ctr for ctr in updated_tkgu_triples if
                    len(set(ctr['tkgu_operations']).intersection({'e-triples', 'd-triples',
                                                                  'ee-triples', 'ee-kg-triples'})) > 0
                ]
                if len(emerging_knowledge_triples) == 0:
                    nr_lines_ignored += 1
                    logger.warning(f'{nr_lines_ignored} nr_lines_ignored_2 ignoring the line as no '
                                   f'emerging triples')
                    continue

                overlapping_with_x_triples = list()

                for curr_ctr in updated_tkgu_triples:
                    curr_tkgu_ops = set(curr_ctr['tkgu_operations'])
                    if 'x-triples' in curr_tkgu_ops:
                        if len(curr_tkgu_ops.difference({'x-triples'})) > 0:
                            overlapping_with_x_triples.append(curr_ctr)

                assert len(overlapping_with_x_triples) == 0

                prompt_types_to_assessors = config['prompt_type_to_assessors']
                wout_qualifiers_assertion_llms = prompt_types_to_assessors['without_qualifiers'] \
                    [ACTION_CATEGORY_ASSERT]
                wout_qualifiers_deprecation_llms = prompt_types_to_assessors['without_qualifiers'] \
                    [ACTION_CATEGORY_DEPRECATE]
                with_qualifiers_assertion_llms = prompt_types_to_assessors['with_qualifiers'] \
                    [ACTION_CATEGORY_ASSERT]
                with_qualifiers_deprecation_llms = prompt_types_to_assessors['with_qualifiers'] \
                    [ACTION_CATEGORY_DEPRECATE]

                supported_d_triples = \
                    get_supported_triples(d_triples_insts, ACTION_CATEGORY_DEPRECATE,
                                          llm_names_with_qualifiers=with_qualifiers_deprecation_llms,
                                          llm_names_wout_qualifiers=wout_qualifiers_deprecation_llms)

                supported_ee_triples = \
                    get_supported_triples(ee_triples_insts, ACTION_CATEGORY_ASSERT,
                                          llm_names_with_qualifiers=with_qualifiers_assertion_llms,
                                          llm_names_wout_qualifiers=wout_qualifiers_assertion_llms)
                supported_e_triples = \
                    get_supported_triples(e_triples_insts, ACTION_CATEGORY_ASSERT,
                                          llm_names_with_qualifiers=with_qualifiers_assertion_llms,
                                          llm_names_wout_qualifiers=wout_qualifiers_assertion_llms)

                supported_ee_kg_triples = \
                    get_supported_triples(ee_kg_triples_insts, ACTION_CATEGORY_ASSERT,
                                          llm_names_with_qualifiers=with_qualifiers_assertion_llms,
                                          llm_names_wout_qualifiers=wout_qualifiers_assertion_llms)

                if len(supported_d_triples) == 0 and len(supported_e_triples) == 0 and len(supported_ee_triples) == 0 \
                        and len(supported_ee_kg_triples) == 0:
                    nr_lines_ignored += 1
                    continue

                #################

                # for curr_triple in curr_parsed_line['tkgu_triples']:
                #     for curr_llm_assessment in curr_triple['llm_assessment']:
                #         if curr_llm_assessment['llm_name'] in assessor_mappings:
                #             curr_llm_assessment['llm_name'] = \
                #                 assessor_mappings[curr_llm_assessment['llm_name']]
                passages_in_file.append(
                    curr_parsed_line
                )

            logger.info(
                f'percentage_ignored_lines in {filename} : '
                f'{(nr_lines_ignored /
                    tot_nr_lines) * 100:.5f} ')

            print('instances loaded, sorting by passage length')
            # sorts instances by passage length incremental, so
            passages_in_file.sort(key=lambda x: len(x['passage']))
            print('instances sorted, calculating similarities')
            #
            de_duplicated_instances_hash_ids = set()
            instances_to_write = list()
            for curr_passage_idx, curr_instance in enumerate(tqdm(passages_in_file)):
                if curr_instance['hash_id'] in de_duplicated_instances_hash_ids:
                    # print('continuing_instance1')
                    continue
                curr_maximum_jaccard_similarity = 0.0
                curr_minimum_ed_distance = 1.0
                curr_passage = curr_instance['passage']

                curr_anchor_page_qid = curr_instance['anchor_page_qid']
                for curr_instance2 in passages_in_file[curr_passage_idx + 1:]:
                    if curr_instance2['hash_id'] in de_duplicated_instances_hash_ids:
                        # print('continuing_instance2')
                        continue
                    curr_passage2 = curr_instance2['passage']
                    curr_anchor_page_qid2 = curr_instance2['anchor_page_qid']
                    min_length = min(len(curr_passage), len(curr_passage2))
                    max_length = max(len(curr_passage), len(curr_passage2))

                    length_ratio = min_length / max_length
                    if curr_anchor_page_qid2 == curr_anchor_page_qid:
                        if max_length >= same_anchor_conf['long_text_cut']:
                            if length_ratio >= same_anchor_conf['min_len_ratio_long_text']:
                                normalized_ed = normalized_edit_distance(curr_passage, curr_passage2)
                                min_length_passage = curr_passage[:min_length]
                                min_length_passage2 = curr_passage2[:min_length]
                                normalized_ed_min_length = normalized_edit_distance(min_length_passage,
                                                                                    min_length_passage2)
                                if normalized_ed <= same_anchor_conf['min_edit_distance_long_text'] or \
                                        normalized_ed_min_length <= same_anchor_conf['ml_edit_distance_long_text']:
                                    logger.debug('*********************long_text_duplicate***********************')
                                    logger.debug(f'duplicate_detected_instance1: {curr_instance}')
                                    logger.debug(f'duplicate_detected_instance2: {curr_instance2}')
                                    curr_instance = deduplicate_instance(instance1=curr_instance,
                                                                         instance2=curr_instance2,
                                                                         config=config)
                                    curr_passage = curr_instance['passage']

                                    logger.debug(f'SOLVED IN FAVOR OF: {curr_instance}')
                                    logger.debug('***************************************************************')
                                    de_duplicated_instances_hash_ids.add(curr_instance2['hash_id'])
                        else:
                            normalized_ed = normalized_edit_distance(curr_passage, curr_passage2)
                            min_length_passage = curr_passage[:min_length]
                            min_length_passage2 = curr_passage2[:min_length]
                            normalized_ed_min_length = normalized_edit_distance(min_length_passage,
                                                                                min_length_passage2)
                            if normalized_ed_min_length <= same_anchor_conf['ml_edit_distance_short_text']:
                                curr_instance = deduplicate_instance(instance1=curr_instance,
                                                                     instance2=curr_instance2,
                                                                     config=config)
                                curr_passage = curr_instance['passage']
                                de_duplicated_instances_hash_ids.add(curr_instance2['hash_id'])
                            elif length_ratio >= same_anchor_conf['min_len_ratio_short_text']:
                                if normalized_ed <= same_anchor_conf['min_edit_distance_short_text'] or \
                                        normalized_ed_min_length <= same_anchor_conf['ml_edit_distance_short_text']:
                                    logger.debug('*********************short_text_duplicate***********************')
                                    logger.debug(f'duplicate_detected_instance1: {curr_instance}')
                                    logger.debug(f'duplicate_detected_instance2: {curr_instance2}')
                                    curr_instance = deduplicate_instance(instance1=curr_instance,
                                                                         instance2=curr_instance2,
                                                                         config=config)
                                    curr_passage = curr_instance['passage']
                                    logger.debug(f'SOLVED IN FAVOR OF: {curr_instance}')
                                    logger.debug('***************************************************************')
                                    de_duplicated_instances_hash_ids.add(curr_instance2['hash_id'])
                    else:
                        if length_ratio > diff_anchor_conf['min_len_ratio']:
                            normalized_ed = normalized_edit_distance(curr_passage, curr_passage2)
                            if normalized_ed <= diff_anchor_conf['min_edit_distance']:
                                logger.debug(
                                    '*********************different_anchor_text_duplicate***********************')
                                logger.debug(f'duplicate_detected_instance1: {curr_instance}')
                                logger.debug(f'duplicate_detected_instance2: {curr_instance2}')
                                curr_instance = deduplicate_instance(instance1=curr_instance,
                                                                     instance2=curr_instance2,
                                                                     config=config)
                                curr_passage = curr_instance['passage']

                                logger.debug(f'SOLVED IN FAVOR OF: {curr_instance}')
                                logger.debug(
                                    '**************************************************************************')
                                de_duplicated_instances_hash_ids.add(curr_instance2['hash_id'])

                    #
                print('**************************')
                print(f'curr_passage {curr_passage_idx}: {curr_instance["passage"]}')
                print('**************************')
                instances_to_write.append(curr_instance)

            random.shuffle(instances_to_write)

            with open(tmp_file_path, 'wt', encoding='utf-8') as out_tmp_file:
                for curr_instance in instances_to_write:
                    out_tmp_file.write(json.dumps(curr_instance, ensure_ascii=False) + '\n')

    logger.info('continuing_with_general_deduplication')
    ##### END pass _v6 general deduplication
    files_in_input = list_files(output_tmp_dir)
    for curr_input_file in files_in_input:
        logger.info(f'reading {curr_input_file}')
        curr_file_path = os.path.join(output_tmp_dir, curr_input_file)
        assert curr_file_path.endswith('.jsonl')
        output_file_name = curr_input_file
        output_file_path = os.path.join(output_dir, output_file_name)
        updates_to_nr_instances = dict()
        nr_lines_in_file = count_lines(curr_file_path)
        with (open(output_file_path, 'wt', encoding='utf-8') as output_file):
            for curr_line in tqdm(open(curr_file_path, 'rt', encoding='utf-8'),
                                  desc='iteration over curr_file_path',
                                  total=nr_lines_in_file):
                parsed_line = json.loads(curr_line)
                tot_nr_lines += 1

                nr_llm_positive_emerging = 0
                nr_llm_positive_emerging_additions = 0
                nr_llm_positive_emerging_removals = 0
                nr_llm_positive_matching_to_kg_removals = 0
                nr_llm_positive_matching_to_kg_additions = 0

                emerging_triple_supported = False
                lst_emerging_knowledge = list()
                lst_existing_knowledge = list()
                lst_ee_kg_knowledge = list()
                curr_existing_knowledge_assessments = list()
                for curr_triple in parsed_line['tkgu_triples']:
                    curr_tkgu_operations = set(curr_triple['tkgu_operations'])

                    is_curr_triple_emerging_knowledge = \
                        len(curr_tkgu_operations. \
                            intersection({'e-triples', 'd-triples', 'ee-triples'})) > 0
                    is_curr_triple_existing_knowledge = \
                        len(curr_tkgu_operations.intersection({'x-triples'})) > 0
                    is_curr_triple_ee_kg_knowledge = \
                        len(curr_tkgu_operations.intersection({'ee-kg-triples'})) > 0

                    if is_curr_triple_emerging_knowledge:
                        lst_emerging_knowledge.append(curr_triple)
                        assessment_addition = [asse for asse in curr_triple['llm_assessment'] \
                                               if
                                               asse['llm_prompt_type'] == 'triple_assertion' and
                                               asse['llm_name'] == config[
                                                   'llm_assessor_addition_triples']]
                        not_assessed = True
                        if len(assessment_addition) == 0:
                            assessment_addition = dict()
                        else:
                            not_assessed = False
                            assessment_addition = assessment_addition[0]
                            logger.debug(f'assessment_addition_found: {assessment_addition}')

                        assessment_deprecation = [asse for asse in curr_triple['llm_assessment'] \
                                                  if
                                                  asse['llm_prompt_type'] == 'triple_deprecation' and
                                                  asse['llm_name'] == config['llm_assessor_deprecation_triples']]
                        if len(assessment_deprecation) == 0:
                            assessment_deprecation = dict()
                        else:
                            not_assessed = False
                            assessment_deprecation = assessment_deprecation[0]

                        if not_assessed:
                            logger.debug(f'not_assessed_2: {curr_triple}')
                            nr_triples_not_assessed[curr_input_file] += 1
                        # else:
                        #     nr_triples_assessed += 1
                        #
                        if len(assessment_addition) > 0 and assessment_addition['llm_assessment']:
                            nr_llm_positive_emerging += 1
                            nr_llm_positive_emerging_additions += 1
                            emerging_triple_supported = True
                        #
                        if len(assessment_deprecation) > 0 and assessment_deprecation['llm_assessment']:
                            if not emerging_triple_supported:
                                nr_llm_positive_emerging += 1
                            emerging_triple_supported = True
                            nr_llm_positive_emerging_removals += 1

                    if is_curr_triple_ee_kg_knowledge:
                        lst_ee_kg_knowledge.append(curr_triple)

                        assessment_addition = [asse for asse in curr_triple['llm_assessment'] \
                                               if
                                               asse['llm_prompt_type'] == 'triple_assertion' and
                                               asse['llm_name'] == config[
                                                   'llm_assessor_addition_triples']]
                        not_assessed = True
                        if len(assessment_addition) == 0:
                            assessment_addition = dict()
                        else:
                            assessment_addition = assessment_addition[0]
                            logger.debug(f'assessment_addition_found: {assessment_addition}')
                            #
                            if assessment_addition['llm_assessment']:
                                nr_llm_positive_matching_to_kg_additions += 1

                            not_assessed = False

                        assessment_deprecation = [asse for asse in curr_triple['llm_assessment'] \
                                                  if
                                                  asse['llm_prompt_type'] == 'triple_deprecation' and
                                                  asse['llm_name'] == config[
                                                      'llm_assessor_deprecation_triples']]
                        if len(assessment_deprecation) == 0:
                            assessment_deprecation = dict()
                        else:
                            assessment_deprecation = assessment_deprecation[0]
                            #
                            if assessment_deprecation['llm_assessment']:
                                nr_llm_positive_matching_to_kg_removals += 1
                            not_assessed = False
                        if not_assessed:
                            logger.debug(f'not_assessed_3: {curr_triple}')
                            nr_triples_not_assessed[curr_input_file] += 1
                        # else:
                        #     nr_triples_assessed += 1

                    if is_curr_triple_existing_knowledge:
                        lst_existing_knowledge.append(curr_triple)
                        not_assessed = True
                        assessment_addition = [asse for asse in curr_triple['llm_assessment'] \
                                               if
                                               asse['llm_prompt_type'] == 'triple_assertion' and
                                               asse['llm_name'] == config[
                                                   'llm_assessor_addition_triples']]
                        if len(assessment_addition) == 0:
                            assessment_addition = dict()
                        else:
                            assessment_addition = assessment_addition[0]
                            not_assessed = False
                        if not_assessed:
                            # logger.debug(f'not_assessed_4: {curr_triple}')
                            print(f'not_assessed_4: {curr_triple}')
                            nr_triples_not_assessed[curr_input_file] += 1
                        assert not not_assessed
                        # else:
                        #     nr_triples_assessed += 1

                nr_llm_positive_existing = \
                    len([cek for cek in curr_existing_knowledge_assessments if cek])
                #################################
                parsed_line['nr_llm_positive_emerging_removals'] = nr_llm_positive_emerging_removals
                parsed_line['nr_llm_positive_emerging_additions'] = nr_llm_positive_emerging_additions
                parsed_line['nr_llm_positive_emerging'] = nr_llm_positive_emerging
                parsed_line['nr_llm_positive_existing'] = nr_llm_positive_existing

                parsed_line['nr_llm_positive_matching_to_kg_removals'] = nr_llm_positive_matching_to_kg_removals
                parsed_line['nr_llm_positive_matching_to_kg_additions'] = nr_llm_positive_matching_to_kg_additions

                ###
                sorted_emerging_knowledge = \
                    sorted(lst_emerging_knowledge, key=lambda entry:
                    f'{entry["triple"][0]}_{entry["triple"][1]}_{entry["triple"][2]}')

                curr_entry = ''
                for curr_emerging_knowledge in sorted_emerging_knowledge:
                    curr_entry += (f'em{curr_emerging_knowledge["triple"][0]}_'
                                   f'{curr_emerging_knowledge["triple"][1]}_'
                                   f'{curr_emerging_knowledge["triple"][2]}')
                    curr_entry_sorted_actions = sorted(curr_emerging_knowledge['tkgu_operations'])
                    curr_entry += ''.join(curr_entry_sorted_actions)
                sorted_existing_knowledge = \
                    sorted(lst_existing_knowledge, key=lambda entry:
                    f'{entry["triple"][0]}_{entry["triple"][1]}_{entry["triple"][2]}')

                for curr_existing_knowledge in sorted_existing_knowledge:
                    curr_entry += (f'ex{curr_existing_knowledge["triple"][0]}_'
                                   f'{curr_existing_knowledge["triple"][1]}_'
                                   f'{curr_existing_knowledge["triple"][2]}')

                sorted_matched_triples_entities_to_kg = \
                    sorted(lst_ee_kg_knowledge, key=lambda entry:
                    f'{entry["triple"][0]}_{entry["triple"][1]}_{entry["triple"][2]}')

                for curr_matched_triples_entities_to_kg in sorted_matched_triples_entities_to_kg:
                    curr_entry += (f'cm{curr_matched_triples_entities_to_kg["triple"][0]}_'
                                   f'{curr_matched_triples_entities_to_kg["triple"][1]}_'
                                   f'{curr_matched_triples_entities_to_kg["triple"][2]}')

                if curr_entry not in updates_to_nr_instances:
                    updates_to_nr_instances[curr_entry] = {
                        'instances': list(),
                        'nr_of_instances': 0
                    }
                updates_to_nr_instances[curr_entry]['nr_of_instances'] += 1
                updates_to_nr_instances[curr_entry]['instances'].append(parsed_line)
                #

            for curr_entry, curr_instances_details in tqdm(updates_to_nr_instances.items(),
                                                           desc='iteration over updates_to_nr_instances',
                                                           total=len(updates_to_nr_instances)):
                sorted_candidate_instances_to_save = list()
                curr_instances = updates_to_nr_instances[curr_entry]['instances']

                if not config['pick_random_duplicates']:
                    # first checks the instances with positive removal if there is any, bases on
                    # that first, working only with the ones with the highest number of supported
                    # removals
                    curr_instances_with_llm_supported_removals = [ci for ci in curr_instances
                                                                  if ci['nr_llm_positive_emerging_removals'] > 0]
                    curr_instances_so_far = list()
                    if len(curr_instances_with_llm_supported_removals) > 0:
                        curr_instances_with_llm_supported_removals.sort(key=lambda x:
                        (x['nr_llm_positive_emerging_removals'],
                         x['nr_llm_positive_emerging_additions'],
                         x['nr_llm_positive_matching_to_kg_additions'],
                         x['nr_llm_positive_existing'],
                         len(x['passage'])), reverse=True)

                        ########################################
                        # BEGIN: new code, which builds a dataset in such a way that the passages are as diverse of possible
                        # measuring minimum edit distance and jaccard similarity between the passages representing same
                        # triples
                        for curr_instance in curr_instances_with_llm_supported_removals:
                            if len(curr_instances_so_far) > max_duplicates_per_delta:
                                break
                            if len(curr_instances_so_far) == 0:
                                curr_instances_so_far.append(curr_instance)
                            else:
                                curr_maximum_jaccard_similarity = 0.0
                                curr_minimum_ed_distance = 1.0
                                for curr_instance_so_far in curr_instances_so_far:
                                    j_similarity = jaccard_similarity(curr_instance_so_far['passage'],
                                                                      curr_instance['passage'])
                                    normalized_ed = normalized_edit_distance(curr_instance_so_far['passage'],
                                                                             curr_instance['passage'])

                                    if j_similarity > curr_maximum_jaccard_similarity:
                                        curr_maximum_jaccard_similarity = j_similarity
                                    if normalized_ed < curr_minimum_ed_distance:
                                        curr_minimum_ed_distance = normalized_ed

                                    if j_similarity < config['maximum_jaccard_similarity'] and \
                                            normalized_ed > config['minimum_edit_distance']:
                                        logger.debug(
                                            '===============START nr_llm_positive_emerging_removals================')
                                        logger.debug(f'PASSAGE1: {curr_instance_so_far["passage"]}')
                                        logger.debug('----')
                                        logger.debug(f'PASSAGE2: {curr_instance["passage"]}')
                                        logger.debug('----')
                                        logger.debug(f'jaccard similarity: {j_similarity}')
                                        logger.debug('----')
                                        logger.debug(f'normalized ed dist: {normalized_ed}')
                                        logger.debug(
                                            '===============START nr_llm_positive_emerging_removals================')

                                if curr_maximum_jaccard_similarity < config['maximum_jaccard_similarity'] and \
                                        curr_minimum_ed_distance > config['minimum_edit_distance']:
                                    curr_instances_so_far.append(curr_instance)
                        ########################################
                    #
                    if len(curr_instances_so_far) < max_duplicates_per_delta:
                        curr_instances_with_llm_supported = [ci for ci in curr_instances
                                                             if (ci['nr_llm_positive_emerging_removals'] == 0
                                                                 and
                                                                 ci['nr_llm_positive_emerging_additions'] > 0)]

                        curr_instances_with_llm_supported.sort(key=lambda x:
                        (x['nr_llm_positive_emerging_additions'],
                         x['nr_llm_positive_matching_to_kg_additions'],
                         x['nr_llm_positive_existing'],
                         len(x['passage'])
                         ),

                                                               reverse=True)

                        ########################################
                        # BEGIN: new code, which builds a dataset in such a way that the passages are as diverse of possible
                        # measuring minimum edit distance and jaccard similarity between the passages representing same
                        # triples
                        for curr_instance in curr_instances_with_llm_supported:
                            if len(curr_instances_so_far) > max_duplicates_per_delta:
                                break
                            if len(curr_instances_so_far) == 0:
                                curr_instances_so_far.append(curr_instance)
                            else:
                                curr_maximum_jaccard_similarity = 0.0
                                curr_minimum_ed_distance = 1.0
                                for curr_instance_so_far in curr_instances_so_far:
                                    j_similarity = jaccard_similarity(curr_instance_so_far['passage'],
                                                                      curr_instance['passage'])
                                    normalized_ed = normalized_edit_distance(curr_instance_so_far['passage'],
                                                                             curr_instance['passage'])
                                    if j_similarity > curr_maximum_jaccard_similarity:
                                        curr_maximum_jaccard_similarity = j_similarity
                                    if normalized_ed < curr_minimum_ed_distance:
                                        curr_minimum_ed_distance = normalized_ed

                                    if j_similarity < config['maximum_jaccard_similarity'] and \
                                            normalized_ed > config['minimum_edit_distance']:
                                        logger.debug(
                                            '===============START nr_llm_positive_emerging_additions================')
                                        logger.debug(f'PASSAGE1: {curr_instance_so_far["passage"]}')
                                        logger.debug('----')
                                        logger.debug(f'PASSAGE2: {curr_instance["passage"]}')
                                        logger.debug('----')
                                        logger.debug(f'jaccard similarity: {j_similarity}')
                                        logger.debug('----')
                                        logger.debug(f'normalized ed dist: {normalized_ed}')
                                        logger.debug(
                                            '================END nr_llm_positive_emerging_additions ================')
                                if curr_maximum_jaccard_similarity < config['maximum_jaccard_similarity'] and \
                                        curr_minimum_ed_distance > config['minimum_edit_distance']:
                                    curr_instances_so_far.append(curr_instance)
                        ########################################
                    if len(curr_instances_so_far) < max_duplicates_per_delta:
                        curr_instances_with_llm_supported = \
                            [ci for ci in curr_instances
                             if (ci['nr_llm_positive_emerging_removals'] == 0
                                 and
                                 ci['nr_llm_positive_emerging_additions'] == 0
                                 and
                                 ci['nr_llm_positive_matching_to_kg_additions'] > 0
                                 # len(ci['matched_triples_entities_to_kg']) > 0
                                 )
                             ]
                        curr_instances_with_llm_supported.sort(key=lambda x:
                        (x['nr_llm_positive_matching_to_kg_removals'],
                         x['nr_llm_positive_matching_to_kg_additions'],
                         len(x['passage'])),
                                                               reverse=True)
                        for curr_instance in curr_instances_with_llm_supported:
                            if len(curr_instances_so_far) > max_duplicates_per_delta:
                                break
                            if len(curr_instances_so_far) == 0:
                                curr_instances_so_far.append(curr_instance)
                            else:
                                curr_maximum_jaccard_similarity = 0.0
                                curr_minimum_ed_distance = 1.0
                                for curr_instance_so_far in curr_instances_so_far:
                                    j_similarity = jaccard_similarity(
                                        curr_instance_so_far['passage'],
                                        curr_instance['passage']
                                    )
                                    normalized_ed = normalized_edit_distance(
                                        curr_instance_so_far['passage'],
                                        curr_instance['passage']
                                    )

                                    if j_similarity > curr_maximum_jaccard_similarity:
                                        curr_maximum_jaccard_similarity = j_similarity
                                    if normalized_ed < curr_minimum_ed_distance:
                                        curr_minimum_ed_distance = normalized_ed

                                    if j_similarity < config['maximum_jaccard_similarity'] and \
                                            normalized_ed > config['minimum_edit_distance']:
                                        logger.debug(
                                            '===============START matched_triples_entities_to_kg================')
                                        logger.debug(f'PASSAGE1: {curr_instance_so_far["passage"]}')
                                        logger.debug('----')
                                        logger.debug(f'PASSAGE2: {curr_instance["passage"]}')
                                        logger.debug('----')
                                        logger.debug(f'jaccard similarity: {j_similarity}')
                                        logger.debug('----')
                                        logger.debug(f'normalized ed dist: {normalized_ed}')
                                        logger.debug(
                                            '================END matched_triples_entities_to_kg===============')
                                if curr_maximum_jaccard_similarity < config['maximum_jaccard_similarity'] and \
                                        curr_minimum_ed_distance > config['minimum_edit_distance']:
                                    curr_instances_so_far.append(curr_instance)
                #
                else:
                    curr_instances_so_far = random.sample(curr_instances, max_duplicates_per_delta)

                random.shuffle(curr_instances_so_far)

                for curr_instance in curr_instances_so_far:
                    if 'interval_id' in curr_instance:
                        del curr_instance['interval_id']
                    del curr_instance['nr_llm_positive_emerging_removals']
                    del curr_instance['nr_llm_positive_emerging_additions']
                    del curr_instance['nr_llm_positive_emerging']
                    del curr_instance['nr_llm_positive_existing']
                    del curr_instance['nr_llm_positive_matching_to_kg_removals']
                    del curr_instance['nr_llm_positive_matching_to_kg_additions']

                    output_file.write(json.dumps(curr_instance) + '\n')
            if tot_nr_triples[curr_input_file] > 0:
                logger.info(
                    f'percentage not assessed triples'
                    f' {nr_triples_not_assessed[curr_input_file]} out of {tot_nr_triples[curr_input_file]}: '
                    f'{(nr_triples_not_assessed[curr_input_file] /
                        (tot_nr_triples[curr_input_file])) * 100:.5f} ')
            # logger.info(
            #     f'percentage not assessed instances -- triples: '
            #     f'{(nr_instances_not_assessed /
            #         (nr_instances_not_assessed + nr_instances_assessed_correctly)) * 100:.5f} '
            #     f' -- '
            #     f'{(nr_triples_not_assessed /
            #         (nr_triples_not_assessed + nr_triples_assessed_correctly)) * 100:.5f} ')
    print('================JOB FINISHED BYE BYE================')
