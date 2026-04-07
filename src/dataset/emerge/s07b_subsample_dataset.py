# 28.08.2025 The goal of this function is to obtain a subsampled dataset
# on which we will test the models.

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


def get_llm_assessment(triple,
                       llm_assessor_name,
                       llm_prompt_type,
                       hash_id):
    """

    :param triple:
    :param llm_assessor_name:
    :param llm_prompt_type: 'triple_deprecation' or 'triple_assessment'
    :return:
    """
    assessment = [ct for ct in triple['llm_assessment'] if
                  ct['llm_name'] == llm_assessor_name and \
                  ct['llm_prompt_type'] == llm_prompt_type]
    if len(assessment) > 0:
        return assessment[0]['llm_assessment']
    else:
        print(f'ERROR for triple {triple} can not find '
              f'llm_assessor_name {llm_assessor_name} and '
              f'llm_prompt_type {llm_prompt_type} '
              f'hash_id {hash_id}')
        return False


def is_valid_triple(curr_triple, llm_assessors, hash_id):
    _contains_valid_ee_kg_triple = False
    is_asserted = True
    for curr_assessor in llm_assessors:
        llm_assessment_r = get_llm_assessment(
            triple=curr_triple,
            llm_assessor_name=curr_assessor,
            llm_prompt_type=ACTION_CATEGORY_ASSERT,
            hash_id=hash_id
        )
        if not llm_assessment_r:
            is_asserted = False
    if is_asserted:
        _contains_valid_ee_kg_triple = True
    #
    return _contains_valid_ee_kg_triple


def is_valid_d_triple(curr_triple, hash_id):
    _contains_valid_d_triple = False
    if len(curr_triple['qualifier_info']) > 0:
        is_asserted = True
        for curr_assessor in d_triple_llm_assessors_names_qualif:
            llm_assessment_r = get_llm_assessment(
                triple=curr_triple,
                llm_assessor_name=curr_assessor,
                llm_prompt_type=ACTION_CATEGORY_DEPRECATE,
                hash_id=hash_id
            )
            if not llm_assessment_r:
                is_asserted = False
        if is_asserted:
            _contains_valid_d_triple = True
    else:
        is_asserted = True
        for curr_assessor in d_triple_llm_assessors_names_no_qualif:
            llm_assessment_r = get_llm_assessment(
                triple=curr_triple,
                llm_assessor_name=curr_assessor,
                llm_prompt_type=ACTION_CATEGORY_DEPRECATE,
                hash_id=hash_id
            )
            if not llm_assessment_r:
                is_asserted = False
        if is_asserted:
            _contains_valid_d_triple = True
    return _contains_valid_d_triple


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
    # parser = argparse.ArgumentParser()
    # Aspects to account during subsampling:
    #  1. all d-triples in the subsampled dataset
    #  2. nr of triples to subsample
    #  3. First subsample the ones that have additions and removals assessed positively by llms.
    #  4. Maybe focus on the ones that have high number of operations associated with text.
    #  5. Maybe increase the minimum size of the text (not for removals).
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/s07b_subsample_dataset/20250828/'
                                's07b_subsample_dataset.json',
                        help='The config file that contains all the parameters')
    ######
    args = parser.parse_args()
    config = json.load(open(args.config_file, 'rt'))

    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    input_dir = config['input_dir']
    d_triple_llm_assessors_names_no_qualif = config['d_triple_llm_assessors_names_no_qualif']
    d_triple_llm_assessors_names_qualif = config['d_triple_llm_assessors_names_qualif']
    llm_ee_kg_assessors = config['llm_ee_kg_assessors']
    llm_ee_assessors = config['llm_ee_assessors']
    llm_e_assessors = config['llm_e_assessors']
    min_nr_d_instances = config['min_nr_d_instances']
    min_nr_ee_kg_instances = config['min_nr_ee_kg_instances']
    min_nr_e_instances = config['min_nr_e_instances']
    min_nr_ee_instances = config['min_nr_ee_instances']
    subsampled_hash_ids = set()
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if not filename.endswith('.jsonl'):
                continue
            print(f'====================== {filename} =======================')
            in_file = os.path.join(root, filename)
            nr_subsampled_d_instances = 0
            nr_subsampled_ee_kg_instances = 0
            nr_subsampled_ee_instances = 0
            nr_subsampled_e_instances = 0
            subsampled_instances = list()
            # adds d-triples
            with open(in_file, 'r', encoding='utf-8') as infile:
                print(f'subsampling d_triples from {filename}')
                for curr_line in infile:
                    if len(subsampled_instances) >= config['nr_samples_per_delta']:
                        break
                    if nr_subsampled_d_instances >= min_nr_d_instances:
                        break
                    parsed_line = json.loads(curr_line)
                    if not should_be_added_final_check(instance=parsed_line):
                        logger.debug(f'ignoring_parsed_line passage: '
                                     f'{parsed_line["passage"]} '
                                     f'and a complete parsed_line: {parsed_line}')
                        continue
                    # if contains d_triple supported by llm
                    contains_valid_d_triple = False
                    contains_valid_ee_kg_triple = False
                    contains_valid_e_triple = False
                    contains_valid_ee_triple = False
                    for curr_triple in parsed_line['tkgu_triples']:
                        curr_tkgu_operations = set(curr_triple['tkgu_operations'])
                        if not contains_valid_d_triple and \
                                'd-triples' in curr_tkgu_operations:
                            contains_valid_d_triple = is_valid_d_triple(
                                curr_triple=curr_triple,
                                hash_id=parsed_line['hash_id']
                            )
                        #
                        if not contains_valid_ee_kg_triple and \
                                'ee-kg-triples' in curr_tkgu_operations:
                            contains_valid_ee_kg_triple = is_valid_triple(
                                curr_triple=curr_triple,
                                llm_assessors=llm_ee_kg_assessors,
                                hash_id=parsed_line['hash_id']
                            )  #
                        if not contains_valid_ee_triple and \
                                'ee-triples' in curr_tkgu_operations:
                            contains_valid_ee_triple = is_valid_triple(
                                curr_triple=curr_triple,
                                llm_assessors=llm_ee_assessors,
                                hash_id=parsed_line['hash_id']
                            )
                        if not contains_valid_e_triple and \
                                'e-triples' in curr_tkgu_operations:
                            contains_valid_e_triple = is_valid_triple(
                                curr_triple=curr_triple,
                                llm_assessors=llm_e_assessors,
                                hash_id=parsed_line['hash_id']
                            )

                    if contains_valid_d_triple:
                        nr_subsampled_d_instances += 1
                        if contains_valid_ee_kg_triple:
                            nr_subsampled_ee_kg_instances += 1
                        if contains_valid_e_triple:
                            nr_subsampled_e_instances += 1
                        if contains_valid_ee_triple:
                            nr_subsampled_ee_instances += 1

                        subsampled_instances.append(parsed_line)
                        subsampled_hash_ids.add(parsed_line['hash_id'])
                        # You can replace the above with your own processing logic
            #######
            with open(in_file, 'r', encoding='utf-8') as infile:
                print(f'subsampling ee_kg_triples from {filename}')
                for curr_line in infile:
                    if len(subsampled_instances) >= config['nr_samples_per_delta']:
                        break
                    if nr_subsampled_ee_kg_instances >= min_nr_ee_kg_instances:
                        break
                    parsed_line = json.loads(curr_line)
                    if parsed_line['hash_id'] in subsampled_hash_ids:
                        continue
                    if not should_be_added_final_check(instance=parsed_line):
                        logger.debug(f'ignoring_parsed_line passage: '
                                     f'{parsed_line["passage"]} '
                                     f'and a complete parsed_line: {parsed_line}')
                        continue

                    contains_valid_ee_kg_triple = False
                    contains_valid_ee_triple = False
                    contains_valid_e_triple = False
                    for curr_triple in parsed_line['tkgu_triples']:
                        curr_tkgu_operations = set(curr_triple['tkgu_operations'])

                        if not contains_valid_ee_kg_triple and \
                                'ee-kg-triples' in curr_tkgu_operations:
                            contains_valid_ee_kg_triple = is_valid_triple(
                                curr_triple=curr_triple,
                                llm_assessors=llm_ee_kg_assessors,
                                hash_id=parsed_line['hash_id']
                            )
                        if not contains_valid_ee_triple and \
                                'ee-triples' in curr_tkgu_operations:
                            contains_valid_ee_triple = is_valid_triple(
                                curr_triple=curr_triple,
                                llm_assessors=llm_ee_assessors,
                                hash_id=parsed_line['hash_id']
                            )
                        if not contains_valid_e_triple and \
                                'e-triples' in curr_tkgu_operations:
                            contains_valid_e_triple = is_valid_triple(
                                curr_triple=curr_triple,
                                llm_assessors=llm_e_assessors,
                                hash_id=parsed_line['hash_id']
                            )

                    if contains_valid_ee_kg_triple:
                        nr_subsampled_ee_kg_instances += 1
                        subsampled_instances.append(parsed_line)
                        subsampled_hash_ids.add(parsed_line['hash_id'])
                        if contains_valid_e_triple:
                            nr_subsampled_e_instances += 1
                        if contains_valid_ee_triple:
                            nr_subsampled_ee_instances += 1

                        # You can replace the above with your own processing logic

            ######
            with open(in_file, 'r', encoding='utf-8') as infile:
                print(f'subsampling ee_triples from {filename}')
                for curr_line in infile:
                    if len(subsampled_instances) >= config['nr_samples_per_delta']:
                        break
                    if nr_subsampled_ee_instances >= min_nr_ee_instances:
                        break
                    parsed_line = json.loads(curr_line)
                    if parsed_line['hash_id'] in subsampled_hash_ids:
                        continue

                    if not should_be_added_final_check(instance=parsed_line):
                        logger.debug(f'ignoring_parsed_line passage: '
                                     f'{parsed_line["passage"]} '
                                     f'and a complete parsed_line: {parsed_line}')
                        continue

                    contains_valid_e_triple = False
                    contains_valid_ee_triple = False
                    for curr_triple in parsed_line['tkgu_triples']:
                        curr_tkgu_operations = set(curr_triple['tkgu_operations'])

                        if not contains_valid_ee_triple and \
                                'ee-triples' in curr_tkgu_operations:
                            contains_valid_ee_triple = is_valid_triple(
                                curr_triple=curr_triple,
                                llm_assessors=llm_ee_assessors,
                                hash_id=parsed_line['hash_id']
                            )
                        if not contains_valid_e_triple and \
                                'e-triples' in curr_tkgu_operations:
                            contains_valid_e_triple = is_valid_triple(
                                curr_triple=curr_triple,
                                llm_assessors=llm_e_assessors,
                                hash_id=parsed_line['hash_id']
                            )

                    if contains_valid_ee_triple:
                        nr_subsampled_ee_instances += 1
                        subsampled_instances.append(parsed_line)
                        subsampled_hash_ids.add(parsed_line['hash_id'])
                        if contains_valid_e_triple:
                            nr_subsampled_e_instances += 1
                        # You can replace the above with your own processing logic
            #####
            with open(in_file, 'r', encoding='utf-8') as infile:
                print(f'subsampling ee_triples from {filename}')
                for curr_line in infile:
                    if len(subsampled_instances) >= config['nr_samples_per_delta']:
                        break
                    if nr_subsampled_e_instances >= min_nr_e_instances:
                        break
                    parsed_line = json.loads(curr_line)
                    if parsed_line['hash_id'] in subsampled_hash_ids:
                        continue

                    if not should_be_added_final_check(instance=parsed_line):
                        logger.debug(f'ignoring_parsed_line passage: '
                                     f'{parsed_line["passage"]} '
                                     f'and a complete parsed_line: {parsed_line}')
                        continue

                    contains_valid_e_triple = False
                    for curr_triple in parsed_line['tkgu_triples']:
                        curr_tkgu_operations = set(curr_triple['tkgu_operations'])

                        if not contains_valid_e_triple and \
                                'e-triples' in curr_tkgu_operations:
                            contains_valid_e_triple = is_valid_triple(
                                curr_triple=curr_triple,
                                llm_assessors=llm_e_assessors,
                                hash_id=parsed_line['hash_id']
                            )

                    if contains_valid_e_triple:
                        nr_subsampled_e_instances += 1
                        subsampled_instances.append(parsed_line)
                        subsampled_hash_ids.add(parsed_line['hash_id'])

            #####
            # adds other triples
            print(f'subsampling other triples from {filename}')
            print(f'len(subsampled_instances) when we got to final subsampler: '
                  f'{len(subsampled_instances)}')
            with open(in_file, 'r', encoding='utf-8') as infile:
                for curr_line in infile:
                    if len(subsampled_instances) >= config['nr_samples_per_delta']:
                        break
                    parsed_line = json.loads(curr_line)
                    ###
                    if not should_be_added_final_check(instance=parsed_line):
                        logger.debug(f'ignoring_parsed_line passage: '
                                     f'{parsed_line["passage"]} '
                                     f'and a complete parsed_line: {parsed_line}')
                        continue
                    ###
                    if parsed_line['hash_id'] in subsampled_hash_ids:
                        continue
                    subsampled_instances.append(parsed_line)
            print(f'nr_subsampled_instances: {len(subsampled_instances)} ------ '
                  f'nr_subsampled_d_instances: {nr_subsampled_d_instances} '
                  f'nr_subsampled_ee_kg_instances: {nr_subsampled_ee_kg_instances} '
                  f'nr_subsampled_ee_instances: {nr_subsampled_ee_instances} '
                  f'nr_subsampled_e_instances: {nr_subsampled_e_instances}')
            print('=================================================================')
            # shuffles again
            print('=====BEGIN writing subsampled to file====')
            random.shuffle(subsampled_instances)
            # Preserve snapshot_*/ directory structure from input if present
            rel_dir = os.path.relpath(root, input_dir)
            if rel_dir != '.':
                out_subdir = os.path.join(output_dir, rel_dir)
                os.makedirs(out_subdir, exist_ok=True)
            else:
                out_subdir = output_dir
            out_file = os.path.join(out_subdir, filename.replace('interesting_snippets',
                                                                 'delta_'))
            with open(out_file, 'wt', encoding='utf-8') as outfile:
                for curr_subsampled_instance in subsampled_instances:
                    outfile.write(json.dumps(curr_subsampled_instance,
                                             ensure_ascii=False) + '\n')
            print('=====END writing subsampled to file====')
    print('================JOB FINISHED BYE BYE================')
