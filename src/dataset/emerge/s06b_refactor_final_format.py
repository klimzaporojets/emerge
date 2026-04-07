# Sanity check such as no overlap between existing and emerging knowledge

import argparse
import base64
import copy
import hashlib
import json
import logging
import os
import os.path
import re
from pathlib import Path
from typing import Dict, Tuple, Any, Set

from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)


def get_triples_to_assessment_in_prompt(p_llm_prompt: str,
                                        p_llm_prompt_type: str,
                                        p_llm_name: str,
                                        p_triples_to_llm_explanation: Dict[Tuple, Any],
                                        p_explored_prompts: Set[Tuple[str, str, str]]
                                        ):
    curr_prompt = (p_llm_name, p_llm_prompt_type, p_llm_prompt)
    if curr_prompt in p_explored_prompts:
        return p_triples_to_llm_explanation, p_explored_prompts
    if 'output_text_of_prompt:' in p_llm_prompt:
        p_llm_prompt_search = p_llm_prompt.split('output_text_of_prompt:', 1)[1].strip()
    else:
        p_llm_prompt_search = p_llm_prompt
    p_explored_prompts.add(curr_prompt)

    # Find all matches in the text
    # pattern = r"\[[\"'](Q\d+ \([^\]]+?\))[\"'], [\"'](P\d+ \([^\]]+?\))[\"'], [\"'](Q\d+ \([^\]]+?\))[\"']\].+?(YES|NO)(.*?)(?=\n\s*\n|$)"
    # pattern = r"\[[\"'](Q\d+ \([^\]]+?\))[\"'], [\"'](P\d+ \([^\]]+?\))[\"'], [\"'](Q\d+ \([^\]]+?\))[\"']\].+?(YES|NO)(.*?)(?=\n{1,5}(?=\d+\.)|$)"
    # pattern = r"\[[\"'](Q\d+ \(.+?\))[\"'], [\"'](P\d+ \(.+?\))[\"'], [\"'](Q\d+ \(.+?\))[\"']\].+?(YES|NO)(.*?)(?=\n{1,5}(?=\d+\.)|$)"
    # pattern = r"\[[\"'](Q\d+ \(.+?\))[\"'], [\"'](P\d+ \(.+?\))[\"'], [\"'](Q\d+ \(.+?)[\"']\].*?(YES|NO)(.*?)(?=\n{1,5}(?=\d+\.)|$)"
    # pattern = r"\[[\"'](Q\d+ \(.+?\))[\"'],\s*[\"'](P\d+ \(.+?\))[\"'],\s*[\"'](Q\d+ \(.+?)[\"']\].*?(YES|NO)(.*?)(?=\n{1,5}(?=\d+\.)|$)"
    # pattern = r"\[[\"'](Q\d+ \(.+?\))[\"'],\s*[\"'](P\d+ \(.+?\))[\"'],\s*[\"'](Q\d+ \(.+?)\].*?(YES|NO)(.*?)(?=\n{1,5}(?=\d+\.)|$)"
    pattern = r"\[[\"'](Q\d+ \(.+?\))[\"'],\s*[\"'](P\d+ \(.+?\))[\"'],\s*[\"'](Q\d+.*?)\].*?(YES|NO)(.*?)(?=\n{1,5}(?=\d+\.)|$)"

    matches = re.findall(pattern, p_llm_prompt_search, re.S)
    logger.debug(f'original_matches_are: {len(matches)} '
                 f'{matches} ****************from_from_from_from************** '
                 f'{p_llm_prompt_search}')
    if matches:
        for match in matches:
            head, relation, tail, llm_assessment, explanation = match
            assert llm_assessment in {'YES', 'NO'}
            # extracts only qids, no labels inside parenthesis
            head, relation, tail = tuple(re.match(r'^[^( ]+', item).group(0) for item in \
                                         [head, relation, tail])

            if (head, relation, tail, p_llm_prompt_type, p_llm_name) not in p_triples_to_llm_explanation:
                p_triples_to_llm_explanation[(head, relation, tail, p_llm_prompt_type, p_llm_name)] = \
                    explanation
    else:
        # pattern = r"\[[\"'](Q\d+ \([^\]]+?\))[\"'], [\"'](P\d+ \([^\]]+?\))[\"'], [\"'](Q\d+ \([^\]]+?\))[\"']\]"
        pattern = r"\[[\"'](Q\d+ \(.+?\))[\"'], [\"'](P\d+ \(.+?\))[\"'], [\"'](Q\d+ \(.+?\))[\"']\]"

        matches = re.findall(pattern, p_llm_prompt, re.S)
        matches = list(set(matches))
        if len(matches) > 0:
            if len(matches) > 1:
                # logger.error(f'matches_should_be_size_1 in '
                #              f'current_matches: {matches} for '
                #              f'p_llm_prompt: {p_llm_prompt}')
                logger.warning(f'matches_should_be_size_1 and is {len(matches)} '
                               f'in current_matches: {matches}')
            for curr_match in matches:
                head, relation, tail = curr_match
                # extracts only qids, no labels inside parenthesis
                head, relation, tail = tuple(re.match(r'^[^( ]+', item).group(0) for item in \
                                             [head, relation, tail])

                explanation = p_llm_prompt_search
                if (head, relation, tail, p_llm_prompt_type, p_llm_name) not in p_triples_to_llm_explanation:
                    p_triples_to_llm_explanation[(head, relation, tail, p_llm_prompt_type, p_llm_name)] = \
                        explanation
                # break
        else:
            err_mess = f'we_need_to_think_what_to do here, matches: {matches} ' \
                       f'prompt: {p_llm_prompt} and ' \
                       f'p_llm_prompt_search: {p_llm_prompt_search}'
            logger.error(err_mess)
    return p_triples_to_llm_explanation, p_explored_prompts


def generate_short_hash(input_string: str, hash_length: int):
    # Create a SHA256 hash of the input string
    hash_object = hashlib.sha256(input_string.encode())
    # Convert the hash to a byte array
    hash_bytes = hash_object.digest()
    # Encode the byte array to a base64 string and strip unwanted characters
    short_hash = base64.urlsafe_b64encode(hash_bytes).decode('utf-8').rstrip('=')
    # Return the first 8 characters for a shorter hash
    return short_hash[:hash_length]


def rename_assessments(assessments):
    new_assess = list()
    for curr_assess in assessments:
        if 'llm_prompt_type' not in curr_assess:

            if 'triple_deprecation' in curr_assess:
                # curr_assess['llm_prompt_type'] = 'triple_deprecation'
                # curr_assess['llm_prompt'] = curr_assess.pop('triple_deprecation_prompt')
                # curr_assess['llm_assessment'] = curr_assess.pop('triple_deprecation')
                new_assess.append({
                    'llm_name': curr_assess['llm_name'],
                    'llm_assessment': curr_assess['triple_deprecation'],
                    'llm_prompt_type': 'triple_deprecation',
                    'llm_prompt': curr_assess['triple_deprecation_prompt']
                }
                )
                # del curr_assess['triple_deprecation']
            elif 'triple_addition' in curr_assess:
                # curr_assess['llm_prompt_type'] = 'triple_assertion'
                # curr_assess['llm_prompt'] = curr_assess.pop('triple_addition_prompt')
                # curr_assess['llm_assessment'] = curr_assess.pop('triple_addition')
                new_assess.append({
                    'llm_name': curr_assess['llm_name'],
                    'llm_assessment': curr_assess['triple_addition'],
                    'llm_prompt_type': 'triple_assertion',
                    'llm_prompt': curr_assess['triple_addition_prompt']
                }
                )
                # del curr_assess['triple_addition']
            elif 'llm_assessment' in curr_assess:
                # curr_assess['llm_prompt_type'] = 'triple_assertion'
                # curr_assess['llm_prompt'] = curr_assess.pop('llm_assessment_prompt')
                new_assess.append({
                    'llm_name': curr_assess['llm_name'],
                    'llm_assessment': curr_assess['llm_assessment'],
                    'llm_prompt_type': 'triple_assertion',
                    'llm_prompt': curr_assess['llm_assessment_prompt']
                }
                )
                # curr_assess['llm_assessment'] = curr_assess['llm_assessment']
                # del curr_assess['triple_addition']
            else:
                raise RuntimeError(f'unrecognized assessment: {curr_assess}')
        else:
            new_assess.append(curr_assess)
    return new_assess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/s04b_sanity_check/20250218/'
                                's04b_sanity_check.json',
                        help='The config file that contains all the parameters')

    ####
    args = parser.parse_args()
    config = json.load(open(args.config_file, 'rt'))
    base_dir = config['base_dir']
    output_dir = os.path.join(base_dir, config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    # tot_nr_entries = 0
    # tot_nr_triples = 0
    # tot_nr_triples_left = 0
    # tot_nr_entries_left = 0
    nr_collissions = 0
    tot_nr_of_triples = 0
    tot_nr_of_assessments = 0
    tot_failed_assessments = 0
    tot_not_assessed_triples = 0
    all_hashed_ids = set()
    for curr_input_dir in config['input_dirs']:
        input_dir = os.path.join(base_dir, curr_input_dir)

        jsonl_files = Path(input_dir).rglob('*.jsonl')

        for file_path in jsonl_files:
            logger.info(f'Reading: {file_path}')
            filename = os.path.basename(file_path)
            output_file_path = os.path.join(output_dir, filename)
            with (open(file_path, 'rt', encoding='utf-8') as infile):
                logger.info(f'processing {file_path}')
                #
                with open(output_file_path, 'wt', encoding='utf-8') as outfile:
                    #####
                    for curr_line_idx, curr_line in \
                            tqdm(enumerate(infile), desc=f'processing {filename}'):
                        set_tkgu_operations = set()
                        all_triples_before = set()
                        all_triples_after = set()
                        parsed_line = json.loads(curr_line)
                        if 'passage' not in parsed_line:
                            curr_passage = parsed_line['chunk']
                        else:
                            curr_passage = parsed_line['passage']
                        if '{{' in curr_passage:
                            logger.info('ignoring_passage_brackets')
                            continue

                        emerging_knowledge_new = list()
                        nr_emerging_knowledge_left = 0
                        nr_emerging_entities_to_kg_left = 0
                        parsed_line['tkgu_triples'] = list()

                        for curr_triple in parsed_line['emerging_knowledge']:
                            # deep copy
                            copied_triple = copy.deepcopy(curr_triple)
                            curr_actions = set(curr_triple['actions'])
                            triple_deprecation_llama_8b_assessment = False
                            triple_deprecation_llama_400b_assessment = False
                            #
                            if curr_triple['qualifier_qid'] is not None:
                                curr_triple['qualifier_info'] = \
                                    {
                                        'qualifier_timestamp': curr_triple['qualifier_timestamp'],
                                        'qualifier_date': curr_triple['qualifier_date'],
                                        'qualifier_qid': curr_triple['qualifier_qid'],
                                        'qualifier_label': curr_triple['qualifier_label']
                                    }
                            else:
                                curr_triple['qualifier_info'] = {}

                            del curr_triple['qualifier_timestamp']
                            del curr_triple['qualifier_date']
                            del curr_triple['qualifier_qid']
                            del curr_triple['qualifier_label']
                            #
                            if 'llm_assessment' in curr_triple:
                                if len(curr_triple['llm_assessment_v2']) > 0:
                                    assert len(curr_triple['llm_assessment_v2']) == 1
                                    assert curr_triple['llm_assessment_v2'][0]['llm_name'] == \
                                           'Meta-Llama-3.1-405B'
                                    assert 'triple_deprecation' in curr_triple['llm_assessment_v2'][0]

                                if 'triple_deprecation' not in curr_triple['llm_assessment']:
                                    if len(curr_triple['llm_assessment']) > 0:
                                        assert 'triple_addition' in curr_triple['llm_assessment']
                                        curr_triple['llm_assessment_v2'].append({
                                            'llm_name': 'Llama-8B',
                                            'llm_assessment': \
                                                curr_triple['llm_assessment']['triple_addition'],
                                            'llm_prompt_type': 'triple_assertion',
                                            'llm_prompt': \
                                                curr_triple['llm_assessment']['triple_addition_prompt']
                                        })
                                    else:
                                        logger.warning('triple_without_assessment_emerging')

                                if 'triple_deprecation' in curr_triple['llm_assessment']:
                                    curr_triple['llm_assessment_v2'].append({
                                        'llm_name': 'Llama-8B',
                                        'llm_assessment': \
                                            curr_triple['llm_assessment']['triple_deprecation'],
                                        'llm_prompt_type': 'triple_deprecation',
                                        'llm_prompt': \
                                            curr_triple['llm_assessment']['triple_deprecation_prompt']
                                    })

                            curr_triple['llm_assessment_v2'] = \
                                rename_assessments(curr_triple['llm_assessment_v2'])

                            for curr_assessment in curr_triple['llm_assessment_v2']:
                                if curr_assessment['llm_name'] in {'Llama-8B', 'Meta-Llama-3.1-8B'}:
                                    if curr_assessment['llm_prompt_type'] == 'triple_deprecation':
                                        triple_deprecation_llama_8b_assessment = \
                                            curr_assessment['llm_assessment']
                                if curr_assessment['llm_name'] == 'Meta-Llama-3.1-405B':
                                    if curr_assessment['llm_prompt_type'] == 'triple_deprecation':
                                        triple_deprecation_llama_400b_assessment = \
                                            curr_assessment['llm_assessment']
                            #
                            # if 'removed_edge' in curr_actions:
                            #     if not triple_deprecation_llama_8b_assessment or \
                            #             not triple_deprecation_llama_400b_assessment:
                            #         curr_actions.remove('removed_edge')

                            del curr_triple['actions']
                            curr_triple['tkgu_operations'] = list()
                            #

                            curr_triple['llm_assessment'] = curr_triple.pop('llm_assessment_v2')
                            #
                            if 'qualifier_removed_edge' in curr_actions or \
                                    'removed_edge' in curr_actions:
                                curr_triple['tkgu_operations'].append('d-triples')
                                set_tkgu_operations.add('d-triples')

                            #
                            if not curr_triple['emerging_head'] and not curr_triple['emerging_tail']:
                                if 'qualifier_added_edge' in curr_actions or \
                                        'added_edge' in curr_actions:
                                    curr_triple['tkgu_operations'].append('e-triples')
                                    set_tkgu_operations.add('e-triples')

                            #
                            if curr_triple['emerging_head'] or curr_triple['emerging_tail']:
                                if 'qualifier_added_edge' in curr_actions or \
                                        'added_edge' in curr_actions:
                                    curr_triple['tkgu_operations'].append('ee-triples')
                                    set_tkgu_operations.add('ee-triples')

                            #
                            if len(curr_triple['tkgu_operations']) > 0:
                                parsed_line['tkgu_triples'].append(curr_triple)
                                all_triples_before.add(tuple(curr_triple['triple']))
                            else:
                                assert len(copied_triple['actions']) == 1
                                assert 'removed_edge' in set(copied_triple['actions'])
                                # logger.warning(f'ignoring_triple: '
                                #                f'{copied_triple}')

                            nr_emerging_knowledge_left += 1

                        del parsed_line['emerging_knowledge']
                        for curr_triple in parsed_line['existing_knowledge']:
                            all_triples_before.add(tuple(curr_triple['triple']))

                            if 'llm_assessment_v2' not in curr_triple:
                                curr_triple['llm_assessment_v2'] = []
                            elif len(curr_triple['llm_assessment_v2']) == 0:
                                logger.warning('triple_without_assessment_IN_EMPTY')

                            if 'llm_assessment' in curr_triple:
                                assert isinstance(curr_triple['llm_assessment'], bool), \
                                    (f'curr_triple["llm_assessment"] '
                                     f'({curr_triple["llm_assessment"]}) '
                                     f'must be a boolean')
                                curr_triple['llm_assessment_v2'].append(
                                    {
                                        'llm_name': 'Llama-8B',
                                        'llm_assessment': curr_triple['llm_assessment'],
                                        'llm_prompt_type': 'triple_assertion',
                                        'llm_prompt': curr_triple['llm_assessment_prompt']
                                    })
                                del curr_triple['llm_assessment_prompt']
                            curr_triple['llm_assessment_v2'] = \
                                rename_assessments(curr_triple['llm_assessment_v2'])
                            if len(curr_triple['llm_assessment_v2']) > 0:
                                curr_triple['llm_assessment_v2'] = \
                                    rename_assessments(curr_triple['llm_assessment_v2'])
                                assert len(curr_triple['llm_assessment_v2']) == 1
                                assert curr_triple['llm_assessment_v2'][0]['llm_name'] in \
                                       {'Llama-8B', 'Meta-Llama-3.1-8B'}
                                if 'llm_assessment' not in curr_triple['llm_assessment_v2'][0]:
                                    curr_triple['llm_assessment_v2'][0]['llm_assessment'] = \
                                        curr_triple['llm_assessment_v2'][0].pop('triple_addition')
                                assert curr_triple['llm_assessment_v2'][0]['llm_prompt_type'] == \
                                       'triple_assertion'
                            else:
                                logger.warning('triple_without_assessment_existing')

                            curr_triple['llm_assessment'] = curr_triple.pop('llm_assessment_v2')

                            curr_triple['tkgu_operations'] = ['x-triples']
                            set_tkgu_operations.add('x-triples')
                            parsed_line['tkgu_triples'].append(curr_triple)
                            # del curr_triple['actions']

                        del parsed_line['existing_knowledge']

                        for curr_triple in parsed_line['matched_triples_entities_to_kg']:
                            all_triples_before.add(tuple(curr_triple['triple']))

                            curr_actions = set(curr_triple['actions'])
                            del curr_triple['actions']
                            if curr_triple['qualifier_qid'] is not None:
                                curr_triple['qualifier_info'] = \
                                    {
                                        'qualifier_timestamp': curr_triple['qualifier_timestamp'],
                                        'qualifier_date': curr_triple['qualifier_date'],
                                        'qualifier_qid': curr_triple['qualifier_qid'],
                                        'qualifier_label': curr_triple['qualifier_label']
                                    }
                            else:
                                curr_triple['qualifier_info'] = {}

                            del curr_triple['qualifier_timestamp']
                            del curr_triple['qualifier_date']
                            del curr_triple['qualifier_qid']
                            del curr_triple['qualifier_label']

                            new_assessments = list()

                            #                                         'llm_name': 'Llama-8B',
                            #                                         'llm_assessment': curr_triple['llm_assessment'],
                            #                                         'llm_prompt_type': 'triple_assertion',
                            #                                         'llm_prompt': curr_triple['llm_assessment_prompt']
                            if 'llm_assessment' in curr_triple:
                                if 'triple_deprecation' not in curr_triple['llm_assessment']:
                                    if len(curr_triple['llm_assessment']) > 0:
                                        assert 'triple_addition' in curr_triple['llm_assessment']
                                        new_assessments.append({
                                            'llm_name': 'Llama-8B',
                                            'llm_assessment': curr_triple['llm_assessment']['triple_addition'],
                                            'llm_prompt_type': 'triple_assertion',
                                            'llm_prompt': \
                                                curr_triple['llm_assessment']['triple_addition_prompt']
                                        })
                                    else:
                                        logger.warning('triple_without_assessment_emerging')

                                if 'triple_deprecation' in curr_triple['llm_assessment']:
                                    new_assessments.append({
                                        'llm_name': 'Llama-8B',
                                        'llm_assessment': \
                                            curr_triple['llm_assessment']['triple_deprecation'],
                                        'llm_prompt_type': 'triple_deprecation',
                                        'llm_prompt': \
                                            curr_triple['llm_assessment']['triple_deprecation_prompt']
                                    })

                            curr_triple['llm_assessment'] = new_assessments
                            curr_triple['llm_assessment_v2'] = \
                                rename_assessments(curr_triple['llm_assessment_v2'])

                            curr_triple['llm_assessment'] = curr_triple['llm_assessment'] + \
                                                            curr_triple['llm_assessment_v2']
                            del curr_triple['llm_assessment_v2']

                            triple_deprecation_llama_8b_assessment = False
                            triple_deprecation_llama_400b_assessment = False
                            for curr_assessment in curr_triple['llm_assessment']:
                                if curr_assessment['llm_name'] in {'Llama-8B', 'Meta-Llama-3.1-8B'}:
                                    if curr_assessment['llm_prompt_type'] == 'triple_deprecation':
                                        triple_deprecation_llama_8b_assessment = \
                                            curr_assessment['llm_assessment']
                                if curr_assessment['llm_name'] == 'Meta-Llama-3.1-405B':
                                    if curr_assessment['llm_prompt_type'] == 'triple_deprecation':
                                        triple_deprecation_llama_400b_assessment = \
                                            curr_assessment['llm_assessment']

                            curr_triple['tkgu_operations'] = ['ee-kg-triples']

                            if 'removed_edge' in curr_actions:
                                if not triple_deprecation_llama_8b_assessment or \
                                        not triple_deprecation_llama_400b_assessment:
                                    curr_actions.remove('removed_edge')

                            if 'qualifier_removed_edge' in curr_actions or \
                                    'removed_edge' in curr_actions:
                                curr_triple['tkgu_operations'].append('d-triples')
                                set_tkgu_operations.add('d-triples')

                            parsed_line['tkgu_triples'].append(curr_triple)

                            set_tkgu_operations.add('ee-kg-triples')

                            # tot_nr_triples_left += 1

                        del parsed_line['matched_triples_entities_to_kg']
                        # tot_nr_entries_left += 1
                        if len(parsed_line['tkgu_triples']) == 0:
                            continue

                        if len(set_tkgu_operations) == 1 and 'x-triples' in set_tkgu_operations:
                            continue

                        if 'passage' not in parsed_line:
                            parsed_line['passage'] = parsed_line.pop('chunk')

                        if 'passage_date' not in parsed_line:
                            parsed_line['passage_date'] = parsed_line.pop('revision_date')

                        if 'passage_timestamp' not in parsed_line:
                            parsed_line['passage_timestamp'] = \
                                parsed_line.pop('revision_timestamp')

                        if 'passage_revision_id' not in parsed_line:
                            parsed_line['passage_revision_id'] = \
                                parsed_line.pop('revision_id')

                        if 'delta_timestamps' not in parsed_line:
                            parsed_line['delta_timestamps'] = \
                                parsed_line.pop('interval_timestamps')
                        if 'delta_dates' not in parsed_line:
                            parsed_line['delta_dates'] = \
                                parsed_line.pop('interval_dates')

                        curr_entry = f"{parsed_line['passage']}, " \
                                     f"{parsed_line['delta_timestamps'][0]}, " \
                                     f"{parsed_line['delta_timestamps'][1]}"
                        curr_hash_id = generate_short_hash(curr_entry,
                                                           hash_length=config['hash_length'])
                        extra_length = 0

                        while curr_hash_id in all_hashed_ids:
                            nr_collissions += 1
                            logger.warning(f'hash collision {nr_collissions}')
                            extra_length += 1
                            curr_hash_id = generate_short_hash(parsed_line['passage'],
                                                               hash_length=extra_length)
                        all_hashed_ids.add(curr_hash_id)
                        assert 'hash_id' not in parsed_line
                        parsed_line['hash_id'] = curr_hash_id

                        #### BEGIN: prompt adjustment
                        explored_prompts = set()
                        triples_to_llm_explanation = dict()

                        # def get_triples_to_assessment_in_prompt(llm_prompt: str,
                        #                                         llm_prompt_type: str,
                        #                                         llm_name: str,
                        #                                         triples_to_llm_explanation: Dict[Tuple, Any],
                        #                                         explored_prompts: Set[Tuple[str, str, str]]
                        #                                         ):
                        #                                     new_assessments.append({
                        #                                         'llm_name': 'Llama-8B',
                        #                                         'llm_assessment': \
                        #                                             curr_triple['llm_assessment']['triple_deprecation'],
                        #                                         'llm_prompt_type': 'triple_deprecation',
                        #                                         'llm_prompt': \
                        #                                             curr_triple['llm_assessment']['triple_deprecation_prompt']
                        #                                     })
                        for curr_triple in parsed_line['tkgu_triples']:
                            all_triples_after.add(tuple(curr_triple['triple']))
                            # if 'llm_assessment' not in curr_triple:
                            #     logger.error('llm_assessment_not_in_curr_triple')
                            #     # pass
                            #     tot_not_assessed_triples += 1
                            # el
                            tot_nr_of_triples += 1
                            if len(curr_triple['llm_assessment']) == 0:
                                tot_not_assessed_triples += 1
                                logger.warning(f'triple_not_assessed, '
                                               f'tot_not_assessed_triples: '
                                               f'{tot_not_assessed_triples} -- '
                                               f'{(tot_not_assessed_triples/tot_nr_of_triples) *100} %')

                            for curr_llm_assessment in curr_triple['llm_assessment']:
                                if curr_llm_assessment['llm_name'] in config['assessor_mappings']:
                                    old_llm_name = curr_llm_assessment['llm_name']
                                    curr_llm_assessment['llm_name'] = \
                                        config['assessor_mappings'][curr_llm_assessment['llm_name']]
                                    logger.debug(f'MAPPING {old_llm_name} to '
                                          f'{curr_llm_assessment["llm_name"]}')
                                tot_nr_of_assessments += 1
                                llm_prompt_type = curr_llm_assessment['llm_prompt_type']
                                llm_name = curr_llm_assessment['llm_name']
                                llm_prompt = curr_llm_assessment['llm_prompt']

                                triples_to_llm_explanation, explored_prompts = \
                                    get_triples_to_assessment_in_prompt(
                                        # p_llm_prompt=llm_decoded_prompt,
                                        p_llm_prompt=llm_prompt,
                                        p_llm_prompt_type=llm_prompt_type,
                                        p_llm_name=llm_name,
                                        p_triples_to_llm_explanation=triples_to_llm_explanation,
                                        p_explored_prompts=explored_prompts
                                    )
                                triple_entry = \
                                    (curr_triple['triple'][0],
                                     curr_triple['triple'][1],
                                     curr_triple['triple'][2],
                                     llm_prompt_type,
                                     llm_name)
                                if triple_entry in triples_to_llm_explanation:
                                    curr_llm_assessment['llm_prompt'] = \
                                        triples_to_llm_explanation[triple_entry]
                                else:
                                    err_msg = f'****could_not_find_triple_entry {triple_entry} ' \
                                              f'given the keys of length ' \
                                              f'{len(triples_to_llm_explanation)}: ' \
                                              f'{triples_to_llm_explanation.keys()} ' \
                                              f'given the following prompt: ' \
                                              f'{llm_prompt}******'
                                    logger.error(err_msg)
                                    tot_failed_assessments += 1

                                    logger.info(f'line_stats_{curr_line_idx}: '
                                                f'{(tot_failed_assessments / tot_nr_of_assessments) * 100} %')

                        # intersec_before_after = all_triples_after.intersection(all_triples_before)
                        # if len(intersec_before_after) > 0:
                        #     logger.error(f'something_wrong_with_intersection: '
                        #                  f'is {len(intersec_before_after)}:'
                        #                  f' {intersec_before_after}')
                        if not (all_triples_before == all_triples_after):
                            logger.error('something_wrong_triples_not_the_same: '
                                         f'all_triples_before: ({len(all_triples_before)}) '
                                         f'{all_triples_before} '
                                         f'all_triples_after: ({len(all_triples_after)}) '
                                         f'{all_triples_after}')
                        assert all_triples_before == all_triples_after
                        #### END: prompt adjustment
                        #
                        if len(parsed_line['tkgu_triples']) > 0:
                            outfile.write(json.dumps(parsed_line,
                                                     ensure_ascii=False) + '\n')

    logger.info('JOB_FINISHED_BYE_BYE')

####
