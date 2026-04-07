# The difference with _v4 is that this _v5 adapts to the new, final format
# of the dataset, which I guess is even the newer than the one adapted by _v4.
# This new format is the result of executing s06b_refactor_final_format.py.

import argparse
import json
import logging
import os
import os.path
import random
from datetime import datetime

from tqdm import tqdm

from dataset.emerge.utils.constants import ACTION_CATEGORY_ADD, ACTION_CATEGORY_DEPRECATE

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/s07c_verify_duplicates/20250828/'
                                's07c_verify_duplicates.json',
                        help='The config file that contains all the parameters')

    # removed_triple_actions = {'qualifier_removed_edge', 'removed_edge'}

    ####
    args = parser.parse_args()
    config = json.load(open(args.config_file, 'rt'))
    input_dir = config['input_dir']

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if not filename.endswith('.jsonl'):
                continue
            print(f'====================== {filename} =======================')
            passages_in_file = list()
            file_path = os.path.join(root, filename)
            print('loading passages')
            for curr_line in open(file_path, 'rt', encoding='utf-8'):
                curr_parsed_line = json.loads(curr_line)
                passages_in_file.append(
                    {
                        'passage': curr_parsed_line['passage'],
                        'anchor_page_qid': curr_parsed_line['anchor_page_qid']
                    }
                )
            print('passages loaded, calculating similarities')
            for curr_passage_idx, curr_instance in enumerate(tqdm(passages_in_file)):
                curr_maximum_jaccard_similarity = 0.0
                curr_minimum_ed_distance = 1.0
                curr_passage = curr_instance['passage']
                curr_anchor_page_qid = curr_instance['anchor_page_qid']
                for curr_instance2 in passages_in_file[curr_passage_idx + 1:]:
                    curr_passage2 = curr_instance2['passage']
                    curr_anchor_page_qid2 = curr_instance2['anchor_page_qid']
                    if curr_anchor_page_qid2 != curr_anchor_page_qid:
                        continue
                    min_length = min(len(curr_passage), len(curr_passage2))
                    max_length = max(len(curr_passage), len(curr_passage2))

                    j_similarity = jaccard_similarity(curr_passage, curr_passage2)
                    normalized_ed = normalized_edit_distance(curr_passage, curr_passage2)
                    min_length_passage = curr_passage[:min_length]
                    min_length_passage2 = curr_passage2[:min_length]

                    j_similarity_min_length = jaccard_similarity(min_length_passage,
                                                                 min_length_passage2)
                    normalized_ed_min_length = normalized_edit_distance(min_length_passage,
                                                                        min_length_passage2)

                    ##
                    if j_similarity > curr_maximum_jaccard_similarity:
                        curr_maximum_jaccard_similarity = j_similarity
                    if normalized_ed < curr_minimum_ed_distance:
                        curr_minimum_ed_distance = normalized_ed

                    if not (j_similarity < config['maximum_jaccard_similarity'] and
                            normalized_ed > config['minimum_edit_distance']):
                        print('===============START similar_passages================')
                        print(f'PASSAGE1 ({len(curr_passage)} chars): {curr_passage}')
                        print('----')
                        print(f'PASSAGE2 ({len(curr_passage2)} chars): {curr_passage2}')
                        print('----')
                        print(f'jaccard similarity: {j_similarity}')
                        print('----')
                        print(f'normalized ed dist: {normalized_ed}')
                        print('----')
                        print(f'jaccard similarity min length: {j_similarity_min_length}')
                        print('----')
                        print(f'normalized ed dist min length: {normalized_ed_min_length}')
                        print('----')
                        print(f'length ratio: {min_length / max_length}')
                        print('================END similar_passages ================')
