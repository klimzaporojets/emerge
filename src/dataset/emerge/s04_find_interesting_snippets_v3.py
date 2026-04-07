# this version s04_find_interesting_snippets_v2 is intended to be a faster
# version compared to s04_find_interesting_snippets due to using batching parameter
# and batching the target entities, which I think takes too long to create tensor from

import argparse
import json
import logging
import os
import subprocess
import time
import numpy as np
from typing import Dict, List

import torch
from torch_geometric.data import Data

from dataset.wikipedia.misc.cleaning import clean_text_from_link_markers
from dataset.emerge.utils.text_utils import calculate_english_word_percentage
import os
from dataset.emerge.utils.s04_find_interesting_snippets_v2_utils import timestamp_to_date, connect_interesting_snippets_with_kg, \
    batch_tensor_method2, batch_tensor_method1, connect_interesting_snippets_with_kg_v3
from dataset.emerge.utils.wiki_utils import load_property_qid_to_label, load_wikidata_qid_to_label, generate_short_hash

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)
from datetime import datetime


def date_string_to_timestamp(date_string):
    # Parse the date string into a datetime object
    dt = datetime.strptime(date_string, "%Y-%m-%d")
    # Convert the datetime object to a timestamp (seconds since epoch)
    timestamp = int(dt.timestamp())
    return timestamp


def is_any_value_between(lst, lower, upper):
    return any(lower <= x <= upper for x in lst)


def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def count_braces(s):
    count_open = s.count('{')
    count_close = s.count('}')
    return count_open + count_close


def count_pipes(s):
    count_pipes = s.count('|')
    return count_pipes


def get_free_gpu_memory(device_id):
    time_begin_get_free_gpu_memory = time.time()
    # Get the total and used memory using nvidia-smi
    if device_id.startswith('cuda:'):
        device_id = int(device_id[device_id.index(':') + 1:])
    else:
        device_id = int(device_id)
    try:
        # Run the nvidia-smi command to get GPU memory info
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,nounits,noheader", "-i",
             str(device_id)]
        ).decode('utf-8')

        # Parse the output
        free_memory, total_memory = map(int, output.strip().split(', '))
        logger.info(f'time to execute get_free_gpu_memory: '
                    f'{time.time() - time_begin_get_free_gpu_memory}')
        return free_memory  # Return free memory in MB
    except Exception as e:
        print(f"Error retrieving GPU memory: {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/s04_find_interesting_snippets/20241129/'
                                's04_find_interesting_snippets.json',
                        help='The config file that contains all the parameters')

    parser.add_argument('--device',
                        help='The device to load tensors to, examples: "cpu", "cuda:1", '
                             '"cuda"....',
                        type=str,
                        required=False,
                        default='cpu')
    parser.add_argument('--device2',
                        help='The device to do the in operation',
                        type=str,
                        required=False,
                        default='cpu')

    parser.add_argument('--batch_size',
                        help='The size of the batch',
                        type=int,
                        required=False,
                        default=1)

    parser.add_argument('--batching_type',
                        help='The type of method to use for batching',
                        type=str,
                        required=False,
                        default='method1')
    ####
    ####
    args = parser.parse_args()
    config = json.load(open(args.config_file, 'rt'))
    str_device = args.device
    str_device2 = args.device2
    batching_type = args.batching_type
    batch_size = args.batch_size
    input_candidate_snippets_dir = config['input_candidate_snippets_dir']
    output_dir = config['output_dir']

    # last_processed_dir points to the directory containing files with the last id of line processed
    # so the next time the script starts, it does not have to process everything from
    # scratch
    last_processed_dir = config['last_processed_dir']

    # last_processed_dir = os.path.dirname(last_processed_status_file)
    os.makedirs(last_processed_dir, exist_ok=True)

    # tensor_timestamps_wdata: torch.Tensor = None
    # wdata_graph: Data = None
    # entity_to_index = dict()
    # relation_to_index = dict()
    # index_to_entity = dict()
    # index_to_relation = dict()
    # wikidata_qid_to_label = dict()
    # property_qid_to_label = dict()
    caches_dir = config['caches_dir']

    # if config['connect_with_whole_kg']:
    whole_kg_path = config['whole_kg_path']
    sha_cache = generate_short_hash(whole_kg_path, hash_length=8)
    caches_wikidata_qid_to_label_path = os.path.join(caches_dir, f'wikidata_qid_to_label_{sha_cache}.pickle')

    logger.info(f'BEGIN loading whole kg from whole_kg_path: {whole_kg_path}')
    whole_kg = torch.load(whole_kg_path, weights_only=False)
    wdata_graph: Data = whole_kg['pyg_graph_wdata']
    # kzaporoj 08.02.2025, also adding the qualifiers in order to be able to
    # determine if existing edges are expired/deprecated or not
    tensor_qualifier_timestamps_wdata = whole_kg['tensor_qualifier_timestamps_wdata']
    #
    wdata_edges: torch.Tensor = wdata_graph.edge_index
    logger.info(f'Maximum value of wdata_edges: {wdata_edges.max()}')
    logger.info(f'BEGIN obtaining entity_ids_in_graph')
    # entity_ids_in_graph = set(wdata_graph.edge_index.numpy())
    unique_ids_values: torch.Tensor = torch.unique(wdata_graph.edge_index)
    entity_ids_in_graph = set(unique_ids_values.tolist())
    logger.info(f'END obtaining entity_ids_in_graph')
    logger.info(f'BEGIN moving to device wdata_graph: {str_device} '
                f'with free_memory of: {get_free_gpu_memory(str_device)}')
    logger.info(f'wdata_graph is: {wdata_graph}')

    wdata_graph.to(str_device)
    tensor_qualifier_timestamps_wdata.to(str_device)
    # 2173MiB / 11264MiB
    logger.info(f'END moving wdata_graph to device: {str_device}, '
                f'free_memory on device now: {get_free_gpu_memory(str_device)}')
    index_to_entity = whole_kg['index_to_entity']
    index_to_relation = whole_kg['index_to_relation']
    entity_to_index = {value: key for key, value in index_to_entity.items()}
    relation_to_index = {value: key for key, value in index_to_relation.items()}
    #
    tensor_timestamps_wdata: torch.Tensor = whole_kg['tensor_timestamps_wdata']
    logger.info(f'BEGIN moving to device tensor_timestamps_wdata: {str_device}')
    tensor_timestamps_wdata = tensor_timestamps_wdata.to(str_device)
    # 3191MiB / 11264MiB
    logger.info(f'END moving to device tensor_timestamps_wdata: {str_device} '
                f'free_memory on device now: {get_free_gpu_memory(str_device)}')
    logger.info(f'END loading whole kg from whole_kg_path: {whole_kg_path}')
    #
    logger.info('BEGIN invoking load_wikidata_qid_to_label')

    entity_qids_in_graph = set([f'Q{index_to_entity[curr_entity_id]}' for curr_entity_id in entity_ids_in_graph])

    caches_property_qid_to_label_path = os.path.join(caches_dir, 'property_qid_to_label.pickle')

    path_wikidata_labels = config['path_wikidata_labels']
    logger.info('BEGIN invoking load_property_qid_to_label')
    path_property_labels = config['path_property_labels']
    property_qid_to_label: Dict = load_property_qid_to_label(
        path_property_labels, caches_property_qid_to_label_path
    )
    logger.info('END invoking load_property_qid_to_label')

    logger.info('BEGIN invoking load_wikidata_qid_to_label')
    wikidata_qid_to_label: Dict = load_wikidata_qid_to_label(
        path_wikidata_labels, caches_wikidata_qid_to_label_path, qids_to_load=entity_qids_in_graph
    )
    logger.info(f'length of wikidata_qid_to_label: {len(wikidata_qid_to_label)}')
    logger.info('END invoking load_wikidata_qid_to_label')

    #
    # exit(0)
    os.makedirs(output_dir, exist_ok=True)
    output_files = dict()

    nr_processed_candidates = 0
    nr_useful_candidates = 0
    files_in_input = list_files(input_candidate_snippets_dir)
    already_processed_chunks = set()
    nr_lines_processed = 0
    time_to_process_lines = 0.0
    tot_nr_ignored = 0
    for curr_input_file in files_in_input:
        #     output_file = open(os.path.join(output_dir, 'interesting_snippets.jsonl'), 'wt',
        #                        encoding='utf-8')

        last_processed_file = os.path.join(last_processed_dir, curr_input_file)
        last_successfully_processed_line = -1
        if os.path.exists(last_processed_file):
            with open(last_processed_file, 'r') as lp_file:
                last_successfully_processed_line = int(lp_file.read())

        if not curr_input_file.endswith('jsonl'):
            continue

        logger.info(f'processing {curr_input_file}')
        if nr_processed_candidates > 0:
            logger.info(f'fraction useful candidates: {nr_useful_candidates / nr_processed_candidates} '
                        f'nr_processed_candidates: {nr_processed_candidates}, '
                        f'nr_useful_candidates: {nr_useful_candidates}')

        batched_entity_ids = list()
        batched_entity_ids_len = list()
        batched_other_data = list()
        curr_batch_max_entity_len = 0

        with (open(os.path.join(input_candidate_snippets_dir, curr_input_file), mode='rt',
                   encoding='utf-8') as infile):
            # last_processed_line = 0

            for line_idx, curr_line in enumerate(infile):
                wikidata_qid_to_mentions: Dict[str, List] = dict()
                if line_idx <= last_successfully_processed_line:
                    continue

                if line_idx - 1 == last_successfully_processed_line:
                    logger.info(f'starting_processing_line_nr_in {curr_input_file} from the '
                                f'line {line_idx}')

                time_begin_line = time.time()
                json_input = json.loads(curr_line)
                torch.cuda.empty_cache()
                anchor_title = json_input['anchor_title']

                # if json_input['chunk'] not in already_processed_chunks:
                if (json_input['chunk'], json_input['interval_id']) not in already_processed_chunks:
                    nr_braces = count_braces(json_input['chunk'])
                    nr_pipes = count_pipes(json_input['chunk'])
                    nr_special = nr_braces + nr_pipes
                    nr_tokens = len(json_input['chunk'].split(' '))
                    nr_chars = len(json_input['chunk'])
                    ratio_chars = 0.0
                    ratio_tokens = 0.0
                    if nr_tokens > 0:
                        ratio_tokens = nr_special / nr_tokens
                    if nr_chars > 0:
                        ratio_chars = nr_special / nr_chars
                    if ratio_tokens > 0.5:
                        # logger.info(f'ignoring {json_input["chunk"]}')
                        # logger.info('===================================')
                        # logger.info(f'ratio_chars: {ratio_chars} '
                        #             f'ratio_tokens: {ratio_tokens}')
                        # logger.info('===================================')
                        # logger.info('===================================')
                        continue
                    nr_processed_candidates += 1
                    already_processed_chunks.add((json_input['chunk'], json_input['interval_id']))

                    interval = list()
                    interval_str = ''
                    ####         #     output_file = open(os.path.join(output_dir, 'interesting_snippets.jsonl'), 'wt',
                    #                        encoding='utf-8')

                    for curr_interval in json_input['interval']:
                        # logger.info(f'curr_interval is: {curr_interval}')
                        if isinstance(json_input['interval'], int):
                            curr_interval_str = f'{timestamp_to_date(curr_interval)}({curr_interval})'
                        else:
                            curr_interval_str = curr_interval

                        interval.append(curr_interval_str)
                        interval_str += curr_interval_str

                    output_file_path = os.path.join(output_dir,
                                                    f'interesting_snippets{interval_str}.jsonl')
                    if output_file_path not in output_files:
                        output_files[output_file_path] = open(output_file_path, 'a', encoding='utf-8')

                    assert len(json_input['interval']) == 2
                    interval_from = json_input['interval'][0]
                    interval_to = json_input['interval'][1]
                    if isinstance(interval_from, str):
                        interval_from = date_string_to_timestamp(interval_from)

                    if isinstance(interval_to, str):
                        interval_to = date_string_to_timestamp(interval_to)
                    json_input['interval_timestamps'] = [interval_from, interval_to]
                    json_input['emerging_knowledge'] = json_input['matched_triples']
                    # in emerging knowledge we do not allow for head and tail to be the same
                    updated_emerging_knowledge = list()
                    for curr_emerging_knowledge in json_input['emerging_knowledge']:
                        if curr_emerging_knowledge['triple_qid'][0] == curr_emerging_knowledge['triple_qid'][2]:
                            continue
                        updated_emerging_knowledge.append(curr_emerging_knowledge)
                    json_input['emerging_knowledge'] = updated_emerging_knowledge
                    # json_input['nr_matched_triples'] = len(updated_emerging_knowledge)
                    if len(updated_emerging_knowledge) == 0:
                        tot_nr_ignored += 1
                        logger.debug(f'{tot_nr_ignored} ignore because nr_matched_triples in 0')
                        continue

                    del json_input['matched_triples']
                    # original_found_entities = json_input['found_entities'].copy()

                    mentions_with_qids = list()
                    for curr_mention in json_input['mentions']:
                        if curr_mention['qid'] is None:
                            logger.debug(f'-- NONE_MENTION_DETECTED!!: {curr_mention} '
                                         f'in {json_input} --')
                            continue
                        mentions_with_qids.append(curr_mention)
                        if curr_mention['qid'] not in wikidata_qid_to_mentions:
                            wikidata_qid_to_mentions[curr_mention['qid']] = list()
                        wikidata_qid_to_mentions[curr_mention['qid']].append(curr_mention['mention_text'])
                        # TODO: this assert needs to go to s03 where the mention positions are calculated
                        # start assert code
                        extracted_mention_text = json_input['chunk'] \
                            [curr_mention['start_char']:curr_mention['end_char']]
                        logger.debug(f'asserting "{extracted_mention_text}" vs '
                                     f'"{curr_mention["mention_text"]}"')
                        assert extracted_mention_text == curr_mention['mention_text']
                        # end assert code
                    json_input['mentions'] = mentions_with_qids
                    original_found_entities = list(wikidata_qid_to_mentions.keys())
                    # json_input['tot_nr_entities'] = len(original_found_entities)
                    # original_matched_triples = json_input['matched_triples']
                    original_matched_triples = json_input['emerging_knowledge']
                    matched_triples = set()
                    for curr_matched_triple in original_matched_triples:
                        assert curr_matched_triple['triple_qid'][0].startswith('Q')
                        assert curr_matched_triple['triple_qid'][1].startswith('P')
                        assert curr_matched_triple['triple_qid'][2].startswith('Q')
                        #
                        c_head_qid = curr_matched_triple['triple_qid'][0]
                        c_rel_qid = curr_matched_triple['triple_qid'][1]
                        c_tail_qid = curr_matched_triple['triple_qid'][2]
                        #
                        # if c_head_qid == 'Q38051551' and c_rel_qid == 'P726' and c_tail_qid == 'Q3131983':
                        #

                        # "Q38051551", "P726", "Q3131983"
                        head_id = entity_to_index[int(curr_matched_triple['triple_qid'][0][1:])]
                        relation_id = relation_to_index[curr_matched_triple['triple_qid'][1]]
                        tail_id = entity_to_index[int(curr_matched_triple['triple_qid'][2][1:])]
                        matched_triples.add((head_id, relation_id, tail_id))

                    idx_up_to_from = None

                    # json_input['interval_id'] = json_input['interval_id']
                    json_input['interval_dates'] = interval
                    # json_input['interval_date'] = json_input['interval']

                    # del json_input['interval_id']
                    del json_input['interval']

                    found_entities = list()
                    found_mentions = list()
                    mentions_to_qids = dict()
                    anchor_title_to_qids = dict()

                    # for curr_title, curr_qid in zip(json_input['entities_titles'],
                    #                                 json_input['found_entities']):
                    #     entity_qid_to_title[curr_qid] = curr_title
                    #     found_entities.append({
                    #         'title': curr_title,
                    #         'qid': curr_qid
                    #     })
                    # found_entities.append(f'{curr_title} ({curr_qid})')

                    # entity_qid_to_title = dict()
                    # entity_title_to_qid = {v: q for q, v in entity_qid_to_title.items()}
                    # for curr_mention, curr_target_title in (json_input['mention_to_title']):
                    #     if curr_mention not in mentions_to_qids:
                    #         mentions_to_qids[curr_mention] = set()
                    #     if curr_target_title in entity_title_to_qid:
                    #         mentions_to_qids[curr_mention].add(entity_title_to_qid[curr_target_title])

                    # json_input['found_entities'] = found_entities
                    # del json_input['entities_titles']
                    # del json_input['mention_to_title']

                    # matched_entities_labels = list()
                    # matched_entities = list()
                    # for curr_matched_entity_qid in json_input['matched_entities']:
                    #     # if curr_matched_entity_qid not in entity_qid_to_title:
                    #     title_curr_entity = curr_matched_entity_qid
                    #     if curr_matched_entity_qid in wikidata_qid_to_label:
                    #         title_curr_entity = wikidata_qid_to_label[curr_matched_entity_qid]
                    #     matched_entities.append({
                    #         'title': f'{title_curr_entity}',
                    #         'qid': f'{curr_matched_entity_qid}'
                    #     })
                    #     # matched_entities_labels.append(f'{entity_qid_to_title[curr_matched_entity_qid]} '
                    #     #                                f'({curr_matched_entity_qid})')
                    #     matched_entities_labels.append(f'{title_curr_entity} '
                    #                                    f'({curr_matched_entity_qid})')
                    del json_input['matched_entities']

                    updated_emerging_knowledge = list()
                    for curr_triple in json_input['emerging_knowledge']:
                        triple_lifespan_date = list()
                        triple_lifespan_timestamp = list()
                        actions = list()
                        # action = ''
                        triple_qid = curr_triple['triple_qid']
                        #
                        # head_label = triple_qid[0]
                        relation_label = triple_qid[1]
                        # tail_label = triple_qid[2]

                        head_label = wikidata_qid_to_mentions[triple_qid[0]][0]
                        tail_label = wikidata_qid_to_mentions[triple_qid[2]][0]
                        #
                        if triple_qid[0] in wikidata_qid_to_label:
                            head_label = wikidata_qid_to_label[triple_qid[0]]
                        # #
                        if triple_qid[1] in property_qid_to_label:
                            relation_label = property_qid_to_label[triple_qid[1]]
                        # #
                        if triple_qid[2] in wikidata_qid_to_label:
                            tail_label = wikidata_qid_to_label[triple_qid[2]]
                        #

                        triple_labels = [head_label, relation_label, tail_label]

                        curr_triple['triple_labels'] = triple_labels

                        # user (20250226) - BEGIN comment this code because property label is not a mention
                        # for curr_qid, curr_label in zip(triple_qid, triple_labels):
                        #     if curr_label not in mentions_to_qids:
                        #         mentions_to_qids[curr_label] = set()
                        #     mentions_to_qids[curr_label].add(curr_qid)
                        # user (20250226) - BEGIN comment this code because property label is not a mention

                        curr_triple['triple'] = curr_triple['triple_qid']
                        triple_lifespan_timestamp.append(int(curr_triple['triple_from']))
                        triple_lifespan_timestamp.append(
                            None if int(curr_triple['triple_to']) == 0 else int(curr_triple['triple_to']))
                        curr_triple['triple_lifespan_timestamp'] = triple_lifespan_timestamp

                        if curr_triple['action'] == 0:
                            if curr_triple['triple_lifespan_timestamp'][1] is not None and \
                                    curr_triple['triple_lifespan_timestamp'][1] <= json_input['interval_timestamps'][1]:
                                actions.append('removed_edge')
                            else:
                                # 2025.03.05 - atadura de alambre, based on the fix in s03_get_deltas_pyg_v3
                                # (see msca_latent_space_status.txt notes)
                                logger.error(f'atadura_inconsistent_removal for triple: {curr_triple} '
                                             f'in the context of interval '
                                             f'{json_input["interval_timestamps"]} '
                                             f'for json_input {json_input}')
                        if curr_triple['action'] == 1:
                            actions.append('added_edge')
                        if curr_triple['qualifier_action_idx'] == 0:
                            actions.append('qualifier_removed_edge')
                        if curr_triple['qualifier_action_idx'] == 1:
                            actions.append('qualifier_added_edge')
                        # 2025.03.05 - atadura de alambre, based on the fix in s03_get_deltas_pyg_v3
                        # (see msca_latent_space_status.txt notes)
                        if len(actions) == 0:
                            logger.error(f'atadura_actions_in_0_for the following line, ignoring: '
                                         f'{json_input}')
                            continue
                        curr_triple['actions'] = actions

                        # curr_triple['qualifier_label'] = (f'{curr_triple["qualifier_label"]} '
                        #                                   f'({curr_triple["qualifier_qid"]})')
                        # curr_triple['qualifier_label'] = (f'{curr_triple["qualifier_label"]} '
                        #                                   f'({curr_triple["qualifier_qid"]})')
                        # curr_triple['qualifier_id'] = curr_triple['qualifier_qid']
                        # head_creation_date = (f'{timestamp_to_date(curr_triple["head_creation_date"])}'
                        #                       f'({curr_triple["head_creation_date"]})')
                        # tail_creation_date = (f'{timestamp_to_date(curr_triple["tail_creation_date"])}'
                        #                       f'({curr_triple["tail_creation_date"]})')
                        head_creation_date = f'{timestamp_to_date(curr_triple["head_creation_date"])}'
                        tail_creation_date = f'{timestamp_to_date(curr_triple["tail_creation_date"])}'
                        ####
                        head_creation_timestamp = int(curr_triple["head_creation_date"])
                        tail_creation_timestamp = int(curr_triple["tail_creation_date"])
                        #
                        triple_lifespan_date.append(f'{timestamp_to_date(curr_triple["triple_from"])}')
                        triple_lifespan_date.append(
                            None if curr_triple['triple_to'] == 0 else f'{timestamp_to_date(curr_triple["triple_to"])}')
                        # triple_lifespan_date.append(f'{timestamp_to_date(curr_triple['triple_from'])}'
                        #                        f'({curr_triple['triple_from']})')
                        # triple_lifespan_date.append(f'{timestamp_to_date(curr_triple['triple_to'])}'
                        #                        f'({curr_triple['triple_to']})')
                        # None if triple_attrs['triple_timestamp_to'] == 0 else triple_attrs['triple_date_to']

                        # triple_lifespan_timestamp.append(int(curr_triple['triple_to']))

                        curr_triple['triple_lifespan_date'] = triple_lifespan_date
                        curr_triple['head_creation_date'] = head_creation_date
                        curr_triple['tail_creation_date'] = tail_creation_date
                        curr_triple['head_creation_timestamp'] = head_creation_timestamp
                        curr_triple['tail_creation_timestamp'] = tail_creation_timestamp
                        if curr_triple['qualifier_qid'] == 'none':
                            curr_triple['qualifier_qid'] = None
                        # else:
                        #
                        if curr_triple['qualifier_qid'] in property_qid_to_label:
                            curr_triple['qualifier_label'] = \
                                property_qid_to_label[curr_triple['qualifier_qid']]
                        else:
                            curr_triple['qualifier_label'] = curr_triple['qualifier_qid']

                        #
                        if curr_triple['qualifier_label'] == 'none':
                            curr_triple['qualifier_label'] = None
                        if curr_triple['qualifier_timestamp'] == -1:
                            curr_triple['qualifier_timestamp'] = None

                        curr_triple['qualifier_date'] = None
                        if curr_triple['qualifier_qid'] is not None and curr_triple['qualifier_timestamp'] is not None:
                            curr_triple['qualifier_date'] = timestamp_to_date(curr_triple['qualifier_timestamp'])

                        del curr_triple['triple_qid']
                        del curr_triple['triple_from']
                        del curr_triple['triple_to']
                        del curr_triple['action_label']
                        del curr_triple['qualifier_id']
                        del curr_triple['action']
                        # del curr_triple['qualifier_qid']
                        # del curr_triple['qualifier_label']
                        del curr_triple['qualifier_action_idx']
                        del curr_triple['qualifier_action_label']
                        # del curr_triple['qualifier_timestamp']
                        # atadura fix (see above), this should be removed once the
                        # s03_get_deltas_pyg_v3.py is run again (with the fix which still has to be
                        # tested)
                        # and change in-place without re-assigning emerging_knowledge
                        updated_emerging_knowledge.append(curr_triple)
                    json_input['emerging_knowledge'] = updated_emerging_knowledge

                    # TODO - the loop below is a repetition of the loop above,
                    #  but looping over json_input['matched_triples_entities_to_kg']
                    #  instead of json_input['emerging_knowledge']
                    #  , make it a single function in the final version
                    #  OR EVEN BETTER, BUT THIS LOGIC INTO s03 (previous step), so the json is
                    #  formatted as in the final version of dataset, except the labels maybe, which
                    #  will need to be added here.
                    for curr_matched_triple_entity_to_kg in json_input['matched_triples_entities_to_kg']:
                        triple_lifespan_date = list()
                        triple_lifespan_timestamp = list()
                        actions = list()
                        triple_qid = curr_matched_triple_entity_to_kg['triple_qid']
                        #
                        head_label = triple_qid[0]
                        relation_label = triple_qid[1]
                        tail_label = triple_qid[2]

                        #
                        if triple_qid[0] in wikidata_qid_to_label:
                            head_label = wikidata_qid_to_label[triple_qid[0]]
                        # #
                        if triple_qid[1] in property_qid_to_label:
                            relation_label = property_qid_to_label[triple_qid[1]]
                        # #
                        if triple_qid[2] in wikidata_qid_to_label:
                            tail_label = wikidata_qid_to_label[triple_qid[2]]
                        #

                        triple_labels = [head_label, relation_label, tail_label]

                        curr_matched_triple_entity_to_kg['triple_labels'] = triple_labels
                        curr_matched_triple_entity_to_kg['triple'] = \
                            curr_matched_triple_entity_to_kg['triple_qid']

                        triple_lifespan_timestamp.append(int(curr_matched_triple_entity_to_kg['triple_from']))
                        triple_lifespan_timestamp.append(
                            None if int(curr_matched_triple_entity_to_kg['triple_to']) == 0 \
                                else int(curr_matched_triple_entity_to_kg['triple_to']))
                        curr_matched_triple_entity_to_kg['triple_lifespan_timestamp'] = triple_lifespan_timestamp

                        # if curr_matched_triple_entity_to_kg['action'] == 0:
                        #     if curr_matched_triple_entity_to_kg['triple_lifespan_timestamp'][1] is not None and \
                        #             curr_matched_triple_entity_to_kg['triple_lifespan_timestamp'][1] <= json_input['interval_timestamps'][1]:
                        #         actions.append('removed_edge')
                        #     else:
                        #         # 2025.03.05 - atadura de alambre, based on the fix in s03_get_deltas_pyg_v3
                        #         # (see msca_latent_space_status.txt notes)
                        #         logger.error(f'atadura_inconsistent_removal for triple: '
                        #                      f'{curr_matched_triple_entity_to_kg} '
                        #                      f'in the context of interval '
                        #                      f'{json_input["interval_timestamps"]} '
                        #                      f'for json_input {json_input}')
                        assert curr_matched_triple_entity_to_kg['action'] == 1
                        if curr_matched_triple_entity_to_kg['action'] == 1:
                            actions.append('added_edge')
                        if curr_matched_triple_entity_to_kg['qualifier_action_idx'] == 0:
                            actions.append('qualifier_removed_edge')
                        if curr_matched_triple_entity_to_kg['qualifier_action_idx'] == 1:
                            actions.append('qualifier_added_edge')

                        curr_matched_triple_entity_to_kg['actions'] = actions
                        head_creation_date = \
                            f'{timestamp_to_date(curr_matched_triple_entity_to_kg["head_creation_date"])}'
                        tail_creation_date = \
                            f'{timestamp_to_date(curr_matched_triple_entity_to_kg["tail_creation_date"])}'
                        ####
                        head_creation_timestamp = int(curr_matched_triple_entity_to_kg['head_creation_date'])
                        tail_creation_timestamp = int(curr_matched_triple_entity_to_kg['tail_creation_date'])
                        #
                        triple_lifespan_date.append(
                            f'{timestamp_to_date(curr_matched_triple_entity_to_kg["triple_from"])}'
                        )
                        triple_lifespan_date.append(
                            None if curr_matched_triple_entity_to_kg['triple_to'] == 0 else \
                                f'{timestamp_to_date(curr_matched_triple_entity_to_kg["triple_to"])}')
                        # triple_lifespan_date.append(f'{timestamp_to_date(curr_triple['triple_from'])}'
                        #                        f'({curr_triple['triple_from']})')
                        # triple_lifespan_date.append(f'{timestamp_to_date(curr_triple['triple_to'])}'
                        #                        f'({curr_triple['triple_to']})')
                        # None if triple_attrs['triple_timestamp_to'] == 0 else triple_attrs['triple_date_to']
                        # triple_lifespan_timestamp.append(int(curr_triple['triple_to']))

                        curr_matched_triple_entity_to_kg['triple_lifespan_date'] = triple_lifespan_date
                        curr_matched_triple_entity_to_kg['head_creation_date'] = head_creation_date
                        curr_matched_triple_entity_to_kg['tail_creation_date'] = tail_creation_date
                        curr_matched_triple_entity_to_kg['head_creation_timestamp'] = head_creation_timestamp
                        curr_matched_triple_entity_to_kg['tail_creation_timestamp'] = tail_creation_timestamp
                        if curr_matched_triple_entity_to_kg['qualifier_qid'] == 'none':
                            curr_matched_triple_entity_to_kg['qualifier_qid'] = None
                        # else:
                        #
                        if curr_matched_triple_entity_to_kg['qualifier_qid'] in property_qid_to_label:
                            curr_matched_triple_entity_to_kg['qualifier_label'] = \
                                property_qid_to_label[curr_matched_triple_entity_to_kg['qualifier_qid']]
                        else:
                            curr_matched_triple_entity_to_kg['qualifier_label'] = curr_matched_triple_entity_to_kg[
                                'qualifier_qid']

                        #
                        if curr_matched_triple_entity_to_kg['qualifier_label'] == 'none':
                            curr_matched_triple_entity_to_kg['qualifier_label'] = None
                        if curr_matched_triple_entity_to_kg['qualifier_timestamp'] == -1:
                            curr_matched_triple_entity_to_kg['qualifier_timestamp'] = None

                        curr_matched_triple_entity_to_kg['qualifier_date'] = None
                        if curr_matched_triple_entity_to_kg['qualifier_qid'] is not None and \
                                curr_matched_triple_entity_to_kg['qualifier_timestamp'] is not None:
                            curr_matched_triple_entity_to_kg['qualifier_date'] = \
                                timestamp_to_date(curr_matched_triple_entity_to_kg['qualifier_timestamp'])

                        del curr_matched_triple_entity_to_kg['triple_qid']
                        del curr_matched_triple_entity_to_kg['triple_from']
                        del curr_matched_triple_entity_to_kg['triple_to']
                        del curr_matched_triple_entity_to_kg['action_label']
                        del curr_matched_triple_entity_to_kg['qualifier_id']
                        del curr_matched_triple_entity_to_kg['action']
                        del curr_matched_triple_entity_to_kg['qualifier_action_idx']
                        del curr_matched_triple_entity_to_kg['qualifier_action_label']
                    #
                    all_removed_edges = all([('removed_edge' in curr_triple['actions'])
                                             for curr_triple in json_input['emerging_knowledge']])
                    qualifiers_removed = [curr_triple
                                          for curr_triple in json_input['emerging_knowledge']
                                          if 'qualifier_removed_edge' in curr_triple['actions']]
                    edges_removed = [curr_triple
                                     for curr_triple in json_input['emerging_knowledge']
                                     if 'removed_edge' in curr_triple['actions']]
                    #
                    #
                    if (len(qualifiers_removed) > 0 or
                        len(edges_removed) > 0) and \
                            config['only_passages_after_kg_edge_removal']:
                        ####
                        if len(qualifiers_removed) > 0:
                            qualif_timestamps = \
                                [edge_rem['qualifier_timestamp'] for edge_rem in qualifiers_removed]
                            # contains_null = any(item is None for item in qualif_timestamps)
                            # if contains_null:
                            #     logger.error('qualifier_timestamp_in_null, '
                            #                  f'what is going on? : {json_input}')
                            #     raise RuntimeError('qualifier_timestamp_in_null, '
                            #                        f'what is going on? : {json_input}')
                            latest_timestamp_removed_q_edge = max(
                                qualif_timestamps
                            )
                        else:
                            latest_timestamp_removed_q_edge = 0

                        if len(edges_removed) > 0:
                            removed_timestamps = [edge_rem['triple_lifespan_timestamp'][1] for edge_rem in
                                                  edges_removed]
                            # contains_null = any(item is None for item in removed_timestamps)
                            latest_timestamp_removed_edge = max(
                                removed_timestamps
                            )
                        else:
                            latest_timestamp_removed_edge = 0

                        if latest_timestamp_removed_q_edge is None:
                            logger.error(f'latest_timestamp_removed_q_edge in None for the '
                                         f'following line: {json_input}')
                        if latest_timestamp_removed_edge is None:
                            logger.error(f'latest_timestamp_removed_edge in None for the '
                                         f'following line: {json_input}')
                        ####
                        latest_timestamp_removal = max(latest_timestamp_removed_q_edge,
                                                       latest_timestamp_removed_edge)
                        timestamp_wikipedia_snippet_revision = json_input['revision_timestamp']
                        # logger.info('timestamp_wikipedia_snippet_revision is '
                        #             f'{timestamp_wikipedia_snippet_revision}')
                        if timestamp_wikipedia_snippet_revision < latest_timestamp_removal:
                            logger.debug('ignoring_because_removal_of_edge_comes '
                                         'after snippet and only_passages_after_kg_edge_removal in '
                                         'true: '
                                         f'edges_removed: {edges_removed} \n'
                                         f'qualifiers_removed: {qualifiers_removed} \n'
                                         f'json_input: {json_input}')
                            continue
                    # qualifier_removed_edge
                    emerging_entities_in_triples = set()
                    # emerging_entities_in_triples = set([curr_triple [] curr_json_response['matched_triples']])
                    s_original_found_entities = set(original_found_entities)
                    for curr_triple_to_check in json_input['emerging_knowledge']:
                        if curr_triple_to_check['emerging_head'] and \
                                curr_triple_to_check['triple'][0] in s_original_found_entities:
                            emerging_entities_in_triples.add(curr_triple_to_check['triple'][0])
                        if curr_triple_to_check['emerging_tail'] and \
                                curr_triple_to_check['triple'][2] in s_original_found_entities:
                            emerging_entities_in_triples.add(curr_triple_to_check['triple'][2])
                        # in theory, we can not have emerging if they are not in text, controlling for this!!!
                        if curr_triple_to_check['emerging_head']:
                            assert curr_triple_to_check['triple'][0] in s_original_found_entities
                            # if curr_triple_to_check['triple'][0] not in s_original_found_entities:
                            #     logger.error(f'ERROR - emerging_knowledge, '
                            #                  f'emerging_head({curr_triple_to_check["triple"][0]}) '
                            #                  f'not in found entities ({s_original_found_entities}) for line: '
                            #                  f'{json_input}')
                        if curr_triple_to_check['emerging_tail']:
                            assert curr_triple_to_check['triple'][2] in s_original_found_entities
                            # if curr_triple_to_check['triple'][2] not in s_original_found_entities:
                            #     logger.error(f'ERROR - emerging_knowledge, '
                            #                  f'emerging_head({curr_triple_to_check["triple"][2]}) '
                            #                  f'not in found entities ({s_original_found_entities}) for line: '
                            #                  f'{json_input}')
                    for curr_triple_to_check in json_input['matched_triples_entities_to_kg']:
                        if curr_triple_to_check['emerging_head'] and \
                                curr_triple_to_check['triple'][0] in s_original_found_entities:
                            emerging_entities_in_triples.add(curr_triple_to_check['triple'][0])
                        if curr_triple_to_check['emerging_tail'] and \
                                curr_triple_to_check['triple'][2] in s_original_found_entities:
                            emerging_entities_in_triples.add(curr_triple_to_check['triple'][2])

                        # in theory, we can not have emerging if they are not in text, controlling for this!!!
                        if curr_triple_to_check['emerging_head']:
                            assert curr_triple_to_check['triple'][0] in s_original_found_entities
                            # if curr_triple_to_check['triple'][0] not in s_original_found_entities:
                            #     logger.error(f'ERROR - matched_triples_entities_to_kg, '
                            #                  f'emerging_head({curr_triple_to_check["triple"][0]}) '
                            #                  f'not in found entities ({s_original_found_entities}) for line: '
                            #                  f'{json_input}')

                        if curr_triple_to_check['emerging_tail']:
                            assert curr_triple_to_check['triple'][2] in s_original_found_entities
                            # if curr_triple_to_check['triple'][2] not in s_original_found_entities:
                            #     logger.error(f'ERROR - matched_triples_entities_to_kg, '
                            #                  f'emerging_head({curr_triple_to_check["triple"][2]}) '
                            #                  f'not in found entities ({s_original_found_entities}) for line: '
                            #                  f'{json_input}')

                    if len(emerging_entities_in_triples) > 0:
                        logger.info(f'emerging_entities_in_triples_is: {emerging_entities_in_triples}')

                    len_matched_triples = len(json_input['emerging_knowledge']) + len(
                        json_input['matched_triples_entities_to_kg'])
                    if (((len_matched_triples >= len(original_found_entities) / 2 or
                          len_matched_triples >= config['min_nr_emerging_triples'])
                         and not all_removed_edges
                         and float(len_matched_triples) / float(len(original_found_entities)) > 0.2)
                            # if at least one qualifier is removed, we are interested because there are very
                            # few edge deprecation cases
                            or len(qualifiers_removed) > 0
                            or len(emerging_entities_in_triples) > 0):
                        # and float(json_input['nr_matched_triples']) / float(json_input['tot_nr_entities']) > 0.2:
                        # time_begin_complex_if_prev1 = time.time()
                        time_begin_complex_if = time.time()
                        nr_useful_candidates += 1
                        if nr_useful_candidates > 0 and nr_useful_candidates % 100 == 0:
                            logger.info(f'nr of processed candidates: {nr_useful_candidates}')
                        # processes the chunk, to it does not contain the text
                        # json_input['chunk_without_links'] = \
                        #     clean_text_from_link_markers(json_input['chunk']).strip()

                        # chunk_without_links = json_input['chunk_without_links']

                        # percentage_english_words = calculate_english_word_percentage(
                        #     chunk_without_links.lower())
                        percentage_english_words = calculate_english_word_percentage(
                            json_input['chunk'].lower())
                        # logger.info(f'percentage_english_words: {percentage_english_words} '
                        #             f'{chunk_without_links}')
                        if percentage_english_words < config['min_percentage_english_words']:
                            tot_nr_ignored += 1
                            logger.debug(f'ignoring_chunk_not_english '
                                         f'percentage: {percentage_english_words}, '
                                         f'chunk: {json_input["chunk"]}')
                            continue

                        if len(json_input['chunk'].split(' ')) < config['min_passage_length']:
                            tot_nr_ignored += 1
                            logger.debug(f'{tot_nr_ignored} ignore passage too short: '
                                         f'{json_input["chunk"]}')
                            continue

                        if len(json_input['chunk'].split(' ')) > config['max_passage_length']:
                            tot_nr_ignored += 1
                            logger.debug(f'{tot_nr_ignored} ignore passage too long: '
                                         f'{json_input["chunk"]}')
                            continue

                        logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                                     f'time it took for prev_1')

                        #### BEGIN: obtaining all the triple connections in KG at that point in time
                        # if tensor_timestamps_wdata is not None and wdata_graph is not None:
                        # curr_batch_other['output_path_file']
                        batched_other_data.append({'interval_to': interval_to,
                                                   'interval_from': interval_from,
                                                   'json_input': json_input,
                                                   'output_file_path': output_file_path,
                                                   'matched_triples': matched_triples})
                        if config['connect_with_whole_kg']:
                            entity_ids = list()
                            # 6197MiB / 11264MiB
                            curr_entity_qid: str
                            for curr_entity_qid in original_found_entities:
                                # if curr_entity_qid is None:
                                #     continue
                                assert curr_entity_qid.startswith('Q')
                                curr_entity_qid_int = int(curr_entity_qid[1:])
                                if curr_entity_qid_int in entity_to_index:
                                    curr_entity_id = entity_to_index[curr_entity_qid_int]
                                    entity_ids.append(curr_entity_id)

                            batched_entity_ids.append(entity_ids)
                            batched_entity_ids_len.append(len(entity_ids))
                            if len(entity_ids) > curr_batch_max_entity_len:
                                curr_batch_max_entity_len = len(entity_ids)

                        if nr_useful_candidates > 0 and nr_useful_candidates % batch_size == 0:
                            res_batch = None
                            if len(batched_entity_ids) > 0:
                                time_res_batch = time.time()
                                if batching_type == 'method1':
                                    res_batch = batch_tensor_method1(
                                        curr_batch_max_length=curr_batch_max_entity_len,
                                        batched_entity_ids=batched_entity_ids,
                                        device=str_device2
                                    )
                                elif batching_type == 'method2':
                                    res_batch = batch_tensor_method2(
                                        batched_entity_ids=batched_entity_ids,
                                        device=str_device2
                                    )
                                else:
                                    raise RuntimeError(f'batching_type not recognized: f{batching_type}')

                                logger.debug(f'{time.time() - time_res_batch} '
                                             f'sec. time it took to get res_batch of shape: '
                                             f'{res_batch.shape}')
                            #
                            batch_idx_to_content = dict()
                            for idx_in_batch, curr_batch_other in enumerate(batched_other_data):
                                curr_output_path_file = curr_batch_other['output_file_path']
                                if res_batch is not None:
                                    tn_entity_ids = res_batch[idx_in_batch]
                                    len_entity_ids = batched_entity_ids_len[idx_in_batch]
                                    json_input = connect_interesting_snippets_with_kg(
                                        property_qid_to_label=property_qid_to_label,
                                        interval_from=curr_batch_other['interval_from'],
                                        interval_to=curr_batch_other['interval_to'],
                                        json_input=curr_batch_other['json_input'],
                                        str_device=str_device,
                                        str_device2=str_device2,
                                        tensor_timestamps_wdata=tensor_timestamps_wdata,
                                        wdata_graph=wdata_graph,
                                        tn_entity_ids=tn_entity_ids,
                                        len_entity_ids=len_entity_ids,
                                        index_to_entity=index_to_entity,
                                        index_to_relation=index_to_relation,
                                        matched_triples=curr_batch_other['matched_triples'],
                                        # matched_triples=matched_triples,
                                        wikidata_qid_to_label=wikidata_qid_to_label,
                                        tensor_qualifier_timestamps_wdata=tensor_qualifier_timestamps_wdata
                                    )
                                #### END: obtaining all the triple connections in KG at that point in time
                                batch_idx_to_content[idx_in_batch] = {
                                    'curr_output_path_file': curr_output_path_file,
                                    'json_input': json_input
                                }

                            time_begin_write_flush = time.time()
                            for curr_batch_idx, curr_batch_content in batch_idx_to_content.items():
                                curr_output_path_file = curr_batch_content['curr_output_path_file']
                                curr_json_input = curr_batch_content['json_input']
                                output_files[curr_output_path_file].write(
                                    json.dumps(curr_json_input) + '\n')
                                output_files[curr_output_path_file].flush()

                            logger.debug(f'sec. {time.time() - time_begin_write_flush} '
                                         f'to complete the write and flush')
                            #
                            # as soon as the batch was written the line is registered
                            with open(last_processed_file, 'w', encoding='utf-8') as lp_file:
                                lp_file.write(str(line_idx))
                                lp_file.flush()
                            batched_entity_ids = list()
                            batched_entity_ids_len = list()
                            curr_batch_max_entity_len = 0
                            batched_other_data = list()

                            curr_time_complex_if = time.time() - time_begin_line
                            logger.debug(f'sec. {curr_time_complex_if} '
                                         f'to complete the complex if')
                            time_to_process_lines += curr_time_complex_if
                        nr_lines_processed += 1
                        avg_time_to_process_lines = (time_to_process_lines / nr_lines_processed)
                        logger.debug('+++++++++++++++++++++++++++++')
                        logger.debug(f'sec. AVG to process line: {avg_time_to_process_lines}')
                        logger.debug('+++++++++++++++++++++++++++++')
            # here should start
            # logger.info(f'processing {curr_input_file}')
            #
            logger.info(f'file {curr_input_file} finished, processing whatever is left in the batch')
            batch_idx_to_content = dict()

            ###########################
            res_batch = None
            if len(batched_entity_ids) > 0:
                time_res_batch = time.time()
                if batching_type == 'method1':
                    res_batch = batch_tensor_method1(
                        curr_batch_max_length=curr_batch_max_entity_len,
                        batched_entity_ids=batched_entity_ids,
                        device=str_device2
                    )
                elif batching_type == 'method2':
                    res_batch = batch_tensor_method2(
                        batched_entity_ids=batched_entity_ids,
                        device=str_device2
                    )
                else:
                    raise RuntimeError(f'batching_type not recognized: f{batching_type}')

                logger.debug(f'{time.time() - time_res_batch} '
                             f'sec. time it took to get res_batch of shape: '
                             f'{res_batch.shape}')

            ###########################

            for idx_in_batch, curr_batch_other in enumerate(batched_other_data):

                curr_output_path_file = curr_batch_other['output_file_path']
                if res_batch is not None:
                    tn_entity_ids = res_batch[idx_in_batch]
                    len_entity_ids = batched_entity_ids_len[idx_in_batch]
                    json_input = connect_interesting_snippets_with_kg(
                        property_qid_to_label=property_qid_to_label,
                        interval_from=curr_batch_other['interval_from'],
                        interval_to=curr_batch_other['interval_to'],
                        json_input=curr_batch_other['json_input'],
                        str_device=str_device,
                        str_device2=str_device2,
                        tensor_timestamps_wdata=tensor_timestamps_wdata,
                        wdata_graph=wdata_graph,
                        tn_entity_ids=tn_entity_ids,
                        len_entity_ids=len_entity_ids,
                        index_to_entity=index_to_entity,
                        index_to_relation=index_to_relation,
                        matched_triples=curr_batch_other['matched_triples'],
                        wikidata_qid_to_label=wikidata_qid_to_label,
                        tensor_qualifier_timestamps_wdata=tensor_qualifier_timestamps_wdata
                    )
                batch_idx_to_content[idx_in_batch] = {
                    'curr_output_path_file': curr_output_path_file,
                    'json_input': json_input
                }
                #### END: obtaining all the triple connections in KG at that point in time
                # time_begin_write_flush = time.time()
                # output_files[curr_output_path_file].write(json.dumps(json_input) + '\n')
                # output_files[curr_output_path_file].flush()
                # logger.debug(f'sec. {time.time() - time_begin_write_flush} '
                #              f'to complete the write and flush')

            #######
            time_begin_write_flush = time.time()
            for curr_batch_idx, curr_batch_content in batch_idx_to_content.items():
                curr_output_path_file = curr_batch_content['curr_output_path_file']
                curr_json_input = curr_batch_content['json_input']
                output_files[curr_output_path_file].write(
                    json.dumps(curr_json_input) + '\n')
                output_files[curr_output_path_file].flush()
            #
            # last_successfully_processed_line = line_idx
            with open(last_processed_file, 'w', encoding='utf-8') as lp_file:
                lp_file.write(str(line_idx))
                lp_file.flush()

            logger.debug(f'sec. {time.time() - time_begin_write_flush} '
                         f'to complete the write and flush')

            #######
            batched_entity_ids = list()
            batched_entity_ids_len = list()
            curr_batch_max_entity_len = 0
            batched_other_data = list()
            # logger.info(f'sec. {time.time() - time_begin_line} for complete the line')
