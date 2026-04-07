"""
difference of s03_API_v6_wiki_mapping wrt s03_API_v5: only functions related
to mapping wikidata qids to titles, wikipedia titles to qids, etc.... This takes
lots of memory and it is fast, so a separate api is creted.
"""

import argparse
import csv
import json
import logging
import os
import pickle
import traceback
from argparse import Action
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Annotated, Tuple, Set

import psutil
import uvicorn
from fastapi import FastAPI, Query
from tqdm import tqdm

from dataset.wikipedia.misc.load_wiki_sql_tables import load_wiki_page_title_to_wiki_page_id, load_wiki_page_id_to_redirected_page_id, \
    load_wiki_page_id_to_wikidata_qid, load_wdata_qid_to_page_ids
from dataset.wikipedia.s02b_normalize_history_graph import get_page_title_changes, get_page_id_of_most_recent_title
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
# logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAPI command-line arguments, including config file.')
    parser.add_argument('--config_file',
                        default='experiments/s03_obtain_textual_delta_snippets_v3/20241122/'
                                's03_config_obtain_textual_delta.json',
                        type=str,
                        required=False,
                        help='Path to config file.')

    parser.add_argument('--dry_run',
                        help='Dry run, API returns hardcored values, nothing '
                             'is loaded. ', action='store_true')

    parser.add_argument('--debug_size',
                        help='The debug size.',
                        type=int,
                        default=-1)

    # parser.add_argument('--device',
    #                     help='The device to load tensors to, examples: "cpu", "cuda:1", '
    #                          '"cuda"....',
    #                     type=str,
    #                     required=False,
    #                     default='cpu')

    parser.add_argument('--api_port',
                        help='The port to deploy the API on.',
                        type=int,
                        required=True,
                        default=8000)

    return parser.parse_args()


# Load data from files specified in command-line arguments
args = parse_arguments()
config_file = args.config_file

config = json.load(open(config_file, 'rt'))

config['dry_run'] = args.dry_run
config['debug_size'] = args.debug_size
# config['device'] = args.device
config['api_port'] = args.api_port


def get_qid_from_page_id():
    pass


def show_free_memory(step: str):
    # Get the available memory
    free_memory = psutil.virtual_memory().available

    # Convert bytes to megabytes
    free_memory_mb = free_memory / (1024 * 1024)

    logger.info(f"{free_memory_mb:.2f} MB free_memory at step {step}")


def get_lifespan_objects():
    caches_dir = config['caches_dir']

    os.makedirs(caches_dir, exist_ok=True)
    # input_page_id_to_revisions_path = config['input_page_id_to_revisions_path']
    path_wikipedia_wikidata_map = config['path_wikipedia_wikidata_map']
    path_wikipedia_page_info = config['path_wikipedia_page_info']
    path_wikipedia_page_redirects = config['path_wikipedia_page_redirects']
    path_extracted_title_changes = config['path_extracted_title_changes']
    logger.debug('=====invoked get_page_title_changes=======')
    show_free_memory('get_page_title_changes before')
    page_title_changes: Dict = get_page_title_changes(path_extracted_title_changes)
    show_free_memory('get_page_title_changes after')

    # logger.info(f'loading revisions from {input_page_id_to_revisions_path}')
    path_cache_wikipedia_page_id_to_wikidata_qid = os.path.join(caches_dir, 'wikipedia_page_id_to_wikidata_qid.pickle')
    path_cache_wikipedia_page_title_to_wikipedia_page_id = os.path.join(caches_dir,
                                                                        'wikipedia_page_title_to_wikipedia_page_id.pickle')
    path_cache_wikipedia_page_id_to_wikipedia_page_title = os.path.join(caches_dir,
                                                                        'wikipedia_page_id_to_wikipedia_page_title.pickle')
    path_cache_wikipedia_page_id_to_redirected_page_id = os.path.join(caches_dir,
                                                                      'wikipedia_page_id_to_redirected_page_id.pickle')
    # path_cache
    wikidata_qid_to_creation_timestamps = dict()
    with open(config['qids_to_creation_date'], 'rt') as infile:
        for curr_line in tqdm(infile, desc='loading to wikidata_qid_to_creation_timestamps'):
            splitted_line = curr_line.split('\t')
            curr_qid = splitted_line[0].strip()
            curr_timestamp = int(splitted_line[1].strip())
            wikidata_qid_to_creation_timestamps[curr_qid] = curr_timestamp

    # app.state.wikidata_qid_to_creation_timestamps

    caches_dir = config['caches_dir']
    # config['input_delta_triples_path']
    # curr_hash = generate_short_hash(config['input_delta_triples_path'], hash_length=8)
    # caches_wikidata_qid_to_label_path = os.path.join(caches_dir,
    #                                                  f'wikidata_qid_to_label_{curr_hash}.pickle')
    logger.info('BEGIN invoking load_wikidata_qid_to_label')
    # path_wikidata_labels = config['path_wikidata_labels']

    # caches_property_qid_to_label_path = os.path.join(caches_dir, 'property_qid_to_label.pickle')
    # logger.info('BEGIN invoking load_property_qid_to_label')
    # path_property_labels = config['path_property_labels']

    # show_free_memory('load_property_qid_label before')
    # property_qid_to_label: Dict = load_property_qid_to_label(
    #     path_property_labels, caches_property_qid_to_label_path
    # )
    show_free_memory('load_property_qid_label after')
    logger.info('END invoking load_property_qid_to_label')

    path_cache_qids_to_page_ids = os.path.join(caches_dir,
                                               'path_cache_qids_to_page_ids.pickle')

    logger.info('BEGIN invoking load_wiki_page_title_to_wiki_page_id')
    show_free_memory('load_wiki_page_title_to_wiki_page_id before')
    wikipedia_page_title_to_wikipedia_page_id, wikipedia_page_id_to_wikipedia_page_title \
        = load_wiki_page_title_to_wiki_page_id(
        path_cache_wikipedia_page_title_to_wikipedia_page_id,
        path_cache_wikipedia_page_id_to_wikipedia_page_title,
        path_wikipedia_page_info)
    show_free_memory('load_wiki_page_title_to_wiki_page_id after')
    logger.info('END invoking load_wiki_page_title_to_wiki_page_id')

    logger.info('BEGIN invoking load_wiki_page_id_to_redirected_page_id')
    show_free_memory('load_wiki_page_id_to_redirected_page_id before')
    wikipedia_page_id_to_redirected_page_id = \
        load_wiki_page_id_to_redirected_page_id(path_cache_wikipedia_page_id_to_redirected_page_id,
                                                wikipedia_page_title_to_wikipedia_page_id,
                                                path_wikipedia_page_redirects)
    show_free_memory('load_wiki_page_id_to_redirected_page_id after')
    logger.info('END invoking load_wiki_page_id_to_redirected_page_id')

    logger.info('BEGIN invoking load_wiki_page_id_to_wikidata_qid')
    show_free_memory('load_wiki_page_id_to_wikidata_qid before')
    wikipedia_page_id_to_wikidata_qid = load_wiki_page_id_to_wikidata_qid(path_cache_wikipedia_page_id_to_wikidata_qid,
                                                                          path_wikipedia_wikidata_map)
    show_free_memory('load_wiki_page_id_to_wikidata_qid after')
    logger.info('END invoking load_wiki_page_id_to_wikidata_qid')

    logger.info('BEGIN obtaining wdata_qid_to_wpedia_page_id')
    wdata_qid_to_wpedia_page_id = {value: key for key, value in wikipedia_page_id_to_wikidata_qid.items()}
    logger.info('END obtaining wdata_qid_to_wpedia_page_id')

    logger.info('BEGIN invoking load_delta_triples')
    wdata_qid_to_wpedia_page_ids = dict()
    wikipedia_page_id_to_wdata_qid_from_history = dict()

    if config['include_all_page_ids_in_qid']:
        show_free_memory('include_all_page_ids_in_qid before')
        wdata_qid_to_wpedia_page_ids = load_wdata_qid_to_page_ids(
            path_cache_qids_to_page_ids=path_cache_qids_to_page_ids,
            qids_to_page_ids_path=config['qids_to_page_ids']
        )
        logger.info(f'BEGIN obtaining wikipedia_page_id_to_wdata_qid_from_history')
        for curr_wdata_qid, curr_wpedia_page_ids in wdata_qid_to_wpedia_page_ids.items():
            for curr_wpedia_page_id in curr_wpedia_page_ids:
                wikipedia_page_id_to_wdata_qid_from_history[curr_wpedia_page_id] = curr_wdata_qid
        logger.info(f'END obtaining wikipedia_page_id_to_wdata_qid_from_history')
        show_free_memory('include_all_page_ids_in_qid after')
        # for
    page_id_to_main_page_id: Dict

    show_free_memory('load_delta_triples_from_paths before')
    # loaded_delta = (
    #     load_delta_triples_from_paths(
    #         # wdata_qid_to_wpedia_page_id=wdata_qid_to_wpedia_page_id,
    #         # wdata_qid_to_wpedia_page_ids=wdata_qid_to_wpedia_page_ids,
    #         config=config
    #     ))
    # show_free_memory('load_delta_triples_from_paths after')
    #
    # logger.info('BEGIN invoking load_wikidata_qid_to_label')
    # show_free_memory('load_wikidata_qid_to_label before')
    # wikidata_qid_to_label: Dict = load_wikidata_qid_to_label(
    #     path_wikidata_labels,
    #     caches_wikidata_qid_to_label_path,
    #     set(loaded_delta['entity_to_index'].keys())
    # )
    show_free_memory('load_wikidata_qid_to_label after')

    logger.info('END invoking load_wikidata_qid_to_label')

    return {
        # 'entity_to_index': loaded_delta['entity_to_index'],
        # 'index_to_entity': loaded_delta['index_to_entity'],
        # 'index_to_relation': loaded_delta['index_to_relation'],
        # 'interval_ids_to_interval': loaded_delta['interval_ids_to_interval'],
        # 'intervals_to_interval_id': loaded_delta['intervals_to_interval_id'],
        # 'interval_ids_to_delta_intersection': loaded_delta['interval_ids_to_delta_intersection'],
        # 'interval_ids_to_delta_emerging': loaded_delta['interval_ids_to_delta_emerging'],
        # 'interval_ids_to_delta_all': loaded_delta['interval_ids_to_delta_all'],
        'page_title_changes': page_title_changes,
        'wikipedia_page_title_to_wikipedia_page_id': wikipedia_page_title_to_wikipedia_page_id,
        'wikipedia_page_id_to_wikipedia_page_title': wikipedia_page_id_to_wikipedia_page_title,
        'wikipedia_page_id_to_redirected_page_id': wikipedia_page_id_to_redirected_page_id,
        'wikipedia_page_id_to_wikidata_qid': wikipedia_page_id_to_wikidata_qid,
        'wikipedia_page_id_to_wdata_qid_from_history': wikipedia_page_id_to_wdata_qid_from_history,
        'wikidata_qid_to_creation_timestamps':wikidata_qid_to_creation_timestamps
        # 'wikidata_qid_to_label': wikidata_qid_to_label,
        # 'property_qid_to_label': property_qid_to_label
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    show_free_memory('starting lifespan')

    if config['dry_run']:
        yield
    else:
        l_o = get_lifespan_objects()

        logger.info('BEGIN assigning global variables')

        # app.state.entity_to_index = l_o['entity_to_index']
        # app.state.index_to_entity = l_o['index_to_entity']
        # app.state.index_to_relation = l_o['index_to_relation']
        # #
        # app.state.interval_ids_to_interval = l_o['interval_ids_to_interval']
        # app.state.intervals_to_interval_id = l_o['intervals_to_interval_id']
        # app.state.interval_ids_to_delta_intersection = l_o['interval_ids_to_delta_intersection']
        # app.state.interval_ids_to_delta_emerging = l_o['interval_ids_to_delta_emerging']
        # app.state.interval_ids_to_delta_all = l_o['interval_ids_to_delta_all']

        app.state.page_title_changes = l_o['page_title_changes']
        app.state.wikipedia_page_title_to_wikipedia_page_id = l_o['wikipedia_page_title_to_wikipedia_page_id']
        app.state.wikipedia_page_id_to_wikipedia_page_title = l_o['wikipedia_page_id_to_wikipedia_page_title']
        app.state.wikipedia_page_id_to_redirected_page_id = l_o['wikipedia_page_id_to_redirected_page_id']
        app.state.wikipedia_page_id_to_wikidata_qid = l_o['wikipedia_page_id_to_wikidata_qid']
        app.state.wikipedia_page_id_to_wdata_qid_from_history = l_o['wikipedia_page_id_to_wdata_qid_from_history']
        app.state.wikidata_qid_to_creation_timestamps = l_o['wikidata_qid_to_creation_timestamps']
        # app.state.wikidata_qid_to_label = l_o['wikidata_qid_to_label']
        # app.state.property_qid_to_label = l_o['property_qid_to_label']
        app.state.nr_found_triples = 0
        app.state.nr_calls_triples = 0
        app.state.tot_size_triples = 0

        logger.info('END assigning global variables')
        # Yield control back to the application
        yield

        # Cleanup can be done here if needed
        # app.state.page_id_to_rev_to_timestamps = None
        # app.state.page_id_to_rev_to_tail_ids = None
        # app.state.data_from_file2 = None


app = FastAPI(lifespan=lifespan)


@app.get('/get_entities_ids')
def get_entities_ids(timestamp: int, page_titles: Annotated[list[str] | None, Query()]):
    to_ret_page_ids_qids = []
    to_ret_page_titles = []

    for curr_title in page_titles:
        get_pid_result = get_page_id_func(timestamp=timestamp,
                                          page_title=curr_title)
        if get_pid_result['page_qid'] is None:
            continue
        to_ret_page_ids_qids.append(get_pid_result['page_qid'])
        to_ret_page_titles.append(get_pid_result['page_title'])

    return {
        "page_titles": to_ret_page_titles,
        "page_qids": to_ret_page_ids_qids
    }


@app.get('/should_process_page')
def should_process_page(page_id: int):
    logger.debug(f'invoked should_process_page with page_id {page_id}')
    if not config['dry_run']:
        # emulates page_id_to_main_page_id to obtain current qid
        page_qid = -1
        if page_id in app.state.wikipedia_page_id_to_redirected_page_id:
            page_id = app.state.wikipedia_page_id_to_redirected_page_id[page_id]
        if page_id in app.state.wikipedia_page_id_to_wikidata_qid:
            page_qid = app.state.wikipedia_page_id_to_wikidata_qid[page_id]
        elif page_id in app.state.wikipedia_page_id_to_wdata_qid_from_history:
            page_qid = app.state.wikipedia_page_id_to_wdata_qid_from_history[page_id]
        # all the pages get processed
        wikidata_creation_timestamp = -1
        if page_qid in app.state.wikidata_qid_to_creation_timestamps:
            wikidata_creation_timestamp = \
                app.state.wikidata_qid_to_creation_timestamps[page_qid]
        else:
            logger.warning(f'wikidata_creation_timestamp not found for {page_qid}')
        #
        should_process_page_v = True

        if should_process_page_v:
            return {
                'should_process_page': should_process_page_v,
                'page_qid': page_qid,
                'wikidata_creation_timestamp': wikidata_creation_timestamp
            }
        else:
            return {
                'should_process_page': should_process_page_v,
                'page_qid': ''
            }
    else:
        to_ret = {
            'should_process_page': True,
            'page_qid': 'Q123456'
        }
        logger.debug(f'dry_run should_process_page to return: {to_ret}')
        return to_ret


def get_page_id_func(timestamp: int, page_title: str):
    to_ret_page_id = None
    if page_title not in app.state.wikipedia_page_title_to_wikipedia_page_id:
        to_ret_page_id = get_page_id_of_most_recent_title(
            page_title,
            int(timestamp),
            app.state.page_title_changes,
            page_id_to_page_title=app.state.wikipedia_page_id_to_wikipedia_page_title,
            page_id=None,
            do_not_return_disambiguations=True
        )
    else:
        to_ret_page_id = app.state.wikipedia_page_title_to_wikipedia_page_id[page_title]
    to_ret_page_qid = None
    to_ret_page_title = None
    if to_ret_page_id is not None:
        nr_redirects_found = 0

        try:
            while to_ret_page_id in app.state.wikipedia_page_id_to_redirected_page_id:
                to_ret_page_id = app.state.wikipedia_page_id_to_redirected_page_id[to_ret_page_id]
                nr_redirects_found += 1
                if nr_redirects_found > 100:
                    break
            # assert target_page_id not in wikipedia_page_id_to_redirected_page_id
            if to_ret_page_id in app.state.wikipedia_page_id_to_redirected_page_id:
                logger.error('following target_page_id in wikipedia_page_id_to_redirected_page_id: '
                             f'{to_ret_page_id} with value of '
                             f'{app.state.
                             wikipedia_page_id_to_redirected_page_id[to_ret_page_id]}')
        except Exception as e:
            logger.error(f'An error occurred: {e}')
            traceback.print_exc()

        if to_ret_page_id in app.state.wikipedia_page_id_to_wikidata_qid:
            to_ret_page_qid = app.state.wikipedia_page_id_to_wikidata_qid[to_ret_page_id]
        else:
            logger.warning(f'to_ret_page_id ({to_ret_page_id}) not in wikipedia_page_id_to_wikidata_qid '
                           f'when calling get_page_id_func with timestamp {timestamp} and '
                           f'page_title {page_title}')
            return {
                'page_id': None,
                'page_qid': None,
                'page_title': None
            }
        if to_ret_page_id in app.state.wikipedia_page_id_to_wikipedia_page_title:
            to_ret_page_title = app.state.wikipedia_page_id_to_wikipedia_page_title[to_ret_page_id]
        else:
            logger.warning(f'to_ret_page_id ({to_ret_page_id}) not in wikipedia_page_id_to_wikipedia_page_title '
                           f'when calling get_page_id_func with timestamp {timestamp} and '
                           f'page_title {page_title} , extracted qid: {to_ret_page_qid}')
            return {
                'page_id': None,
                'page_qid': None,
                'page_title': None
            }

    to_ret = {
        'page_id': to_ret_page_id,
        'page_qid': to_ret_page_qid,
        'page_title': page_title,
        'page_title_normalized': to_ret_page_title
    }
    return to_ret


@app.get('/get_page_ids')
def get_page_ids(timestamp, page_titles):
    if config['dry_run']:
        to_ret = {
            'page_ids': [1],
            'page_qids': ['Q123'],
            'page_titles': ['Hello_world']
        }
        logger.debug(f'dry_run get_page_id to return: {to_ret}')
        return to_ret

    to_ret = dict()
    to_ret_page_ids = list()
    to_ret_page_qids = list()
    # to_ret_page_titles = list()
    for curr_title in page_titles:
        # to_ret.append(get_page_id_func(timestamp=timestamp,
        #                                page_title=curr_title))
        page_info = get_page_id_func(timestamp=timestamp,
                                     page_title=curr_title)
        if page_info['page_qid'] is None:
            continue
        page_id = page_info['page_id']
        page_qid = page_info['page_qid']
        to_ret_page_ids.append(page_id)
        to_ret_page_qids.append(page_qid)

    to_ret['page_ids'] = to_ret_page_ids
    to_ret['page_titles'] = page_titles
    to_ret['page_qids'] = to_ret_page_qids
    return to_ret


@app.get('/get_page_id')
def get_page_id(timestamp, page_title):
    if config['dry_run']:
        to_ret = {
            'page_id': 1,
            'page_qid': 'Q123',
            'page_title': 'Hello_world'
        }
        logger.debug(f'dry_run get_page_id to return: {to_ret}')
        return to_ret

    return get_page_id_func(timestamp=timestamp,
                            page_title=page_title)


if __name__ == '__main__':
    # parse_arguments()
    # Run the FastAPI app on http://127.0.0.1:8000
    # uvicorn.run(app, host=config['api_host'], port=config['api_port'], log_level=logging.WARNING)
    uvicorn.run(app, host=config['api_host'], port=config['api_port'], log_level=logging.WARNING)

# if not curr_is_in_intersection:
#     continue
#         rows_to_write.append([
#                               0-  subgraph_idx,
#                               1-  curr_action_label,
#                               2-  curr_head_qid,
#                               3-  curr_head_label,
#                               4-  curr_relation_qid,
#                               5-  curr_relation_label,
#                               6-  curr_target_qid,
#                               7-  curr_tail_label,
#                               8-  head_changes,
#                               9-  head_normalized_changes,
#                               10- tail_changes,
#                               11- tail_normalized_changes,
#                               12- is_in_intersection,
#                               13- curr_head_creation_timestamp,
#                               14- curr_tail_creation_timestamp,
#                               15- timestamp_from,
#                               16- timestamp_to,
#                               17- curr_triple_timestamp_from,
#                               18- curr_triple_timestamp_to
#                               ])
