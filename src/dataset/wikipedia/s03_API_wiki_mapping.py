"""
FastAPI server that loads Wikipedia SQL tables, redirect maps,
and QID mappings into memory. Used by s03_extract_entity_descriptions.py to
resolve page IDs to QIDs and check entity creation timestamps.

Based on s08_API_v1_wiki_mapping.py from wikidata-temp/wikipedia-temp.
"""

import argparse
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict

import psutil
import uvicorn
from fastapi import FastAPI
from tqdm import tqdm

from .misc.load_wiki_sql_tables import (load_wiki_page_title_to_wiki_page_id,
                                         load_wiki_page_id_to_redirected_page_id,
                                         load_wiki_page_id_to_wikidata_qid,
                                         load_wdata_qid_to_page_ids)

_LOG_LEVEL = logging._nameToLevel.get(
    os.environ.get('LOGGING_LEVEL', '').strip(), logging.INFO
)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=_LOG_LEVEL)
logger = logging.getLogger(__name__)


# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAPI command-line arguments, including config file.')
    parser.add_argument('--config_file',
                        default='config/dataset/wikipedia/s03_extract_entity_descriptions/'
                                '20251101_slurm_english/config.json',
                        type=str,
                        required=False,
                        help='Path to config file.')

    parser.add_argument('--dry_run',
                        help='Dry run, API returns hardcoded values, nothing '
                             'is loaded. ', action='store_true')

    parser.add_argument('--debug_size',
                        help='The debug size.',
                        type=int,
                        default=-1)

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
config['api_port'] = args.api_port


def show_free_memory(step: str):
    # Get the available memory
    free_memory = psutil.virtual_memory().available

    # Convert bytes to megabytes
    free_memory_mb = free_memory / (1024 * 1024)

    logger.info(f"{free_memory_mb:.2f} MB free_memory at step {step}")


def get_lifespan_objects():
    caches_dir = config['caches_dir']

    os.makedirs(caches_dir, exist_ok=True)
    path_wikipedia_wikidata_map = config['path_wikipedia_wikidata_map']
    path_wikipedia_page_info = config['path_wikipedia_page_info']
    path_wikipedia_page_redirects = config['path_wikipedia_page_redirects']

    path_cache_wikipedia_page_id_to_wikidata_qid = os.path.join(caches_dir, 'wikipedia_page_id_to_wikidata_qid.pickle')
    path_cache_wikipedia_page_title_to_wikipedia_page_id = os.path.join(caches_dir,
                                                                        'wikipedia_page_title_to_wikipedia_page_id.pickle')
    path_cache_wikipedia_page_id_to_wikipedia_page_title = os.path.join(caches_dir,
                                                                        'wikipedia_page_id_to_wikipedia_page_title.pickle')
    path_cache_wikipedia_page_id_to_redirected_page_id = os.path.join(caches_dir,
                                                                      'wikipedia_page_id_to_redirected_page_id.pickle')
    wikidata_qid_to_creation_timestamps = dict()
    with open(config['qids_to_creation_date'], 'rt') as infile:
        for curr_line in tqdm(infile, desc='loading to wikidata_qid_to_creation_timestamps'):
            splitted_line = curr_line.split('\t')
            curr_qid = splitted_line[0].strip()
            curr_timestamp = int(splitted_line[1].strip())
            wikidata_qid_to_creation_timestamps[curr_qid] = curr_timestamp

    path_cache_qids_to_page_ids = os.path.join(caches_dir,
                                               'path_cache_qids_to_page_ids.pickle')

    # load_wiki_page_title_to_wiki_page_id is needed as input to load_wiki_page_id_to_redirected_page_id
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

    return {
        'wikipedia_page_id_to_redirected_page_id': wikipedia_page_id_to_redirected_page_id,
        'wikipedia_page_id_to_wikidata_qid': wikipedia_page_id_to_wikidata_qid,
        'wikipedia_page_id_to_wdata_qid_from_history': wikipedia_page_id_to_wdata_qid_from_history,
        'wikidata_qid_to_creation_timestamps': wikidata_qid_to_creation_timestamps
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    show_free_memory('starting lifespan')

    if config['dry_run']:
        yield
    else:
        l_o = get_lifespan_objects()

        logger.info('BEGIN assigning global variables')

        app.state.wikipedia_page_id_to_redirected_page_id = l_o['wikipedia_page_id_to_redirected_page_id']
        app.state.wikipedia_page_id_to_wikidata_qid = l_o['wikipedia_page_id_to_wikidata_qid']
        app.state.wikipedia_page_id_to_wdata_qid_from_history = l_o['wikipedia_page_id_to_wdata_qid_from_history']
        app.state.wikidata_qid_to_creation_timestamps = l_o['wikidata_qid_to_creation_timestamps']

        logger.info('END assigning global variables')
        # Yield control back to the application
        yield


app = FastAPI(lifespan=lifespan)


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

        return {
            'should_process_page': True,
            'page_qid': page_qid,
            'wikidata_creation_timestamp': wikidata_creation_timestamp
        }
    else:
        to_ret = {
            'should_process_page': True,
            'page_qid': 'Q123456'
        }
        logger.debug(f'dry_run should_process_page to return: {to_ret}')
        return to_ret


if __name__ == '__main__':
    uvicorn.run(app, host=config['api_host'], port=config['api_port'], log_level=logging.WARNING)
