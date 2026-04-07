"""
This _v8 extends _v6 and intends to differentiate between the triples involving only
mentions in text and the ones where only head or tail are found in text the other
entity (head or tail) is in delta. The function changed is
get_entities_in_triples_from_deltas.
"""

import argparse
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Annotated, Tuple, Set

import psutil
import torch
import uvicorn
from dateutil.relativedelta import relativedelta
from fastapi import FastAPI, Query
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr

import os
from dataset.emerge.utils.constants import AttrIndexes
from dataset.emerge.utils.s03_v2_temporal_interval import TemporalInterval
from dataset.emerge.utils.wiki_utils import load_wikidata_qid_to_label, load_property_qid_to_label, generate_short_hash
from torch_geometric.data.data import DataTensorAttr

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
# logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def load_delta_triples_from_paths(config: Dict):
    head_id_2_temporal_interval_2_target_ids = dict()
    page_id_to_main_page_id = dict()

    interval_ids_to_interval = dict()
    intervals_to_interval_id = dict()
    interval_ids_to_delta_intersection = dict()
    interval_ids_to_delta_emerging = dict()
    interval_ids_to_delta_all = dict()

    input_idxs_path = os.path.join(config['input_delta_triples_path'],
                                   'idx_entities_rels.pt')
    idxs_entities_rels: Dict
    idxs_entities_rels = torch.load(input_idxs_path, weights_only=False)

    entity_to_index = idxs_entities_rels['entity_to_index']
    index_to_entity = idxs_entities_rels['index_to_entity']
    index_to_relation = idxs_entities_rels['index_to_relation']

    interval_id = 0
    for curr_delta_granularity in config['delta_intervals_granularities']:
        # for interval_id, curr_delta_interval_start in enumerate(config['delta_intervals_start']):
        for curr_delta_interval_start_str in config['delta_intervals_start']:
            curr_nr_delta_intervals = curr_delta_granularity['nr_delta_intervals']

            # Convert to datetime object with time set to 00:00:00
            date_object = datetime.strptime(curr_delta_interval_start_str, "%Y-%m-%d")

            # Get the timestamp
            curr_delta_interval_start = int(date_object.timestamp())

            date_from = datetime.fromtimestamp(curr_delta_interval_start)
            date_from_str = date_from.strftime('%Y%m%d')
            curr_delta_type = str(curr_delta_granularity['granularity'])
            curr_path = os.path.join(str(config['input_delta_triples_path']),
                                     curr_delta_type,
                                     date_from_str
                                     )
            # date_to = datetime.fromtimestamp(curr_delta_interval_start)
            date_to = None
            for curr_interval_delta_offset in range(curr_nr_delta_intervals):
                if curr_delta_type == 'weekly':
                    date_to = date_from + relativedelta(weeks=(curr_interval_delta_offset + 1))  # Add 1 weeks
                elif curr_delta_type == 'monthly':
                    date_to += relativedelta(months=(curr_interval_delta_offset + 1))
                else:
                    raise RuntimeError(f'curr_delta_granularity not recognized: '
                                       f'{curr_delta_type}')
                date_to_str = date_to.strftime('%Y%m%d')

                curr_file_delta_intersec_path = (
                    os.path.join(curr_path,
                                 f'{date_from_str}_{date_to_str}_delta_intersection.txt.pt'))

                curr_file_delta_emerging_path = (
                    os.path.join(curr_path,
                                 f'{date_from_str}_{date_to_str}_delta_emerging_wdata.txt.pt'))

                # 20230101_20230305_delta_wdata.txt.pt
                curr_file_delta_all_path = (
                    os.path.join(curr_path,
                                 f'{date_from_str}_{date_to_str}_delta_wdata.txt.pt'))

                curr_interval = TemporalInterval(
                    granularity=curr_delta_type,
                    interval_start=curr_delta_interval_start,
                    interval_end=int(date_to.timestamp()),
                    nr_deltas=curr_nr_delta_intervals
                )
                logger.info(f'created_interval curr_interval: {curr_interval}')
                #
                if not os.path.exists(curr_file_delta_intersec_path):
                    logger.warning(f'The following path does not exist: {curr_file_delta_intersec_path}')
                    raise RuntimeError(f'The following path does not exist: {curr_file_delta_intersec_path}')
                    # continue
                    #
                if not os.path.exists(curr_file_delta_emerging_path):
                    logger.warning(f'The following path does not exist: {curr_file_delta_emerging_path}')
                    raise RuntimeError(f'The following path does not exist: {curr_file_delta_emerging_path}')
                    # continue

                if not os.path.exists(curr_file_delta_all_path):
                    logger.warning(f'The following path does not exist: {curr_file_delta_emerging_path}')
                    raise RuntimeError(f'The following path does not exist: {curr_file_delta_emerging_path}')
                    # continue

                loaded_delta_intersection: Data = torch.load(curr_file_delta_intersec_path,
                                                             map_location=config['device'],
                                                             weights_only=False)
                loaded_delta_emerging: Data = torch.load(curr_file_delta_emerging_path,
                                                         map_location=config['device'],
                                                         weights_only=False)
                loaded_delta_all: Data = torch.load(curr_file_delta_all_path,
                                                    map_location=config['device'],
                                                    weights_only=False)
                #
                interval_ids_to_delta_intersection[interval_id] = loaded_delta_intersection
                interval_ids_to_delta_emerging[interval_id] = loaded_delta_emerging
                interval_ids_to_delta_all[interval_id] = loaded_delta_all

                interval_ids_to_interval[interval_id] = curr_interval
                intervals_to_interval_id[curr_interval] = interval_id
                #
                interval_id += 1

    # (interval_ids_to_interval, intervals_to_interval_id, interval_ids_to_delta,
    #      entity_to_index, index_to_entity, index_to_relation)

    return {
        'interval_ids_to_interval': interval_ids_to_interval,
        'intervals_to_interval_id': intervals_to_interval_id,
        'interval_ids_to_delta_intersection': interval_ids_to_delta_intersection,
        'interval_ids_to_delta_emerging': interval_ids_to_delta_emerging,
        'interval_ids_to_delta_all': interval_ids_to_delta_all,
        'entity_to_index': entity_to_index,
        'index_to_entity': index_to_entity,
        'index_to_relation': index_to_relation
    }
    # return interval_ids_to_interval, intervals_to_interval_id, interval_ids_to_delta_intersection, \
    #     entity_to_index, index_to_entity, index_to_relation


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

    parser.add_argument('--device',
                        help='The device to load tensors to, examples: "cpu", "cuda:1", '
                             '"cuda"....',
                        type=str,
                        required=False,
                        default='cpu')

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
config['device'] = args.device
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
    torch.serialization.add_safe_globals([DataTensorAttr])

    os.makedirs(caches_dir, exist_ok=True)
    input_page_id_to_revisions_path = config['input_page_id_to_revisions_path']
    path_wikipedia_wikidata_map = config['path_wikipedia_wikidata_map']
    path_wikipedia_page_info = config['path_wikipedia_page_info']
    path_wikipedia_page_redirects = config['path_wikipedia_page_redirects']
    path_extracted_title_changes = config['path_extracted_title_changes']
    logger.debug('=====invoked get_page_title_changes=======')
    # show_free_memory('get_page_title_changes before')
    # page_title_changes: Dict = get_page_title_changes(path_extracted_title_changes)
    # show_free_memory('get_page_title_changes after')

    logger.info(f'loading revisions from {input_page_id_to_revisions_path}')
    path_cache_wikipedia_page_id_to_wikidata_qid = os.path.join(caches_dir, 'wikipedia_page_id_to_wikidata_qid.pickle')
    path_cache_wikipedia_page_title_to_wikipedia_page_id = os.path.join(caches_dir,
                                                                        'wikipedia_page_title_to_wikipedia_page_id.pickle')
    path_cache_wikipedia_page_id_to_wikipedia_page_title = os.path.join(caches_dir,
                                                                        'wikipedia_page_id_to_wikipedia_page_title.pickle')
    path_cache_wikipedia_page_id_to_redirected_page_id = os.path.join(caches_dir,
                                                                      'wikipedia_page_id_to_redirected_page_id.pickle')

    caches_dir = config['caches_dir']
    # config['input_delta_triples_path']
    curr_hash = generate_short_hash(config['input_delta_triples_path'], hash_length=8)
    # caches_wikidata_qid_to_label_path = os.path.join(caches_dir,
    #                                                  f'wikidata_qid_to_label_{curr_hash}.pickle')
    logger.info('BEGIN invoking load_wikidata_qid_to_label')
    path_wikidata_labels = config['path_wikidata_labels']

    caches_property_qid_to_label_path = os.path.join(caches_dir, 'property_qid_to_label.pickle')
    logger.info('BEGIN invoking load_property_qid_to_label')
    path_property_labels = config['path_property_labels']

    # show_free_memory('load_property_qid_label before')
    # property_qid_to_label: Dict = load_property_qid_to_label(
    #     path_property_labels, caches_property_qid_to_label_path
    # )
    # show_free_memory('load_property_qid_label after')
    logger.info('END invoking load_property_qid_to_label')

    path_cache_qids_to_page_ids = os.path.join(caches_dir,
                                               'path_cache_qids_to_page_ids.pickle')

    logger.info('BEGIN invoking load_wiki_page_title_to_wiki_page_id')
    # show_free_memory('load_wiki_page_title_to_wiki_page_id before')
    # wikipedia_page_title_to_wikipedia_page_id, wikipedia_page_id_to_wikipedia_page_title \
    #     = load_wiki_page_title_to_wiki_page_id(
    #     path_cache_wikipedia_page_title_to_wikipedia_page_id,
    #     path_cache_wikipedia_page_id_to_wikipedia_page_title,
    #     path_wikipedia_page_info)
    # show_free_memory('load_wiki_page_title_to_wiki_page_id after')
    logger.info('END invoking load_wiki_page_title_to_wiki_page_id')

    logger.info('BEGIN invoking load_wiki_page_id_to_redirected_page_id')
    # show_free_memory('load_wiki_page_id_to_redirected_page_id before')
    # wikipedia_page_id_to_redirected_page_id = \
    #     load_wiki_page_id_to_redirected_page_id(path_cache_wikipedia_page_id_to_redirected_page_id,
    #                                             wikipedia_page_title_to_wikipedia_page_id,
    #                                             path_wikipedia_page_redirects)
    # show_free_memory('load_wiki_page_id_to_redirected_page_id after')
    logger.info('END invoking load_wiki_page_id_to_redirected_page_id')

    logger.info('BEGIN invoking load_wiki_page_id_to_wikidata_qid')
    # show_free_memory('load_wiki_page_id_to_wikidata_qid before')
    # wikipedia_page_id_to_wikidata_qid = load_wiki_page_id_to_wikidata_qid(path_cache_wikipedia_page_id_to_wikidata_qid,
    #                                                                       path_wikipedia_wikidata_map)
    # show_free_memory('load_wiki_page_id_to_wikidata_qid after')
    logger.info('END invoking load_wiki_page_id_to_wikidata_qid')

    # logger.info('BEGIN obtaining wdata_qid_to_wpedia_page_id')
    # wdata_qid_to_wpedia_page_id = {value: key for key, value in wikipedia_page_id_to_wikidata_qid.items()}
    # logger.info('END obtaining wdata_qid_to_wpedia_page_id')

    logger.info('BEGIN invoking load_delta_triples')
    wdata_qid_to_wpedia_page_ids = dict()
    wikipedia_page_id_to_wdata_qid_from_history = dict()

    # if config['include_all_page_ids_in_qid']:
    #     show_free_memory('include_all_page_ids_in_qid before')
    #     wdata_qid_to_wpedia_page_ids = load_wdata_qid_to_page_ids(
    #         path_cache_qids_to_page_ids=path_cache_qids_to_page_ids,
    #         qids_to_page_ids_path=config['qids_to_page_ids']
    #     )
    #     logger.info(f'BEGIN obtaining wikipedia_page_id_to_wdata_qid_from_history')
    #     for curr_wdata_qid, curr_wpedia_page_ids in wdata_qid_to_wpedia_page_ids.items():
    #         for curr_wpedia_page_id in curr_wpedia_page_ids:
    #             wikipedia_page_id_to_wdata_qid_from_history[curr_wpedia_page_id] = curr_wdata_qid
    #     logger.info(f'END obtaining wikipedia_page_id_to_wdata_qid_from_history')
    #     show_free_memory('include_all_page_ids_in_qid after')
    #     # for
    page_id_to_main_page_id: Dict

    # show_free_memory('load_delta_triples_from_paths before')
    loaded_delta = (
        load_delta_triples_from_paths(
            # wdata_qid_to_wpedia_page_id=wdata_qid_to_wpedia_page_id,
            # wdata_qid_to_wpedia_page_ids=wdata_qid_to_wpedia_page_ids,
            config=config
        ))
    # show_free_memory('load_delta_triples_from_paths after')

    logger.info('BEGIN invoking load_wikidata_qid_to_label')
    # show_free_memory('load_wikidata_qid_to_label before')
    # wikidata_qid_to_label: Dict = load_wikidata_qid_to_label(
    #     path_wikidata_labels,
    #     caches_wikidata_qid_to_label_path,
    #     set(loaded_delta['entity_to_index'].keys())
    # )
    # show_free_memory('load_wikidata_qid_to_label after')

    logger.info('END invoking load_wikidata_qid_to_label')

    return {
        'entity_to_index': loaded_delta['entity_to_index'],
        'index_to_entity': loaded_delta['index_to_entity'],
        'index_to_relation': loaded_delta['index_to_relation'],
        'interval_ids_to_interval': loaded_delta['interval_ids_to_interval'],
        'intervals_to_interval_id': loaded_delta['intervals_to_interval_id'],
        'interval_ids_to_delta_intersection': loaded_delta['interval_ids_to_delta_intersection'],
        'interval_ids_to_delta_emerging': loaded_delta['interval_ids_to_delta_emerging'],
        'interval_ids_to_delta_all': loaded_delta['interval_ids_to_delta_all'],
        # 'page_title_changes': page_title_changes,
        # 'wikipedia_page_title_to_wikipedia_page_id': wikipedia_page_title_to_wikipedia_page_id,
        # 'wikipedia_page_id_to_wikipedia_page_title': wikipedia_page_id_to_wikipedia_page_title,
        # 'wikipedia_page_id_to_redirected_page_id': wikipedia_page_id_to_redirected_page_id,
        # 'wikipedia_page_id_to_wikidata_qid': wikipedia_page_id_to_wikidata_qid,
        # 'wikipedia_page_id_to_wdata_qid_from_history': wikipedia_page_id_to_wdata_qid_from_history,
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

        app.state.entity_to_index = l_o['entity_to_index']
        app.state.index_to_entity = l_o['index_to_entity']
        app.state.index_to_relation = l_o['index_to_relation']
        #
        app.state.interval_ids_to_interval = l_o['interval_ids_to_interval']
        app.state.intervals_to_interval_id = l_o['intervals_to_interval_id']
        app.state.interval_ids_to_delta_intersection = l_o['interval_ids_to_delta_intersection']
        app.state.interval_ids_to_delta_emerging = l_o['interval_ids_to_delta_emerging']
        app.state.interval_ids_to_delta_all = l_o['interval_ids_to_delta_all']

        #
        # app.state.page_title_changes = l_o['page_title_changes']
        # app.state.wikipedia_page_title_to_wikipedia_page_id = l_o['wikipedia_page_title_to_wikipedia_page_id']
        # app.state.wikipedia_page_id_to_wikipedia_page_title = l_o['wikipedia_page_id_to_wikipedia_page_title']
        # app.state.wikipedia_page_id_to_redirected_page_id = l_o['wikipedia_page_id_to_redirected_page_id']
        # app.state.wikipedia_page_id_to_wikidata_qid = l_o['wikipedia_page_id_to_wikidata_qid']
        # app.state.wikipedia_page_id_to_wdata_qid_from_history = l_o['wikipedia_page_id_to_wdata_qid_from_history']
        # app.state.wikidata_qid_to_label = l_o['wikidata_qid_to_label']
        # app.state.property_qid_to_label = l_o['property_qid_to_label']
        #
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


@app.get('/get_temporal_intervals')
def get_temporal_intervals():
    if not config['dry_run']:
        to_ret_interval_ids_to_interval = {
            iid: (curr_int.interval_start, curr_int.interval_end)
            for iid, curr_int in
            app.state.interval_ids_to_interval.items()
        }
        to_ret_intervals_to_interval_id = {
            (curr_int.interval_start, curr_int.interval_end): iid
            for curr_int, iid in
            app.state.intervals_to_interval_id.items()
        }
        to_ret = {
            'interval_ids_to_interval': to_ret_interval_ids_to_interval,
        }
        return to_ret
    else:
        pass


def get_triples(
        already_added_triples: Set[Tuple[int, int, int]],
        already_added_triples_to_kg: Set[Tuple[int, int, int]],
        already_added_entities: Set[int], delta_type: str,
        tn_entities_idxs_to_search: torch.Tensor, interval_id_delta: Data, index_to_entity: Dict[int, str],
        index_to_relation: Dict[int, str],
        interval_from_timestamp: int,
        interval_to_timestamp: int,
        nr_triples_with_emerging_heads: int,
        nr_triples_with_emerging_tails: int,
        triples_lst: List,
        triples_head_or_tail_to_kg_lst: List
):
    """

    :param already_added_triples_to_kg:
    :param triples_head_or_tail_to_kg_lst:
    :param interval_to_timestamp:
    :param triples_lst: the list of triples, only adds the triples that are new and have not been added before
    (i.e., do not exist in already_added_triples)
    :param nr_triples_with_emerging_heads: nr of triples where head entities appear in the interval starting at
    interval_from_timestamp
    :param nr_triples_with_emerging_tails: nr of triples where tail entities appear in the interval starting at
    interval_from_timestamp
    :param interval_from_timestamp: the timestamp from which an entity is considered emerging
    :param index_to_relation: relation idx in pytorch geometric interval_id_delta to Wikidata property qid
    :param index_to_entity: entity idx in pytorch geometric interval_id_delta to Wikidata entity qid
    :param interval_id_delta: the delta against which the entities_to_search will be matched.
    :param tn_entities_idxs_to_search: the entities to search in deltas (e.g., coming from a paragraph in Wikipedia)
    :param already_added_triples:
    :param already_added_entities:
    :param delta_type: like "intersection" or "emerging_entities" or "all", not sure if it useful, since in practice each
    triple can have different types, like be containing "emerging_entities", but also be in "intersection" with wikipedia
    :return:
    """
    edge_index: torch.Tensor = interval_id_delta.edge_index
    edge_attr: torch.Tensor = interval_id_delta.edge_attr

    edge_index_mask = torch.isin(edge_index, tn_entities_idxs_to_search)

    #### 2025.01.23 - BEGIN introducing s03_API_v5 core functionality
    # triples_mask_between_mentions = (edge_index_mask[0, :] & edge_index_mask[1, :]) | \
    #                                 (edge_index_mask[0, :] & (
    #                                         edge_attr[:,
    #                                         AttrIndexes.IDX_DELTA_WDATA_HEAD_CREATION_DATE] > interval_from_timestamp)) | \
    #                                 (edge_index_mask[1, :] & (
    #                                         edge_attr[:,
    #                                         AttrIndexes.IDX_DELTA_WDATA_TAIL_CREATION_DATE] > interval_from_timestamp))
    triples_mask_between_mentions = (edge_index_mask[0, :] & edge_index_mask[1, :])
    # logger.info('==============================')
    # logger.info(f'triples_mask.sum_after_first: {triples_mask.sum()}')
    # deleteme_triples_mask_sum1 = triples_mask_between_mentions.sum()
    # deleteme_triples_mask_sum2 = 0
    # deleteme_triples_mask_sum3 = 0
    triples_mask_from_head = None
    if config['match_all_emerging_relations_from_head']:
        if config['match_only_emerging_entities_to_kg']:
            # here connects only emerging entities in snippet to KG when they are head
            triples_mask_from_head = (edge_index_mask[0, :] &
                                      (edge_attr[:,
                                       AttrIndexes.IDX_DELTA_WDATA_HEAD_CREATION_DATE] > interval_from_timestamp) &
                                      (edge_attr[:,
                                       AttrIndexes.IDX_DELTA_WDATA_TAIL_CREATION_DATE] < interval_from_timestamp) &  # be sure that the entity emerging entity is connected to already exists in the graph and is not emerging itself
                                      ((edge_attr[:, AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_TO] >
                                        interval_to_timestamp + config['triple_stability_offset_in_secs'])
                                       |
                                       (edge_attr[:, AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_TO] == 0))
                                      )


        else:
            # here connects all the entities mentioned in snippet to KG when they are head
            triples_mask_from_head = (edge_index_mask[0, :] & (
                    edge_attr[:, AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_FROM] > interval_from_timestamp) &
                                      ((edge_attr[:, AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_TO] >
                                        interval_to_timestamp + config['triple_stability_offset_in_secs'])
                                       |
                                       (edge_attr[:, AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_TO] == 0)
                                       )
                                      )

    triples_mask_from_tail = None
    if config['match_all_emerging_relations_from_tail']:
        if config['match_only_emerging_entities_to_kg']:
            # here connects only emerging entities in snippet to KG when they are tail
            triples_mask_from_tail = (edge_index_mask[1, :] &
                                      ((
                                               edge_attr[:,
                                               AttrIndexes.IDX_DELTA_WDATA_TAIL_CREATION_DATE] > interval_from_timestamp) &
                                       (
                                               edge_attr[:,
                                               AttrIndexes.IDX_DELTA_WDATA_HEAD_CREATION_DATE] < interval_from_timestamp)
                                       ) &
                                      (
                                              (edge_attr[:, AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_TO] >
                                               interval_to_timestamp + config['triple_stability_offset_in_secs'])
                                              |
                                              (edge_attr[:, AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_TO] == 0)
                                      )
                                      )
        #                 (edge_attr[:,AttrIndexes.IDX_DELTA_WDATA_TAIL_CREATION_DATE] < interval_from_timestamp) & # be sure that the entity emerging entity is connected to already exists in the graph and is not emerging itself
        #                                       ((edge_attr[:, AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_TO] >
        #                                        interval_to_timestamp + config['triple_stability_offset_in_secs'])
        #                                       |
        #                                       (edge_attr[:, AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_TO] == 0))
        #                                       )
        else:
            # here connects all the entities mentioned in snippet to KG when they are tail
            triples_mask_from_tail = (edge_index_mask[1, :] & (
                    edge_attr[:, AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_FROM] > interval_from_timestamp) &
                                      ((edge_attr[:,
                                        AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_TO] >
                                        interval_to_timestamp + config[
                                            'triple_stability_offset_in_secs'])
                                       |
                                       (edge_attr[:,
                                        AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_TO] == 0)
                                       )
                                      )

    triples_mask_from_head_or_tail = triples_mask_from_head
    if triples_mask_from_head_or_tail is not None and triples_mask_from_tail is not None:
        triples_mask_from_head_or_tail = triples_mask_from_head_or_tail | triples_mask_from_tail
    elif triples_mask_from_head_or_tail is None or triples_mask_from_tail is not None:
        triples_mask_from_head_or_tail = triples_mask_from_tail

    if triples_mask_from_head_or_tail is not None:
        # makes sure that it is disjoint from the  triples_mask_between_mentions
        triples_mask_from_head_or_tail = triples_mask_from_head_or_tail & ~triples_mask_between_mentions
    # deleteme_triples_mask_sum3 = triples_mask_from_head_or_tail.sum()
    # if deleteme_triples_mask_sum3 > deleteme_triples_mask_sum1 > 1:
    #     logger.info(
    #         '----------------------------------------------------------- \n'
    #         f'deleteme_triples_mask_sum1: {deleteme_triples_mask_sum1} \n'
    #         f'deleteme_triples_mask_sum2: {deleteme_triples_mask_sum2} \n'
    #         f'deleteme_triples_mask_sum3: {deleteme_triples_mask_sum3} \n'
    #         '-----------------------------------------------------------'
    #     )

    #### 2025.01.23 - END introducing s03_API_v5 core functionality

    edges_between_mentions = edge_index[:, triples_mask_between_mentions].T
    #
    #
    entities_in_triples = edges_between_mentions.flatten().tolist()
    nr_matched_triples = edges_between_mentions.shape[0]
    #
    app.state.nr_calls_triples += 1
    if nr_matched_triples > 0:

        # logger.info(f'triples_mask1.sum(): {triples_mask1.sum().item()} \t'
        #             f'triples_mask2.sum(): {triples_mask2.sum().item()} \t'
        #             f'triples_mask3.sum(): {triples_mask3.sum().item()}')

        app.state.nr_found_triples += 1
        app.state.tot_size_triples += nr_matched_triples

        if triples_mask_from_head_or_tail is not None:
            edges_head_or_tail_to_kg = edge_index[:, triples_mask_from_head_or_tail].T
            entities_head_or_tail_to_kg = edges_head_or_tail_to_kg.flatten().tolist()
            nr_matched_triples_h_or_t_to_kg = edges_head_or_tail_to_kg.shape[0]
            edge_attr_to_kg = edge_attr[triples_mask_from_head_or_tail, :].tolist()
            edges_to_kg_lst = edges_head_or_tail_to_kg.tolist()
            assert len(edges_to_kg_lst) == len(edge_attr_to_kg)

            for curr_triple in zip(edges_to_kg_lst, edge_attr_to_kg):
                head_idx = curr_triple[0][0]
                tail_idx = curr_triple[0][1]
                property_idx = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_RELATION_TYPE]
                triple_idx = (head_idx, property_idx, tail_idx)
                if head_idx == tail_idx:
                    continue
                if triple_idx in already_added_triples_to_kg:
                    continue
                already_added_triples_to_kg.add(triple_idx)

                head_qid = f'Q{index_to_entity[head_idx]}'
                tail_qid = f'Q{index_to_entity[tail_idx]}'
                property_id = index_to_relation[property_idx]
                action_idx = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_ACTION]
                qualifier_action_idx = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_QUALIFIER_ACTION]
                qualifier_timestamp = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_QUALIFIER_TSTMP]
                triple_from = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_FROM]
                triple_to = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_TO]
                qualifier_id = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_QUALIFIER_ID]
                head_creation_date = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_HEAD_CREATION_DATE]
                tail_creation_date = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_TAIL_CREATION_DATE]

                qualifier_action_label = 'none_qualifier'
                if qualifier_action_idx == 0:
                    qualifier_action_label = 'removed_qualifier'
                elif qualifier_action_idx == 1:
                    qualifier_action_label = 'added_qualifier'

                action_label = 'none_edge'
                if action_idx == 0:
                    action_label = 'removed_edge'
                elif action_idx == 1:
                    action_label = 'added_edge'

                qualifier_label = 'none'
                qualifier_qid = 'none'
                # qualifier_label = 'none'

                if qualifier_id in index_to_relation:
                    qualifier_qid = index_to_relation[qualifier_id]
                    # if qualifier_qid in app.state.property_qid_to_label:
                    #     qualifier_label = app.state.property_qid_to_label[qualifier_qid]
                #
                is_emerging_head = head_creation_date >= interval_from_timestamp
                if is_emerging_head:
                    nr_triples_with_emerging_heads += 1
                #
                is_emerging_tail = tail_creation_date >= interval_from_timestamp
                if is_emerging_tail:
                    nr_triples_with_emerging_tails += 1

                triples_head_or_tail_to_kg_lst.append({
                    'triple_qid': [head_qid, property_id, tail_qid],
                    # 'triple_labels': [head_label, property_label, tail_label],
                    'action': action_idx,
                    'action_label': action_label,
                    'qualifier_id': qualifier_id,
                    'qualifier_qid': qualifier_qid,
                    # 'qualifier_label': qualifier_label,
                    'qualifier_action_idx': qualifier_action_idx,
                    'qualifier_action_label': qualifier_action_label,
                    'qualifier_timestamp': qualifier_timestamp,
                    'triple_from': triple_from,
                    'triple_to': triple_to,
                    'head_creation_date': head_creation_date,
                    'tail_creation_date': tail_creation_date,
                    'emerging_head': is_emerging_head,
                    'emerging_tail': is_emerging_tail,
                    'source_delta_type': delta_type
                })

        edge_attr_with_entities = edge_attr[triples_mask_between_mentions, :].tolist()
        edges_with_entities_lst = edges_between_mentions.tolist()
        assert len(edges_with_entities_lst) == len(edge_attr_with_entities)
        for curr_triple in zip(edges_with_entities_lst, edge_attr_with_entities):
            head_idx = curr_triple[0][0]
            tail_idx = curr_triple[0][1]
            property_idx = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_RELATION_TYPE]
            triple_idx = (head_idx, property_idx, tail_idx)
            if head_idx == tail_idx:
                continue
            if triple_idx in already_added_triples:
                continue
            already_added_triples.add(triple_idx)

            head_qid = f'Q{index_to_entity[head_idx]}'
            tail_qid = f'Q{index_to_entity[tail_idx]}'
            property_id = index_to_relation[property_idx]
            action_idx = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_ACTION]
            qualifier_action_idx = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_QUALIFIER_ACTION]
            qualifier_timestamp = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_QUALIFIER_TSTMP]
            triple_from = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_FROM]
            triple_to = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_TO]
            qualifier_id = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_QUALIFIER_ID]
            head_creation_date = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_HEAD_CREATION_DATE]
            tail_creation_date = curr_triple[1][AttrIndexes.IDX_DELTA_WDATA_TAIL_CREATION_DATE]
            # head_label = head_qid
            # if head_qid in app.state.wikidata_qid_to_label:
            #     head_label = app.state.wikidata_qid_to_label[head_qid]
            # tail_label = tail_qid
            # if tail_qid in app.state.wikidata_qid_to_label:
            #     tail_label = app.state.wikidata_qid_to_label[tail_qid]
            # property_label = property_id
            # if property_id in app.state.property_qid_to_label:
            #     property_label = app.state.property_qid_to_label[property_id]

            qualifier_action_label = 'none_qualifier'
            if qualifier_action_idx == 0:
                qualifier_action_label = 'removed_qualifier'
            elif qualifier_action_idx == 1:
                qualifier_action_label = 'added_qualifier'

            action_label = 'none_edge'
            if action_idx == 0:
                action_label = 'removed_edge'
            elif action_idx == 1:
                action_label = 'added_edge'

            qualifier_label = 'none'
            qualifier_qid = 'none'
            # qualifier_label = 'none'

            if qualifier_id in index_to_relation:
                qualifier_qid = index_to_relation[qualifier_id]
                # if qualifier_qid in app.state.property_qid_to_label:
                #     qualifier_label = app.state.property_qid_to_label[qualifier_qid]
            #
            is_emerging_head = head_creation_date >= interval_from_timestamp
            if is_emerging_head:
                nr_triples_with_emerging_heads += 1
            #
            is_emerging_tail = tail_creation_date >= interval_from_timestamp
            if is_emerging_tail:
                nr_triples_with_emerging_tails += 1

            triples_lst.append({
                'triple_qid': [head_qid, property_id, tail_qid],
                # 'triple_labels': [head_label, property_label, tail_label],
                'action': action_idx,
                'action_label': action_label,
                'qualifier_id': qualifier_id,
                'qualifier_qid': qualifier_qid,
                # 'qualifier_label': qualifier_label,
                'qualifier_action_idx': qualifier_action_idx,
                'qualifier_action_label': qualifier_action_label,
                'qualifier_timestamp': qualifier_timestamp,
                'triple_from': triple_from,
                'triple_to': triple_to,
                'head_creation_date': head_creation_date,
                'tail_creation_date': tail_creation_date,
                'emerging_head': is_emerging_head,
                'emerging_tail': is_emerging_tail,
                'source_delta_type': delta_type
            })
    #
    matched_entities = set(entities_in_triples)
    #
    already_added_entities = already_added_entities | matched_entities
    to_ret = {
        'nr_triples_with_emerging_heads': nr_triples_with_emerging_heads,
        'nr_triples_with_emerging_tails': nr_triples_with_emerging_tails,
        'triples_lst': triples_lst,
        'triples_head_or_tail_to_kg_lst': triples_head_or_tail_to_kg_lst,
        'already_added_entities': already_added_entities,
        'already_added_triples': already_added_triples,
        'already_added_triples_to_kg': already_added_triples_to_kg
    }
    if nr_matched_triples > 0 and app.state.tot_size_triples > 0:
        logger.info(f'nr found triples: {app.state.nr_found_triples}, nr_calls_triples: {app.state.nr_calls_triples}'
                    f' fraction found: {app.state.nr_found_triples / app.state.nr_calls_triples}, '
                    f' avg size found triples: {app.state.tot_size_triples / app.state.nr_found_triples}')

    return to_ret


@app.get('/get_entities_in_triples_from_deltas')
def get_entities_in_triples_from_deltas(timestamp: int,
                                        interval_ids: Annotated[list[int] | None, Query()],
                                        page_titles: Annotated[list[str] | None, Query()],
                                        page_qids: Annotated[list[str] | None, Query()]
                                        ):
    if config['dry_run']:
        page_ids_qids = ['Q1', 'Q3', 'Q5', 'Q10']
        logger.debug(f'dry_run get_page_id to return: {page_ids_qids}')
        return page_ids_qids

    entity_to_index: Dict[int, int] = app.state.entity_to_index
    index_to_entity: Dict[int, str] = app.state.index_to_entity
    index_to_relation: Dict[int, str] = app.state.index_to_relation
    #

    page_ids_qids = []

    logger.debug(f'get_entities_in_triples_from_deltas received '
                 f'interval_ids: {interval_ids} '
                 f'page_titles: {page_titles}')

    if isinstance(page_titles, str):
        # logger.info(f'putting the following page_titles in a list: {page_titles}')
        page_titles = [page_titles]  # put in the list if it is a string
    #
    # for curr_title in page_titles:
    #     get_pid_result = get_page_id_func(timestamp=timestamp,
    #                                       page_title=curr_title)
    #     if get_pid_result['page_qid'] is None:
    #         continue
    #     page_ids_qids.append(get_pid_result)
    logger.debug(f'get_entities_in_triples_from_deltas received page_titles: '
                 f'{page_titles} and obtained page_ids_qids: {page_ids_qids}')
    # entity_idxs_list = list(set({(entity_to_index[int(curr_qid['page_qid'][1:])],
    # entities_list = list(set({(entity_to_index[int(curr_qid['page_qid'][1:])],
    #                            curr_qid['page_qid'],
    #                            curr_qid['page_title']
    #                            )
    #                           for
    #                           curr_qid in page_ids_qids
    #                           if int(curr_qid['page_qid'][1:]) in entity_to_index}))
    entities_list = list(set({(entity_to_index[int(curr_page_qid[1:])],
                               curr_page_qid,
                               curr_page_title
                               )
                              for
                              curr_page_qid, curr_page_title in zip(page_qids, page_titles)
                              if int(curr_page_qid[1:]) in entity_to_index}))
    entity_idxs_list = []
    entity_titles_list = []
    entity_qids_list = []
    for curre in entities_list:
        entity_idxs_list.append(curre[0])
        entity_qids_list.append(curre[1])
        entity_titles_list.append(curre[2])

    if len(entity_idxs_list) == 0:
        # return [{
        #     'tot_nr_entities': 0,
        #     'nr_matched_triples': 0,
        #     'nr_triples_with_emerging_heads': 0,
        #     'nr_triples_with_emerging_tails': 0,
        #     'nr_matched_entities': 0,
        #     'matched_entities': [],
        #     'entities': [],
        #     'matched_triples': []
        # }]
        return list()  # better return empty list

    tn_entities_idxs_to_search = torch.tensor(entity_idxs_list, dtype=torch.int64,
                                              device=config['device'])

    to_ret_list = list()
    for curr_interval_id in interval_ids:
        interval_id_delta_intersection: Data = app.state.interval_ids_to_delta_intersection[curr_interval_id]
        interval_id_delta_emerging: Data = app.state.interval_ids_to_delta_emerging[curr_interval_id]
        interval_id_delta_all: Data = app.state.interval_ids_to_delta_all[curr_interval_id]
        #

        already_added_triples: Set[Tuple[int, int, int]] = set()
        already_added_triples_to_kg: Set[Tuple[int, int, int]] = set()
        already_added_entities: Set[int] = set()
        delta_type = 'wikipedia_intersection'

        curr_interval: TemporalInterval = app.state.interval_ids_to_interval[curr_interval_id]
        nr_triples_with_emerging_heads = 0
        nr_triples_with_emerging_tails = 0
        triples_lst = []
        triples_head_or_tail_to_kg_lst = []
        res_get_triples = get_triples(
            already_added_triples_to_kg=already_added_triples_to_kg,
            already_added_triples=already_added_triples,
            already_added_entities=already_added_entities,
            delta_type=delta_type,
            tn_entities_idxs_to_search=tn_entities_idxs_to_search,
            interval_id_delta=interval_id_delta_intersection,
            index_to_entity=index_to_entity,
            index_to_relation=index_to_relation,
            interval_from_timestamp=curr_interval.interval_start,
            interval_to_timestamp=curr_interval.interval_end,
            nr_triples_with_emerging_heads=nr_triples_with_emerging_heads,
            nr_triples_with_emerging_tails=nr_triples_with_emerging_tails,
            triples_lst=triples_lst,
            triples_head_or_tail_to_kg_lst=triples_head_or_tail_to_kg_lst
        )
        # BEGIN: delta_type: emerging
        already_added_triples = res_get_triples['already_added_triples']
        already_added_entities = res_get_triples['already_added_entities']
        nr_triples_with_emerging_heads = res_get_triples['nr_triples_with_emerging_heads']
        nr_triples_with_emerging_tails = res_get_triples['nr_triples_with_emerging_tails']
        triples_lst = res_get_triples['triples_lst']
        delta_type = 'emerging'
        res_get_triples = get_triples(
            already_added_triples_to_kg=already_added_triples_to_kg,
            already_added_triples=already_added_triples,
            already_added_entities=already_added_entities,
            delta_type=delta_type,
            tn_entities_idxs_to_search=tn_entities_idxs_to_search,
            interval_id_delta=interval_id_delta_emerging,
            index_to_entity=index_to_entity,
            index_to_relation=index_to_relation,
            interval_from_timestamp=curr_interval.interval_start,
            interval_to_timestamp=curr_interval.interval_end,
            nr_triples_with_emerging_heads=nr_triples_with_emerging_heads,
            nr_triples_with_emerging_tails=nr_triples_with_emerging_tails,
            triples_lst=triples_lst,
            triples_head_or_tail_to_kg_lst=triples_head_or_tail_to_kg_lst
        )

        # BEGIN: delta_type: all
        already_added_triples = res_get_triples['already_added_triples']
        already_added_entities = res_get_triples['already_added_entities']
        nr_triples_with_emerging_heads = res_get_triples['nr_triples_with_emerging_heads']
        nr_triples_with_emerging_tails = res_get_triples['nr_triples_with_emerging_tails']
        triples_lst = res_get_triples['triples_lst']
        delta_type = 'all'
        res_get_triples = get_triples(
            already_added_triples_to_kg=already_added_triples_to_kg,
            already_added_triples=already_added_triples,
            already_added_entities=already_added_entities,
            delta_type=delta_type,
            tn_entities_idxs_to_search=tn_entities_idxs_to_search,
            interval_id_delta=interval_id_delta_all,
            index_to_entity=index_to_entity,
            index_to_relation=index_to_relation,
            interval_from_timestamp=curr_interval.interval_start,
            interval_to_timestamp=curr_interval.interval_end,
            nr_triples_with_emerging_heads=nr_triples_with_emerging_heads,
            nr_triples_with_emerging_tails=nr_triples_with_emerging_tails,
            triples_lst=triples_lst,
            triples_head_or_tail_to_kg_lst=triples_head_or_tail_to_kg_lst
        )

        tot_nr_entities = len(entity_idxs_list)

        nr_matched_triples = len(res_get_triples['already_added_triples'])
        nr_matched_triples_to_kg = len(res_get_triples['already_added_triples_to_kg'])

        nr_matched_entities = len(res_get_triples['already_added_entities'])

        matched_entities_qids = [f'Q{app.state.index_to_entity[idx_ent]}'
                                 for idx_ent in res_get_triples['already_added_entities']]
        #
        to_ret = {
            'interval_id': curr_interval_id,
            'tot_nr_entities': tot_nr_entities,
            'nr_matched_triples_to_kg': nr_matched_triples_to_kg,
            'nr_matched_triples': nr_matched_triples,
            'nr_triples_with_emerging_heads': nr_triples_with_emerging_heads,
            'nr_triples_with_emerging_tails': nr_triples_with_emerging_tails,
            'nr_matched_entities': nr_matched_entities,
            'matched_entities': matched_entities_qids,
            # TODO - BEGIN the entities extraction has to be moved to the client!
            # 'entities': entity_qids_list,
            # TODO - END the entities extraction has to be moved to the client!
            'titles': entity_titles_list,
            'matched_triples': res_get_triples['triples_lst'],
            'matched_triples_entities_to_kg': res_get_triples['triples_head_or_tail_to_kg_lst']
        }

        if nr_matched_triples > 0:
            logger.debug(f'returning from get_entities_in_triples_from_deltas: {to_ret}')
            to_ret_list.append(to_ret)

    logger.debug(f'returning from get_entities_in_triples_from_deltas: {to_ret_list}')
    # return to_ret
    return to_ret_list


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
