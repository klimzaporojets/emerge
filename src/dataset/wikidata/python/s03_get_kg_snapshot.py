# s03_get_kg_snapshot.py
# Given one or more timestamps, extracts the Wikidata knowledge graph as it existed at each timestamp.
# Reuses the graph-loading logic from s03_get_deltas.py.
import argparse
import csv
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict

import torch

from dataset.wikidata.python.misc.wiki_utils import generate_short_hash, load_wikidata_qid_to_label, \
    load_property_qid_to_label, get_git_commit_hash, load_wiki_page_id_to_wdata_qid_only_parsed
from dataset.wikidata.python.s03_get_deltas import load_into_pyg_dynamically_data
from dataset.wikipedia.misc.load_wiki_sql_tables import load_wiki_page_id_to_wikidata_qid

logger = logging.getLogger(__name__)


def extract_snapshot(edge_timestamps, pyg_graph, snapshot_timestamp,
                     index_to_entity, index_to_relation, output_dir,
                     wikidata_qid_to_label=None, property_qid_to_label=None,
                     output_with_labels=False):
    """
    Given a loaded PyG graph with edge timestamps, extracts all triples active at snapshot_timestamp
    and writes them to a TSV file.

    A triple is active at time T if:
      - it was added before T (edge_timestamps[:, 1] < T)
      - AND it was either deleted after T (edge_timestamps[:, 2] > T) or never deleted (edge_timestamps[:, 2] == 0)

    Output format depends on output_with_labels:
      - False (triples_only): head_qid  relation_id  tail_qid
      - True  (with_labels):  head_qid  relation_id  tail_qid  head_label  relation_label  tail_label
    """
    start_time = time.time()

    active_mask = ((edge_timestamps[:, 1] < snapshot_timestamp) &
                   ((edge_timestamps[:, 2] > snapshot_timestamp) | (edge_timestamps[:, 2] == 0)))

    active_indices = edge_timestamps[active_mask, 0]
    active_edges = pyg_graph.edge_index[:, active_indices]
    active_attrs = pyg_graph.edge_attr[active_indices, :]

    nr_active_triples = active_edges.shape[1]
    logger.info(f'snapshot at {snapshot_timestamp}: {nr_active_triples:,} active triples')

    os.makedirs(output_dir, exist_ok=True)
    output_triples_path = os.path.join(output_dir, 'triples.tsv')

    with open(output_triples_path, 'wt') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        batch_rows = []
        for idx in range(nr_active_triples):
            head_idx = active_edges[0, idx].item()
            tail_idx = active_edges[1, idx].item()
            head_qid = f'Q{index_to_entity[head_idx]}'
            tail_qid = f'Q{index_to_entity[tail_idx]}'

            # relation index is in edge_attr column 1 (column 0 is triple_id)
            relation_idx = active_attrs[idx, 1].item()
            relation_id = index_to_relation[relation_idx]

            if output_with_labels:
                head_label = wikidata_qid_to_label.get(head_qid, '')
                tail_label = wikidata_qid_to_label.get(tail_qid, '')
                relation_label = property_qid_to_label.get(relation_id, '')
                batch_rows.append([head_qid, relation_id, tail_qid,
                                   head_label, relation_label, tail_label])
            else:
                batch_rows.append([head_qid, relation_id, tail_qid])

            if len(batch_rows) >= 100000:
                writer.writerows(batch_rows)
                batch_rows = []
        if batch_rows:
            writer.writerows(batch_rows)

    elapsed = time.time() - start_time
    logger.info(f'snapshot written to {output_triples_path} in {elapsed:.1f}s')

    return nr_active_triples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_file',
        required=True,
        type=str,
        help='The config file that contains the config parameters')

    parser.add_argument(
        '--debug_nr_triples',
        help='If -1 loads all the triples, if not limits to the nr passed.',
        type=int,
        default=-1)

    args = parser.parse_args()
    config = json.load(open(args.config_file, 'rt'))

    output_dir_data = config['output_dir_data']
    os.makedirs(output_dir_data, exist_ok=True)
    commit_hash = get_git_commit_hash()

    caches_dir = config['caches_dir']
    os.makedirs(caches_dir, exist_ok=True)

    debug_nr_triples = args.debug_nr_triples

    input_triples_path_wdata = config['path_extracted_history_triples_wdata']
    output_with_labels = config['output_with_labels']

    # load property labels (always needed for relation IDs during loading)
    path_property_labels = config['path_property_labels']
    caches_property_qid_to_label_path = os.path.join(caches_dir, 'property_qid_to_label.pickle')
    property_qid_to_label: Dict = load_property_qid_to_label(
        path_property_labels, caches_property_qid_to_label_path
    )

    # load entity labels (only needed for with_labels output format)
    wikidata_qid_to_label: Dict = dict()
    if output_with_labels:
        path_wikidata_labels = config['path_wikidata_labels']
        caches_wikidata_qid_to_label_path = os.path.join(caches_dir, 'wikidata_qid_to_label.pickle')
        logger.info('loading wikidata_qid_to_label')
        wikidata_qid_to_label = load_wikidata_qid_to_label(
            path_wikidata_labels, caches_wikidata_qid_to_label_path
        )
        logger.info(f'loaded {len(wikidata_qid_to_label)} entity labels')

    # load wikipedia-wikidata mapping if filtering is enabled
    wikidata_qid_to_wikipedia_page_id = dict()
    if config['heads_in_wikipedia'] or config['tails_in_wikipedia']:
        path_wikipedia_wikidata_map = config['path_wikipedia_wikidata_map']

        wdata_entity_creation_date_path = config['wdata_entity_creation_date_path']
        wpedia_entity_creation_date_path = config['wpedia_entity_creation_date_path']

        if config['qids_in_both_wdata_and_wpedia_parsed_entities']:
            path_cache = os.path.join(caches_dir, 'wpedia_wdata_parsed_mapping.pickle')
            wikipedia_page_id_to_wikidata_qid = load_wiki_page_id_to_wdata_qid_only_parsed(
                path_cache,
                wpedia_entity_creation_date_path,
                wdata_entity_creation_date_path
            )
        else:
            path_cache = os.path.join(caches_dir, 'wikipedia_page_id_to_wikidata_qid.pickle')
            wikipedia_page_id_to_wikidata_qid = load_wiki_page_id_to_wikidata_qid(
                path_cache, path_wikipedia_wikidata_map)
        wikidata_qid_to_wikipedia_page_id = {v: k for k, v in wikipedia_page_id_to_wikidata_qid.items()}
    else:
        # no filtering: all entities are included
        wikidata_qid_to_wikipedia_page_id = None

    # Build cache path
    wdata_entity_creation_date_path = config['wdata_entity_creation_date_path']
    wpedia_entity_creation_date_path = config['wpedia_entity_creation_date_path']
    wpedia_wdata_path_hash = generate_short_hash(
        f'{wpedia_entity_creation_date_path}_{wdata_entity_creation_date_path}',
        hash_length=8)

    if debug_nr_triples == -1:
        cache_path = os.path.join(caches_dir,
                                  f'loaded_graph_snapshot_all_{config["heads_in_wikipedia"]}_'
                                  f'{config["tails_in_wikipedia"]}_'
                                  f'{config["qids_in_both_wdata_and_wpedia_parsed_entities"]}_'
                                  f'{wpedia_wdata_path_hash}.pt')
    else:
        cache_path = os.path.join(caches_dir,
                                  f'loaded_graph_snapshot_{debug_nr_triples}_'
                                  f'{config["heads_in_wikipedia"]}_'
                                  f'{config["tails_in_wikipedia"]}_'
                                  f'{config["qids_in_both_wdata_and_wpedia_parsed_entities"]}_'
                                  f'{wpedia_wdata_path_hash}.pt')

    if not os.path.exists(cache_path):
        logger.info(f'NOT IN CACHE {cache_path}, loading graph')
        entity_to_index: Dict = dict()
        relation_to_index: Dict = dict()

        (pyg_graph_wdata, edge_timestamps_wdata, entity_to_index,
         index_to_entity, relation_to_index, triple_to_index,
         qualifier_timestamps_wdata) = \
            load_into_pyg_dynamically_data(
                input_triples_path=input_triples_path_wdata,
                debug_nr_triples=debug_nr_triples,
                entity_to_index=entity_to_index,
                relation_to_index=relation_to_index,
                qualifier_to_details=dict(),
                wikidata_qid_to_wikipedia_page_id=wikidata_qid_to_wikipedia_page_id,
                config=config,
                kb_type='wikidata',
                property_qid_to_label=property_qid_to_label,
                precision=config['precision_wdata'])

        index_to_relation = {v: k for k, v in relation_to_index.items()}

        logger.info(f'graph loaded, saving to cache {cache_path}')
        torch.save({
            'pyg_graph_wdata': pyg_graph_wdata,
            'tensor_timestamps_wdata': edge_timestamps_wdata,
            'index_to_entity': index_to_entity,
            'index_to_relation': index_to_relation
        }, cache_path)
    else:
        logger.info(f'loading from cache {cache_path}')
        loaded_data = torch.load(cache_path, weights_only=False)
        pyg_graph_wdata = loaded_data['pyg_graph_wdata']
        edge_timestamps_wdata = loaded_data['tensor_timestamps_wdata']
        index_to_entity = loaded_data['index_to_entity']
        index_to_relation = loaded_data['index_to_relation']
        logger.info(f'graph loaded from cache')

    logger.info(f'graph stats: edge_index.shape={pyg_graph_wdata.edge_index.shape}, '
                f'edge_timestamps.shape={edge_timestamps_wdata.shape}')

    # Extract snapshots for each requested timestamp
    snapshot_timestamps = config['snapshot_timestamps']

    for snapshot_ts_str in snapshot_timestamps:
        date_object = datetime.strptime(snapshot_ts_str, "%Y-%m-%d")
        snapshot_epoch = int(date_object.timestamp())

        precision = config['precision_wdata']
        if precision == 'seconds':
            snapshot_ts_query = snapshot_epoch
        elif precision == 'milliseconds':
            # edge_timestamps are converted to seconds during loading in load_into_pyg_dynamically_data
            snapshot_ts_query = snapshot_epoch
        else:
            raise RuntimeError(f'unknown precision: {precision}')

        logger.info(f'extracting snapshot for {snapshot_ts_str} (epoch={snapshot_epoch})')

        snapshot_output_dir = os.path.join(output_dir_data, f'snapshot_{snapshot_ts_str}')

        nr_triples = extract_snapshot(
            edge_timestamps=edge_timestamps_wdata,
            pyg_graph=pyg_graph_wdata,
            snapshot_timestamp=snapshot_ts_query,
            index_to_entity=index_to_entity,
            index_to_relation=index_to_relation,
            output_dir=snapshot_output_dir,
            wikidata_qid_to_label=wikidata_qid_to_label,
            property_qid_to_label=property_qid_to_label,
            output_with_labels=output_with_labels
        )

        # Save metadata
        metadata = {
            'snapshot_timestamp': snapshot_ts_str,
            'snapshot_epoch': snapshot_epoch,
            'nr_active_triples': nr_triples,
            'commit_hash': commit_hash,
            'config': config
        }
        with open(os.path.join(snapshot_output_dir, 'metadata.json'), 'wt') as f:
            json.dump(metadata, f, indent=4)

        logger.info(f'snapshot {snapshot_ts_str}: {nr_triples:,} triples written to {snapshot_output_dir}')

    logger.info('all snapshots extracted')
