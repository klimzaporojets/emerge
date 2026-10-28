# s03_get_kg_snapshot.py
# Given one or more timestamps, extracts the Wikidata knowledge graph as it existed at each timestamp.
# Reuses the graph-loading logic from s03_get_deltas.py.
import argparse
import csv
import gzip
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict
from zoneinfo import ZoneInfo

import torch

from dataset.wikidata.python.misc.wiki_utils import generate_short_hash, input_files_fingerprint, \
    load_wikidata_qid_to_label_qids, \
    load_property_qid_to_label, get_git_commit_hash, load_wiki_page_id_to_wdata_qid_only_parsed
from dataset.wikidata.python.s03_get_deltas import load_into_pyg_dynamically_data
from dataset.wikipedia.misc.load_wiki_sql_tables import load_wiki_page_id_to_wikidata_qid

logger = logging.getLogger(__name__)

QID_RE = re.compile(r"Q[1-9][0-9]*\Z")
PID_RE = re.compile(r"P[1-9][0-9]*\Z")


def load_entity_filter_qids(path: str) -> Dict:
    """Read an entity filter (one QID per line, plain or .gz) as a membership dict {qid: None}.
    HARD: every line must be Q<digits>, no duplicates, not empty (a wrong file must never filter silently)."""
    opener = gzip.open if path.endswith('.gz') else open
    qids: Dict = dict()
    with opener(path, 'rt', encoding='utf-8') as f:
        for nr, line in enumerate(f, start=1):
            qid = line.rstrip('\n')
            if not QID_RE.match(qid):
                raise ValueError(f'{path} line {nr}: not a QID: {qid!r}')
            if qid in qids:
                raise ValueError(f'{path} line {nr}: duplicate QID {qid}')
            qids[qid] = None
    if not qids:
        raise ValueError(f'{path}: empty entity filter')
    return qids


# ---------------- fail-early checks of the assumptions this script relies on ----------------
# HARD checks raise (an invariant of the code / format is broken: continuing would write wrong files).
# SOFT checks are data properties that are not strict invariants: counted, logged and stored in metadata.json.

# Wikidata went online on 2012-10-29: no history timestamp can be earlier (2012-01-01 leaves a margin).
EARLIEST_PLAUSIBLE_EPOCH = 1325376000
# Above this share of entities without a label, the labels file does not belong to this graph.
MAX_MISSING_LABEL_SHARE = 0.10


def snapshot_epoch(date_str: str, tz_name):
    """Epoch of 00:00 on date_str in tz_name; tz_name None = the machine's local time (the 2025 convention)."""
    date_object = datetime.strptime(date_str, '%Y-%m-%d')
    if tz_name is None:
        return int(date_object.timestamp())
    return int(date_object.replace(tzinfo=ZoneInfo(tz_name)).timestamp())


def check_config(config: Dict) -> None:
    """HARD: everything that can be checked before the ~1 h graph load."""
    dates = config['snapshot_timestamps']
    if not dates:
        raise ValueError('snapshot_timestamps is empty')
    parsed = [datetime.strptime(d, '%Y-%m-%d') for d in dates]  # raises on a malformed date
    if any(a >= b for a, b in zip(parsed, parsed[1:])):
        raise ValueError(f'snapshot_timestamps must be strictly increasing: {dates}')
    if config['precision_wdata'] not in ('seconds', 'milliseconds'):
        raise ValueError(f'unknown precision_wdata: {config["precision_wdata"]}')
    if 'snapshot_timezone' in config:
        ZoneInfo(config['snapshot_timezone'])  # raises on an unknown zone
    history_dir = config['path_extracted_history_triples_wdata']
    if not os.path.isdir(history_dir) or not os.listdir(history_dir):
        raise FileNotFoundError(f'history dir missing or empty: {history_dir}')
    required_files = [config['path_property_labels']]
    if config['output_with_labels']:
        required_files.append(config['path_wikidata_labels'])
    if 'path_entity_filter_qids' in config:
        if not (config['heads_in_wikipedia'] and config['tails_in_wikipedia']):
            raise ValueError('path_entity_filter_qids filters heads AND tails: set heads_in_wikipedia and '
                             'tails_in_wikipedia to true')
        required_files.append(config['path_entity_filter_qids'])
    elif config['heads_in_wikipedia'] or config['tails_in_wikipedia']:
        required_files += [config['wdata_entity_creation_date_path'], config['wpedia_entity_creation_date_path']]
    for path in required_files:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise FileNotFoundError(f'input file missing or empty: {path}')


def check_expected_graph_size(config: Dict, graph_checks: Dict) -> None:
    """HARD (optional config keys): the loaded graph must be exactly the expected one, e.g. the graph the dataset
    labels were built from -> stop right after the load, before any snapshot is written."""
    for key, found in (('expected_nr_graph_triples', graph_checks['nr_edges']),
                       ('expected_nr_graph_entities', graph_checks['nr_entities'])):
        if key in config:
            if found != config[key]:
                raise ValueError(f'{key}: expected {config[key]:,}, the loaded graph has {found:,}')
            logger.info(f'check {key}: PASS ({found:,})')


def check_labels(name: str, labels: Dict) -> None:
    """HARD: a TAB / CR / LF inside a label would split a snapshot line and break every reader (they split on TAB)."""
    bad = [k for k, v in labels.items() if '\t' in v or '\n' in v or '\r' in v]
    if bad:
        raise ValueError(f'{len(bad):,} {name} contain TAB/CR/LF, e.g. {[(k, labels[k]) for k in bad[:5]]}')
    n_quote = sum(1 for v in labels.values() if '"' in v)
    logger.info(f'check {name}: {len(labels):,} labels, none with TAB/CR/LF; {n_quote:,} contain a double quote '
                f'(csv.writer quotes such fields, as in the released snapshots)')


def check_graph(pyg_graph, edge_timestamps, index_to_entity: Dict, index_to_relation: Dict,
                wikidata_qid_to_wikipedia_page_id, both_filters: bool) -> Dict:
    """HARD checks of the loaded graph; returns SOFT counts."""
    edge_index, edge_attr = pyg_graph.edge_index, pyg_graph.edge_attr
    nr_edges = edge_index.shape[1]
    if edge_index.dim() != 2 or edge_index.shape[0] != 2 or nr_edges == 0:
        raise ValueError(f'edge_index must be [2, E>0], got {tuple(edge_index.shape)}')
    if edge_attr.dim() != 2 or edge_attr.shape[0] != nr_edges or edge_attr.shape[1] < 2:
        raise ValueError(f'edge_attr must be [E={nr_edges}, >=2], got {tuple(edge_attr.shape)}')
    if edge_timestamps.dim() != 2 or edge_timestamps.shape[1] != 3:
        raise ValueError(f'edge_timestamps must be [N, 3], got {tuple(edge_timestamps.shape)}')
    if sorted(index_to_entity) != list(range(len(index_to_entity))):
        raise ValueError('index_to_entity keys must be exactly 0..n-1')
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= len(index_to_entity):
        raise ValueError(f'edge_index values outside 0..{len(index_to_entity) - 1}')
    if any(not isinstance(v, int) or v <= 0 for v in index_to_entity.values()):
        raise ValueError('index_to_entity values must be positive ints (numeric part of the QID)')
    edge_ids = edge_timestamps[:, 0]
    if int(edge_ids.min()) < 0 or int(edge_ids.max()) >= nr_edges:
        raise ValueError(f'edge_timestamps[:, 0] outside 0..{nr_edges - 1}')
    used_relations = set(torch.unique(edge_attr[:, 1]).tolist())
    if not used_relations <= set(index_to_relation):
        raise ValueError(f'relation indexes without an id: {sorted(used_relations - set(index_to_relation))[:5]}')
    bad_rel = [r for r in (index_to_relation[i] for i in used_relations) if not PID_RE.match(r)]
    if bad_rel:
        raise ValueError(f'relation ids not of the form P<digits>: {bad_rel[:5]}')
    added, removed = edge_timestamps[:, 1], edge_timestamps[:, 2]
    latest = int(time.time()) + 86400
    if int(added.min()) < EARLIEST_PLAUSIBLE_EPOCH or int(added.max()) > latest:
        raise ValueError(f'added timestamps outside [2012-01-01, now]: min={int(added.min())} '
                         f'max={int(added.max())} (milliseconds not converted to seconds?)')
    if int(removed.max()) > latest or bool(((removed != 0) & (removed <= added)).any()):
        raise ValueError('removal timestamps must be 0 (never removed) or after the addition, and not in the future')
    if both_filters:
        outside = [f'Q{q}' for q in index_to_entity.values() if f'Q{q}' not in wikidata_qid_to_wikipedia_page_id]
        if outside:
            raise ValueError(f'{len(outside):,} entities outside the Wikipedia filter, e.g. {outside[:5]}')
    soft = {'nr_edges': nr_edges, 'nr_intervals': int(edge_timestamps.shape[0]), 'nr_entities': len(index_to_entity),
            'nr_relations_used': len(used_relations),
            'nr_edges_without_interval': nr_edges - int(torch.unique(edge_ids).numel())}
    logger.info(f'check graph: PASS (hard checks); soft counts: {soft}')
    return soft


def extract_snapshot(edge_timestamps, pyg_graph, snapshot_timestamp,
                     index_to_entity, index_to_relation, output_dir,
                     wikidata_qid_to_label=None, property_qid_to_label=None,
                     output_with_labels=False, max_missing_label_share=MAX_MISSING_LABEL_SHARE):
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
    if nr_active_triples == 0:
        raise ValueError(f'snapshot at {snapshot_timestamp} is empty (date before the history starts?)')
    # SOFT: one edge active through two overlapping intervals, or two edges with the same (head, relation, tail),
    # would each give a duplicate line (consumers read the lines into sets; the validator reports duplicates)
    nr_overlapping_intervals = nr_active_triples - int(torch.unique(active_indices).numel())
    nr_entities_total = max(index_to_entity) + 1
    nr_relations_total = int(active_attrs[:, 1].max()) + 1
    if nr_entities_total * nr_relations_total * nr_entities_total < 2 ** 63:   # one int64 key per triple fits
        triple_keys = ((active_edges[0].long() * nr_relations_total + active_attrs[:, 1].long())
                       * nr_entities_total + active_edges[1].long())
        nr_unique_triples = int(torch.unique(triple_keys).numel())
    else:                                                                      # e.g. an unfiltered graph
        nr_unique_triples = int(torch.unique(torch.stack([active_edges[0].long(), active_attrs[:, 1].long(),
                                                          active_edges[1].long()]), dim=1).shape[1])
    nr_duplicate_triples = nr_active_triples - nr_unique_triples
    if nr_overlapping_intervals or nr_duplicate_triples:
        logger.warning(f'snapshot at {snapshot_timestamp}: {nr_overlapping_intervals:,} overlapping intervals, '
                       f'{nr_duplicate_triples:,} duplicate (head, relation, tail) lines')

    os.makedirs(output_dir, exist_ok=True)
    output_triples_path = os.path.join(output_dir, 'triples.tsv')

    last_progress_time = time.time()
    nr_rows_written = 0
    entities_without_label = set()
    relations_without_label = set()
    with open(output_triples_path, 'wt', encoding='utf-8') as outfile:
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
                # a missing label falls back to the id itself (format of the released snapshots)
                if head_qid in wikidata_qid_to_label:
                    head_label = wikidata_qid_to_label[head_qid]
                else:
                    head_label = head_qid
                    entities_without_label.add(head_qid)
                if tail_qid in wikidata_qid_to_label:
                    tail_label = wikidata_qid_to_label[tail_qid]
                else:
                    tail_label = tail_qid
                    entities_without_label.add(tail_qid)
                if relation_id in property_qid_to_label:
                    relation_label = property_qid_to_label[relation_id]
                else:
                    relation_label = relation_id
                    relations_without_label.add(relation_id)
                batch_rows.append([head_qid, relation_id, tail_qid,
                                   head_label, relation_label, tail_label])
            else:
                batch_rows.append([head_qid, relation_id, tail_qid])

            if len(batch_rows) >= 100000:
                writer.writerows(batch_rows)
                nr_rows_written += len(batch_rows)
                batch_rows = []
                if time.time() - last_progress_time >= 30:
                    logger.info(f'snapshot at {snapshot_timestamp}: written {idx + 1:,} / {nr_active_triples:,} '
                                f'triples ({time.time() - start_time:.0f}s)')
                    last_progress_time = time.time()
        if batch_rows:
            writer.writerows(batch_rows)
            nr_rows_written += len(batch_rows)

    if nr_rows_written != nr_active_triples:
        raise RuntimeError(f'wrote {nr_rows_written:,} rows but {nr_active_triples:,} triples are active')
    nr_entities = int(torch.unique(active_edges).numel())
    if output_with_labels and len(entities_without_label) > max_missing_label_share * nr_entities:
        raise ValueError(f'{len(entities_without_label):,} of {nr_entities:,} entities have no label: the labels '
                         f'file does not belong to this graph? e.g. {sorted(entities_without_label)[:5]}')
    elapsed = time.time() - start_time
    logger.info(f'snapshot written to {output_triples_path} in {elapsed:.1f}s')

    stats = {'nr_active_triples': nr_active_triples, 'nr_entities': nr_entities,
             'nr_entities_without_label': len(entities_without_label),
             'nr_relations_without_label': len(relations_without_label),
             'nr_overlapping_intervals': nr_overlapping_intervals, 'nr_duplicate_triples': nr_duplicate_triples}
    logger.info(f'snapshot at {snapshot_timestamp}: {stats}')
    return stats


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
    check_config(config)  # fail in seconds, not after the ~1 h graph load
    # snapshot cut-off = 00:00 in this zone; without the key the machine's local time is used (2025 convention)
    snapshot_tz = config['snapshot_timezone'] if 'snapshot_timezone' in config else None
    if snapshot_tz is None:
        logger.warning('no snapshot_timezone in the config: cut-offs use this machine\'s local time '
                       f'({time.tzname})')

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
    # every cached pickle is keyed on its input files' fingerprint (a fixed name would serve a stale pickle)
    caches_property_qid_to_label_path = os.path.join(
        caches_dir, f'property_qid_to_label_{input_files_fingerprint([path_property_labels])}.pickle')
    property_qid_to_label: Dict = load_property_qid_to_label(
        path_property_labels, caches_property_qid_to_label_path
    )
    check_labels('property labels', property_qid_to_label)

    # entity filter: an explicit QID list (path_entity_filter_qids), else the Wikipedia mapping if filtering is enabled
    wikidata_qid_to_wikipedia_page_id = dict()
    if 'path_entity_filter_qids' in config:
        # the exact entity set of a previous build (e.g. the graph the dataset labels were built from); it replaces
        # the creation-date-based Wikipedia filter for heads AND tails (check_config enforces both flags)
        path_entity_filter_qids = config['path_entity_filter_qids']
        wikidata_qid_to_wikipedia_page_id = load_entity_filter_qids(path_entity_filter_qids)  # only membership is used
        logger.info(f'entity filter from {path_entity_filter_qids}: {len(wikidata_qid_to_wikipedia_page_id):,} entities')
    elif config['heads_in_wikipedia'] or config['tails_in_wikipedia']:
        path_wikipedia_wikidata_map = config['path_wikipedia_wikidata_map']

        wdata_entity_creation_date_path = config['wdata_entity_creation_date_path']
        wpedia_entity_creation_date_path = config['wpedia_entity_creation_date_path']

        if config['qids_in_both_wdata_and_wpedia_parsed_entities']:
            mapping_hash = input_files_fingerprint([wpedia_entity_creation_date_path, wdata_entity_creation_date_path])
            path_cache = os.path.join(caches_dir, f'wpedia_wdata_parsed_mapping_{mapping_hash}.pickle')
            wikipedia_page_id_to_wikidata_qid = load_wiki_page_id_to_wdata_qid_only_parsed(
                path_cache,
                wpedia_entity_creation_date_path,
                wdata_entity_creation_date_path
            )
        else:
            mapping_hash = input_files_fingerprint([path_wikipedia_wikidata_map])
            path_cache = os.path.join(caches_dir, f'wikipedia_page_id_to_wikidata_qid_{mapping_hash}.pickle')
            wikipedia_page_id_to_wikidata_qid = load_wiki_page_id_to_wikidata_qid(
                path_cache, path_wikipedia_wikidata_map)
        wikidata_qid_to_wikipedia_page_id = {v: k for k, v in wikipedia_page_id_to_wikidata_qid.items()}
        # HARD: the loader tests 'Q<digits>' strings against these keys; another key type/format (wrong file or
        # column) would silently filter out every triple. A few odd keys (e.g. a header row present in both
        # creation-date files) are harmless for membership tests -> only a warning.
        bad_keys = [k for k in wikidata_qid_to_wikipedia_page_id if not isinstance(k, str) or not QID_RE.match(k)]
        if not wikidata_qid_to_wikipedia_page_id or len(bad_keys) > 0.01 * len(wikidata_qid_to_wikipedia_page_id):
            raise ValueError(f'Wikipedia filter must be a non-empty set of Q<digits> strings; '
                             f'{len(wikidata_qid_to_wikipedia_page_id):,} keys, {len(bad_keys):,} bad, '
                             f'e.g. {bad_keys[:5]}')
        if bad_keys:
            logger.warning(f'Wikipedia filter: {len(bad_keys):,} keys are not Q<digits>: {bad_keys[:5]}')
        logger.info(f'Wikipedia filter: {len(wikidata_qid_to_wikipedia_page_id):,} entities')
    else:
        # no filtering: all entities are included
        wikidata_qid_to_wikipedia_page_id = None

    # Build cache path
    if 'path_entity_filter_qids' in config:
        filter_inputs = [path_entity_filter_qids]
        filter_tag = f'qidlist_{generate_short_hash(path_entity_filter_qids, hash_length=8)}'
    else:
        wdata_entity_creation_date_path = config['wdata_entity_creation_date_path']
        wpedia_entity_creation_date_path = config['wpedia_entity_creation_date_path']
        filter_inputs = [wdata_entity_creation_date_path, wpedia_entity_creation_date_path]
        wpedia_wdata_path_hash = generate_short_hash(
            f'{wpedia_entity_creation_date_path}_{wdata_entity_creation_date_path}',
            hash_length=8)
        filter_tag = (f'{config["heads_in_wikipedia"]}_{config["tails_in_wikipedia"]}_'
                      f'{config["qids_in_both_wdata_and_wpedia_parsed_entities"]}_{wpedia_wdata_path_hash}')
    # The paths alone do not identify the inputs: a graph cached from an older version of the same files would be
    # reused silently (this produced stale released snapshots once). Also hash every input file's size + mtime.
    inputs_hash = input_files_fingerprint([input_triples_path_wdata] + filter_inputs)
    logger.info(f'input files fingerprint: {inputs_hash}')

    nr_part = 'all' if debug_nr_triples == -1 else str(debug_nr_triples)
    cache_path = os.path.join(caches_dir, f'loaded_graph_snapshot_{nr_part}_{filter_tag}_{inputs_hash}.pt')

    # Entity labels: written into the snapshot (output_with_labels) and used by the graph loader for the example
    # triple of each qualifier in qualifiers.jsonl. With both Wikipedia filters on, every head and tail the loader
    # keeps is in wikidata_qid_to_wikipedia_page_id, so loading only those labels gives the same lookups as loading
    # all ~110M (and the snapshot's entities are a subset of them); otherwise all labels are loaded.
    wikidata_qid_to_label: Dict = dict()
    if output_with_labels:
        path_wikidata_labels = config['path_wikidata_labels']
        if config['heads_in_wikipedia'] and config['tails_in_wikipedia']:
            qids_to_load = set(wikidata_qid_to_wikipedia_page_id)
        else:
            qids_to_load = None
        labels_hash = generate_short_hash(
            f'{os.path.basename(cache_path)}_{input_files_fingerprint([path_wikidata_labels])}', hash_length=8)
        caches_wikidata_qid_to_label_path = os.path.join(caches_dir, f'wikidata_qid_to_label_{labels_hash}.pickle')
        logger.info('loading wikidata_qid_to_label for '
                    + (f'the {len(qids_to_load):,} Wikipedia-filtered entities' if qids_to_load is not None
                       else 'all entities'))
        wikidata_qid_to_label = load_wikidata_qid_to_label_qids(
            path_wikidata_labels, caches_wikidata_qid_to_label_path, qids_to_load=qids_to_load
        )
        logger.info(f'loaded {len(wikidata_qid_to_label):,} entity labels')
        check_labels('entity labels', wikidata_qid_to_label)

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
                wikidata_qid_to_label=wikidata_qid_to_label,
                precision=config['precision_wdata'])

        index_to_relation = {v: k for k, v in relation_to_index.items()}

        logger.info(f'graph loaded, saving to cache {cache_path}')
        # write to a temporary file first: an interrupted save must never leave a truncated cache behind
        torch.save({
            'pyg_graph_wdata': pyg_graph_wdata,
            'tensor_timestamps_wdata': edge_timestamps_wdata,
            'index_to_entity': index_to_entity,
            'index_to_relation': index_to_relation
        }, cache_path + '.tmp')
        os.replace(cache_path + '.tmp', cache_path)
        logger.info(f'graph saved to cache {cache_path}')
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
    # after the cache is saved: a failing check costs a quick rerun from the cache, not the ~1 h load
    graph_checks = check_graph(pyg_graph_wdata, edge_timestamps_wdata, index_to_entity, index_to_relation,
                               wikidata_qid_to_wikipedia_page_id,
                               both_filters=config['heads_in_wikipedia'] and config['tails_in_wikipedia'])
    check_expected_graph_size(config, graph_checks)

    # Extract snapshots for each requested timestamp
    snapshot_timestamps = config['snapshot_timestamps']
    previous_nr_triples = None

    for snapshot_ts_str in snapshot_timestamps:
        # edge_timestamps are in seconds (load_into_pyg_dynamically_data converts milliseconds)
        snapshot_ts_query = snapshot_epoch(snapshot_ts_str, snapshot_tz)
        machine_local_epoch = snapshot_epoch(snapshot_ts_str, None)
        logger.info(f'extracting snapshot for {snapshot_ts_str} (epoch={snapshot_ts_query}, timezone={snapshot_tz}; '
                    f'machine local time would give {machine_local_epoch})')
        if machine_local_epoch != snapshot_ts_query:
            logger.warning(f'machine local time differs from snapshot_timezone={snapshot_tz}: using the config')

        snapshot_output_dir = os.path.join(output_dir_data, f'snapshot_{snapshot_ts_str}')
        if os.path.exists(os.path.join(snapshot_output_dir, 'metadata.json')):
            raise FileExistsError(f'{snapshot_output_dir} is already complete (metadata.json exists): '
                                  f'remove it or leave this date out of snapshot_timestamps')

        stats = extract_snapshot(
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
        nr_triples = stats['nr_active_triples']
        # SOFT: a KG snapshot normally grows over time (deletions could in principle outweigh additions)
        if previous_nr_triples is not None and nr_triples < previous_nr_triples:
            logger.warning(f'snapshot {snapshot_ts_str} is SMALLER than the previous one '
                           f'({nr_triples:,} < {previous_nr_triples:,})')
        previous_nr_triples = nr_triples

        # Save metadata (written last: its presence marks the snapshot as complete)
        metadata = {
            'snapshot_timestamp': snapshot_ts_str,
            'snapshot_epoch': snapshot_ts_query,
            'snapshot_timezone': snapshot_tz,
            'nr_active_triples': nr_triples,
            'checks': {'snapshot': stats, 'graph': graph_checks},
            'commit_hash': commit_hash,
            'config': config
        }
        with open(os.path.join(snapshot_output_dir, 'metadata.json'), 'wt') as f:
            json.dump(metadata, f, indent=4)

        logger.info(f'snapshot {snapshot_ts_str}: {nr_triples:,} triples written to {snapshot_output_dir}')

    logger.info('all snapshots extracted')
