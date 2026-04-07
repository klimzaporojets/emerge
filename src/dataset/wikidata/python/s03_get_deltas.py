# s03_get_deltas_pyg_v3 differences wrt s03_get_deltas_pyg_v2
#   - Essentially in s03_get_deltas_pyg_v3 aims to produce accumulative deltas as depicted in the following slide: https://docs.google.com/presentation/d/1MXpy8QG1OB_tVARobuJQNVR0QiV_9IdISgK4v0SssoU/edit#slide=id.g315ef8dc105_0_0
#   - the idea is to also include the entities mentioned in a paragraph, and not only base on the target entities pointed from a particular paragraph as in _v2.
#   - The idea is to also account for:
#       - qualifiers (especially for removal of triples, but also for addition),
#       - when delta is in intersection, but has been for some time in wikipedia.
#       - when delta is in intersection, and has been introduced in delta of wikipedia
#       - when one of the entities, either head or tail is emergent in the delta
import argparse
import csv
import json
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Dict, List, Set

import networkx as nx
import torch
from dateutil.relativedelta import relativedelta
from networkx import Graph
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import psutil
import ctypes

from dataset.wikidata.python.misc.s03_constant import AttrIndexes, QualifIdsToActions
from dataset.wikidata.python.misc.s03_volatility import obtain_changes_per_entity, save_most_volatile_entities
from dataset.wikidata.python.misc.wiki_utils import generate_short_hash, load_wikidata_qid_to_label, \
    load_property_qid_to_label, get_git_commit_hash, load_wiki_page_id_to_wdata_qid_only_parsed
from dataset.wikipedia.misc.load_wiki_sql_tables import load_wiki_page_id_to_wikidata_qid

logger = logging.getLogger(__name__)


def get_disconnected_subgraphs(graph_delta: Data):
    # Convert to NetworkX graph
    G: Graph = to_networkx(graph_delta, to_undirected=True)
    # Find connected components
    connected_components = list(nx.connected_components(G))
    # ignore components composed by a single node
    connected_components = [set_elem for set_elem in connected_components if len(set_elem) > 1]
    # TODO: for each connected component create a mask
    # logger.info(f'The following are the connected components: {connected_components}')
    list_of_disconnected_subgraphs: List[Data] = list()
    connected_components = sorted(connected_components,
                                  key=lambda component: len(component),
                                  reverse=True)
    for curr_component in connected_components:
        tn_component = torch.tensor(list(curr_component), dtype=torch.long)
        mask_is_in = torch.isin(graph_delta.edge_index, tn_component)
        # logger.info(f'mask_is_in.shape is: {mask_is_in.shape} '
        #             f'mask_is_in content is: {mask_is_in}')
        index_triples = mask_is_in[0, :] | mask_is_in[1, :]
        edge_index_component = graph_delta.edge_index[:, index_triples]
        edge_attrs_component = graph_delta.edge_attr[index_triples, :]
        curr_component_pyg = Data(edge_index=edge_index_component,
                                  edge_attr=edge_attrs_component)
        list_of_disconnected_subgraphs.append(curr_component_pyg)

    return list_of_disconnected_subgraphs


def get_deltas_data(pyg_graph: Data,
                    kb_type: str,
                    edge_timestamps,
                    timestamp_from,
                    timestamp_to,
                    qualifier_timestamps_wdata,
                    index_to_entity: Dict,
                    index_to_relation: Dict,
                    wikidata_qid_to_label: Dict,
                    property_qid_to_label: Dict,
                    return_timestamp_to: bool = True,
                    return_timestamp_to_or_from: bool = True):
    logger.info('BEGIN get_deltas_data')
    start_time = time.time()
    graph_delta_added_and_removed = Data()
    graph_delta_added_only = Data()
    graph_delta_removed_only = Data()

    # 2025.02.12 - edge_timestamps[:, 2] == 0 means the triple still exists in the version of wikidata that was parsed
    time_from_mask = ((edge_timestamps[:, 1] < timestamp_from) &
                      ((edge_timestamps[:, 2] > timestamp_from) | (edge_timestamps[:, 2] == 0)))
    time_to_mask = ((edge_timestamps[:, 1] < timestamp_to) &
                    ((edge_timestamps[:, 2] > timestamp_to) | (edge_timestamps[:, 2] == 0)))

    # TODO BEGIN 11.03.2025 - Here the intuition is to not allow removals of things that are added just little after
    #  (e.g., before 1 year after removal) or mark somehow
    time_extended_to_mask = (
            (edge_timestamps[:, 1] < timestamp_to + config['triple_removal_stability_offset_in_secs']) &
            (edge_timestamps[:, 1] > timestamp_to))

    # TODO END 11.03.2025 - Here the intuition is to not allow removals of things that are added just little after
    #  (e.g., before 1 year after removal) or mark somehow

    qualifier_mask_added_edge = None
    qualifier_mask_removed_edge = None
    idx_qualifier_added_edge = None
    idx_qualifier_removed_edge = None
    assert kb_type in {'wikidata', 'wikipedia'}
    # if qualifier_timestamps_wdata is not None:
    if kb_type == 'wikidata':
        # qualifier_mask = (timestamp_from < qualifier_timestamps_wdata[:, 2]) & \
        #                  (qualifier_timestamps_wdata[:, 2] < timestamp_to)
        qualifier_mask_added_edge = (timestamp_from < qualifier_timestamps_wdata[:, 2]) & \
                                    (qualifier_timestamps_wdata[:, 2] < timestamp_to) & \
                                    (qualifier_timestamps_wdata[:, 3] == 1)
        qualifier_mask_removed_edge = (timestamp_from < qualifier_timestamps_wdata[:, 2]) & \
                                      (qualifier_timestamps_wdata[:, 2] < timestamp_to) & \
                                      (qualifier_timestamps_wdata[:, 3] == 0)
        #
        idx_qualifier_added_edge = qualifier_timestamps_wdata[qualifier_mask_added_edge, 0]
        idx_qualifier_removed_edge = qualifier_timestamps_wdata[qualifier_mask_removed_edge, 0]
        #

    # idx_time_from = all_timestamps[time_from_mask, 0]
    # idx_time_to = all_timestamps[time_to_mask, 0]

    graph_edges = pyg_graph.edge_index

    # edges_from = graph_edges[:, idx_time_from].T  # --> shape --> [114, 2]
    # edges_to = graph_edges[:, idx_time_to].T  # --> shape --> [188, 2]

    masked_time_to_not_in_from = time_to_mask & (~time_from_mask)
    masked_time_from_not_in_to = time_from_mask & (~time_to_mask)
    masked_time_to_or_from = time_to_mask | time_from_mask

    # idx_qualifier_removed

    idx_time_to_not_in_from = edge_timestamps[masked_time_to_not_in_from, 0]
    idx_time_from_not_in_to = edge_timestamps[masked_time_from_not_in_to, 0]

    # TODO 11.03.2025 begin extended
    idx_time_extended_to = edge_timestamps[time_extended_to_mask, 0]
    # TODO 11.03.2025 end bextended

    ##### 2025.03.05 --- BEGIN: fix bug makes sure there is no overlap between addition and removal

    mask_added = ~torch.isin(idx_time_to_not_in_from, idx_time_from_not_in_to)
    mask_removed = ~torch.isin(idx_time_from_not_in_to, idx_time_to_not_in_from)
    # TODO 11.03.2025 begin extended
    time_extended_start = time.time()
    mask_triple_removed_and_not_added_shortly_later = ~torch.isin(idx_time_from_not_in_to, idx_time_extended_to)
    time_extended_end = time.time()
    logger.info('=======')
    logger.info('nr_of_secs to obtain mask_triple_removed_and_not_added_shortly_later using isin operator: '
                f'{time_extended_end - time_extended_start} with the following shapes: '
                f' -- idx_time_from_not_in_to.shape: {idx_time_from_not_in_to.shape}, '
                f'-- idx_time_extended_to.shape: {idx_time_extended_to.shape} '
                f' -- mask_triple_removed_and_not_added_shortly_later.sum() '
                f'{mask_triple_removed_and_not_added_shortly_later.sum()} '
                f' -- mask_removed.sum() {mask_removed.sum()}')
    mask_removed = mask_removed & mask_triple_removed_and_not_added_shortly_later
    logger.info(f'mask_removed.sum() after: {mask_removed.sum()}')
    logger.info('=======')
    # TODO 11.03.2025 end extended
    # masked_time_to_not_in_from[masked_time_to_not_in_from, :][~mask_not_removed] = False
    # masked_time_to_not_in_from[masked_time_to_not_in_from][~mask_not_removed] = False
    # if masked_time_to_not_in_from[masked_time_to_not_in_from][~mask_not_removed].sum() > 0:
    # assert masked_time_to_not_in_from[masked_time_to_not_in_from][~mask_not_removed].sum() == 0
    # masked_time_from_not_in_to[masked_time_from_not_in_to][~mask_removed] = False
    # assert masked_time_from_not_in_to[masked_time_from_not_in_to][~mask_removed].sum() == 0

    if mask_added.sum() < idx_time_to_not_in_from.shape[0]:
        logger.info('oh_yeahhhhh we had an overlap_between_action_add_and_remove_for_mask_not_removed!')

    if mask_removed.sum() < idx_time_from_not_in_to.shape[0]:
        logger.info('oh_yeahhhhh2 we had an overlap_between_action_add_and_remove_for_mask_removed!')

    idx_time_to_not_in_from = idx_time_to_not_in_from[mask_added]
    idx_time_from_not_in_to = idx_time_from_not_in_to[mask_removed]
    ##### 2025.03.05 --- END: fix bug

    idx_time_to = edge_timestamps[time_to_mask, 0]
    idx_time_to_or_from = edge_timestamps[masked_time_to_or_from, 0]

    edge_attrs = pyg_graph.edge_attr
    idx_attr = 0
    if edge_attrs is None:
        edge_attrs = (-1) * torch.ones((graph_edges.shape[1], 1), dtype=torch.int64)
    else:
        col_attrs = (-1) * torch.ones((graph_edges.shape[1], 1), dtype=torch.int64)
        idx_attr = edge_attrs.shape[1]
        edge_attrs = torch.hstack((edge_attrs, col_attrs))

    # TODO!! idea: might work if we add an extra for wikipedia here!! so no intersection is needed!!!
    edge_attrs[idx_time_to, idx_attr] = 2  # existence of edges in to
    edge_attrs[idx_time_to_not_in_from, idx_attr] = 1  # additions of edges
    edge_attrs[idx_time_from_not_in_to, idx_attr] = 0  # removals of edges

    # timestamp_attrs_delta_added = all_timestamps[idx_time_to_not_in_from, 1:3]
    # timestamp_attrs_delta_removed = all_timestamps[idx_time_from_not_in_to, 1:3]
    col_attrs = (-1) * torch.ones((graph_edges.shape[1], 2), dtype=torch.int64)
    idx_attr = edge_attrs.shape[1]
    edge_attrs = torch.hstack((edge_attrs, col_attrs))
    #
    # edge_attrs[idx_time_to_not_in_from, idx_attr:idx_attr + 2] = edge_timestamps[masked_time_to_not_in_from, 1:3]
    edge_attrs[idx_time_to_not_in_from, idx_attr:idx_attr + 2] = edge_timestamps[masked_time_to_not_in_from, 1:3][
        mask_added]
    # edge_attrs[idx_time_from_not_in_to, idx_attr:idx_attr + 2] = edge_timestamps[masked_time_from_not_in_to, 1:3]
    edge_attrs[idx_time_from_not_in_to, idx_attr:idx_attr + 2] = edge_timestamps[masked_time_from_not_in_to, 1:3][
        mask_removed]
    edge_attrs[idx_time_to, idx_attr:idx_attr + 2] = edge_timestamps[time_to_mask, 1:3]

    #
    ##### TODO - BEGIN wip here, save the relation types with the number of triples
    relation_stats: List[Dict] = list()

    if kb_type == 'wikidata':
        tn_rel_types = edge_attrs[idx_time_to, AttrIndexes.IDX_DELTA_WDATA_RELATION_TYPE]
        unique_values, counts = torch.unique(tn_rel_types, return_counts=True)
        unique_values_lst = unique_values.tolist()
        counts_lst = counts.tolist()
        for curr_unique_value, curr_count in zip(unique_values_lst, counts_lst):
            relation_qid = index_to_relation[curr_unique_value]
            relation_label = relation_qid
            if relation_qid in property_qid_to_label:
                relation_label = property_qid_to_label[relation_qid]
            relation_stats.append({
                'relation_label': relation_label,
                'relation_qid': relation_qid,
                'count': curr_count
            })

        # pass
    ##### TODO - END wip here, save the relation types with the number of triples
    #

    if kb_type == 'wikidata':
        col_attrs = (-1) * torch.ones((graph_edges.shape[1], 1), dtype=torch.int64)
        idx_attr = edge_attrs.shape[1]
        edge_attrs = torch.hstack((edge_attrs, col_attrs))
        edge_attrs[idx_qualifier_added_edge, idx_attr] = 1  # additions of facts according to qualifiers
        edge_attrs[idx_qualifier_removed_edge, idx_attr] = 0  # removals of facts according to qualifiers
        ####
        col_attrs = (-1) * torch.ones((graph_edges.shape[1], 2), dtype=torch.int64)
        idx_attr = edge_attrs.shape[1]
        edge_attrs = torch.hstack((edge_attrs, col_attrs))
        edge_attrs[idx_qualifier_added_edge, idx_attr:idx_attr + 2] = \
            (qualifier_timestamps_wdata)[qualifier_mask_added_edge, 1:3]
        edge_attrs[idx_qualifier_removed_edge, idx_attr:idx_attr + 2] = \
            (qualifier_timestamps_wdata)[qualifier_mask_removed_edge, 1:3]

    # # there is a bug somewhere here
    # timestamp_attrs_delta_added_and_removed = torch.cat(
    #     [timestamp_attrs_delta_added, timestamp_attrs_delta_removed], dim=0)
    #
    # attrs_delta_added_and_removed = torch.cat(
    #     [attrs_delta_added_mark, attrs_delta_removed_mark], dim=0)
    #
    # attrs_delta_added_and_removed = torch.cat([timestamp_attrs_delta_added_and_removed,
    #                                            attrs_delta_added_and_removed], dim=1)
    #
    # attrs_delta_removed = torch.cat([timestamp_attrs_delta_removed,
    #                                  attrs_delta_removed], dim=1)
    #
    # attrs_delta_added = torch.cat([timestamp_attrs_delta_added,
    #                                attrs_delta_added], dim=1)

    graph_timestamp_to = Data()
    graph_timestamp_to_or_from = Data()

    if return_timestamp_to:
        edges_to = pyg_graph.edge_index[:, idx_time_to]
        attrs_to = pyg_graph.edge_attr[idx_time_to, :]
        # timestamp_attrs = all_timestamps[idx_time_to, 1:3]
        timestamp_attrs = edge_timestamps[time_to_mask, 1:3]
        attrs_to = torch.cat([attrs_to, timestamp_attrs], dim=1)
        graph_timestamp_to.edge_index = edges_to
        graph_timestamp_to.edge_attr = attrs_to

    if return_timestamp_to_or_from:
        edges_to_or_from = pyg_graph.edge_index[:, idx_time_to_or_from]
        attrs_to_or_from = pyg_graph.edge_attr[idx_time_to_or_from, :]
        timestamp_attrs = edge_timestamps[masked_time_to_or_from, 1:3]
        # timestamp_attrs = all_timestamps[idx_time_to_or_from, 1:3]
        attrs_to_or_from = torch.cat([attrs_to_or_from, timestamp_attrs], dim=1)

        graph_timestamp_to_or_from.edge_index = edges_to_or_from
        graph_timestamp_to_or_from.edge_attr = attrs_to_or_from

    if kb_type == 'wikidata':
        mask_added_and_removed = (edge_attrs[:, AttrIndexes.IDX_DELTA_WDATA_ACTION] == 1) | \
                                 (edge_attrs[:, AttrIndexes.IDX_DELTA_WDATA_ACTION] == 0) | \
                                 ((edge_attrs[:,
                                 AttrIndexes.IDX_DELTA_WDATA_ACTION] == 2) &  # this last one means that edges are in to (and potentially maybe in from), but some qualifier action is taking place
                                  (edge_attrs[:, AttrIndexes.IDX_DELTA_WDATA_QUALIFIER_ACTION] > -1))
        graph_delta_added_and_removed.edge_index = graph_edges[:, mask_added_and_removed]
        graph_delta_added_and_removed.edge_attr = edge_attrs[mask_added_and_removed, :]
    elif kb_type == 'wikipedia':
        edges_delta_added = graph_edges[:, idx_time_to_not_in_from]
        edges_delta_removed = graph_edges[:, idx_time_from_not_in_to]
        edges_delta_added_and_removed = torch.cat([edges_delta_added, edges_delta_removed], dim=1)
        attrs_delta_added = edge_attrs[idx_time_to_not_in_from, :]
        attrs_delta_removed = edge_attrs[idx_time_from_not_in_to, :]
        graph_delta_added_and_removed.edge_index = edges_delta_added_and_removed
        graph_delta_added_and_removed.edge_attr = torch.cat([attrs_delta_added,
                                                             attrs_delta_removed], dim=0)
    else:
        raise RuntimeError(f'kb_type not recognized: {kb_type}')

    curr_time = time.time()
    logger.info(f'{(curr_time - start_time) / 60:.4f} mins to END get_deltas_data')

    return (
        graph_delta_added_and_removed,
        # graph_delta_removed_only,
        # graph_delta_added_only,
        graph_timestamp_to,
        graph_timestamp_to_or_from,
        relation_stats
    )


def concatenate_graphs_data(batch_triples: Dict, pyg_graph, nr_loaded_triples,
                            batch_timestamps: List,
                            batch_qualifier_timestamps: List,
                            all_timestamps,
                            qualifier_timestamps,
                            entity_to_index: Dict,
                            relation_to_index: Dict):
    if len(batch_triples['edges']) > 0:
        edge_index_tn = torch.tensor(batch_triples['edges'], dtype=torch.long).T
        curr_edge_index_tn = pyg_graph.edge_index
        if curr_edge_index_tn is not None:
            curr_edge_index_tn = torch.cat([curr_edge_index_tn, edge_index_tn], dim=1)
        else:
            curr_edge_index_tn = edge_index_tn
        pyg_graph.edge_index = curr_edge_index_tn

        edge_attrs_triple_index_tn = torch.tensor(batch_triples['triple_id'], dtype=torch.int).unsqueeze(1)
        if len(batch_triples['relations']) > 0:
            assert len(batch_triples['relations']) == len(batch_triples['triple_id'])
            edge_attrs_relations_tn = torch.tensor(batch_triples['relations'], dtype=torch.int).unsqueeze(1)
            curr_edge_attr_tn = torch.cat([edge_attrs_triple_index_tn, edge_attrs_relations_tn], dim=1)
        else:
            curr_edge_attr_tn = edge_attrs_triple_index_tn

        edge_attr = pyg_graph.edge_attr
        if edge_attr is None:
            # pyg_graph.edge_attr = edge_attrs_relations_tn
            pyg_graph.edge_attr = curr_edge_attr_tn
        else:
            # pyg_graph.edge_attr = torch.cat([edge_attr, edge_attrs_relations_tn], dim=0)
            pyg_graph.edge_attr = torch.cat([edge_attr, curr_edge_attr_tn], dim=0)
        logger.info(f'pyg_graph.edge_attr.shape: {pyg_graph.edge_attr.shape} --- '
                    f'pyg_graph.edge_index.shape: {pyg_graph.edge_index.shape}')

    if len(batch_timestamps) > 0:
        # to_tensorize_list = batch_timestamps
        list_tensor = torch.tensor(batch_timestamps, dtype=torch.long)
        if all_timestamps is None:
            all_timestamps = list_tensor
        else:
            all_timestamps = torch.cat([all_timestamps, list_tensor])

    if len(batch_qualifier_timestamps) > 0:
        # to_tensorize_list = batch_timestamps
        list_tensor = torch.tensor(batch_qualifier_timestamps, dtype=torch.long)
        if qualifier_timestamps is None:
            qualifier_timestamps = list_tensor
        else:
            qualifier_timestamps = torch.cat([qualifier_timestamps, list_tensor])

    memory_info = psutil.virtual_memory()
    logger.info(f'nr_loaded_triples is: {nr_loaded_triples:,}')
    free_memory_gb = memory_info.available / (1024 ** 3)  # Convert bytes to gigabytes
    logger.info(f'current free memory is: {free_memory_gb:.2f}')

    return pyg_graph, all_timestamps, qualifier_timestamps


def extend_with_creation_dates(
        pyg_graph: Data,
        entity_to_creation_date_path,
        entity_to_index,
        index_to_entity,
        precision,
        wikidata_qid_to_label: Dict
):
    logger.info('inside extend_with_creation_dates')
    # entity_to_creation_date = dict()
    entity_creation_date_batch = list()
    nr_appended_entities = 0
    entity_creation_date_tensor = torch.zeros((pyg_graph.edge_index.max() + 1, 1), dtype=torch.long)

    # entity_creation_date_tensor = None
    with open(entity_to_creation_date_path, 'rt') as infile:
        tsv_reader = csv.reader(infile, delimiter='\t')
        logger.info(f'reading file: {entity_to_creation_date_path}')
        for row in tsv_reader:
            curr_qid = row[0]
            assert curr_qid.startswith('Q')
            curr_qid_int = int(row[0][1:])
            if curr_qid_int in entity_to_index:
                nr_appended_entities += 1
                curr_timestamp = int(row[1])
                if precision == 'milliseconds':
                    curr_timestamp = int(curr_timestamp / 1000)

                # logic to always assign the lower timestamp for a particular qid, this is because
                # many pages in wikipedia for example can redirect to the same qid, and we want to
                # be sure to have the lowest creation date of those , which will represent when
                # the conceptual entity was really added to wikipedia
                if torch.eq(entity_creation_date_tensor[entity_to_index[curr_qid_int], 0], 0):
                    entity_creation_date_tensor[entity_to_index[curr_qid_int], 0] = curr_timestamp
                elif torch.gt(entity_creation_date_tensor[entity_to_index[curr_qid_int], 0], curr_timestamp):
                    entity_creation_date_tensor[entity_to_index[curr_qid_int], 0] = curr_timestamp

                if nr_appended_entities % 100000 == 0:
                    logger.info(f'nr_appended_entities: {nr_appended_entities}'
                                f' in entity_to_creation_date_path {entity_to_creation_date_path}')

    logger.info(f'obtained entity_creation_data_tensor of shape {entity_creation_date_tensor.shape}')
    pyg_edge_attributes = pyg_graph.edge_attr

    creation_date_zero_mask: torch.Tensor = (entity_creation_date_tensor == 0)
    zero_mask_nr = creation_date_zero_mask.sum().item()
    if zero_mask_nr > 0:
        logger.error(f'SOMETHING WRONG, creation_date_zero_mask.sum() LARGER than 0: {zero_mask_nr} '
                     f'total shape: {creation_date_zero_mask.shape}')

        for curr_entity_idx, curr_timestamp in enumerate(entity_creation_date_tensor.squeeze(1).tolist()):
            if curr_timestamp == 0:
                if curr_entity_idx in index_to_entity:
                    curr_qid = f'Q{index_to_entity[curr_entity_idx]}'
                    label_entity = ''
                    if curr_qid in wikidata_qid_to_label:
                        label_entity = wikidata_qid_to_label[curr_qid]

                    logger.error(f'The following qid has timestamp in zero: {curr_qid} '
                                 f'({label_entity})')
                else:
                    logger.error(f'!!The following entity idx is not in index_to_entity: Q{curr_entity_idx}')

    pyg_edge_index = pyg_graph.edge_index

    pyg_edge_head_timestamps = entity_creation_date_tensor[pyg_edge_index[0, :], 0]
    pyg_edge_tail_timestamps = entity_creation_date_tensor[pyg_edge_index[1, :], 0]

    pyg_edge_head_timestamps_mask = (pyg_edge_head_timestamps == 0)
    pyg_edge_tail_timestamps_mask = (pyg_edge_tail_timestamps == 0)

    zero_mask_nr = pyg_edge_head_timestamps_mask.sum().item()

    if zero_mask_nr > 0:
        logger.error(f'SOMETHING WRONG, pyg_edge_head_timestamps_mask.sum() LARGER than 0: {zero_mask_nr} '
                     f'total shape: {pyg_edge_head_timestamps_mask.shape}')

    zero_mask_nr = pyg_edge_tail_timestamps_mask.sum().item()
    if zero_mask_nr > 0:
        logger.error(f'SOMETHING WRONG, pyg_edge_tail_timestamps_mask.sum() LARGER than 0: {zero_mask_nr}, '
                     f'total shape: {pyg_edge_tail_timestamps_mask.shape}')

    # if pyg_edge_attributes is not None:
    # assert pyg_edge_head_timestamps.shape[0] == pyg_edge_attributes.shape[0]
    # assert pyg_edge_tail_timestamps.shape[0] == pyg_edge_attributes.shape[0]
    assert pyg_edge_head_timestamps.shape[0] == pyg_edge_index.shape[1]
    assert pyg_edge_tail_timestamps.shape[0] == pyg_edge_index.shape[1]

    if pyg_edge_attributes is not None:
        pyg_edge_attributes = torch.cat([pyg_edge_attributes,
                                         pyg_edge_head_timestamps.unsqueeze(1)], dim=1)
    else:
        pyg_edge_attributes = pyg_edge_head_timestamps.unsqueeze(1)
    pyg_edge_attributes = torch.cat([pyg_edge_attributes,
                                     pyg_edge_tail_timestamps.unsqueeze(1)], dim=1)
    # pyg_edge_attributes.shape --> torch.Size([203, 3]) or torch.Size([14485, 2]) in case of wikipedia
    pyg_graph.edge_attr = pyg_edge_attributes
    return pyg_graph, entity_creation_date_tensor


def extract_qualifier_timestamp(qualifier_date_str):
    # Input string
    # date_string = "Y1966MM3D16"
    # logger.info(f'============ START extract_qualifier_timestamp=================')

    # Define the format
    if 'D' in qualifier_date_str:
        # format_string = 'Y%YMM%mD%d'
        year = qualifier_date_str[qualifier_date_str.index('Y') + 1:qualifier_date_str.index('MM')]
        year = int(year)
        month = qualifier_date_str[qualifier_date_str.index('MM') + 2:qualifier_date_str.index('D')]
        month = int(month)
        if 'H' not in qualifier_date_str:
            day = qualifier_date_str[qualifier_date_str.index('D') + 1:]
        else:
            day = qualifier_date_str[qualifier_date_str.index('D') + 1:qualifier_date_str.index('H')]
        day = int(day)
    else:
        year = qualifier_date_str[qualifier_date_str.index('Y') + 1:qualifier_date_str.index('MM')]
        year = int(year)
        month = qualifier_date_str[qualifier_date_str.index('MM') + 2:]
        month = int(month)
        # day = qualifier_date_str[qualifier_date_str.index('D') + 1:]
        day = 1

    return (year, month, day)


def load_into_pyg_dynamically_data(input_triples_path, debug_nr_triples,
                                   entity_to_index, relation_to_index,
                                   qualifier_to_details,
                                   wikidata_qid_to_wikipedia_page_id,
                                   config, kb_type, precision,
                                   property_qid_to_label,
                                   triple_to_index: Dict = None):
    pyg_graph = Data()

    heads_in_wikipedia = config['heads_in_wikipedia']
    tails_in_wikipedia = config['tails_in_wikipedia']

    memory_info = psutil.virtual_memory()

    # Calculate free memory in gigabytes
    free_memory_gb = memory_info.available / (1024 ** 3)  # Convert bytes to gigabytes

    logger.info(f'Free memory before starting: {free_memory_gb:.2f} GB')
    nr_loaded_triples = 0
    exit_loading = False
    if len(entity_to_index) > 0:
        curr_entity_index = max(entity_to_index.values()) + 1
    else:
        curr_entity_index = 0

    batch_triples = {'edges': list(), 'relations': list(), 'triple_id': list()}
    if len(relation_to_index) > 0:
        curr_relation_index = max(relation_to_index.values()) + 1
    else:
        curr_relation_index = 0
    batch_timestamps = list()
    batch_qualifier_timestamps = list()
    all_timestamps = None
    qualifier_timestamps = None
    showed = False
    idx_triple = 0

    output_dir_data = config['output_dir_data']
    os.makedirs(output_dir_data, exist_ok=True)
    qualifiers_file = None
    if kb_type == 'wikidata':
        qualifiers_list_path = os.path.join(output_dir_data, 'qualifiers.jsonl')
        qualifiers_file = open(qualifiers_list_path, 'wt')

    files_to_process = os.listdir(input_triples_path)
    notified_qualifiers = dict()
    if kb_type == 'wikidata':
        triple_to_index = dict()
    for curr_file in files_to_process:
        curr_file_path = os.path.join(input_triples_path, curr_file)
        with (open(curr_file_path, 'r') as input_file):
            tsv_reader = csv.reader(input_file, delimiter='\t')
            for row in tsv_reader:
                add_triple = True
                if kb_type == 'wikidata':
                    tail_type = row[3]
                    curr_head = int(row[0])
                    curr_relation = row[1]
                    curr_tail = row[2]
                    qid_head = f'Q{curr_head}'
                elif kb_type == 'wikipedia':
                    tail_type = 'wikibase-entityid'
                    curr_head = row[1]
                    assert curr_head.startswith('Q')
                    qid_head = curr_head
                    curr_head = int(curr_head[1:])
                    curr_relation = 'none'
                    curr_tail = row[3]
                    assert curr_tail.startswith('Q')
                    curr_tail = curr_tail[1:]
                    # qid_head = f'Q{curr_head}'
                else:
                    raise RuntimeError(f'kb_type not recognized: {kb_type}')
                if tail_type == 'wikibase-entityid':
                    curr_tail = int(curr_tail)
                    qid_tail = f'Q{curr_tail}'
                    ###########
                    if heads_in_wikipedia and add_triple:
                        add_triple = qid_head in wikidata_qid_to_wikipedia_page_id
                    if tails_in_wikipedia and add_triple:
                        add_triple = qid_tail in wikidata_qid_to_wikipedia_page_id

                    if add_triple:
                        showed = False
                        if curr_head not in entity_to_index:
                            entity_to_index[curr_head] = curr_entity_index
                            curr_entity_index += 1

                        if curr_tail not in entity_to_index:
                            entity_to_index[curr_tail] = curr_entity_index
                            curr_entity_index += 1

                        if kb_type == 'wikidata' and curr_relation not in relation_to_index:
                            relation_to_index[curr_relation] = curr_relation_index
                            curr_relation_index += 1

                        curr_head_idx = entity_to_index[curr_head]
                        curr_tail_idx = entity_to_index[curr_tail]
                        if kb_type == 'wikidata':
                            if (curr_head_idx, curr_tail_idx) not in triple_to_index:
                                triple_to_index[(curr_head_idx, curr_tail_idx)] = len(triple_to_index)
                                if len(triple_to_index) % 1000000 == 0:
                                    logger.info(f'number of elements in triple_to_index: '
                                                f'{len(triple_to_index)}')
                            triple_id = triple_to_index[(curr_head_idx, curr_tail_idx)]
                        elif kb_type == 'wikipedia':
                            if (curr_head_idx, curr_tail_idx) in triple_to_index:
                                triple_id = triple_to_index[(curr_head_idx, curr_tail_idx)]
                            else:
                                triple_id = -1
                        else:
                            raise RuntimeError(f'kb_type not recognized {kb_type}')
                        batch_triples['edges'].append([curr_head_idx, curr_tail_idx])
                        batch_triples['triple_id'].append(triple_id)
                        if kb_type == 'wikidata':
                            batch_triples['relations'].append(relation_to_index[curr_relation])
                        elif kb_type == 'wikipedia':
                            pass
                            # batch_triples['relations'].append(0)
                        else:
                            raise RuntimeError(f'kb_type not recognized: {kb_type}')

                        # ]([curr_head_idx, relation_to_index[curr_relation], curr_tail_idx])
                        nr_loaded_triples += 1

                        ######
                        history_target = row[4]
                        history_target = history_target.split(',')
                        if history_target[-1] == '':
                            history_target = history_target[:-1]
                        history_target_no_qualifiers = [hist for hist in history_target
                                                        if not hist.startswith('P')]
                        history_target_qualifiers = [hist for hist in history_target
                                                     if hist.startswith('P')]
                        if len(history_target_qualifiers) > 0:
                            # TODO - map qualifiers to from or to
                            # TODO - obtain the timestamp from qualifiers
                            # logger.info(f'history of qualifiers > 0: {history_target_qualifiers}')
                            timestamps_qualifiers_from = None
                            timestamps_qualifiers_to = None
                            for curr_qualifier in history_target_qualifiers:
                                # logger.info(f'curr_qualifier: {curr_qualifier}')
                                qualifier_id = curr_qualifier[:curr_qualifier.index(':')]

                                qualifier_precision = curr_qualifier[curr_qualifier.rindex(':') + 1:]
                                # logger.info(f'qualifier precision {qualifier_precision}')
                                qualifier_precision = int(qualifier_precision)
                                if qualifier_precision in {10, 11}:
                                    qualifier_date_str = curr_qualifier[
                                        curr_qualifier.index(':') + 1:
                                        curr_qualifier.rindex(':')
                                    ]
                                    # logger.info(f'qualifier_date_str: {qualifier_date_str}')
                                    # logger.info(f'curr_qualifier: {curr_qualifier}')
                                    try:
                                        qualifier_year, qualifier_month, qualifier_day = extract_qualifier_timestamp(
                                            qualifier_date_str)
                                        # BEGIN - commented as of 07.03.2025, qualifiers ignored
                                        # if qualifier_year < 2012:
                                        #     continue
                                        # END - commented as of 07.03.2025, qualifiers ignored
                                        keep_trying = True

                                        while keep_trying:
                                            try:
                                                dt = datetime(year=qualifier_year, month=qualifier_month,
                                                              day=qualifier_day)
                                                qualifier_timestamp = int(dt.timestamp())
                                                keep_trying = False
                                            except ValueError as e:
                                                logger.error('------------------------------------')
                                                logger.error(traceback.format_exc())
                                                logger.error(f'ValueError with the following qualifier '
                                                             f'{curr_qualifier}')
                                                logger.error(
                                                    f'ValueError extracted from the qualifier: '
                                                    f'qualifier_year: {qualifier_year}, '
                                                    f'qualifier_month: {qualifier_month}, '
                                                    f'qualifier_day: {qualifier_day}, '
                                                    f'timestamp: {qualifier_timestamp}'
                                                )
                                                if qualifier_day >= 29:
                                                    qualifier_day_new = qualifier_day - 1
                                                    logger.error('ValueError adjusting qualifier_day '
                                                                 f'from {qualifier_day} to {qualifier_day_new}')
                                                    qualifier_day = qualifier_day_new
                                                    logger.error('------------------------------------')
                                                else:
                                                    logger.error('dont_know_how_to_fix, continuing')
                                                    logger.error('------------------------------------')
                                                    keep_trying = False
                                    except Exception as e:
                                        logger.error('------------------------------------')
                                        logger.error(f'Exception_error with the following qualifier '
                                                     f'{curr_qualifier}')
                                        logger.error(
                                            f'Exception_extracted from the qualifier: '
                                            f'qualifier_year: {qualifier_year}, '
                                            f'qualifier_month: {qualifier_month}, '
                                            f'qualifier_day: {qualifier_day}, '
                                            f'timestamp: {qualifier_timestamp}'
                                        )
                                        logger.error(traceback.format_exc())
                                        logger.error('------------------------------------')
                                        continue

                                    if qualifier_id not in relation_to_index:
                                        relation_to_index[qualifier_id] = curr_relation_index
                                        curr_relation_index += 1

                                    action_idx = -1
                                    if qualifier_id in QualifIdsToActions.QUALIF_IDS_TO_ACTION:
                                        action_idx = QualifIdsToActions.QUALIF_IDS_TO_ACTION[qualifier_id]
                                        # logger.info(f'action_idx for qualifier_id {qualifier_id} is '
                                        #             f'{action_idx}')

                                    batch_qualifier_timestamps.append(
                                        (
                                            idx_triple,
                                            relation_to_index[qualifier_id],
                                            qualifier_timestamp,
                                            action_idx
                                        )
                                    )

                                #
                                if qualifier_id not in qualifier_to_details and \
                                        qualifier_id not in notified_qualifiers:
                                    #
                                    label_head = qid_head
                                    label_tail = qid_tail
                                    label_relation = curr_relation

                                    if qid_head in wikidata_qid_to_label:
                                        label_head = wikidata_qid_to_label[qid_head]

                                    if qid_tail in wikidata_qid_to_label:
                                        label_tail = wikidata_qid_to_label[qid_tail]

                                    if curr_relation in property_qid_to_label:
                                        label_relation = property_qid_to_label[curr_relation]

                                    notified_qualifiers[qualifier_id] = \
                                        {'label': property_qid_to_label[qualifier_id],
                                         'idx': len(notified_qualifiers),
                                         'type': 0,
                                         'example': curr_qualifier,
                                         'history_target_no_qualifiers': history_target_no_qualifiers,
                                         'triple': [qid_head, curr_relation, qid_tail],
                                         'triple_str': [label_head, label_relation, label_tail]
                                         }
                                    #
                                    if qualifiers_file is not None:
                                        qualifiers_file.write(json.dumps(notified_qualifiers[qualifier_id]) + '\n')
                                        qualifiers_file.flush()
                                    #
                                    # logger.info(f'curr_qualifier has to be added to details: {curr_qualifier}')
                                    # logger.info(
                                    #     f'curr_qualifier has to be added to details: {notified_qualifiers[qualifier_id]}')
                                    # logger.info(f'notified_qualifiers: {notified_qualifiers}')

                        timestamps_from = [int(ht[:-2])
                                           for ht in history_target_no_qualifiers
                                           if ht.endswith('A')]
                        timestamps_to = [int(ht[:-2])
                                         for ht in history_target_no_qualifiers
                                         if ht.endswith('D')]
                        # precision = config['precision']
                        if precision == 'milliseconds':
                            timestamps_from = \
                                [int(tf / 1000) for tf in timestamps_from]
                            timestamps_to = \
                                [int(tf / 1000) for tf in timestamps_to]

                        if len(timestamps_to) < len(timestamps_from):
                            timestamps_to.append(0)
                            # timestamps_to.append(int(time.time()))
                        assert len(timestamps_to) == len(timestamps_from)

                        for curr_timestamp_from, curr_timestamp_to in \
                                zip(timestamps_from, timestamps_to):
                            assert curr_timestamp_from < curr_timestamp_to or curr_timestamp_to == 0
                            batch_timestamps.append(
                                (
                                    idx_triple,
                                    curr_timestamp_from,
                                    curr_timestamp_to
                                )
                            )

                        idx_triple += 1
                        ######

                if -1 < debug_nr_triples < nr_loaded_triples:
                    logger.info(f'nr_loaded_triples ({nr_loaded_triples:,}) exceeds '
                                f'debug_nr_triples ({debug_nr_triples:,}), exiting')
                    exit_loading = True
                    break

                #### BEGIN add to PyG
                if nr_loaded_triples % 10000000 == 0 and not showed:
                    # if nr_loaded_triples % 10 == 0 and not showed:
                    logger.info(f'nr_loaded_triples: {nr_loaded_triples}')
                    showed = True
                    pyg_graph, all_timestamps, qualifier_timestamps = concatenate_graphs_data(
                        batch_triples=batch_triples,
                        pyg_graph=pyg_graph,
                        nr_loaded_triples=nr_loaded_triples,
                        batch_timestamps=batch_timestamps,
                        batch_qualifier_timestamps=batch_qualifier_timestamps,
                        all_timestamps=all_timestamps,
                        qualifier_timestamps=qualifier_timestamps,
                        entity_to_index=entity_to_index,
                        relation_to_index=relation_to_index
                    )

                    batch_timestamps = list()
                    batch_qualifier_timestamps = list()
                    batch_triples = {'edges': list(), 'relations': list(), 'triple_id': list()}

            pyg_graph, all_timestamps, qualifier_timestamps = concatenate_graphs_data(
                batch_triples=batch_triples,
                pyg_graph=pyg_graph,
                nr_loaded_triples=nr_loaded_triples,
                batch_timestamps=batch_timestamps,
                batch_qualifier_timestamps=batch_qualifier_timestamps,
                all_timestamps=all_timestamps,
                qualifier_timestamps=qualifier_timestamps,
                entity_to_index=entity_to_index,
                relation_to_index=relation_to_index
            )

            batch_timestamps = list()
            batch_qualifier_timestamps = list()

            batch_triples = {'edges': list(), 'relations': list(), 'triple_id': list()}
            # exits the loading if set above
            if exit_loading:
                break

    # logger.info(f'graph loaded and contains {pyg_graph} '
    #             f'all_timestamps: {all_timestamps}')
    logger.info(f'notified_qualifiers are: {notified_qualifiers}')
    index_to_entity = {value: key for key, value in entity_to_index.items()}

    return (pyg_graph, all_timestamps, entity_to_index, index_to_entity, relation_to_index,
            triple_to_index, qualifier_timestamps)


def append_graph_to_file(
        csv_writer,
        subgraph_idx: int,
        wikidata_qid_to_label: Dict,
        property_qid_to_label: Dict,
        delta_graph: Data,
        kb_type: str,
        index_to_relation: Dict,
        index_to_entity: Dict,
        timestamp_from: int,
        timestamp_to: int,
        sorted_entities: torch.Tensor,
        sorted_changes: torch.Tensor,
        sorted_normalized_changes: torch.Tensor,
        config: Dict,
        entities_to_changes: Dict,
        delta_intersection: Data = None
):
    logger.info('BEGIN append_graph_to_file')
    start_time_super = time.time()
    curr_entity_idx_to_changes = dict()
    curr_entity_idx_to_normalized_changes = dict()
    curr_tensor_row: torch.Tensor
    rows_to_write = list()
    nr_processed = 0
    accum_time_nr_changes = 0.0
    accum_time_nr_sub_changes = 0.0
    accum_time_relation_type = 0.0
    accum_time_action = 0.0
    accum_time_intersection = 0.0
    accum_time_write = 0.0

    if kb_type == 'wikidata':
        action_attr_index = AttrIndexes.IDX_DELTA_WDATA_ACTION
        # action_attr_index = delta_graph.edge_attr.shape[1] - 1
    elif kb_type == 'wikipedia':
        action_attr_index = AttrIndexes.IDX_DELTA_WPEDIA_ACTION
    else:
        raise RuntimeError(f'kb_type not recognized: {kb_type}')

    for idx_row, curr_tensor_row in enumerate(delta_graph.edge_index.T):
        nr_processed += 1
        curr_head_idx = curr_tensor_row[0].item()
        curr_head_qid = f'Q{index_to_entity[curr_head_idx]}'
        curr_tail_idx = curr_tensor_row[1].item()
        curr_tail_qid = f'Q{index_to_entity[curr_tail_idx]}'
        is_in_intersection = False

        start_time = time.time()
        if curr_head_idx not in curr_entity_idx_to_changes:
            start_sub_time = time.time()
            curr_entity_idx_to_changes[curr_head_idx] = entities_to_changes[curr_head_idx][0]
            curr_entity_idx_to_normalized_changes[curr_head_idx] = entities_to_changes[curr_head_idx][1]
            # mask = (sorted_entities == curr_head_idx)
            # nr_changes = sorted_changes[mask].item()
            # nr_normalized_changes = sorted_normalized_changes[mask].item()
            # curr_entity_idx_to_changes[curr_head_idx] = nr_changes
            # curr_entity_idx_to_normalized_changes[curr_head_idx] = nr_normalized_changes
            accum_time_nr_sub_changes += (time.time() - start_sub_time)

        if (not config['volatile_only_changes_in_head'] and
                curr_tail_idx not in curr_entity_idx_to_normalized_changes):
            mask = (sorted_entities == curr_tail_idx)
            nr_changes_tn = sorted_changes[mask]
            if nr_changes_tn.numel() > 0:
                nr_changes = nr_changes_tn.item()
                sorted_changes_tn = sorted_normalized_changes[mask]
                nr_normalized_changes = sorted_changes_tn.item()
                curr_entity_idx_to_changes[curr_tail_idx] = nr_changes
                curr_entity_idx_to_normalized_changes[curr_tail_idx] = nr_normalized_changes
            else:
                curr_entity_idx_to_changes[curr_tail_idx] = -1
                curr_entity_idx_to_normalized_changes[curr_tail_idx] = -1
        else:
            if curr_tail_idx not in curr_entity_idx_to_changes:
                curr_entity_idx_to_changes[curr_tail_idx] = -1
                curr_entity_idx_to_normalized_changes[curr_tail_idx] = -1

        accum_time_nr_changes += (time.time() - start_time)

        tail_normalized_changes = curr_entity_idx_to_normalized_changes[curr_tail_idx]
        tail_changes = curr_entity_idx_to_changes[curr_tail_idx]
        head_normalized_changes = curr_entity_idx_to_normalized_changes[curr_head_idx]
        head_changes = curr_entity_idx_to_changes[curr_head_idx]

        curr_head_label = ''
        curr_tail_label = ''
        if curr_head_qid in wikidata_qid_to_label:
            curr_head_label = wikidata_qid_to_label[curr_head_qid]

        if curr_tail_qid in wikidata_qid_to_label:
            curr_tail_label = wikidata_qid_to_label[curr_tail_qid]

        curr_relation_label = ''
        curr_relation_qid = ''
        start_time = time.time()

        if kb_type == 'wikidata':
            curr_relation_idx = delta_graph.edge_attr[idx_row, AttrIndexes.IDX_DELTA_WDATA_RELATION_TYPE].item()
            curr_relation_qid = index_to_relation[curr_relation_idx]
            if curr_relation_qid in property_qid_to_label:
                curr_relation_label = property_qid_to_label[curr_relation_qid]
            curr_head_creation_timestamp = delta_graph.edge_attr[
                idx_row, AttrIndexes.IDX_DELTA_WDATA_HEAD_CREATION_DATE].item()
            curr_tail_creation_timestamp = delta_graph.edge_attr[
                idx_row, AttrIndexes.IDX_DELTA_WDATA_TAIL_CREATION_DATE].item()
            curr_triple_timestamp_from = delta_graph.edge_attr[
                idx_row, AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_FROM].item()
            curr_triple_timestamp_to = delta_graph.edge_attr[
                idx_row, AttrIndexes.IDX_DELTA_WDATA_TRIPLE_TSTMP_TO].item()
        else:
            curr_head_creation_timestamp = delta_graph.edge_attr[
                idx_row, AttrIndexes.IDX_DELTA_WPEDIA_HEAD_CREATION_DATE].item()
            curr_tail_creation_timestamp = delta_graph.edge_attr[
                idx_row, AttrIndexes.IDX_DELTA_WPEDIA_TAIL_CREATION_DATE].item()
            curr_triple_timestamp_from = delta_graph.edge_attr[
                idx_row, AttrIndexes.IDX_DELTA_WPEDIA_TRIPLE_TSTMP_FROM].item()
            curr_triple_timestamp_to = delta_graph.edge_attr[
                idx_row, AttrIndexes.IDX_DELTA_WPEDIA_TRIPLE_TSTMP_TO].item()

        accum_time_relation_type += (time.time() - start_time)

        start_time = time.time()
        curr_action = delta_graph.edge_attr[idx_row, action_attr_index].item()
        if curr_action == 1:
            curr_action_label = 'added'
        elif curr_action == 0:
            curr_action_label = 'removed'
        else:
            if kb_type == 'wikidata':
                qualifier_action = delta_graph.edge_attr[idx_row, \
                    AttrIndexes.IDX_DELTA_WDATA_QUALIFIER_ACTION].item()

                # qualifier_action = curr_tensor_row[1][AttrIndexes.IDX_DELTA_WDATA_QUALIFIER_ACTION]
                if qualifier_action == 0:
                    curr_action_label = 'removed_qualifier'
                elif qualifier_action == 1:
                    curr_action_label = 'added_qualifier'
                else:
                    raise RuntimeError(f'curr_action not recognized: {curr_action} '
                                       f'for kb_type: {kb_type} '
                                       f'and action of qualifier: {qualifier_action}')
            else:
                raise RuntimeError(f'curr_action not recognized: {curr_action} '
                                   f'for kb_type: {kb_type}')
        accum_time_action += (time.time() - start_time)

        start_time = time.time()
        if delta_intersection is not None and delta_intersection.edge_index.numel() > 0:
            intersec_bool = (curr_tensor_row == delta_intersection.edge_index.T)
            intersec_bool = intersec_bool[:, 0] & intersec_bool[:, 1]
            is_in_intersection = (intersec_bool.sum().item() > 0)
        accum_time_intersection += (time.time() - start_time)

        start_time = time.time()
        rows_to_write.append([subgraph_idx,
                              curr_action_label,
                              curr_head_qid,
                              curr_head_label,
                              curr_relation_qid,
                              curr_relation_label,
                              curr_tail_qid,
                              curr_tail_label,
                              head_changes,
                              head_normalized_changes,
                              tail_changes,
                              tail_normalized_changes,
                              is_in_intersection,
                              curr_head_creation_timestamp,
                              curr_tail_creation_timestamp,
                              timestamp_from,
                              timestamp_to,
                              curr_triple_timestamp_from,
                              curr_triple_timestamp_to])
        if nr_processed % 1000 == 0:
            csv_writer.writerows(rows_to_write)
            rows_to_write = list()
        accum_time_write += (time.time() - start_time)
    if len(rows_to_write) > 0:
        csv_writer.writerows(rows_to_write)

    end_time = time.time()
    #     accum_time_nr_changes = 0.0
    #     accum_time_relation_type = 0.0
    #     accum_time_action = 0.0
    #     accum_time_intersection = 0.0
    #     accum_time_write = 0.0
    logger.debug(
        f'accumultated times: \n\t '
        f'accum_time_nr_changes: {accum_time_nr_changes} \n\t'
        f'accum_time_relation_type: {accum_time_relation_type} \n\t'
        f'accum_time_action: {accum_time_action} \n\t'
        f'accum_time_intersection: {accum_time_intersection} \n\t'
        f'accum_time_write: {accum_time_write} \n\t'
        f'accum_time_nr_sub_changes: {accum_time_nr_sub_changes}'
    )
    logger.info(f'{(end_time - start_time_super) / 60:.4f} mins to END append_graph_to_file')


#         append_graph_target_entities_to_file(
#             csv_writer=writer,
#             head_to_target_entities_in_intersection=head_to_target_entities_in_intersection,
#             wikidata_qid_to_label=wikidata_qid_to_label,
#             property_qid_to_label=property_qid_to_label,
#             kb_type='wikidata',
#             index_to_relation=index_to_relation,
#             index_to_entity=index_to_entity,
#             timestamp_from=timestamp_from,
#             timestamp_to=timestamp_to,
#             config=config
#         )

def append_graph_target_entities_to_file(
        csv_writer,
        head_to_target_entities_in_intersection: Dict[int, Set[int]],
        wikidata_qid_to_label: Dict,
        property_qid_to_label: Dict,
        kb_type: str,
        index_to_relation: Dict,
        index_to_entity: Dict,
        timestamp_from: int,
        timestamp_to: int,
        config: Dict
):
    assert kb_type in {'wikipedia', 'wikidata'}
    if kb_type != 'wikidata':
        return
    logger.info('BEGIN append_graph_to_file')
    start_time_super = time.time()
    rows_to_write = list()
    nr_processed = 0
    accum_time_nr_changes = 0.0
    accum_time_nr_sub_changes = 0.0
    accum_time_relation_type = 0.0
    accum_time_action = 0.0
    accum_time_intersection = 0.0
    accum_time_write = 0.0

    # Expected output to s03_API_v2.py:
    #                         curr_head_qid = row[0]
    #                         curr_head_label = row[1]
    #                         curr_target_qids = row[2].split('##')
    #                         curr_target_labels = row[3].split('##')
    curr_head_id: int
    curr_tail_ids: Set[int]
    for curr_head_id, curr_tail_ids in head_to_target_entities_in_intersection.items():
        nr_processed += 1
        curr_head_idx = curr_head_id
        curr_head_qid = f'Q{index_to_entity[curr_head_idx]}'
        curr_tail_idxs: List[int] = list(curr_tail_ids)
        curr_target_qids: List[str] = [f'Q{index_to_entity[curr_tail_idx]}' for curr_tail_idx in curr_tail_idxs]

        if curr_head_qid in wikidata_qid_to_label:
            curr_head_label = wikidata_qid_to_label[curr_head_qid]
        else:
            curr_head_label = curr_head_qid

        curr_target_labels = list()
        for curr_tail_qid in curr_target_qids:
            if curr_tail_qid in wikidata_qid_to_label:
                curr_target_labels.append(wikidata_qid_to_label[curr_tail_qid])
            else:
                curr_target_labels.append(curr_tail_qid)

        start_time = time.time()
        rows_to_write.append([curr_head_qid,
                              curr_head_label,
                              '##'.join(curr_target_qids),
                              '##'.join(curr_target_labels)])
        if nr_processed % 1000 == 0:
            csv_writer.writerows(rows_to_write)
            rows_to_write = list()
        accum_time_write += (time.time() - start_time)
    if len(rows_to_write) > 0:
        csv_writer.writerows(rows_to_write)

    end_time = time.time()
    logger.debug(
        f'accumultated times: \n\t '
        f'accum_time_nr_changes: {accum_time_nr_changes} \n\t'
        f'accum_time_relation_type: {accum_time_relation_type} \n\t'
        f'accum_time_action: {accum_time_action} \n\t'
        f'accum_time_intersection: {accum_time_intersection} \n\t'
        f'accum_time_write: {accum_time_write} \n\t'
        f'accum_time_nr_sub_changes: {accum_time_nr_sub_changes}'
    )
    logger.info(f'{(end_time - start_time_super) / 60:.4f} mins to END append_graph_to_file')


def show_and_save_delta(
        timestamp_from: int,
        timestamp_to: int,
        config,
        output_file_path,
        output_file_path_subgraphs,
        output_file_path_volatile,
        delta_added_and_removed: Data,
        complete_graph: Data,
        kb_type,
        data_timestamp_to_or_from: Data,
        delta_intersection: Data
):
    start_time = time.time()
    logger.info('BEGIN show_and_save_delta')
    dir_path = os.path.dirname(output_file_path)
    os.makedirs(dir_path, exist_ok=True)
    sorted_entities, sorted_changes, sorted_normalized_changes, entities_to_changes = (
        obtain_changes_per_entity(
            delta_added_and_removed=delta_added_and_removed,
            config=config,
            wdata_timestamp_to=data_timestamp_to_or_from,
            only_changes_in_head=config['volatile_only_changes_in_head']
        ))
    # curr_time = time.time()
    # logger.info(f'END obtain_changes_per_entity {((curr_time - start_time) / 60):.3f} mins')
    save_most_volatile_entities(
        delta_added_and_removed=delta_added_and_removed,
        # complete_graph=complete_graph,
        config=config,
        output_file_path_volatile=output_file_path_volatile,
        index_to_entity=index_to_entity,
        index_to_relation=index_to_relation,
        wikidata_qid_to_label=wikidata_qid_to_label,
        property_qid_to_label=property_qid_to_label,
        kb_type=kb_type,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        # wdata_timestamp_to=data_timestamp_to_or_from
        sorted_entities=sorted_entities,
        sorted_changes=sorted_changes,
        sorted_normalized_changes=sorted_normalized_changes
    )

    if config['extract_disconnected_subgraphs']:
        logger.info(f'BEGIN obtaining subgraphs for graph of size '
                    f'{delta_added_and_removed.edge_index.shape}')
        disconnected_subgraphs_wdata = (get_disconnected_subgraphs(delta_added_and_removed))
        logger.info('END obtaining subgraphs')

        with open(output_file_path_subgraphs, 'wt') as outfile:
            writer = csv.writer(outfile, delimiter='\t')
            for subgraph_idx, curr_disconnected_subgraph in enumerate(disconnected_subgraphs_wdata):
                # outfile.write(f'=====DISCONNECTED SUBGRAPH NR {subgraph_idx}======\n')
                append_graph_to_file(
                    csv_writer=writer,
                    subgraph_idx=subgraph_idx,
                    wikidata_qid_to_label=wikidata_qid_to_label,
                    property_qid_to_label=property_qid_to_label,
                    delta_graph=curr_disconnected_subgraph,
                    kb_type=kb_type,
                    index_to_entity=index_to_entity,
                    index_to_relation=index_to_relation,
                    timestamp_from=timestamp_from,
                    timestamp_to=timestamp_to,
                    sorted_entities=sorted_entities,
                    sorted_changes=sorted_changes,
                    sorted_normalized_changes=sorted_normalized_changes,
                    config=config,
                    entities_to_changes=entities_to_changes,
                    delta_intersection=delta_intersection
                )
    else:
        logger.debug(f'\'extract_disconnected_subgraphs\' '
                     f'in false so NOT obtaining subgraphs '
                     f'for graph of size '
                     f'{delta_added_and_removed.edge_index.shape}')

    torch.save(delta_added_and_removed, f'{output_file_path}.pt')

    with open(output_file_path, 'wt') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        append_graph_to_file(
            subgraph_idx=-1,
            csv_writer=writer,
            wikidata_qid_to_label=wikidata_qid_to_label,
            property_qid_to_label=property_qid_to_label,
            delta_graph=delta_added_and_removed,
            kb_type=kb_type,
            index_to_entity=index_to_entity,
            index_to_relation=index_to_relation,
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            sorted_entities=sorted_entities,
            sorted_changes=sorted_changes,
            sorted_normalized_changes=sorted_normalized_changes,
            config=config,
            entities_to_changes=entities_to_changes,
            delta_intersection=delta_intersection
        )

    logger.info(f'{(time.time() - start_time) / 60:.4f} mins to END show_and_save_delta')


def show_intersection_wdata_wpedia(
        timestamp_from: int,
        timestamp_to: int,
        config,
        output_file_path: str,
        output_file_path_subgraphs: str,
        output_file_path_target_entities: str,
        delta_added_and_removed_wpedia: Data,
        # all the edges in wikipedia at the timestamp_to point in time
        wpedia_timestamp_to: Data,
        delta_added_and_removed_wdata: Data,
        complete_graph: Data,
        index_to_entity: Dict,
        index_to_relation: Dict,
        property_qid_to_label: Dict,
        output_file_path_volatile: str,
        wdata_timestamp_to: Data
):
    logger.info('BEGIN show_intersection_wdata_wpedia')
    start_time = time.time()

    edge_attrs_wpedia = delta_added_and_removed_wpedia.edge_attr
    edge_attrs_wdata = delta_added_and_removed_wdata.edge_attr
    # actions_wdata = edge_attrs_wdata[:, edge_attrs_wdata.shape[1] - 1]
    # actions_wpedia = edge_attrs_wpedia[:, edge_attrs_wpedia.shape[1] - 1]
    actions_wdata = edge_attrs_wdata[:, AttrIndexes.IDX_DELTA_WDATA_ACTION]
    actions_wdata_qualifier = edge_attrs_wdata[:, AttrIndexes.IDX_DELTA_WDATA_QUALIFIER_ACTION]
    actions_wpedia = edge_attrs_wpedia[:, AttrIndexes.IDX_DELTA_WPEDIA_ACTION]

    edge_idxs_wpedia = delta_added_and_removed_wpedia.edge_index
    edge_idxs_wdata = delta_added_and_removed_wdata.edge_index

    # added_edges_wdata_mask = (actions_wdata == 1)
    added_edges_wdata_mask = (actions_wdata == 1) | (actions_wdata_qualifier == 1)
    added_edges_wpedia_mask = (actions_wpedia == 1)

    added_edges_wdata = edge_idxs_wdata[:, added_edges_wdata_mask]
    # added_edges_wdata = edge_idxs_wdata[:, added_edges_wdata_mask]

    # added_edges_wpedia = edge_idxs_wpedia[:, added_edges_wpedia_mask]
    added_edge_attrs_wpedia = edge_attrs_wpedia[added_edges_wpedia_mask, :]
    added_edge_attrs_wdata = edge_attrs_wdata[added_edges_wdata_mask, :]
    #
    deleteme_removed_wdata_qualifier_mask = (actions_wdata_qualifier == 0)
    removed_edges_wdata_mask = (actions_wdata == 0) | (actions_wdata_qualifier == 0)
    removed_edges_wdata = edge_idxs_wdata[:, removed_edges_wdata_mask]
    removed_edge_attrs_wdata = edge_attrs_wdata[removed_edges_wdata_mask, :]

    # curr_sub_time = time.time()
    # logger.info(f'{(time.time() - curr_sub_time) / 60:.4f} '
    #             f'mins to complete show_intersection_wdata_wpedia 1')

    # curr_sub_time = time.time()
    # curr_wdata_edge: Tensor
    # logger.info(f'{(time.time() - curr_sub_time) / 60:.4f} '
    #             f'mins to complete show_intersection_wdata_wpedia 3')
    curr_sub_time = time.time()
    curr_wdata_edge: Tensor

    # FOR ADDED EDGES
    unique_triple_ids_added_edges_wdata = added_edge_attrs_wdata[:, AttrIndexes.IDX_DELTA_WDATA_TRIPLE_ID].unique()

    unique_triple_ids_added_edges_wpedia = added_edge_attrs_wpedia[:, AttrIndexes.IDX_DELTA_WPEDIA_TRIPLE_ID].unique()

    # common_triples_ids = torch.intersect1d(unique_triple_ids_added_edges_wdata, unique_triple_ids_added_edges_wpedia)
    common_triples_ids = torch.masked_select(unique_triple_ids_added_edges_wdata,
                                             torch.isin(unique_triple_ids_added_edges_wdata,
                                                        unique_triple_ids_added_edges_wpedia))

    mask_wdata = torch.isin(added_edge_attrs_wdata[:, AttrIndexes.IDX_DELTA_WDATA_TRIPLE_ID], common_triples_ids)

    edge_indexes_added_wdata_intersect = added_edges_wdata[:, mask_wdata]
    edge_attrs_added_wdata_intersect = added_edge_attrs_wdata[mask_wdata, :]

    # FOR REMOVED EDGES
    unique_triple_ids_removed_edges_wdata = removed_edge_attrs_wdata[:, AttrIndexes.IDX_DELTA_WDATA_TRIPLE_ID].unique()

    unique_triple_ids_edges_wpedia_to = wpedia_timestamp_to.edge_attr[:,
    AttrIndexes.IDX_DELTA_WPEDIA_TRIPLE_ID].unique()

    # common_triples_removed = torch.intersect1d(unique_triple_ids_removed_edges_wdata, unique_triple_ids_edges_wpedia_to)
    common_triples_removed = torch.masked_select(unique_triple_ids_removed_edges_wdata,
                                                 torch.isin(unique_triple_ids_removed_edges_wdata,
                                                            unique_triple_ids_edges_wpedia_to))

    mask_wdata_removed = torch.isin(removed_edge_attrs_wdata[:, AttrIndexes.IDX_DELTA_WDATA_TRIPLE_ID],
                                    common_triples_removed)

    edge_indexes_removed_wdata_intersect = removed_edges_wdata[:, mask_wdata_removed]
    edge_attrs_removed_wdata_intersect = removed_edge_attrs_wdata[mask_wdata_removed, :]

    intersection_edge_indexes = torch.cat(
        [edge_indexes_added_wdata_intersect, edge_indexes_removed_wdata_intersect], dim=1)
    intersection_edge_attrs = torch.cat([edge_attrs_added_wdata_intersect, edge_attrs_removed_wdata_intersect], dim=0)
    logger.info(f'{(time.time() - curr_sub_time) / 60:.4f} '
                f'mins to complete show_intersection_wdata_wpedia 1')

    delta_intersection = Data(
        edge_index=intersection_edge_indexes,
        edge_attr=intersection_edge_attrs
    )
    ##### BEGIN: obtain the head entities with candidate mentions

    masked_tail_present = torch.isin(wpedia_timestamp_to.edge_index[1, :], intersection_edge_indexes)

    candidate_anchor_pages = wpedia_timestamp_to.edge_index[:, masked_tail_present]

    unique_values, counts = torch.unique(candidate_anchor_pages[0, :], return_counts=True)

    # min_nr_target_entities_per_page --> 10
    min_nr_target_entities_per_page = config['min_nr_target_entities_per_page']
    anchor_pages_many_mentions = unique_values[counts > min_nr_target_entities_per_page]

    mask_target_mentions = torch.isin(candidate_anchor_pages[0, :], anchor_pages_many_mentions)

    anchor_pages_w_target_mentions = candidate_anchor_pages[:, mask_target_mentions]

    head_to_target_entities_in_intersection = {}

    for row in anchor_pages_w_target_mentions.T:
        head_id = row[0].item()  # Get the value from the first column
        tail_id = row[1].item()  # Get the value from the second column

        # Add the value to the set associated with the key
        if head_id not in head_to_target_entities_in_intersection:
            head_to_target_entities_in_intersection[head_id] = set()  # Initialize a new set if the key is not present
        head_to_target_entities_in_intersection[head_id].add(tail_id)  # Add the value to the set

    # BEGIN: only during debugging for statistic purposes only
    # candidate_head_to_triples = dict()
    # list_of_tuples = [tuple(row.numpy()) for row in delta_intersection.edge_index.T]
    # for idx_row, row in enumerate(list_of_tuples):
    #     if idx_row % 50 == 0:
    #         logger.info(f'idx_row: {idx_row}, '
    #                     f'len of candidate_head_to_triples: {len(candidate_head_to_triples)} '
    #                     f'len of head_to_target_entities_in_intersection: {len(head_to_target_entities_in_intersection)}')
    #     for idx_head, (head, entities) in enumerate(head_to_target_entities_in_intersection.items()):
    #         to_check_in = entities | {head}
    #         triple_as_set = set(row)
    #         triple_as_tuple = row
    #         intersect = to_check_in & triple_as_set
    #         if len(intersect) == 2:
    #             if head not in candidate_head_to_triples:
    #                 candidate_head_to_triples[head] = set()
    #             candidate_head_to_triples[head].add(triple_as_tuple)
    # logger.info(f'len of candidate_head_to_triples: {len(candidate_head_to_triples)} '
    #             f'len of head_to_target_entities_in_intersection: {len(head_to_target_entities_in_intersection)}')
    # END: only during debugging for statistic purposes only
    ##### END: obtain the head entities with candidate mentions

    sorted_entities, sorted_changes, sorted_normalized_changes, entities_to_changes = (
        obtain_changes_per_entity(
            delta_added_and_removed=delta_intersection,
            config=config,
            wdata_timestamp_to=wdata_timestamp_to,
            only_changes_in_head=config['volatile_only_changes_in_head']
        ))

    save_most_volatile_entities(delta_added_and_removed=delta_intersection,
                                config=config,
                                output_file_path_volatile=output_file_path_volatile,
                                index_to_entity=index_to_entity,
                                index_to_relation=index_to_relation,
                                wikidata_qid_to_label=wikidata_qid_to_label,
                                property_qid_to_label=property_qid_to_label,
                                kb_type='wikidata',
                                timestamp_from=timestamp_from,
                                timestamp_to=timestamp_to,
                                # wdata_timestamp_to=wdata_timestamp_to,
                                sorted_entities=sorted_entities,
                                sorted_changes=sorted_changes,
                                sorted_normalized_changes=sorted_normalized_changes
                                )

    if config['extract_disconnected_subgraphs']:
        logger.info(f'BEGIN obtaining subgraphs for intersection for graph of size '
                    f'{delta_intersection.edge_index.shape}')
        disconnected_subgraphs_intersection = (get_disconnected_subgraphs(delta_intersection))
        logger.info('END obtaining subgraphs for intersection')
        #
        subgraph_idx = 0
        with open(output_file_path_subgraphs, 'wt') as outfile:
            writer = csv.writer(outfile, delimiter='\t')

            for curr_disconnected_subgraph in disconnected_subgraphs_intersection:
                # outfile.write(f'=====DISCONNECTED SUBGRAPH NR {subgraph_idx}======\n')
                append_graph_to_file(
                    csv_writer=writer,
                    subgraph_idx=subgraph_idx,
                    wikidata_qid_to_label=wikidata_qid_to_label,
                    property_qid_to_label=property_qid_to_label,
                    delta_graph=curr_disconnected_subgraph,
                    kb_type='wikidata',
                    index_to_entity=index_to_entity,
                    index_to_relation=index_to_relation,
                    timestamp_from=timestamp_from,
                    timestamp_to=timestamp_to,
                    sorted_entities=sorted_entities,
                    sorted_changes=sorted_changes,
                    config=config,
                    entities_to_changes=entities_to_changes,
                    sorted_normalized_changes=sorted_normalized_changes
                )
                subgraph_idx += 1
    else:
        logger.debug(f'\'extract_disconnected_subgraphs\' '
                     f'in false so NOT obtaining subgraphs'
                     f' for intersection for graph of size '
                     f'{delta_intersection.edge_index.shape}')

    with open(output_file_path, 'wt') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        append_graph_to_file(
            csv_writer=writer,
            subgraph_idx=-1,
            wikidata_qid_to_label=wikidata_qid_to_label,
            property_qid_to_label=property_qid_to_label,
            delta_graph=delta_intersection,
            kb_type='wikidata',
            # outfile=outfile,
            index_to_relation=index_to_relation,
            index_to_entity=index_to_entity,
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            sorted_entities=sorted_entities,
            sorted_changes=sorted_changes,
            config=config,
            entities_to_changes=entities_to_changes,
            sorted_normalized_changes=sorted_normalized_changes,
            delta_intersection=delta_intersection
        )

    torch.save(delta_intersection, f'{output_file_path}.pt')

    with open(output_file_path_target_entities, 'wt') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        append_graph_target_entities_to_file(
            csv_writer=writer,
            head_to_target_entities_in_intersection=head_to_target_entities_in_intersection,
            wikidata_qid_to_label=wikidata_qid_to_label,
            property_qid_to_label=property_qid_to_label,
            kb_type='wikidata',
            index_to_relation=index_to_relation,
            index_to_entity=index_to_entity,
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            config=config
        )

    curr_time = time.time()
    logger.info(f'{(curr_time - start_time) / 60:.4f} to END show_intersection_wdata_wpedia')

    return delta_intersection


def get_emerging_entities_graph(
        timestamp_from: int,
        timestamp_to: int,
        config,
        delta_added_and_removed: Data,
        kb_type: str
):
    edge_attrs = delta_added_and_removed.edge_attr
    if kb_type == 'wikipedia':
        # mask_emerging_entities_head = (
        #         edge_attrs[:, AttrIndexes.IDX_WPEDIA_TIMESTAMP_START] >= timestamp_from)
        mask_emerging_entities_head = edge_attrs[:, AttrIndexes.IDX_DELTA_WPEDIA_HEAD_CREATION_DATE] >= timestamp_from
        mask_emerging_entities_tail = edge_attrs[:, AttrIndexes.IDX_DELTA_WPEDIA_TAIL_CREATION_DATE] >= timestamp_from
    elif kb_type == 'wikidata':
        mask_emerging_entities_head = edge_attrs[:, AttrIndexes.IDX_DELTA_WDATA_HEAD_CREATION_DATE] >= timestamp_from
        mask_emerging_entities_tail = edge_attrs[:, AttrIndexes.IDX_DELTA_WDATA_TAIL_CREATION_DATE] >= timestamp_from
    else:
        raise RuntimeError('kb_type in get_emerging_entities_graph not recognized: '
                           f'{kb_type}')
    mask_emerging = (mask_emerging_entities_head | mask_emerging_entities_tail)
    emerging_edge_attrs = delta_added_and_removed.edge_attr[mask_emerging, :]
    emerging_edge_index = delta_added_and_removed.edge_index[:, mask_emerging]
    delta_emerging: Data = Data(edge_attr=emerging_edge_attrs, edge_index=emerging_edge_index)
    return delta_emerging


def show_and_save_deltas(
        timestamp_from: int,
        timestamp_to: int,
        delta_added_and_removed_wdata: Data,
        delta_added_and_removed_wpedia: Data,
        pyg_graph_wdata: Data,
        pyg_graph_wpedia: Data,
        config,
        index_to_entity: Dict,
        index_to_relation: Dict,
        property_qid_to_label: Dict,
        wpedia_timestamp_to: Data,
        wpedia_timestamp_to_or_from: Data,
        wdata_timestamp_to_or_from: Data,
        delta_interval: str,
        relation_stats: List[Dict] = None
):
    logger.info('BEGIN show_and_save_deltas')
    start_time = time.time()

    # Convert timestamp to datetime object
    date_from = datetime.fromtimestamp(timestamp_from)
    timestamp_from_str = date_from.strftime('%Y%m%d')

    date_to = datetime.fromtimestamp(timestamp_to)
    timestamp_to_str = date_to.strftime('%Y%m%d')

    ##### here saves the relation stats (number of triples per each relation type).
    if relation_stats is not None:
        output_file_path = os.path.join(config['output_dir_data'],
                                        delta_interval,
                                        timestamp_from_str,
                                        f'{timestamp_from_str}_{timestamp_to_str}_relations_stats.csv')
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, mode='wt', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter='\t')
            for curr_entry in relation_stats:
                writer.writerow([curr_entry['relation_qid'],
                                 curr_entry['relation_label'],
                                 curr_entry['count']])
    #####
    output_file_path = os.path.join(config['output_dir_data'],
                                    delta_interval,
                                    timestamp_from_str,
                                    f'{timestamp_from_str}_{timestamp_to_str}_delta_intersection.txt')

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    output_file_path_subgraphs = os.path.join(
        config['output_dir_data'], delta_interval, timestamp_from_str,
        f'{timestamp_from_str}_{timestamp_to_str}_delta_intersection_subgraphs.txt'
    )
    output_file_path_volatile = os.path.join(
        config['output_dir_data'], delta_interval, timestamp_from_str,
        f'{timestamp_from_str}_{timestamp_to_str}_delta_intersection_volatile.txt'
    )

    output_file_path_target_entities = os.path.join(config['output_dir_data'],
                                                    delta_interval, timestamp_from_str,
                                                    f'{timestamp_from_str}_{timestamp_to_str}_target_entities_delta_intersection.txt')

    delta_intersection: Data = show_intersection_wdata_wpedia(
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        config=config,
        output_file_path=output_file_path,
        output_file_path_target_entities=output_file_path_target_entities,
        output_file_path_subgraphs=output_file_path_subgraphs,
        delta_added_and_removed_wpedia=delta_added_and_removed_wpedia,
        delta_added_and_removed_wdata=delta_added_and_removed_wdata,
        wpedia_timestamp_to=wpedia_timestamp_to,
        index_to_entity=index_to_entity,
        index_to_relation=index_to_relation,
        property_qid_to_label=property_qid_to_label,
        complete_graph=pyg_graph_wdata,
        output_file_path_volatile=output_file_path_volatile,
        wdata_timestamp_to=wdata_timestamp_to_or_from
    )
    output_file_path = os.path.join(config['output_dir_data'],
                                    delta_interval, timestamp_from_str,
                                    f'{timestamp_from_str}_{timestamp_to_str}_delta_wdata.txt')
    output_file_path_subgraphs = os.path.join(
        config['output_dir_data'],
        delta_interval, timestamp_from_str,
        f'{timestamp_from_str}_{timestamp_to_str}_delta_wdata_subgraphs.txt'
    )
    output_file_path_volatile = os.path.join(
        config['output_dir_data'],
        delta_interval, timestamp_from_str,
        f'{timestamp_from_str}_{timestamp_to_str}_delta_wdata_volatile.txt'
    )
    show_and_save_delta(
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        config=config,
        output_file_path=output_file_path,
        output_file_path_subgraphs=output_file_path_subgraphs,
        output_file_path_volatile=output_file_path_volatile,
        delta_added_and_removed=delta_added_and_removed_wdata,
        complete_graph=pyg_graph_wdata,
        kb_type='wikidata',
        data_timestamp_to_or_from=wdata_timestamp_to_or_from,
        delta_intersection=delta_intersection
    )

    output_file_path = os.path.join(config['output_dir_data'],
                                    delta_interval, timestamp_from_str,
                                    f'{timestamp_from_str}_{timestamp_to_str}_delta_wpedia.txt')
    output_file_path_subgraphs = os.path.join(
        config['output_dir_data'],
        delta_interval, timestamp_from_str,
        f'{timestamp_from_str}_{timestamp_to_str}_delta_wpedia_subgraphs.txt'
    )
    output_file_path_volatile = os.path.join(
        config['output_dir_data'],
        delta_interval, timestamp_from_str,
        f'{timestamp_from_str}_{timestamp_to_str}_delta_wpedia_volatile.txt'
    )
    show_and_save_delta(
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        config=config,
        output_file_path=output_file_path,
        output_file_path_subgraphs=output_file_path_subgraphs,
        delta_added_and_removed=delta_added_and_removed_wpedia,
        complete_graph=pyg_graph_wpedia,
        kb_type='wikipedia',
        output_file_path_volatile=output_file_path_volatile,
        data_timestamp_to_or_from=wpedia_timestamp_to_or_from,
        delta_intersection=delta_intersection
    )

    # delta with edges involving emerging entities created in the time period between timestamp_from
    # and timestamp_to
    delta_emerging_entities_wpedia: Data = get_emerging_entities_graph(
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        config=config,
        delta_added_and_removed=delta_added_and_removed_wpedia,
        kb_type='wikipedia'
    )
    delta_emerging_entities_wdata: Data = get_emerging_entities_graph(
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        config=config,
        delta_added_and_removed=delta_added_and_removed_wdata,
        kb_type='wikidata'
    )
    output_file_path = os.path.join(config['output_dir_data'],
                                    delta_interval, timestamp_from_str,
                                    f'{timestamp_from_str}_{timestamp_to_str}_delta_emerging_wpedia.txt')

    output_file_path_subgraphs = os.path.join(
        config['output_dir_data'],
        delta_interval, timestamp_from_str,
        f'{timestamp_from_str}_{timestamp_to_str}_delta_emerging_wpedia_subgraphs.txt'
    )
    output_file_path_volatile = os.path.join(
        config['output_dir_data'],
        delta_interval, timestamp_from_str,
        f'{timestamp_from_str}_{timestamp_to_str}_delta_emerging_wpedia_volatile.txt'
    )
    show_and_save_delta(
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        config=config,
        output_file_path=output_file_path,
        output_file_path_subgraphs=output_file_path_subgraphs,
        delta_added_and_removed=delta_emerging_entities_wpedia,
        kb_type='wikipedia',
        complete_graph=pyg_graph_wpedia,
        output_file_path_volatile=output_file_path_volatile,
        data_timestamp_to_or_from=wpedia_timestamp_to_or_from,
        delta_intersection=delta_intersection
    )
    output_file_path = os.path.join(config['output_dir_data'],
                                    delta_interval, timestamp_from_str,
                                    f'{timestamp_from_str}_{timestamp_to_str}_delta_emerging_intersection.txt')
    output_file_path_target_entities = os.path.join(config['output_dir_data'],
                                                    delta_interval, timestamp_from_str,
                                                    f'{timestamp_from_str}_{timestamp_to_str}_target_entities_delta_emerging_intersection.txt')
    output_file_path_subgraphs = os.path.join(
        config['output_dir_data'],
        delta_interval, timestamp_from_str,
        f'{timestamp_from_str}_{timestamp_to_str}_delta_emerging_intersection_subgraphs.txt'
    )
    output_file_path_volatile = os.path.join(
        config['output_dir_data'],
        delta_interval, timestamp_from_str,
        f'{timestamp_from_str}_{timestamp_to_str}_delta_emerging_intersection_volatile.txt'
    )
    output_file_path_target_entities = os.path.join(config['output_dir_data'],
                                                    delta_interval, timestamp_from_str,
                                                    f'{timestamp_from_str}_{timestamp_to_str}_target_entities_delta_emerging_intersection.txt')

    delta_intersection: Data = show_intersection_wdata_wpedia(
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        config=config,
        output_file_path=output_file_path,
        output_file_path_target_entities=output_file_path_target_entities,
        output_file_path_subgraphs=output_file_path_subgraphs,
        delta_added_and_removed_wpedia=delta_emerging_entities_wpedia,
        delta_added_and_removed_wdata=delta_emerging_entities_wdata,
        wpedia_timestamp_to=wpedia_timestamp_to,
        index_to_entity=index_to_entity,
        index_to_relation=index_to_relation,
        property_qid_to_label=property_qid_to_label,
        complete_graph=pyg_graph_wdata,
        output_file_path_volatile=output_file_path_volatile,
        wdata_timestamp_to=wdata_timestamp_to_or_from
    )

    output_file_path = os.path.join(config['output_dir_data'],
                                    delta_interval, timestamp_from_str,
                                    f'{timestamp_from_str}_{timestamp_to_str}_delta_emerging_wdata.txt')

    output_file_path_subgraphs = os.path.join(
        config['output_dir_data'],
        delta_interval, timestamp_from_str,
        f'{timestamp_from_str}_{timestamp_to_str}_delta_emerging_wdata_subgraphs.txt'
    )
    output_file_path_volatile = os.path.join(
        config['output_dir_data'],
        delta_interval, timestamp_from_str,
        f'{timestamp_from_str}_{timestamp_to_str}_delta_emerging_wdata_volatile.txt'
    )

    show_and_save_delta(
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        config=config,
        output_file_path=output_file_path,
        output_file_path_subgraphs=output_file_path_subgraphs,
        delta_added_and_removed=delta_emerging_entities_wdata,
        kb_type='wikidata',
        complete_graph=pyg_graph_wdata,
        output_file_path_volatile=output_file_path_volatile,
        data_timestamp_to_or_from=wdata_timestamp_to_or_from,
        delta_intersection=delta_intersection
    )
    curr_time = time.time()
    logger.info(f'{(curr_time - start_time) / 60:.4f} mins to END show_and_save_deltas')


def interactive_querying_deltas(
        pyg_graph_wdata: Data,
        pyg_graph_wpedia: Data,
        edge_timestamps_wdata: torch.Tensor,
        qualifier_timestamps_wdata: torch.Tensor,
        edge_timestamps_wpedia: torch.Tensor,
        index_to_entity: Dict,
        index_to_relation: Dict,
        wikidata_qid_to_label: Dict,
        property_qid_to_label: Dict,
        config
):
    logger.info(
        'about to start iterative querying, the following are some of the statistics of '
        f'the graphs: \n'
        f'\t - pyg_graph_wdata.edge_index.shape - {pyg_graph_wdata.edge_index.shape} \n'
        f'\t - pyg_graph_wpedia.edge_index.shape - {pyg_graph_wpedia.edge_index.shape} \n'
        f'\t - pyg_graph_wdata.edge_attr.shape - {pyg_graph_wdata.edge_attr.shape} \n'
        f'\t - pyg_graph_wpedia.edge_attr.shape - {pyg_graph_wpedia.edge_attr.shape}'
    )
    delta_intervals_granularities = config['delta_intervals_granularities']
    # delta_intervals_start: List[int] = config['delta_intervals_start']

    nr_deltas_processed = 0
    for curr_delta_interval_type in delta_intervals_granularities:
        logger.info(f'=======================================================')
        logger.info(f'curr_delta_interval_type: {curr_delta_interval_type}')
        nr_delta_intervals = curr_delta_interval_type['nr_delta_intervals']
        curr_delta_granularity = curr_delta_interval_type['granularity']
        # if delta_interval_end == -1:
        #     delta_interval_end = datetime.now().timestamp()
        # Initialize the current date to the starting date
        start_time_deltas_time = time.time()
        # Iterate until the current date exceeds the target date
        # while curr_timestamp_to <= delta_interval_end:
        # for curr_interval_idx in range(1, nr_delta_intervals + 1):
        for curr_delta_interval_start_str in config['delta_intervals_start']:
            # Convert to datetime object with time set to 00:00:00
            date_object = datetime.strptime(curr_delta_interval_start_str, "%Y-%m-%d")

            # Get the timestamp
            curr_delta_interval_start = date_object.timestamp()

            curr_timestamp_from = curr_delta_interval_start
            curr_timestamp_to = curr_delta_interval_start
            for curr_interval_idx in range(nr_delta_intervals):
                logger.info(f'=+===+===+====+====+====+====+====+====+====+====+====+')
                if nr_deltas_processed > 0:
                    logger.info(f'nr deltas processed: {nr_deltas_processed}')
                    logger.info(f'avg_deltas_per_minute for type {curr_delta_granularity}: '
                                f'{(nr_deltas_processed / ((time.time() - start_time_deltas_time) / 60)):.4f} '
                                f'per minute')
                nr_deltas_processed += 1
                # curr_timestamp_from = curr_timestamp_to
                readable_timestmap_from = datetime.fromtimestamp(curr_timestamp_from)
                readable_timestamp_to = datetime.fromtimestamp(curr_timestamp_to)
                # Convert the timestamp to a datetime object

                # Print the readable date
                if curr_delta_granularity == 'weekly':
                    readable_timestamp_to += relativedelta(weeks=1)  # Add 1 weeks
                elif curr_delta_granularity == 'monthly':
                    readable_timestamp_to += relativedelta(months=1)
                elif curr_delta_granularity == 'yearly':
                    readable_timestamp_to += relativedelta(years=1)
                else:
                    raise RuntimeError(f'curr_delta_granularity not recognized: '
                                       f'{curr_delta_granularity}')
                curr_timestamp_to = int(readable_timestamp_to.timestamp())

                logger.info(f'curr_timestamp_to is: '
                            f'{readable_timestamp_to.strftime("%Y-%m-%d %H:%M:%S")}')
                logger.info(f'curr_timestamp_from is: '
                            f'{readable_timestmap_from.strftime("%Y-%m-%d %H:%M:%S")}')

                try:
                    if config['fixed_timestamp_from'] > -1:
                        curr_timestamp_from = config['fixed_timestamp_from']
                    if config['fixed_timestamp_to'] > -1:
                        curr_timestamp_to = config['fixed_timestamp_to']
                    # locally try with: 1562771677 1722771677
                    # in production try with (2023-08-20 to 2023-09-01): 1692482400 1693519200
                    # in production try with (2023-08-20 to 2023-08-23): 1692482400 1692741600

                    delta_added_and_removed_wdata: Data
                    delta_removed_only_wdata: Data
                    delta_added_only_wdata: Data

                    relation_stats: List[Dict]

                    (delta_added_and_removed_wdata,
                     # delta_removed_only_wdata,
                     # delta_added_only_wdata,
                     _,
                     wdata_timestamp_to_or_from,
                     relation_stats) = (
                        get_deltas_data(pyg_graph=pyg_graph_wdata,
                                        kb_type='wikidata',
                                        edge_timestamps=edge_timestamps_wdata,
                                        qualifier_timestamps_wdata=qualifier_timestamps_wdata,
                                        timestamp_from=curr_timestamp_from,
                                        timestamp_to=curr_timestamp_to,
                                        index_to_entity=index_to_entity,
                                        index_to_relation=index_to_relation,
                                        wikidata_qid_to_label=wikidata_qid_to_label,
                                        property_qid_to_label=property_qid_to_label,
                                        return_timestamp_to=False,
                                        return_timestamp_to_or_from=True))

                    logger.debug(f'delta_added_and_removed_wdata.edge_index.shape: '
                                 f'{delta_added_and_removed_wdata.edge_index.shape}')
                    logger.debug(f'delta_added_and_removed_wdata.edge_attr.shape: '
                                 f'{delta_added_and_removed_wdata.edge_attr.shape}')

                    delta_added_and_removed_wpedia: Data
                    delta_removed_only_wpedia: Data
                    delta_added_only_wpedia: Data

                    (delta_added_and_removed_wpedia,
                     # delta_removed_only_wpedia,
                     # delta_added_only_wpedia,
                     wpedia_timestamp_to, wpedia_timestamp_to_or_from, _) = (
                        get_deltas_data(pyg_graph=pyg_graph_wpedia,
                                        kb_type='wikipedia',
                                        edge_timestamps=edge_timestamps_wpedia,
                                        qualifier_timestamps_wdata=None,
                                        timestamp_from=curr_timestamp_from,
                                        timestamp_to=curr_timestamp_to,
                                        index_to_entity=index_to_entity,
                                        index_to_relation=index_to_relation,
                                        wikidata_qid_to_label=wikidata_qid_to_label,
                                        property_qid_to_label=property_qid_to_label,
                                        return_timestamp_to=True,
                                        return_timestamp_to_or_from=True))

                    logger.debug(f'delta_added_and_removed_wpedia.edge_index.shape: '
                                 f'{delta_added_and_removed_wpedia.edge_index.shape}')
                    logger.debug(f'delta_added_and_removed_wpedia.edge_attr.shape: '
                                 f'{delta_added_and_removed_wpedia.edge_attr.shape}')

                    # logger.info('BEGIN obtaining subgraphs from the wikidata delta')
                    # disconnected_subgraphs_wdata = (
                    #     get_disconnected_subgraphs(delta_added_and_removed_wdata))
                    # logger.info('END obtaining subgraphs from the wikipedia delta')
                    #
                    # logger.info('BEGIN obtaining subgraphs from the wikipedia delta')
                    # disconnected_subgraphs_wpedia = (
                    #     get_disconnected_subgraphs(delta_added_and_removed_wpedia))
                    # logger.info('END obtaining subgraphs from the wikipedia delta')

                    show_and_save_deltas(
                        timestamp_from=curr_timestamp_from,
                        timestamp_to=curr_timestamp_to,
                        delta_added_and_removed_wdata=delta_added_and_removed_wdata,
                        delta_added_and_removed_wpedia=delta_added_and_removed_wpedia,
                        pyg_graph_wdata=pyg_graph_wdata,
                        pyg_graph_wpedia=pyg_graph_wpedia,
                        config=config,
                        index_to_entity=index_to_entity,
                        index_to_relation=index_to_relation,
                        property_qid_to_label=property_qid_to_label,
                        wpedia_timestamp_to=wpedia_timestamp_to,
                        wdata_timestamp_to_or_from=wdata_timestamp_to_or_from,
                        wpedia_timestamp_to_or_from=wpedia_timestamp_to_or_from,
                        delta_interval=curr_delta_granularity,
                        relation_stats=relation_stats
                        # delta_interval_start=curr_delta_interval_start
                    )
                except Exception as e:
                    logger.error(f'Following error during input: {e} ')
                    logger.error(f'Stack trace: ')
                    traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_file',
        required=False,
        type=str,
        default='experiments/s03_get_deltas/20241009/s03_config_get_deltas_local.json',
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
    output_data = {
        'commit_hash': commit_hash,
        'parameters_args': vars(args),
        'parameters_config': config
    }
    with open(os.path.join(output_dir_data, 'config.json'), 'wt') as json_file:
        json.dump(output_data, json_file, indent=4)

    path_wikidata_labels = config['path_wikidata_labels']
    path_property_labels = config['path_property_labels']

    wpedia_entity_creation_date_path = config['wpedia_entity_creation_date_path']
    wdata_entity_creation_date_path = config['wdata_entity_creation_date_path']

    wpedia_wdata_path_hash = generate_short_hash(f'{wpedia_entity_creation_date_path}_'
                                                 f'{wdata_entity_creation_date_path}',
                                                 hash_length=8)

    # wikidata_qid_to_label = dict()
    caches_dir = config['caches_dir']
    os.makedirs(caches_dir, exist_ok=True)
    caches_wikidata_qid_to_label_path = os.path.join(caches_dir, 'wikidata_qid_to_label.pickle')
    logger.info('BEGIN invoking load_wikidata_qid_to_label')
    wikidata_qid_to_label: Dict = load_wikidata_qid_to_label(
        path_wikidata_labels, caches_wikidata_qid_to_label_path
    )
    logger.info('END invoking load_wikidata_qid_to_label')

    caches_property_qid_to_label_path = os.path.join(caches_dir, 'property_qid_to_label.pickle')
    logger.info('BEGIN invoking load_property_qid_to_label')
    property_qid_to_label: Dict = load_property_qid_to_label(
        path_property_labels, caches_property_qid_to_label_path
    )
    logger.info('END invoking load_property_qid_to_label')

    debug_nr_triples = args.debug_nr_triples

    input_triples_path_wdata = config['path_extracted_history_triples_wdata']
    input_triples_path_wpedia = config['path_extracted_history_triples_wpedia']

    files_wdata = os.listdir(input_triples_path_wdata)
    logger.info(f'Current input files wikidata: {files_wdata}')
    #
    files_wpedia = os.listdir(input_triples_path_wpedia)
    logger.info(f'Current input files wikipedia: {files_wpedia}')

    # Get the virtual memory object
    memory_info = psutil.virtual_memory()

    # Calculate free memory in gigabytes
    free_memory_gb = memory_info.available / (1024 ** 3)  # Convert bytes to gigabytes
    os.makedirs(caches_dir, exist_ok=True)
    #
    # obtains hashes of path_extracted_history_triples_wpedia and path_extracted_history_triples_wdata
    if debug_nr_triples == -1:
        cache_path = os.path.join(caches_dir, f'loaded_graph_all_v3_{config["heads_in_wikipedia"]}_'
                                              f'{config["tails_in_wikipedia"]}_'
                                              f'{config["qids_in_both_wdata_and_wpedia_parsed_entities"]}_'
                                              f'{wpedia_wdata_path_hash}.pt')
    else:
        cache_path = os.path.join(caches_dir, f'loaded_graph_v3_{debug_nr_triples}_'
                                              f'{config["heads_in_wikipedia"]}_'
                                              f'{config["tails_in_wikipedia"]}_'
                                              f'{config["qids_in_both_wdata_and_wpedia_parsed_entities"]}_'
                                              f'{wpedia_wdata_path_hash}.pt')

    path_wikipedia_wikidata_map = config['path_wikipedia_wikidata_map']

    path_cache_wikipedia_page_id_to_wikidata_qid = os.path.join(caches_dir,
                                                                'wikipedia_page_id_to_wikidata_qid.pickle')

    path_cache_wdata_and_wpedia_parsed_mapping = os.path.join(caches_dir,
                                                              'wpedia_wdata_parsed_mapping.pickle')

    if config['qids_in_both_wdata_and_wpedia_parsed_entities']:
        wikipedia_page_id_to_wikidata_qid = load_wiki_page_id_to_wdata_qid_only_parsed(
            path_cache_wdata_and_wpedia_parsed_mapping,
            wpedia_entity_creation_date_path,
            wdata_entity_creation_date_path
        )
    else:
        logger.info('BEGIN invoking load_wiki_page_id_to_wikidata_qid')
        wikipedia_page_id_to_wikidata_qid = load_wiki_page_id_to_wikidata_qid(
            path_cache_wikipedia_page_id_to_wikidata_qid,
            path_wikipedia_wikidata_map)
        logger.info('END invoking load_wiki_page_id_to_wikidata_qid')

    logger.info('loading the dictionary wikidata_qid_to_wikipedia_page_id')
    wikidata_qid_to_wikipedia_page_id = {value: key for key, value in wikipedia_page_id_to_wikidata_qid.items()}
    logger.info('loaded the dictionary wikidata_qid_to_wikipedia_page_id')

    index_to_entity = dict()
    index_to_relation = dict()
    if not os.path.exists(cache_path):
        logger.info(f'NOT IN CACHE {cache_path}, loading graph')
        entity_to_index: Dict = dict()
        relation_to_index: Dict = dict()
        qualifier_to_details: Dict = dict()

        # pyg_graph_wpedia = None
        # all_timestamps_wpedia = None

        (pyg_graph_wdata, edge_timestamps_wdata, entity_to_index,
         index_to_entity, relation_to_index, triple_to_index,
         qualifier_timestamps_wdata) = \
            load_into_pyg_dynamically_data(input_triples_path=input_triples_path_wdata,
                                           debug_nr_triples=debug_nr_triples,
                                           entity_to_index=entity_to_index,
                                           relation_to_index=relation_to_index,
                                           # qualifier_to_details=qualifier_to_details,
                                           qualifier_to_details=dict(),
                                           wikidata_qid_to_wikipedia_page_id=wikidata_qid_to_wikipedia_page_id,
                                           config=config,
                                           # args,
                                           kb_type='wikidata',
                                           property_qid_to_label=property_qid_to_label,
                                           precision=config['precision_wdata'])

        logger.info('extend_with_creation_dates for wikidata')
        pyg_graph_wdata, entity_creation_date_tensor_wdata = \
            extend_with_creation_dates(
                pyg_graph=pyg_graph_wdata,
                # all_timestamps=all_timestamps_wdata,
                entity_to_creation_date_path=wdata_entity_creation_date_path,
                # entity_to_creation_date_path=entity_to_creation_date_path,
                entity_to_index=entity_to_index,
                index_to_entity=index_to_entity,
                # relation_to_index=relation_to_index,
                precision=config['precision_wdata'],
                wikidata_qid_to_label=wikidata_qid_to_label
            )
        #
        (pyg_graph_wpedia, edge_timestamps_wpedia, entity_to_index,
         index_to_entity,
         relation_to_index, _, _) = \
            load_into_pyg_dynamically_data(input_triples_path=input_triples_path_wpedia,
                                           debug_nr_triples=debug_nr_triples,
                                           entity_to_index=entity_to_index,
                                           relation_to_index=relation_to_index,
                                           qualifier_to_details=qualifier_to_details,
                                           wikidata_qid_to_wikipedia_page_id=wikidata_qid_to_wikipedia_page_id,
                                           config=config,
                                           # args,
                                           kb_type='wikipedia',
                                           precision=config['precision_wpedia'],
                                           property_qid_to_label=property_qid_to_label,
                                           triple_to_index=triple_to_index)

        logger.info('extend_with_creation_dates for wikipedia')
        # extend with creation date
        pyg_graph_wpedia, entity_creation_date_tensor_wpedia = \
            extend_with_creation_dates(
                pyg_graph=pyg_graph_wpedia,
                # all_timestamps=all_timestamps_wdata,
                entity_to_creation_date_path=wpedia_entity_creation_date_path,
                # entity_to_creation_date_path=entity_to_creation_date_path,
                entity_to_index=entity_to_index,
                index_to_entity=index_to_entity,
                # relation_to_index=relation_to_index,
                precision=config['precision_wpedia'],
                wikidata_qid_to_label=wikidata_qid_to_label
            )

        logger.info(f'inverting entity_to_index and relation_to_index')
        # index_to_entity = {value: key for key, value in entity_to_index.items()}
        index_to_relation = {value: key for key, value in relation_to_index.items()}
        logger.info(f'graph loaded, pickling to {cache_path}')
        torch.save({
            'pyg_graph_wpedia': pyg_graph_wpedia,
            'pyg_graph_wdata': pyg_graph_wdata,
            'tensor_timestamps_wdata': edge_timestamps_wdata,
            'tensor_qualifier_timestamps_wdata': qualifier_timestamps_wdata,
            'tensor_timestamps_wpedia': edge_timestamps_wpedia,
            # 'entity_creation_date_tensor_wpedia': entity_creation_date_tensor_wpedia,
            # 'entity_creation_date_tensor_wdata': entity_creation_date_tensor_wdata,
            'index_to_entity': index_to_entity,
            'index_to_relation': index_to_relation
        }, cache_path)
        logger.info(f'graph saved to pickle')
    else:
        logger.info(f'cache exists, loading from {cache_path}')
        loaded_data = torch.load(cache_path, weights_only=False)
        pyg_graph_wdata = loaded_data['pyg_graph_wdata']
        pyg_graph_wpedia = loaded_data['pyg_graph_wpedia']
        edge_timestamps_wdata = loaded_data['tensor_timestamps_wdata']
        edge_timestamps_wpedia = loaded_data['tensor_timestamps_wpedia']
        qualifier_timestamps_wdata = loaded_data['tensor_qualifier_timestamps_wdata']
        index_to_entity = loaded_data['index_to_entity']
        index_to_relation = loaded_data['index_to_relation']
        # entity_creation_date_tensor_wdata = loaded_data['entity_creation_date_tensor_wdata']
        # entity_creation_date_tensor_wpedia = loaded_data['entity_creation_date_tensor_wdata']
        logger.info(f'GRAPH LOADED from cache: {cache_path}')
        # logger.info(f'loaded pyg_graph: {pyg_graph_wdata}')
        # logger.info(f'loaded batch_timestamps: {all_timestamps_wdata}')
    entity_to_index = {v: k for k, v in index_to_entity.items()}

    idxs_to_save = {
        'entity_to_index': entity_to_index,
        'index_to_entity': index_to_entity,
        'index_to_relation': index_to_relation
    }

    output_idxs_path = os.path.join(output_dir_data, 'idx_entities_rels.pt')
    torch.save(idxs_to_save, output_idxs_path)
    interactive_querying_deltas(
        pyg_graph_wdata=pyg_graph_wdata,
        pyg_graph_wpedia=pyg_graph_wpedia,
        edge_timestamps_wpedia=edge_timestamps_wpedia,
        edge_timestamps_wdata=edge_timestamps_wdata,
        qualifier_timestamps_wdata=qualifier_timestamps_wdata,
        index_to_entity=index_to_entity,
        index_to_relation=index_to_relation,
        wikidata_qid_to_label=wikidata_qid_to_label,
        property_qid_to_label=property_qid_to_label,
        config=config
    )

    logger.info('obtained deltas of both wikipedia and wikidata')
