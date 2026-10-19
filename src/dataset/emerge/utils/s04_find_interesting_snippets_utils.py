import logging
import time
from datetime import datetime
from typing import Dict

import torch
from torch.nn.utils.rnn import pack_sequence
from torch_geometric.data import Data

import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)


def batch_tensor_method2(batched_entity_ids, device):
    ##### BEGIN METHOD 2 OF LIST OF LISTS TO TORCH TENSOR
    # import torch
    # from torch.nn.utils.rnn import pack_sequence
    #
    # # List of lists with different sizes
    # ragged_list = [[1, 2, 3], [4, 5], [6]]

    # Convert each sublist to a tensor
    tensors = [torch.tensor(sublist) for sublist in batched_entity_ids]

    # Pack the sequence
    packed_sequence = pack_sequence(tensors, enforce_sorted=False).to(device=device)

    return packed_sequence


##### END METHOD 2 OF LIST OF LISTS TO TORCH TENSOR

def batch_tensor_method1(curr_batch_max_length, batched_entity_ids, device):
    # batched_entity_ids = list()
    # batched_entity_ids_len = list()
    # curr_batch_max_entity_len = 0

    ##### BEGIN METHOD 1 OF LIST OF LISTS TO TORCH TENSOR
    # import torch
    #
    # # List of lists with different sizes
    # ragged_list = [[1, 2, 3], [4, 5], [6]]
    #
    # # Find the maximum length of sublists
    # max_length = max(len(sublist) for sublist in ragged_list)
    #
    # # Pad the sublists with zeros (or another value)
    padded_list = [sublist + [0] * (curr_batch_max_length - len(sublist)) for
                   sublist in batched_entity_ids]

    # Convert to a torch tensor
    tensor = torch.tensor(padded_list).to(device)
    return tensor


#
# print(tensor)
##### END METHOD 1 OF LIST OF LISTS TO TORCH TENSOR


def timestamp_to_date(timestamp: int, return_time=False):
    # Convert timestamp to datetime object
    dt_object = datetime.fromtimestamp(timestamp)

    # Format the datetime object to a string
    if return_time:
        formatted_date = dt_object.strftime('%Y-%m-%d - %H:%M')
    else:
        formatted_date = dt_object.strftime('%Y-%m-%d')

    # print(formatted_date)  # Output: 2021-10-01
    return formatted_date


def connect_interesting_snippets_with_kg(property_qid_to_label,
                                         interval_from,
                                         interval_to,
                                         str_device: str,
                                         str_device2: str,
                                         tensor_timestamps_wdata,
                                         wdata_graph: Data,
                                         tn_entity_ids: torch.Tensor,
                                         len_entity_ids,
                                         index_to_entity,
                                         index_to_relation,
                                         matched_triples,
                                         wikidata_qid_to_label,
                                         json_input: Dict,
                                         tensor_qualifier_timestamps_wdata: torch.Tensor
                                         ):
    # logger.info('inside connect_with_whole_kg')
    time_begin_complex_if = time.time()
    existing_triples = list()
    # mask_timestamps_wdata_at_from = (tensor_timestamps_wdata[:, 1] < interval_from) & (
    #         tensor_timestamps_wdata[:, 2] > interval_to)
    mask_timestamps_wdata_at_from = \
        (tensor_timestamps_wdata[:, 1] < interval_from) & (
                (tensor_timestamps_wdata[:, 2] > interval_to) | (tensor_timestamps_wdata[:, 2] == 0))  # .to(str_device)
    logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                 f'time it took for prev_2')

    idx_graph_wdata_at_from = tensor_timestamps_wdata[
        mask_timestamps_wdata_at_from, 0]  # .to(str_device)

    mask_qualifier_timestamps_remove_action = (
            (tensor_qualifier_timestamps_wdata[:, 2] < interval_from) &
            (tensor_qualifier_timestamps_wdata[:, 3] == 0)
    )
    idx_graph_wdata_qualifier_remove_action = \
        tensor_qualifier_timestamps_wdata[mask_qualifier_timestamps_remove_action, 0]

    logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                 f'time it took for prev_3')

    # 3869MiB / 11264MiB
    masked_tensor_timestamps = tensor_timestamps_wdata[mask_timestamps_wdata_at_from, :]
    logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                 f'time it took for prev_4')
    # edges_wdata_at_from = wdata_graph.edge_index[:, idx_graph_wdata_at_from]
    edges_wdata_at_from = wdata_graph.edge_index[:, idx_graph_wdata_at_from].to(str_device2)
    logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                 f'time it took for prev_5')
    attrs_wdata_at_from = wdata_graph.edge_attr[idx_graph_wdata_at_from, :]
    logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                 f'time it took for prev_6')

    edges_wdata_at_from_qualifier_remove_action = wdata_graph.edge_index \
        [:, idx_graph_wdata_qualifier_remove_action].to(str_device2)
    attrs_wdata_at_from_qualifier_remove_action = wdata_graph.edge_attr \
        [idx_graph_wdata_qualifier_remove_action, :].to(str_device2)
    qualifier_details_remove_action = tensor_qualifier_timestamps_wdata[mask_qualifier_timestamps_remove_action, 1:].to(
        str_device2)
    # qualifier_details_remove_action.to(str_device2)


    del idx_graph_wdata_at_from
    del idx_graph_wdata_qualifier_remove_action
    # logger.info(f'shape of tensor_qualifier_timestamps_wdata: '
    #             f'{tensor_qualifier_timestamps_wdata.shape}')
    # 4009MiB / 11264MiB
    logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                 f'time it took for prev_7, tn_entity_ids.shape): '
                 f'{tn_entity_ids.shape} and len_entity_ids: {len_entity_ids}')
    # tn_entity_ids = torch.tensor(entity_ids, dtype=torch.int64, device=str_device2)
    # 6197MiB / 11264MiB
    logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                 f'time it took for prev_8')
    tn_entity_ids = tn_entity_ids[:len_entity_ids]
    logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                 f'time it took for to obtain the first len_entity_ids , prev_9')

    logger.debug(f'DEBUGGING with tn_entity_ids.shape of '
                 f'{tn_entity_ids.shape} and '
                 f'edges_wdata_at_from[0, :].shape '
                 f'of {edges_wdata_at_from[0, :].shape} ')
    # f'free gpu memory: {get_free_gpu_memory(str_device)}')
    time_start = time.time()
    mask_head = torch.isin(edges_wdata_at_from[0, :], tn_entity_ids)
    mask_tail = torch.isin(edges_wdata_at_from[1, :], tn_entity_ids)

    mask_head_qualifier_remove_action = torch.isin(edges_wdata_at_from_qualifier_remove_action[0, :], tn_entity_ids)
    mask_tail_qualifier_remove_action = torch.isin(edges_wdata_at_from_qualifier_remove_action[1, :], tn_entity_ids)

    # TODO: starting from here can be all in batch again, as the only operation that does
    #   need one by one execution is is the isin, particularly edges_wdata_at_from[:, mask_triples].T.tolist()
    #   seems to perform slowly.
    logger.debug(f'sec. {time.time() - time_start} '
                 f'time the operations in isin took a')
    del tn_entity_ids
    mask_triples = mask_head & mask_tail
    logger.debug(f'sec. {time.time() - time_start} '
                 f'time the operations in isin took b')
    #
    logger.debug(f'sec. {time.time() - time_start} '
                 f'time the operations in isin took c')
    mask_triples_qualifier_remove_action = mask_head_qualifier_remove_action & mask_tail_qualifier_remove_action

    time_begin_complex_if_to_list = time.time()
    found_triples_lst = edges_wdata_at_from[:, mask_triples].T.tolist()
    found_triples_lst_qualifier_remove_action = edges_wdata_at_from_qualifier_remove_action[:,
                                                mask_triples_qualifier_remove_action].T.tolist()
    found_attrs_lst_qualifier_remove_action = attrs_wdata_at_from_qualifier_remove_action[
                                              mask_triples_qualifier_remove_action, :].tolist()
    found_qualifier_details_remove_action = qualifier_details_remove_action[mask_triples_qualifier_remove_action,
                                            :].tolist()
    # found_triples_set_qualifier_remove_action = set(tuple(sublist) for sublist in found_triples_lst_qualifier_remove_action)
    found_triples_set_qualifier_remove_action = dict()
    for curr_found_triple, curr_found_attr, curr_qualif_details in \
            zip(found_triples_lst_qualifier_remove_action, found_attrs_lst_qualifier_remove_action,
                found_qualifier_details_remove_action):
        ########
        # assert curr_triple_time[0] == curr_found_attr[0]
        #
        curr_head_id = curr_found_triple[0]
        curr_relation_id = curr_found_attr[1]
        curr_tail_id = curr_found_triple[1]
        triple_ids = (curr_head_id, curr_relation_id, curr_tail_id)
        found_triples_set_qualifier_remove_action[triple_ids] = curr_qualif_details

    logger.debug(f'sec. {time.time() - time_begin_complex_if_to_list} '
                 f'time it took to execute to_list and some del 1 '
                 f'found_triples_lst lengths: {len(found_triples_lst)} '
                 f'dtype: {edges_wdata_at_from.dtype} '
                 f'device edges: {edges_wdata_at_from.device} '
                 f'dtype attributes: {attrs_wdata_at_from.dtype}')
    mask_triples = mask_triples.to(str_device)
    logger.debug(f'sec. {time.time() - time_begin_complex_if_to_list} '
                 f'time it took to execute to_list and some del 2')
    found_attrs_lst = attrs_wdata_at_from[mask_triples, :].tolist()
    logger.debug(f'sec. {time.time() - time_begin_complex_if_to_list} '
                 f'time it took to execute to_list and some del 3')
    temporal_attrs = masked_tensor_timestamps[mask_triples, :].tolist()
    logger.debug(f'sec. {time.time() - time_begin_complex_if_to_list} '
                 f'time it took to execute to_list and some del 4')
    del edges_wdata_at_from
    del attrs_wdata_at_from
    del edges_wdata_at_from_qualifier_remove_action
    del attrs_wdata_at_from_qualifier_remove_action

    found_triples_in_from = set()
    triple_to_attributes = dict()
    time_begin_complex_if_zip = time.time()
    for curr_found_triple, curr_found_attr, curr_triple_time in \
            zip(found_triples_lst, found_attrs_lst, temporal_attrs):
        ########
        # assert curr_triple_time[0] == curr_found_attr[0]
        #
        curr_head_id = curr_found_triple[0]
        curr_relation_id = curr_found_attr[1]
        curr_tail_id = curr_found_triple[1]
        triple_ids = (curr_head_id, curr_relation_id, curr_tail_id)
        qualifier_info = dict()
        if triple_ids in found_triples_set_qualifier_remove_action:
            found_qualifier = found_triples_set_qualifier_remove_action[triple_ids]
            logger.debug('triple_ids in found_triples_set_qualifier_remove_action! '
                         f'{triple_ids} and the details are: '
                         f'{found_qualifier}')
            qualifier_info['qualifier_timestamp'] = found_qualifier[1]
            qualifier_info['qualifier_date'] = timestamp_to_date(qualifier_info['qualifier_timestamp'])
            qualifier_info['qualifier_qid'] = index_to_relation[found_qualifier[0]]
            qualifier_info['qualifier_label'] = property_qid_to_label[qualifier_info['qualifier_qid']]

        found_triples_in_from.add(triple_ids)
        emerging_head = False
        emerging_tail = False
        if interval_from <= curr_found_attr[2]:
            emerging_head = True

        if interval_from <= curr_found_attr[3]:
            emerging_tail = True

        # if interval_from <
        triple_attrs = {
            'emerging_head': emerging_head,
            'emerging_tail': emerging_tail,
            'head_creation_date': f'{timestamp_to_date(curr_found_attr[2])}',
            'tail_creation_date': f'{timestamp_to_date(curr_found_attr[3])}',
            'head_creation_timestamp': int(curr_found_attr[2]),
            'tail_creation_timestamp': int(curr_found_attr[3]),
            'triple_date_from': f'{timestamp_to_date(curr_triple_time[1])}',
            'triple_date_to': f'{timestamp_to_date(curr_triple_time[2])}',
            'triple_timestamp_from': int(curr_triple_time[1]),
            'triple_timestamp_to': int(curr_triple_time[2]),
            'qualifier_info': qualifier_info
        }

        triple_to_attributes[triple_ids] = triple_attrs

    logger.debug(f'sec. {time.time() - time_begin_complex_if_zip} '
                 f'to assign triple_to_attributes')
    #
    time_begin_complex_if_difference = time.time()
    del temporal_attrs
    # logger.info(f'the difference is on {found_triples_in_from} ==AND== {matched_triples}')
    additional_triple_in_from_not_in_matched = found_triples_in_from.difference(matched_triples)
    logger.debug(f'sec. {time.time() - time_begin_complex_if_difference}'
                 f' time for del and difference')
    time_begin_complex_if_print = time.time()
    for curr_additional_triple in additional_triple_in_from_not_in_matched:
        # triple_attrs = triple_to_attributes[triple_ids]
        triple_attrs = triple_to_attributes[curr_additional_triple]
        qid_head = index_to_entity[curr_additional_triple[0]]
        qid_head = f'Q{qid_head}'
        label_head = qid_head
        if qid_head in wikidata_qid_to_label:
            label_head = wikidata_qid_to_label[qid_head]
        qid_relation = index_to_relation[curr_additional_triple[1]]
        label_relation = qid_relation
        #
        if qid_relation in property_qid_to_label:
            label_relation = property_qid_to_label[qid_relation]

        qid_tail = index_to_entity[curr_additional_triple[2]]
        qid_tail = f'Q{qid_tail}'
        label_tail = qid_tail
        if qid_tail in wikidata_qid_to_label:
            label_tail = wikidata_qid_to_label[qid_tail]
        # curr_triple_field = (f'{qid_head}({label_head})',
        #                      f'{qid_relation}({label_relation})',
        #                      f'{qid_tail}({label_tail})')
        if qid_head == qid_tail:
            continue
        curr_triple_field = (f'{qid_head}',
                             f'{qid_relation}',
                             f'{qid_tail}')
        curr_triple_labels_field = (f'{label_head}',
                                    f'{label_relation}',
                                    f'{label_tail}')
        existing_triples.append({
            'qualifier_info': triple_attrs['qualifier_info'],
            'emerging_head': triple_attrs['emerging_head'],
            'emerging_tail': triple_attrs['emerging_tail'],
            'head_creation_date': triple_attrs['head_creation_date'],
            'tail_creation_date': triple_attrs['tail_creation_date'],
            'head_creation_timestamp': triple_attrs['head_creation_timestamp'],
            'tail_creation_timestamp': triple_attrs['tail_creation_timestamp'],
            'triple_lifespan_date': [
                triple_attrs['triple_date_from'],
                None if triple_attrs['triple_timestamp_to'] == 0 else triple_attrs['triple_date_to']
                # triple_attrs['triple_date_to']
                # value_if_true if condition else value_if_false
            ],
            'triple_lifespan_timestamp': [
                triple_attrs['triple_timestamp_from'],
                None if triple_attrs['triple_timestamp_to'] == 0 else triple_attrs['triple_timestamp_to']
            ],
            'triple': curr_triple_field,
            'triple_labels': curr_triple_labels_field
        })
        # if len(triple_attrs['qualifier_info']) > 0:
    # json_input['existing_triples'] = existing_triples
    json_input['existing_knowledge'] = existing_triples

    curr_time_complex_if = time.time() - time_begin_complex_if_print
    logger.debug(f'sec. {curr_time_complex_if} to output'
                 f' the complex if')
    return json_input


def connect_interesting_snippets_with_kg_v3(property_qid_to_label,
                                            interval_from,
                                            interval_to,
                                            str_device: str,
                                            str_device2: str,
                                            tensor_timestamps_wdata,
                                            wdata_graph: Data,
                                            tn_entity_ids: torch.Tensor,
                                            len_entity_ids,
                                            index_to_entity,
                                            index_to_relation,
                                            matched_triples,
                                            wikidata_qid_to_label,
                                            wikidata_qid_to_mentions,
                                            json_input: Dict,
                                            tensor_qualifier_timestamps_wdata: torch.Tensor
                                            ):
    # logger.info('inside connect_with_whole_kg')
    time_begin_complex_if = time.time()
    existing_triples = list()
    # mask_timestamps_wdata_at_from = (tensor_timestamps_wdata[:, 1] < interval_from) & (
    #         tensor_timestamps_wdata[:, 2] > interval_to)
    mask_timestamps_wdata_at_from = \
        (tensor_timestamps_wdata[:, 1] < interval_from) & (
                (tensor_timestamps_wdata[:, 2] > interval_to) | (tensor_timestamps_wdata[:, 2] == 0))  # .to(str_device)
    logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                 f'time it took for prev_2')

    idx_graph_wdata_at_from = tensor_timestamps_wdata[
        mask_timestamps_wdata_at_from, 0]  # .to(str_device)

    mask_qualifier_timestamps_remove_action = (
            (tensor_qualifier_timestamps_wdata[:, 2] < interval_from) &
            (tensor_qualifier_timestamps_wdata[:, 3] == 0)
    )
    idx_graph_wdata_qualifier_remove_action = \
        tensor_qualifier_timestamps_wdata[mask_qualifier_timestamps_remove_action, 0]

    logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                 f'time it took for prev_3')

    # 3869MiB / 11264MiB
    masked_tensor_timestamps = tensor_timestamps_wdata[mask_timestamps_wdata_at_from, :]
    logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                 f'time it took for prev_4')
    # edges_wdata_at_from = wdata_graph.edge_index[:, idx_graph_wdata_at_from]
    edges_wdata_at_from = wdata_graph.edge_index[:, idx_graph_wdata_at_from].to(str_device2)
    logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                 f'time it took for prev_5')
    attrs_wdata_at_from = wdata_graph.edge_attr[idx_graph_wdata_at_from, :]
    logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                 f'time it took for prev_6')

    edges_wdata_at_from_qualifier_remove_action = wdata_graph.edge_index \
        [:, idx_graph_wdata_qualifier_remove_action].to(str_device2)
    attrs_wdata_at_from_qualifier_remove_action = wdata_graph.edge_attr \
        [idx_graph_wdata_qualifier_remove_action, :].to(str_device2)
    qualifier_details_remove_action = tensor_qualifier_timestamps_wdata[mask_qualifier_timestamps_remove_action, 1:].to(
        str_device2)
    # qualifier_details_remove_action.to(str_device2)


    del idx_graph_wdata_at_from
    del idx_graph_wdata_qualifier_remove_action
    # logger.info(f'shape of tensor_qualifier_timestamps_wdata: '
    #             f'{tensor_qualifier_timestamps_wdata.shape}')
    # 4009MiB / 11264MiB
    logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                 f'time it took for prev_7, tn_entity_ids.shape): '
                 f'{tn_entity_ids.shape} and len_entity_ids: {len_entity_ids}')
    # tn_entity_ids = torch.tensor(entity_ids, dtype=torch.int64, device=str_device2)
    # 6197MiB / 11264MiB
    logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                 f'time it took for prev_8')
    tn_entity_ids = tn_entity_ids[:len_entity_ids]
    logger.debug(f'sec. {time.time() - time_begin_complex_if} '
                 f'time it took for to obtain the first len_entity_ids , prev_9')

    logger.debug(f'DEBUGGING with tn_entity_ids.shape of '
                 f'{tn_entity_ids.shape} and '
                 f'edges_wdata_at_from[0, :].shape '
                 f'of {edges_wdata_at_from[0, :].shape} ')
    # f'free gpu memory: {get_free_gpu_memory(str_device)}')
    time_start = time.time()
    mask_head = torch.isin(edges_wdata_at_from[0, :], tn_entity_ids)
    mask_tail = torch.isin(edges_wdata_at_from[1, :], tn_entity_ids)

    mask_head_qualifier_remove_action = torch.isin(edges_wdata_at_from_qualifier_remove_action[0, :], tn_entity_ids)
    mask_tail_qualifier_remove_action = torch.isin(edges_wdata_at_from_qualifier_remove_action[1, :], tn_entity_ids)

    # TODO: starting from here can be all in batch again, as the only operation that does
    #   need one by one execution is is the isin, particularly edges_wdata_at_from[:, mask_triples].T.tolist()
    #   seems to perform slowly.
    logger.debug(f'sec. {time.time() - time_start} '
                 f'time the operations in isin took a')
    del tn_entity_ids
    mask_triples = mask_head & mask_tail
    logger.debug(f'sec. {time.time() - time_start} '
                 f'time the operations in isin took b')
    #
    logger.debug(f'sec. {time.time() - time_start} '
                 f'time the operations in isin took c')
    mask_triples_qualifier_remove_action = mask_head_qualifier_remove_action & mask_tail_qualifier_remove_action

    time_begin_complex_if_to_list = time.time()
    found_triples_lst = edges_wdata_at_from[:, mask_triples].T.tolist()
    found_triples_lst_qualifier_remove_action = edges_wdata_at_from_qualifier_remove_action[:,
                                                mask_triples_qualifier_remove_action].T.tolist()
    found_attrs_lst_qualifier_remove_action = attrs_wdata_at_from_qualifier_remove_action[
                                              mask_triples_qualifier_remove_action, :].tolist()
    found_qualifier_details_remove_action = qualifier_details_remove_action[mask_triples_qualifier_remove_action,
                                            :].tolist()
    # found_triples_set_qualifier_remove_action = set(tuple(sublist) for sublist in found_triples_lst_qualifier_remove_action)
    found_triples_set_qualifier_remove_action = dict()
    for curr_found_triple, curr_found_attr, curr_qualif_details in \
            zip(found_triples_lst_qualifier_remove_action, found_attrs_lst_qualifier_remove_action,
                found_qualifier_details_remove_action):
        ########
        # assert curr_triple_time[0] == curr_found_attr[0]
        #
        curr_head_id = curr_found_triple[0]
        curr_relation_id = curr_found_attr[1]
        curr_tail_id = curr_found_triple[1]
        triple_ids = (curr_head_id, curr_relation_id, curr_tail_id)
        found_triples_set_qualifier_remove_action[triple_ids] = curr_qualif_details

    logger.debug(f'sec. {time.time() - time_begin_complex_if_to_list} '
                 f'time it took to execute to_list and some del 1 '
                 f'found_triples_lst lengths: {len(found_triples_lst)} '
                 f'dtype: {edges_wdata_at_from.dtype} '
                 f'device edges: {edges_wdata_at_from.device} '
                 f'dtype attributes: {attrs_wdata_at_from.dtype}')
    mask_triples = mask_triples.to(str_device)
    logger.debug(f'sec. {time.time() - time_begin_complex_if_to_list} '
                 f'time it took to execute to_list and some del 2')
    found_attrs_lst = attrs_wdata_at_from[mask_triples, :].tolist()
    logger.debug(f'sec. {time.time() - time_begin_complex_if_to_list} '
                 f'time it took to execute to_list and some del 3')
    temporal_attrs = masked_tensor_timestamps[mask_triples, :].tolist()
    logger.debug(f'sec. {time.time() - time_begin_complex_if_to_list} '
                 f'time it took to execute to_list and some del 4')
    del edges_wdata_at_from
    del attrs_wdata_at_from
    del edges_wdata_at_from_qualifier_remove_action
    del attrs_wdata_at_from_qualifier_remove_action

    found_triples_in_from = set()
    triple_to_attributes = dict()
    time_begin_complex_if_zip = time.time()
    for curr_found_triple, curr_found_attr, curr_triple_time in \
            zip(found_triples_lst, found_attrs_lst, temporal_attrs):
        ########
        # assert curr_triple_time[0] == curr_found_attr[0]
        #
        curr_head_id = curr_found_triple[0]
        curr_relation_id = curr_found_attr[1]
        curr_tail_id = curr_found_triple[1]
        triple_ids = (curr_head_id, curr_relation_id, curr_tail_id)
        qualifier_info = dict()
        if triple_ids in found_triples_set_qualifier_remove_action:
            found_qualifier = found_triples_set_qualifier_remove_action[triple_ids]
            logger.debug('triple_ids in found_triples_set_qualifier_remove_action! '
                         f'{triple_ids} and the details are: '
                         f'{found_qualifier}')
            qualifier_info['qualifier_timestamp'] = found_qualifier[1]
            qualifier_info['qualifier_date'] = timestamp_to_date(qualifier_info['qualifier_timestamp'])
            qualifier_info['qualifier_qid'] = index_to_relation[found_qualifier[0]]
            qualifier_info['qualifier_label'] = property_qid_to_label[qualifier_info['qualifier_qid']]

        found_triples_in_from.add(triple_ids)
        emerging_head = False
        emerging_tail = False
        if interval_from <= curr_found_attr[2]:
            emerging_head = True

        if interval_from <= curr_found_attr[3]:
            emerging_tail = True

        # if interval_from <
        triple_attrs = {
            'emerging_head': emerging_head,
            'emerging_tail': emerging_tail,
            'head_creation_date': f'{timestamp_to_date(curr_found_attr[2])}',
            'tail_creation_date': f'{timestamp_to_date(curr_found_attr[3])}',
            'head_creation_timestamp': int(curr_found_attr[2]),
            'tail_creation_timestamp': int(curr_found_attr[3]),
            'triple_date_from': f'{timestamp_to_date(curr_triple_time[1])}',
            'triple_date_to': f'{timestamp_to_date(curr_triple_time[2])}',
            'triple_timestamp_from': int(curr_triple_time[1]),
            'triple_timestamp_to': int(curr_triple_time[2]),
            'qualifier_info': qualifier_info
        }

        triple_to_attributes[triple_ids] = triple_attrs

    logger.debug(f'sec. {time.time() - time_begin_complex_if_zip} '
                 f'to assign triple_to_attributes')
    #
    time_begin_complex_if_difference = time.time()
    del temporal_attrs
    # logger.info(f'the difference is on {found_triples_in_from} ==AND== {matched_triples}')
    additional_triple_in_from_not_in_matched = found_triples_in_from.difference(matched_triples)
    logger.debug(f'sec. {time.time() - time_begin_complex_if_difference}'
                 f' time for del and difference')
    time_begin_complex_if_print = time.time()
    for curr_additional_triple in additional_triple_in_from_not_in_matched:
        # triple_attrs = triple_to_attributes[triple_ids]
        triple_attrs = triple_to_attributes[curr_additional_triple]
        qid_head = index_to_entity[curr_additional_triple[0]]
        qid_head = f'Q{qid_head}'
        label_head = qid_head
        if qid_head in wikidata_qid_to_label:
            label_head = wikidata_qid_to_label[qid_head]
        qid_relation = index_to_relation[curr_additional_triple[1]]
        label_relation = qid_relation
        #
        if qid_relation in property_qid_to_label:
            label_relation = property_qid_to_label[qid_relation]

        qid_tail = index_to_entity[curr_additional_triple[2]]
        qid_tail = f'Q{qid_tail}'
        label_tail = qid_tail
        if qid_tail in wikidata_qid_to_label:
            label_tail = wikidata_qid_to_label[qid_tail]
        # curr_triple_field = (f'{qid_head}({label_head})',
        #                      f'{qid_relation}({label_relation})',
        #                      f'{qid_tail}({label_tail})')
        curr_triple_field = (f'{qid_head}',
                             f'{qid_relation}',
                             f'{qid_tail}')
        curr_triple_labels_field = (f'{label_head}',
                                    f'{label_relation}',
                                    f'{label_tail}')
        existing_triples.append({
            'qualifier_info': triple_attrs['qualifier_info'],
            'emerging_head': triple_attrs['emerging_head'],
            'emerging_tail': triple_attrs['emerging_tail'],
            'head_creation_date': triple_attrs['head_creation_date'],
            'tail_creation_date': triple_attrs['tail_creation_date'],
            'head_creation_timestamp': triple_attrs['head_creation_timestamp'],
            'tail_creation_timestamp': triple_attrs['tail_creation_timestamp'],
            'triple_lifespan_date': [
                triple_attrs['triple_date_from'],
                None if triple_attrs['triple_timestamp_to'] == 0 else triple_attrs['triple_date_to']
                # triple_attrs['triple_date_to']
                # value_if_true if condition else value_if_false
            ],
            'triple_lifespan_timestamp': [
                triple_attrs['triple_timestamp_from'],
                None if triple_attrs['triple_timestamp_to'] == 0 else triple_attrs['triple_timestamp_to']
            ],
            'triple': curr_triple_field,
            'triple_labels': curr_triple_labels_field
        })
        # if len(triple_attrs['qualifier_info']) > 0:
    # json_input['existing_triples'] = existing_triples
    json_input['existing_knowledge'] = existing_triples

    curr_time_complex_if = time.time() - time_begin_complex_if_print
    logger.debug(f'sec. {curr_time_complex_if} to output'
                 f' the complex if')
    return json_input
