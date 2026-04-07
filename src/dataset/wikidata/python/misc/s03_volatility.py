import csv
import logging
import time
from typing import Dict, Tuple

import torch
from torch_geometric.data import Data

from dataset.wikidata.python.misc.s03_constant import AttrIndexes

logger = logging.getLogger(__name__)


def obtain_changes_per_entity(
        delta_added_and_removed: Data,
        config: Dict,
        wdata_timestamp_to: Data,
        only_changes_in_head: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    logger.info(f'BEGIN obtain_changes_per_entity')
    start_time = time.time()

    if only_changes_in_head and delta_added_and_removed.edge_index.numel() > 0:
        # index 0 indicates it is head
        unique_entities, nr_changes_per_entity = delta_added_and_removed.edge_index[0, :].unique(return_counts=True)
    else:
        unique_entities, nr_changes_per_entity = delta_added_and_removed.edge_index.unique(return_counts=True)
    sorted_entities, all_sorted_indices = torch.sort(unique_entities, descending=True)
    nr_changes_per_entity = nr_changes_per_entity[all_sorted_indices]

    if only_changes_in_head and wdata_timestamp_to.edge_index.numel() > 0:
        all_unique_entities, all_nr_changes_per_entity = wdata_timestamp_to.edge_index[0, :].unique(return_counts=True)
    else:
        all_unique_entities, all_nr_changes_per_entity = wdata_timestamp_to.edge_index.unique(return_counts=True)

    mask_present_in_delta = torch.isin(all_unique_entities, unique_entities)
    all_nr_changes_per_entity = all_nr_changes_per_entity[mask_present_in_delta]
    all_unique_entities = all_unique_entities[mask_present_in_delta]
    all_sorted_entities, all_sorted_indices = torch.sort(all_unique_entities, descending=True)
    all_nr_changes_per_entity = all_nr_changes_per_entity[all_sorted_indices]
    numerator = torch.ones_like(nr_changes_per_entity)
    mask_entities_in_all_entities = torch.isin(sorted_entities, all_sorted_entities)
    numerator[mask_entities_in_all_entities] = all_nr_changes_per_entity

    # assert torch.equal(all_sorted_entities, sorted_entities)
    assert torch.equal(all_sorted_entities, sorted_entities[mask_entities_in_all_entities])

    # normalized_delta_changes = nr_changes_per_entity * (nr_changes_per_entity / all_nr_changes_per_entity)
    normalized_delta_changes = nr_changes_per_entity * (nr_changes_per_entity / numerator)

    sorted_normalized_changes, sorted_indices = \
        torch.sort(normalized_delta_changes, descending=True)

    sorted_entities = sorted_entities[sorted_indices]
    sorted_changes = nr_changes_per_entity[sorted_indices]

    tensor_cat = torch.cat([sorted_entities.unsqueeze(1),
                            sorted_changes.unsqueeze(1),
                            sorted_normalized_changes.unsqueeze(1)], dim=1)
    entities_to_changes = {tensor_cat[i, 0].item(): tensor_cat[i, 1:].tolist() for i in range(tensor_cat.size(0))}

    curr_time = time.time()
    logger.info(f'{((curr_time - start_time) / 60):.4f} mins to END obtain_changes_per_entity')

    return sorted_entities, sorted_changes, sorted_normalized_changes, entities_to_changes


def save_most_volatile_entities(
        delta_added_and_removed: Data,
        config: Dict,
        output_file_path_volatile: str,
        index_to_entity: Dict[int, int],
        index_to_relation: Dict[int, int],
        wikidata_qid_to_label: Dict[str, str],
        property_qid_to_label: Dict[str, str],
        kb_type: str,
        timestamp_from: int,
        timestamp_to: int,
        sorted_entities: torch.Tensor,
        sorted_changes: torch.Tensor,
        sorted_normalized_changes: torch.Tensor
):
    logger.debug('sorted_normalized_changes.shape before applying '
                 'min_normalized_delta_nr_of_changes filter: '
                 f'{sorted_normalized_changes.shape}')
    logger.info(f'BEGIN save_most_volatile_entities with {sorted_normalized_changes.shape}')
    start_time = time.time()

    sorted_normalized_changes_mask = (sorted_normalized_changes >=
                                      config['min_normalized_delta_nr_of_changes'])
    sorted_normalized_changes = sorted_normalized_changes[sorted_normalized_changes_mask]
    sorted_entities = sorted_entities[sorted_normalized_changes_mask]
    sorted_changes = sorted_changes[sorted_normalized_changes_mask]

    logger.debug('sorted_normalized_changes.shape after applying '
                 'min_normalized_delta_nr_of_changes filter: '
                 f'{sorted_normalized_changes.shape}')

    sorted_normalized_changes = sorted_normalized_changes[: config['max_nr_of_delta_changed_entities']]
    sorted_entities = sorted_entities[: config['max_nr_of_delta_changed_entities']]
    sorted_changes = sorted_changes[: config['max_nr_of_delta_changed_entities']]

    logger.debug('sorted_normalized_changes.shape after applying '
                 'max_nr_of_delta_changed_entities filter: '
                 f'{sorted_normalized_changes.shape}')

    logger.debug('END obtaining the most dynamic (volatile) entities')

    logger.debug('BEGIN to print the most dynamic (volatile) entities '
                 f'in "{output_file_path_volatile}"')
    with open(output_file_path_volatile, 'wt') as outfile:
        #
        writer = csv.writer(outfile, delimiter='\t')
        for idx_entity, (
                curr_volatile_entity_id,
                curr_volatile_entity_nr_of_changes_normalized,
                curr_volatile_entity_nr_of_changes
        ) in enumerate(zip(
            sorted_entities.tolist(),
            sorted_normalized_changes.tolist(),
            sorted_changes.tolist()
        )):
            curr_volatile_qid = f'Q{index_to_entity[curr_volatile_entity_id]}'
            volatile_triples_head_mask = \
                (delta_added_and_removed.edge_index[0, :] == curr_volatile_entity_id)
            volatile_triples_tail_mask = \
                (delta_added_and_removed.edge_index[1, :] == curr_volatile_entity_id)

            curr_volatile_label = ''
            if curr_volatile_qid in wikidata_qid_to_label:
                curr_volatile_label = wikidata_qid_to_label[curr_volatile_qid]

            volatile_triples_mask = torch.logical_or(volatile_triples_head_mask,
                                                     volatile_triples_tail_mask)

            volatile_edges = delta_added_and_removed.edge_index[:, volatile_triples_mask]
            volatile_attributes = delta_added_and_removed.edge_attr[volatile_triples_mask, :]

            # for idx_row, curr_tensor_row in enumerate(volatile_edges.T):
            for idx_row, curr_tensor_row in enumerate(zip(volatile_edges.T.tolist(), volatile_attributes.tolist())):
                # curr_head_idx = curr_tensor_row[0].item()
                curr_head_idx = curr_tensor_row[0][0]
                curr_head_qid = f'Q{index_to_entity[curr_head_idx]}'
                # curr_tail_idx = curr_tensor_row[1].item()
                curr_tail_idx = curr_tensor_row[0][1]
                curr_tail_qid = f'Q{index_to_entity[curr_tail_idx]}'
                curr_head_label = ''
                curr_tail_label = ''
                if curr_volatile_entity_id == curr_head_idx:
                    # curr_volatile_creation_date = volatile_attributes[idx_row, volatile_attributes.shape[1] - 3]
                    curr_volatile_creation_date = curr_tensor_row[1][volatile_attributes.shape[1] - 3]
                elif curr_volatile_entity_id == curr_tail_idx:
                    # curr_volatile_creation_date = volatile_attributes[idx_row, volatile_attributes.shape[1] - 2]
                    curr_volatile_creation_date = curr_tensor_row[1][volatile_attributes.shape[1] - 2]
                else:
                    raise RuntimeError('curr_volatile_entity_id not found in text nor head: '
                                       f' {curr_volatile_entity_id} {curr_head_idx} {curr_tail_idx}')
                if curr_head_qid in wikidata_qid_to_label:
                    curr_head_label = wikidata_qid_to_label[curr_head_qid]

                if curr_tail_qid in wikidata_qid_to_label:
                    curr_tail_label = wikidata_qid_to_label[curr_tail_qid]

                curr_relation_label = ''
                curr_relation_qid = ''
                if kb_type == 'wikidata':
                    action_attr_index = AttrIndexes.IDX_DELTA_WDATA_ACTION
                    # curr_relation_idx = volatile_attributes[idx_row, 0].item()
                    curr_relation_idx = curr_tensor_row[1][AttrIndexes.IDX_DELTA_WDATA_RELATION_TYPE]
                    curr_relation_qid = index_to_relation[curr_relation_idx]
                    if curr_relation_qid in property_qid_to_label:
                        curr_relation_label = property_qid_to_label[curr_relation_qid]
                elif kb_type == 'wikipedia':
                    action_attr_index = AttrIndexes.IDX_DELTA_WPEDIA_ACTION
                else:
                    raise RuntimeError(f'kb_type not recognized: {kb_type}')
                # curr_action = volatile_attributes[idx_row, action_attr_index].item()
                curr_action = curr_tensor_row[1][action_attr_index]
                curr_action_label = ''
                if curr_action == 1:
                    curr_action_label = 'added'
                elif curr_action == 0:
                    curr_action_label = 'removed'
                else:
                    if kb_type == 'wikidata':
                        qualifier_action = curr_tensor_row[1][AttrIndexes.IDX_DELTA_WDATA_QUALIFIER_ACTION]
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

                writer.writerow(
                    [
                        timestamp_from,
                        timestamp_to,
                        curr_volatile_label,
                        curr_volatile_qid,
                        # curr_volatile_creation_date.item(),
                        curr_volatile_creation_date,
                        curr_volatile_entity_nr_of_changes,  # nr of changes
                        f'{curr_volatile_entity_nr_of_changes_normalized:.2f}',  # nr of normalized changes
                        # volatile_attributes[idx_row, volatile_attributes.shape[1] - 3].item(),  # head creation date
                        curr_tensor_row[1][volatile_attributes.shape[1] - 3],  # head creation date
                        # volatile_attributes[idx_row, volatile_attributes.shape[1] - 2].item(),  # tail creation date
                        curr_tensor_row[1][volatile_attributes.shape[1] - 2],  # tail creation date
                        curr_action_label,
                        curr_head_label,
                        curr_relation_label,
                        curr_tail_label,
                        curr_head_qid,
                        curr_relation_qid,
                        curr_tail_qid
                    ]
                )
    curr_time = time.time()

    logger.debug('END to print the most dynamic (volatile) entities '
                 f'in "{output_file_path_volatile}"')
    logger.info(
        f'{(curr_time - start_time) / 60:.4f} mins to END '
        f'save_most_volatile_entities with {sorted_normalized_changes.shape}'
    )
