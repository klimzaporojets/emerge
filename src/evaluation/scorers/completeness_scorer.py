import logging
import time
from typing import List, Tuple, Dict

from sentence_transformers import SentenceTransformer, util

from evaluation.misc.utils import divide_list, batch_iterator

logger = logging.getLogger(__name__)

import torch
from concurrent.futures import ThreadPoolExecutor

import threading

counter = 0
counter_lock = threading.Lock()


def calculate_batched_completeness(
        gt_pred_triples,
        batch_size,
        max_workers,
        models: List[SentenceTransformer],
        device,
        tkgu_type: str,
        model: str,
        model_alias: str,
        max_records_calculate_completeness_score=-1) -> Tuple[List[Dict], List[Dict]]:
    if max_records_calculate_completeness_score > -1:
        adapted_gt_pred_triples = gt_pred_triples[:max_records_calculate_completeness_score]
    else:
        adapted_gt_pred_triples = gt_pred_triples

    to_ret_tuples = process_multi_jobs_in_batches_batching_in_job_v13(
        gt_pred_triples=adapted_gt_pred_triples,
        models=models,
        device=device,
        batch_size=batch_size,
        max_workers=max_workers
    )

    to_ret_lst_gt = list()
    to_ret_lst_pred = list()
    for curr_ret_record in to_ret_tuples:
        curr_hash_id = curr_ret_record[0]
        curr_triple_qids = curr_ret_record[1]
        curr_batch_scores = curr_ret_record[4]
        curr_batch_scores_gt = curr_batch_scores[0]
        curr_batch_scores_pred = curr_batch_scores[1]
        curr_pred_labels = curr_ret_record[3]
        assert len(curr_triple_qids) == len(curr_batch_scores_gt)
        for curr_idx in range(len(curr_batch_scores_gt)):
            to_ret_lst_gt.append({
                'hash_id': curr_hash_id,
                'triple_head': curr_triple_qids[curr_idx][0],
                'triple_relation': curr_triple_qids[curr_idx][1],
                'triple_tail': curr_triple_qids[curr_idx][2],
                'model': model,
                'tkgu_type': tkgu_type,
                'score': curr_batch_scores_gt[curr_idx],
                'metric': 'completeness',
                'evaluator_model': f'{model_alias}',
                'granularity_level': 'triple'
            })

        assert len(curr_pred_labels) == len(curr_batch_scores_pred)
        for curr_idx in range(len(curr_batch_scores_pred)):
            curr_pred_triple_labels = curr_pred_labels[curr_idx]
            assert len(curr_pred_triple_labels) == 3
            curr_pred_triple_completeness_score = curr_batch_scores_pred[curr_idx]
            to_ret_lst_pred.append({
                'hash_id': curr_hash_id,
                'triple_head_label': curr_pred_triple_labels[0],
                'triple_relation_label': curr_pred_triple_labels[1],
                'triple_tail_label': curr_pred_triple_labels[2],
                'model': model,
                'tkgu_type': tkgu_type,
                'score': curr_pred_triple_completeness_score,
                'metric': 'completeness',
                'evaluator_model': f'{model_alias}',
                'granularity_level': 'triple'
            })

    logger.debug('end invoking calculate_batched_completeness')
    return to_ret_lst_gt, to_ret_lst_pred


# Worker function
def increase_counter(number_to_increase, start_time):
    global counter
    # Simulate some work (optional)

    # Safely increment counter
    with counter_lock:
        counter += number_to_increase
        curr_time = time.time()
        diff_mins = (curr_time - start_time) / 60
        avg_per_min = counter / diff_mins
        logger.debug(f'increase_counter total processed: {counter} in {diff_mins} mins, '
                     f'avg: {avg_per_min} instances per min')


def calculate_score_batch(model, triples_gt_list,
                          triples_gt_llm_asserted_list,
                          triples_pred_list, device, similarity_threshold=0.90):
    """

    :param model:
    :param triples_gt_list: example -->
    :param triples_pred_list: example -->
    :param device:
    :param similarity_threshold:
    :return:
    """
    all_triple_labels_gt = []
    all_triple_labels_pred = []
    # logger.info(f'triples_gt_list --> {triples_gt_list}')
    # logger.info(f'triples_pred_list --> {triples_pred_list}')
    logger.debug('calculate_score_batch - iteration and appending')
    for triples_gt, triples_pred in zip(triples_gt_list, triples_pred_list):
        gt = [' '.join(t).lower().strip() for t in triples_gt]
        pred = [' '.join(t).lower().strip() for t in triples_pred]
        all_triple_labels_gt.append(gt)
        all_triple_labels_pred.append(pred)

    # Flatten and encode in one go
    logger.debug('calculate_score_batch - flattening')
    flat_gt = [item for sublist in all_triple_labels_gt for item in sublist]
    flat_pred = [item for sublist in all_triple_labels_pred for item in sublist]
    flat_gt_assessments = [item for sublist in triples_gt_llm_asserted_list for item in sublist]

    logger.debug('calculate_score_batch - encoding')
    embeddings_gt = model.encode(flat_gt, convert_to_tensor=True, show_progress_bar=False) if flat_gt else \
        torch.empty(0).to(device=device)
    embeddings_pred = model.encode(flat_pred, convert_to_tensor=True, show_progress_bar=False) if flat_pred else \
        torch.empty(0).to(device=device)
    tensor_gt_assessments = torch.tensor(flat_gt_assessments, dtype=torch.bool).to(device=device)

    # Now compute scores batch-by-batch
    results = []
    start_gt = 0
    start_pred = 0
    logger.debug('calculate_score_batch - iterating of all_triple_labels_gt and all_triple_labels_pred')
    for gt, pred in zip(all_triple_labels_gt, all_triple_labels_pred):
        len_gt = len(gt)
        len_pred = len(pred)
        ##################
        if len_gt == 0 or len_pred == 0:
            results.append(([0.0] * len_gt, [0.0] * len_pred))
        else:
            scores = util.cos_sim(
                embeddings_gt[start_gt:start_gt + len_gt],
                embeddings_pred[start_pred:start_pred + len_pred]
            )
            curr_tensor_gt_assessments = tensor_gt_assessments[start_gt:start_gt + len_gt]
            if scores.numel() == 0:  # extra safeguard
                results.append(([0.0] * len_gt, [0.0] * len_pred))
            else:
                max_scores_gt, _ = scores.max(dim=1)
                mask = curr_tensor_gt_assessments  # boolean mask, same length as scores

                if mask.any():
                    # take max ONLY over masked elements
                    max_scores_pred = scores[mask].max(dim=0).values
                else:
                    # no true entries → set to zero with same shape as one row of scores
                    max_scores_pred = torch.zeros_like(scores[0])

                logger.debug(f'----------------------------------')
                logger.debug(
                    f'tensor_gt_assessments.shape: {tensor_gt_assessments.shape} \n'
                    f'curr_tensor_gt_assessments.shape: {curr_tensor_gt_assessments.shape} \n'
                    f'curr_tensor_gt_assessments: {curr_tensor_gt_assessments} \n'
                    f'scores.shape: {scores.shape} \n'
                    f'scores: {scores} \n'
                    f'scores[mask].shape: {scores[mask].shape} \n'
                    f'scores[mask]: {scores[mask]} \n'
                    f'max_scores_gt.shape: {max_scores_gt.shape} \n'
                    f'max_scores_gt: {max_scores_gt} \n'
                    f'max_scores_pred.shape: {max_scores_pred.shape} \n'
                    f'max_scores_pred: {max_scores_pred} '
                )
                logger.debug(f'----------------------------------')
                max_scores_gt_lst = max_scores_gt.tolist()
                max_scores_pred_lst = max_scores_pred.tolist()
                assert len(max_scores_gt_lst) == len_gt
                assert len(max_scores_pred_lst) == len_pred
                results.append((max_scores_gt_lst, max_scores_pred_lst))
        ###################
        start_gt += len_gt
        start_pred += len_pred

    return results


def process_instances_multiprocess(gt_pred_triples, model, device, batch_size, start_time):
    try:
        to_ret = list()
        model = model.to(device)
        for idx_batch, curr_batch in enumerate(batch_iterator(gt_pred_triples, batch_size)):
            #
            hash_ids, triple_qids, gt_list, pred_list, gt_list_llm_asserted, _ = map(list, zip(*curr_batch))
            #
            batch_scores = calculate_score_batch(
                model=model,
                triples_gt_list=gt_list,
                triples_gt_llm_asserted_list=gt_list_llm_asserted,
                triples_pred_list=pred_list,
                device=device
            )
            increase_counter(len(batch_scores), start_time)

            zipped = [[h, q, g, p, gt_and_pred_scores] for h, q, g, p, gt_and_pred_scores in \
                      zip(hash_ids, triple_qids, gt_list, pred_list, batch_scores)]

            to_ret.extend(zipped)
        return to_ret
    finally:
        del model


# this one is used
def process_multi_jobs_in_batches_batching_in_job_v13(gt_pred_triples: List,
                                                      models: List[SentenceTransformer],
                                                      device,
                                                      batch_size=2,
                                                      max_workers=3):
    """
    The idea here is to put all the instances in a job at once, like job_instances --> subset of instances for a particular job.
    Each job gets a fair share of instances and does the batching inside. This way, there is no need to create new jobs
    for each batch as happens in process_multi_jobs_in_batches, avoiding out of memory
    :param models:
    :param gt_pred_triples:
    :param model:
    :param device:
    :param batch_size:
    :param max_workers:
    :return:
    """

    results = []
    divided_gt_pred_triples = divide_list(gt_pred_triples, max_workers)
    logger.debug(f'process_multi_jobs_in_batches len(gt_pred_triples): {len(gt_pred_triples)} '
                 f'batch_size is: {batch_size} '
                 f'max_workers is: {max_workers}')

    start_time = time.time()
    global counter
    counter = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list()
        for idx_instances, curr_job_instances in enumerate(divided_gt_pred_triples):
            logger.debug(f'submitting_to_process_instances_multiprocess curr_job_instances of length: '
                         f'{len(curr_job_instances)}')
            curr_future = executor.submit(process_instances_multiprocess, curr_job_instances,
                                          models[idx_instances],
                                          device, batch_size,
                                          start_time)

            futures.append(curr_future)

        for future in futures:
            future_result = future.result()
            logger.debug(f'process_multi_jobs_in_batches_batching_in_job future size: {len(future_result)}')
            results.extend(future_result)

    return results
