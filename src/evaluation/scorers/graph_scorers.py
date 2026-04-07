#
import logging
import math
import os
import time
from typing import List, Any

from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer

from .misc.graph_matching import \
    split_to_edges, get_tokens, get_bleu_rouge, \
    get_sentence_transformer_score_fast, get_bert_score_fast, prepare_input_to_calculate_graph_scorers

from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import itertools

level_name = os.getenv('LOG_LEVEL', 'INFO').upper()
level = logging.getLevelName(level_name)

logger = logging.getLogger(__name__)


def batch_iter(lst, batch_size):
    """Yield successive batches of batch_size from a list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def calculate_gj_model(
        tkgu_type: str,
        model: str,
        model_scorers: List[Any],
        scorer_workers: int,
        model_alias: str,
        model_batch_size: int,
        data_batch_size: int,
        batches_inter: List,
        shared_model_across_threads=False,
        implementation_class='bert_scorer'
):
    """
    Wrapper that runs `calculate_graph_judge_scores` in parallel batches.
    If max_workers <= 1, executes sequentially without threading.
    """
    # logger.debug('+++ begin_calculate_graph_judge_scores_batched starting_batched +++')
    # if not gt_pred_triples or len(gt_pred_triples) == 0:
    #     logger.debug('gt_pred_triples is empty, returning None')
    #     return None
    #
    # # ------------------------
    # # Split into batches
    # # ------------------------
    results_per_batch_model = []
    # batches_inter = prepare_input_to_calculate(gt_pred_triples=gt_pred_triples)
    # if not batches_inter:
    #     return None
    if scorer_workers > 1:
        batches_model = list(batch_iter(batches_inter, data_batch_size))
        logger.debug(
            f'graph_judge_scorer_v13_processing MODEL '
            f'{len(model)} (batches_inter: {len(batches_inter)}) instances in '
            # f'{len(batches_model)} batches of size {batch_size_model} '
            f'{len(batches_model)} batches of size {data_batch_size} '
            f'using {scorer_workers} workers. '
            f'implementation_class: {implementation_class}'
        )
        with ThreadPoolExecutor(max_workers=scorer_workers) as model_executor:
            if implementation_class == 'bert_scorer':
                if shared_model_across_threads:
                    model_futures = {
                        model_executor.submit(
                            calculate_graph_judge_scores_bert, batch,
                            model_scorers[0],
                            model_batch_size
                        ): idx
                        for idx, batch in enumerate(batches_model)
                    }
                else:
                    model_futures = {}
                    next_batch_idx = 0
                    num_scorers = len(model_scorers)
                    # seed initial wave
                    while next_batch_idx < len(batches_model) and len(model_futures) < min(scorer_workers, num_scorers):
                        idx = next_batch_idx
                        batch = batches_model[idx]
                        scorer = model_scorers[len(model_futures)]
                        fut = model_executor.submit(
                            calculate_graph_judge_scores_bert, batch,
                            scorer,
                            model_batch_size
                        )
                        model_futures[fut] = (idx, scorer)
                        next_batch_idx += 1

                    logger.debug('submitted all the models')

                    # drain + refill, reusing freed scorer (never concurrently)
                    while model_futures:
                        for future in as_completed(list(model_futures.keys())):
                            batch_idx, freed_scorer = model_futures.pop(future)
                            try:
                                curr_res = future.result()
                                results_per_batch_model.append((batch_idx, curr_res))
                            except Exception as e:
                                logger.error(f" BERT batch {batch_idx} failed: {e}")
                                raise

                            if next_batch_idx < len(batches_model):
                                idx = next_batch_idx
                                batch = batches_model[idx]
                                fut = model_executor.submit(
                                    calculate_graph_judge_scores_bert, batch,
                                    freed_scorer,
                                    model_batch_size
                                )
                                model_futures[fut] = (idx, freed_scorer)
                                next_batch_idx += 1
                            break
                    model_futures = None
            elif implementation_class == 'sentence_transformer':
                if shared_model_across_threads:
                    model_futures = {
                        model_executor.submit(
                            calculate_graph_judge_scores_sentence_transformer, batch,
                            model_scorers[0],
                            model_batch_size
                        ): idx
                        for idx, batch in enumerate(batches_model)
                    }
                else:
                    model_futures = {}
                    next_batch_idx = 0
                    num_scorers = len(model_scorers)
                    # seed initial wave
                    while next_batch_idx < len(batches_model) and len(model_futures) < min(scorer_workers, num_scorers):
                        idx = next_batch_idx
                        batch = batches_model[idx]
                        scorer = model_scorers[len(model_futures)]
                        fut = model_executor.submit(
                            calculate_graph_judge_scores_sentence_transformer, batch,
                            scorer,
                            model_batch_size
                        )
                        model_futures[fut] = (idx, scorer)
                        next_batch_idx += 1

                    logger.debug('submitted all the models')

                    # drain + refill, reusing freed scorer (never concurrently)
                    while model_futures:
                        for future in as_completed(list(model_futures.keys())):
                            batch_idx, freed_scorer = model_futures.pop(future)
                            try:
                                curr_res = future.result()
                                results_per_batch_model.append((batch_idx, curr_res))
                            except Exception as e:
                                logger.error(f" BERT batch {batch_idx} failed: {e}")
                                raise

                            if next_batch_idx < len(batches_model):
                                idx = next_batch_idx
                                batch = batches_model[idx]
                                fut = model_executor.submit(
                                    calculate_graph_judge_scores_sentence_transformer, batch,
                                    freed_scorer,
                                    model_batch_size
                                )
                                model_futures[fut] = (idx, freed_scorer)
                                next_batch_idx += 1
                            break
                    model_futures = None
            else:
                raise RuntimeError(f'implementation_class not recognized: {implementation_class}')
        logger.debug('submitted all the models')

        if model_futures is not None:
            for future in as_completed(model_futures):
                batch_idx = model_futures[future]
                try:
                    curr_res = future.result()
                    results_per_batch_model.append((batch_idx, curr_res))
                except Exception as e:
                    logger.error(f" BERT batch {batch_idx} failed: {e}")
                    raise
    else:
        logger.debug('model_single_execution')
        if implementation_class == 'bert_scorer':
            curr_res = calculate_graph_judge_scores_bert(
                batch=batches_inter,
                bert_scorer=model_scorers[0],
                bert_scorer_batch_size=model_batch_size
            )
            results_per_batch_model.append((0, curr_res))
        elif implementation_class == 'sentence_transformer':
            curr_res = calculate_graph_judge_scores_sentence_transformer(
                batch=batches_inter,
                st_scorer=model_scorers[0],
                st_scorer_batch_size=model_batch_size
            )
            results_per_batch_model.append((0, curr_res))
        else:
            raise RuntimeError(f'implementation_class not recognized: {implementation_class}')

    # ------------------------
    # Sort results in batch order
    # ------------------------
    results_per_batch_model.sort(key=lambda x: x[0])
    results_per_batch_model = [r for (_, r) in results_per_batch_model if r is not None]

    # --- Combine instance-level scores ---
    combined_return_results = {}

    first_keys = results_per_batch_model[0].keys()
    for key in first_keys:
        combined_return_results[key] = list(
            itertools.chain.from_iterable(
                batch_res[key]
                for batch_res in results_per_batch_model
            )
        )

    logger.info('begin_calculating_to_ret_final')
    to_ret_final = return_concatenation_all_model_scores(
        model=model,
        tkgu_type=tkgu_type,
        model_alias=model_alias,
        **combined_return_results
    )
    logger.info('end_calculating_to_ret_final')

    return to_ret_final


def calculate_gj_bleu_rouge(
        gt_pred_triples: List,
        tkgu_type: str,
        model: str,
        max_workers: int,
        model_alias='bleu_rouge'
):
    """
    Wrapper that runs `calculate_graph_judge_scores` in parallel batches.
    If max_workers <= 1, executes sequentially without threading.
    """
    logger.debug('+++ begin_calculate_graph_judge_scores_batched starting_batched +++')
    if len(gt_pred_triples) == 0:
        return None

    # ------------------------
    # Split into batches
    # ------------------------
    results_per_batch_bleu_rouge = []
    batches_inter = prepare_input_to_calculate_graph_scorers(gt_pred_triples=gt_pred_triples)
    if not batches_inter:
        return None
    batch_size = math.ceil(len(batches_inter) / max_workers)
    batches_bleu_rouge = list(batch_iter(batches_inter, batch_size))
    logger.debug(
        f'[INFO] Processing BLEU/ROUGE {len(gt_pred_triples)} '
        f'(batches_inter: {len(batches_inter)}) instances in '
        f'{len(batches_bleu_rouge)} batches of size {batch_size} '
        f'using {max_workers} workers. NR batches: {len(batches_bleu_rouge)}'
    )
    # ============================================================
    # CASE 1 — Sequential execution (max_workers <= 1)
    # ============================================================
    if max_workers <= 1:
        logger.info(' Running sequentially (no threading)')
        for batch_idx, batch in enumerate(batches_bleu_rouge):
            try:
                res = calculate_graph_judge_scores_bleu_rouge(
                    batch=batch
                )
                results_per_batch_bleu_rouge.append((batch_idx, res))
            except Exception as e:
                logger.error(f'Batch {batch_idx} failed: {e}')
                raise

    # ============================================================
    # CASE 2 — Parallel execution using ProcessPoolExecutor
    # ============================================================
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    calculate_graph_judge_scores_bleu_rouge,
                    batch
                ): idx
                for idx, batch in enumerate(batches_bleu_rouge)
            }

            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    res = future.result()
                    results_per_batch_bleu_rouge.append((batch_idx, res))
                except Exception as e:
                    logger.error(f'Batch {batch_idx} failed: {e}')
                    raise

    # ------------------------
    # Sort results in batch order
    # ------------------------
    results_per_batch_bleu_rouge.sort(key=lambda x: x[0])
    results_per_batch_bleu_rouge = [r for (_, r) in results_per_batch_bleu_rouge if r is not None]
    if len(results_per_batch_bleu_rouge) == 0:
        return None

    # ------------------------
    # Combine results across batches
    # ------------------------

    # --- Combine instance-level scores ---
    combined_return_results = {}
    first_keys = results_per_batch_bleu_rouge[0].keys()

    for key in first_keys:
        combined_return_results[key] = list(
            itertools.chain.from_iterable(
                batch_res[key]
                for batch_res in results_per_batch_bleu_rouge
            )
        )

    to_ret_final = return_concatenation_all_bleu_rouge(
        model=model,
        tkgu_type=tkgu_type,
        model_alias=model_alias,
        **combined_return_results
    )

    return to_ret_final


def calculate_graph_judge_scores_sentence_transformer(
        batch,
        st_scorer: SentenceTransformer,
        st_scorer_batch_size: int
):
    (hash_ids, gold_tokens, pred_tokens, gold_edges, pred_edges, gold_graphs, gt_triple_qids,
     pred_graphs) = \
        map(list, zip(*batch))
    logger.debug('Get bert score...')
    start = time.time()

    (precisions_TS, recalls_TS, f1s_TS, all_triples_transformer_scores,
     per_gt_triple_scores_all, per_pred_triple_scores_all) = \
        get_sentence_transformer_score_fast(
            all_gold_edges=gold_edges,
            all_pred_edges=pred_edges,
            scorer=st_scorer,
            st_batch_size=st_scorer_batch_size
        )

    end = time.time()
    elapsed = end - start
    logger.debug(f'elapsed_time_get_bert_score: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')
    # if not len(hash_ids) == len(precisions_TS) == len(recalls_TS) == len(f1s_TS) == \
    #        len(all_triples_transformer_scores) == len(gold_graphs) == len(gt_triple_qids) == \
    #     len(pred_graphs) == len(per_gt_triple_scores_all) == len(per_pred_triple_scores_all):

    assert len(hash_ids) == len(precisions_TS) == len(recalls_TS) == len(f1s_TS) == \
           len(all_triples_transformer_scores) == len(gold_graphs) == len(gt_triple_qids) == \
           len(pred_graphs) == len(per_gt_triple_scores_all) == len(per_pred_triple_scores_all)

    return {
        'hash_ids': hash_ids,
        'precisions_MS': precisions_TS,
        'recalls_MS': recalls_TS,
        'f1s_MS': f1s_TS,
        'all_triples_model_scores': all_triples_transformer_scores,
        'per_gt_triple_scores_all': per_gt_triple_scores_all,
        'per_pred_triple_scores_all': per_pred_triple_scores_all,
        'gold_graphs': gold_graphs,
        'pred_graphs': pred_graphs,
        'gt_triple_qids': gt_triple_qids
    }


def calculate_graph_judge_scores_bert(
        batch,
        bert_scorer: BERTScorer,
        bert_scorer_batch_size: int
):
    hash_ids, gold_tokens, pred_tokens, gold_edges, pred_edges, gold_graphs, gt_triple_qids, pred_graphs = \
        map(list, zip(*batch))
    logger.debug('Get bert score...')
    start = time.time()

    # precisions, recalls, f1s, per_triple_scores, per_gt_triple_scores_all, per_pred_triple_scores_all
    precisions_BS, recalls_BS, f1s_BS, all_triples_bert_scores, per_gt_triple_scores_all, per_pred_triple_scores_all = \
        get_bert_score_fast(
            all_gold_edges=gold_edges,
            all_pred_edges=pred_edges,
            scorer=bert_scorer,
            bert_scorer_batch_size=bert_scorer_batch_size
        )

    end = time.time()
    elapsed = end - start
    logger.debug(f'elapsed_time_get_bert_score: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')
    assert len(hash_ids) == len(precisions_BS) == len(recalls_BS) == len(f1s_BS) == \
           len(all_triples_bert_scores) == len(gold_graphs) == len(gt_triple_qids) == \
           len(pred_graphs) == len(per_gt_triple_scores_all) == len(per_pred_triple_scores_all)
    return {
        'hash_ids': hash_ids,
        'precisions_MS': precisions_BS,
        'recalls_MS': recalls_BS,
        'f1s_MS': f1s_BS,
        'all_triples_model_scores': all_triples_bert_scores,
        'per_gt_triple_scores_all': per_gt_triple_scores_all,
        'per_pred_triple_scores_all': per_pred_triple_scores_all,
        'gold_graphs': gold_graphs,
        'pred_graphs': pred_graphs,
        'gt_triple_qids': gt_triple_qids
    }


def calculate_graph_judge_scores_bleu_rouge(
        batch: List
):
    hash_ids, gold_tokens, pred_tokens, gold_edges, pred_edges, gold_graphs, gt_triple_qids, pred_graphs = \
        map(list, zip(*batch))
    start = time.time()

    logger.debug('Get bleu rouge...')
    (
        precisions_rouge,
        recalls_rouge,
        f1s_rouge,
        precisions_bleu,
        recalls_bleu,
        f1s_bleu,
        all_triples_rouge_scores,
        all_triples_bleu_scores
    ) = \
        get_bleu_rouge(gold_tokens, pred_tokens, gold_edges, pred_edges)
    end = time.time()
    elapsed = end - start
    logger.debug(f'elapsed_time_4: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')

    logger.debug(f'G-BLEU Precision: {precisions_bleu.sum() / len(gold_graphs):.4f}')
    logger.debug(f'G-BLEU Recall: {recalls_bleu.sum() / len(gold_graphs):.4f}')
    logger.debug(f'G-BLEU F1: {f1s_bleu.sum() / len(gold_graphs):.4f}\n')

    logger.debug(f'G-Rouge Precision: {precisions_rouge.sum() / len(gold_graphs):.4f}')
    logger.debug(f'G-Rouge Recall Score: {recalls_rouge.sum() / len(gold_graphs):.4f}')
    logger.debug(f'G-Rouge F1 Score: {f1s_rouge.sum() / len(gold_graphs):.4f}\n')

    end = time.time()
    elapsed = end - start
    logger.debug(f'elapsed_time_5: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')
    assert len(hash_ids) == len(precisions_rouge) == len(recalls_rouge) == \
           len(f1s_rouge) == len(precisions_bleu) == len(recalls_bleu) == \
           len(f1s_bleu) == len(gold_graphs) == len(all_triples_bleu_scores) == \
           len(all_triples_rouge_scores) == len(gt_triple_qids)

    return {
        'hash_ids': hash_ids,
        'precisions_rouge': precisions_rouge,
        'recalls_rouge': recalls_rouge,
        'f1s_rouge': f1s_rouge,
        'precisions_bleu': precisions_bleu,
        'recalls_bleu': recalls_bleu,
        'f1s_bleu': f1s_bleu,
        'all_triples_rouge_scores': all_triples_rouge_scores,
        'all_triples_bleu_scores': all_triples_bleu_scores,
        'gold_graphs': gold_graphs,
        'gt_triple_qids': gt_triple_qids
    }


def return_concatenation_all_bleu_rouge(
        hash_ids,
        gold_graphs,
        all_triples_bleu_scores,
        all_triples_rouge_scores,
        precisions_bleu,
        recalls_bleu,
        f1s_bleu,
        precisions_rouge,
        recalls_rouge,
        f1s_rouge,
        model,
        tkgu_type,
        gt_triple_qids,
        model_alias
):
    start = time.time()
    to_ret_scores_per_instance = list()
    assert len(hash_ids) == len(gold_graphs) == len(all_triples_bleu_scores) == \
           len(all_triples_rouge_scores) == len(precisions_bleu) == len(recalls_bleu) == len(f1s_bleu) \
           == len(precisions_rouge) == len(recalls_rouge) == len(f1s_rouge) == len(gt_triple_qids)
    #
    to_ret_scores_per_triple = list()
    for curr_hash_id, curr_gold_triples, \
            curr_gold_triples_bleu_scores, curr_gold_triples_rouge_scores, curr_gt_triple_qids, \
            curr_precision_bleu, curr_recall_bleu, curr_f1_bleu, curr_precision_rouge, \
            curr_recall_rouge, curr_f1_rouge in \
            zip(hash_ids, gold_graphs,
                all_triples_bleu_scores, all_triples_rouge_scores, gt_triple_qids, precisions_bleu,
                recalls_bleu, f1s_bleu, precisions_rouge, recalls_rouge, f1s_rouge):
        ## bleu
        to_ret_scores_per_instance.append(
            {
                'hash_id': curr_hash_id,
                'tkgu_type': tkgu_type,
                'metric': 'bleu-f1',
                'model': model,
                'granularity_level': 'instance',
                'score': curr_f1_bleu,
                'evaluator_model': model_alias
            }
        )
        to_ret_scores_per_instance.append(
            {
                'hash_id': curr_hash_id,
                'tkgu_type': tkgu_type,
                'metric': 'bleu-precision',
                'model': model,
                'granularity_level': 'instance',
                'score': curr_precision_bleu,
                'evaluator_model': model_alias
            }
        )
        to_ret_scores_per_instance.append(
            {
                'hash_id': curr_hash_id,
                'tkgu_type': tkgu_type,
                'metric': 'bleu-recall',
                'model': model,
                'granularity_level': 'instance',
                'score': curr_recall_bleu,
                'evaluator_model': model_alias
            }
        )
        ## rouge
        to_ret_scores_per_instance.append(
            {
                'hash_id': curr_hash_id,
                'tkgu_type': tkgu_type,
                'metric': 'rouge-f1',
                'model': model,
                'granularity_level': 'instance',
                'score': curr_f1_rouge,
                'evaluator_model': model_alias
            }
        )
        to_ret_scores_per_instance.append(
            {
                'hash_id': curr_hash_id,
                'tkgu_type': tkgu_type,
                'metric': 'rouge-precision',
                'model': model,
                'granularity_level': 'instance',
                'score': curr_precision_rouge,
                'evaluator_model': model_alias
            }
        )
        to_ret_scores_per_instance.append(
            {
                'hash_id': curr_hash_id,
                'tkgu_type': tkgu_type,
                'metric': 'rouge-recall',
                'model': model,
                'granularity_level': 'instance',
                'score': curr_recall_rouge,
                'evaluator_model': model_alias
            }
        )
        #
        assert len(curr_gold_triples) == len(curr_gold_triples_bleu_scores)
        assert len(curr_gold_triples) == len(curr_gold_triples_rouge_scores)
        assert len(curr_gold_triples) == len(curr_gt_triple_qids)
        for (
                curr_gold_triple,
                curr_gold_triple_bleu_score,
                curr_gold_triple_rouge_score,
                curr_curr_gt_triple_qids,
        ) in \
                zip(
                    curr_gold_triples,
                    curr_gold_triples_bleu_scores,
                    curr_gold_triples_rouge_scores,
                    curr_gt_triple_qids
                ):
            curr_gold_triple_str = ' '.join(' '.join(curr_gold_triple).lower().strip().split())
            assert curr_gold_triple_str == curr_gold_triple_bleu_score[0]
            assert curr_gold_triple_str == curr_gold_triple_rouge_score[0]
            # print('check_here')
            to_ret_scores_per_triple.append(
                {
                    'hash_id': curr_hash_id,
                    'triple_head': curr_curr_gt_triple_qids[0],
                    'triple_relation': curr_curr_gt_triple_qids[1],
                    'triple_tail': curr_curr_gt_triple_qids[2],
                    # 'triple_head_label': curr_gold_triple[0],
                    # 'triple_relation_label': curr_gold_triple[1],
                    # 'triple_tail_label': curr_gold_triple[2],
                    'model': model,
                    'tkgu_type': tkgu_type,
                    'metric': 'bleu-triple',
                    'granularity_level': 'triple',
                    'score': curr_gold_triple_bleu_score[2],
                    'evaluator_model': model_alias
                }
            )
            to_ret_scores_per_triple.append(
                {
                    'hash_id': curr_hash_id,
                    'triple_head': curr_curr_gt_triple_qids[0],
                    'triple_relation': curr_curr_gt_triple_qids[1],
                    'triple_tail': curr_curr_gt_triple_qids[2],
                    # 'triple_head_label': curr_gold_triple[0],
                    # 'triple_relation_label': curr_gold_triple[1],
                    # 'triple_tail_label': curr_gold_triple[2],
                    'model': model,
                    'tkgu_type': tkgu_type,
                    'metric': 'rouge-triple',
                    'granularity_level': 'triple',
                    'score': curr_gold_triple_rouge_score[2],
                    'evaluator_model': model_alias
                }
            )

    #
    end = time.time()
    elapsed = end - start
    logger.debug(f'elapsed_time_6: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')
    #

    return {
        'scores_per_instance': to_ret_scores_per_instance,
        'scores_per_triple': to_ret_scores_per_triple
    }


def return_concatenation_all_model_scores(
        hash_ids: List[str],
        gold_graphs: List,
        pred_graphs: List,
        all_triples_model_scores: List,
        per_gt_triple_scores_all: List,
        per_pred_triple_scores_all: List,
        precisions_MS: List,
        recalls_MS: List,
        f1s_MS: List,
        model: str,
        tkgu_type: str,
        model_alias: str,
        gt_triple_qids: List
):
    #     return {
    #         'hash_ids': hash_ids,
    #         'precisions_MS': precisions_BS,
    #         'recalls_MS': recalls_BS,
    #         'f1s_MS': f1s_BS,
    #         'all_triples_model_scores': all_triples_bert_scores,
    #         'per_gt_triple_scores_all': per_gt_triple_scores_all,
    #         'per_pred_triple_scores_all': per_pred_triple_scores_all,
    #         'gold_graphs': gold_graphs,
    #         'gt_triple_qids': gt_triple_qids
    #     }

    start = time.time()
    assert len(hash_ids) == len(gold_graphs) == len(all_triples_model_scores) == len(gt_triple_qids) \
           == len(precisions_MS) == len(recalls_MS) == len(f1s_MS) == len(per_gt_triple_scores_all) \
           == len(per_pred_triple_scores_all) == len(pred_graphs)
    to_ret_scores_per_triple = list()
    to_ret_scores_per_instance = list()
    to_ret_scores_per_triple_additional_stats = list()

    for (curr_hash_id, curr_gold_triples, curr_gold_triples_bert_scores,
         curr_gt_triple_qids,
         curr_precision_MS, curr_recall_MS, curr_f1_MS, curr_gold_triples_bert_scores_all,
         curr_pred_triples, curr_pred_triple_model_score_all) in \
            zip(hash_ids, gold_graphs, all_triples_model_scores, gt_triple_qids,
                precisions_MS, recalls_MS, f1s_MS, per_gt_triple_scores_all, pred_graphs,
                per_pred_triple_scores_all):
        assert len(curr_gold_triples) == len(curr_gold_triples_bert_scores)
        assert len(curr_gold_triples) == len(curr_gold_triples_bert_scores_all)
        # assert len(curr_gold_triples) == len(curr_hash_id)
        assert len(curr_gold_triples) == len(curr_gt_triple_qids)
        # assert len(curr_gold_triples) == len(curr_precision_MS)
        # assert len(curr_gold_triples) == len(curr_recall_MS)
        # assert len(curr_gold_triples) == len(curr_f1_MS)
        to_ret_scores_per_instance.append(
            {
                'hash_id': curr_hash_id,
                'tkgu_type': tkgu_type,
                # 'metric': f'gj-{model_alias}-precision',
                'metric': f'gj-precision',
                'evaluator_model': model_alias,
                'model': model,
                'granularity_level': 'instance',
                'score': curr_precision_MS
            }
        )
        to_ret_scores_per_instance.append(
            {
                'hash_id': curr_hash_id,
                'tkgu_type': tkgu_type,
                # 'metric': f'gj-{model_alias}-recall',
                'metric': f'gj-recall',
                'evaluator_model': model_alias,
                'model': model,
                'granularity_level': 'instance',
                'score': curr_recall_MS
            }
        )
        to_ret_scores_per_instance.append(
            {
                'hash_id': curr_hash_id,
                'tkgu_type': tkgu_type,
                # 'metric': f'gj-{model_alias}-f1',
                'metric': f'gj-f1',
                'evaluator_model': model_alias,
                'model': model,
                'granularity_level': 'instance',
                'score': curr_f1_MS
            }
        )
        for (
                curr_gold_triple,
                curr_curr_gt_triple_qids,
                curr_gold_triple_model_score,
                curr_gold_triple_model_score_all
        ) in \
                zip(
                    curr_gold_triples,
                    curr_gt_triple_qids,
                    curr_gold_triples_bert_scores,
                    curr_gold_triples_bert_scores_all
                ):
            curr_gold_triple_str = ' '.join(' '.join(curr_gold_triple).lower().strip().split())
            assert curr_gold_triple_str == curr_gold_triple_model_score[0]
            to_ret_scores_per_triple.append(
                {
                    'hash_id': curr_hash_id,
                    'triple_head': curr_curr_gt_triple_qids[0],
                    'triple_relation': curr_curr_gt_triple_qids[1],
                    'triple_tail': curr_curr_gt_triple_qids[2],
                    # 'triple_head_label': curr_gold_triple[0],
                    # 'triple_relation_label': curr_gold_triple[1],
                    # 'triple_tail_label': curr_gold_triple[2],
                    # 'metric': f'gj-{model_alias}-triple',
                    'metric': f'gj-triple',
                    'evaluator_model': model_alias,
                    'model': model,
                    'granularity_level': 'triple',
                    'score': curr_gold_triple_model_score[2],
                    'tkgu_type': tkgu_type
                    # f'score_triple_{model_alias}_{model}_{tkgu_type}': curr_gold_triple_model_score[2],
                }
                # {
                #         'hash_id': curr_hash_id,
                #         'triple_head_label': curr_gold_triple[0],
                #         'triple_relation_label': curr_gold_triple[1],
                #         'triple_tail_label': curr_gold_triple[2],
                #         f'score_triple_{model_alias}_{model}_{tkgu_type}': curr_gold_triple_model_score[2],
                #     }
            )
            # Skip additional stats when no predictions were matched (pred index is None)
            if curr_gold_triple_model_score_all[3] is not None:
                curr_best_pred_triple_matched_with_gold = curr_pred_triples[curr_gold_triple_model_score_all[3]]
                curr_best_pred_triple_matched_with_gold_score = float(curr_gold_triple_model_score_all[2])
                curr_best_pred_triple_matched_with_gold_str = curr_gold_triple_model_score_all[1]
                curr_gold_triple_str = curr_gold_triple_model_score_all[0]
                to_ret_scores_per_triple_additional_stats.append(
                    {
                        'hash_id': curr_hash_id,
                        'gt_triple_head': curr_curr_gt_triple_qids[0],
                        'gt_triple_relation': curr_curr_gt_triple_qids[1],
                        'gt_triple_tail': curr_curr_gt_triple_qids[2],
                        'gt_triple_head_label': curr_gold_triple[0],
                        'gt_triple_relation_label': curr_gold_triple[1],
                        'gt_triple_tail_label': curr_gold_triple[2],
                        'gt_triple_str': curr_gold_triple_str,
                        'pred_triple_head_label': curr_best_pred_triple_matched_with_gold[0],
                        'pred_triple_relation_label': curr_best_pred_triple_matched_with_gold[1],
                        'pred_triple_tail_label': curr_best_pred_triple_matched_with_gold[2],
                        'pred_triple_str': curr_best_pred_triple_matched_with_gold_str,
                        'pred_score': curr_best_pred_triple_matched_with_gold_score,
                        'metric': f'gj-triple-matches-w-gold',
                        'evaluator_model': model_alias,
                        'model': model,
                        'granularity_level': 'triple',
                        'score': float(curr_gold_triple_model_score[2]),
                        'tkgu_type': tkgu_type
                    }
                )
        # for curr_instance_idx, curr_pred_triple_model_score_all in \
        #         enumerate(per_pred_triple_scores_all):
        # curr_pred_triple_model_score_all = per_pred_triple_scores_all[curr_instance_idx]
        for (pred_matched_idx,
             (curr_pred_matched_str, curr_gt_matched_triple_str, curr_pred_matched_score,
              curr_gt_matched_triple_idx)) in \
                enumerate(curr_pred_triple_model_score_all):
            curr_gt_matched_triple_idx = int(curr_gt_matched_triple_idx)
            curr_gt_matched_triple_qids = curr_gt_triple_qids[curr_gt_matched_triple_idx]
            curr_pred_matched_score = float(curr_pred_matched_score)
            curr_gt_matched_triple_labels = curr_gold_triples[curr_gt_matched_triple_idx]
            curr_pred_matched_labels = curr_pred_triples[pred_matched_idx]
            to_ret_scores_per_triple_additional_stats.append(
                {
                    'hash_id': curr_hash_id,
                    'gt_triple_head': curr_gt_matched_triple_qids[0],
                    'gt_triple_relation': curr_gt_matched_triple_qids[1],
                    'gt_triple_tail': curr_gt_matched_triple_qids[2],
                    'gt_triple_head_label': curr_gt_matched_triple_labels[0],
                    'gt_triple_relation_label': curr_gt_matched_triple_labels[1],
                    'gt_triple_tail_label': curr_gt_matched_triple_labels[2],
                    'gt_triple_str': curr_gt_matched_triple_str,
                    'pred_triple_head_label': curr_pred_matched_labels[0],
                    'pred_triple_relation_label': curr_pred_matched_labels[1],
                    'pred_triple_tail_label': curr_pred_matched_labels[2],
                    'pred_triple_str': curr_pred_matched_str,
                    'pred_score': curr_pred_matched_score,
                    'metric': f'gj-triple-matches-w-pred',
                    'evaluator_model': model_alias,
                    'model': model,
                    'granularity_level': 'triple',
                    'score': curr_pred_matched_score,
                    'tkgu_type': tkgu_type
                }
            )

    assert len(precisions_MS) == len(gold_graphs) == len(recalls_MS) == len(f1s_MS)
    logger.debug(f'G-BertScore Precision Score: {sum(precisions_MS) / len(gold_graphs):.4f}')
    logger.debug(f'G-BertScore Recall Score: {sum(recalls_MS) / len(gold_graphs):.4f}')
    logger.debug(f'G-BertScore F1 Score: {sum(f1s_MS) / len(gold_graphs):.4f}\n')

    #
    end = time.time()
    elapsed = end - start
    logger.debug(f'elapsed_time_6: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')

    return {
        'scores_per_triple_additional_stats': to_ret_scores_per_triple_additional_stats,
        'scores_per_instance': to_ret_scores_per_instance,
        'scores_per_triple': to_ret_scores_per_triple
    }
    # return {
    #     'scores_per_instance': {
    #         'hash_id': hash_ids,
    #         f'score_precision_instance_{model_alias}_{model}_{tkgu_type}': precisions_MS,
    #         f'score_recall_instance_{model_alias}_{model}_{tkgu_type}': recalls_MS,
    #         f'score_f1_instance_{model_alias}_{model}_{tkgu_type}': f1s_MS
    #     },
    #     'scores_per_triple': to_ret_scores_per_triple
    # }
