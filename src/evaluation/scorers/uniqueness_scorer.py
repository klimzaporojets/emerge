import logging
import math
import time
from typing import List, Any

from sentence_transformers import SentenceTransformer

from concurrent.futures import as_completed, ThreadPoolExecutor
import itertools

from evaluation.scorers.misc.graph_matching import split_to_edges, get_tokens

logger = logging.getLogger(__name__)


def batch_iter(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def calculate_uniqueness_model(
        gt_pred_triples: List,
        tkgu_type: str,
        model: str,
        model_scorers: List[Any],
        scorer_workers: int,
        model_alias: str,
        phi: float,
        calculate_on_pred: bool,
        model_batch_size=64,
        shared_model_across_threads=False,
        implementation_class='bert_scorer'
):
    """
    Wrapper that runs `calculate_uniqueness_scores` in parallel batches.
    If max_workers <= 1, executes sequentially without threading.
    """
    logger.debug('+++ begin_calculate_uniqueness_scores_batched starting_batched +++')
    if not gt_pred_triples or len(gt_pred_triples) == 0:
        logger.debug('gt_pred_triples is empty, returning None')
        return None

    # ------------------------
    # Split into batches
    # ------------------------
    results_per_batch_model = []
    batches_inter = prepare_input_to_calculate(gt_pred_triples=gt_pred_triples,
                                               calculate_on_pred=calculate_on_pred)

    batch_size_model = math.ceil(len(batches_inter) / scorer_workers)
    batches_model = list(batch_iter(batches_inter, batch_size_model))

    if scorer_workers > 1:
        logger.debug(
            f'[INFO] Processing MODEL {len(gt_pred_triples)} (batches_inter: {len(batches_inter)}) instances in '
            f'{len(batches_model)} batches of size {batch_size_model} using {scorer_workers} workers. '
            f'implementation_class: {implementation_class}')
        with ThreadPoolExecutor(max_workers=scorer_workers) as model_executor:
            if implementation_class == 'sentence_transformer':
                model_futures = {
                    model_executor.submit(
                        calculate_uniqueness_scores_sentence_transformer, batch,
                        model_scorers[0] if shared_model_across_threads else model_scorers[idx],
                        model_batch_size,
                        phi,
                        calculate_on_pred
                    ): idx
                    for idx, batch in enumerate(batches_model)
                }
            else:
                raise RuntimeError(f'implementation_class not recognized: {implementation_class}')
        logger.debug('submitted all the models')
        for future in as_completed(model_futures):
            batch_idx = model_futures[future]
            try:
                curr_res = future.result()
                results_per_batch_model.append((batch_idx, curr_res))
            except Exception as e:
                logger.error(f" batch {batch_idx} failed: {e}")
                raise
    else:
        logger.debug('model_single_execution')
        if implementation_class == 'sentence_transformer':
            curr_res = calculate_uniqueness_scores_sentence_transformer(
                batch=batches_inter,
                st_scorer=model_scorers[0],
                st_scorer_batch_size=model_batch_size,
                phi=phi,
                calculate_on_pred=calculate_on_pred
            )
            results_per_batch_model.append((0, curr_res))
        else:
            raise RuntimeError(f'implementation_class not recognized: {implementation_class}')

    # ------------------------
    # Sort results in batch order
    # ------------------------
    results_per_batch_model.sort(key=lambda x: x[0])
    results_per_batch_model = [r for (_, r) in results_per_batch_model if r is not None]

    # ------------------------
    # Combine results across batches
    # ------------------------

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

    to_ret_final = return_concatenation_all_model_scores(
        model=model,
        tkgu_type=tkgu_type,
        model_alias=model_alias,
        **combined_return_results
    )

    return to_ret_final


def prepare_input_to_calculate(gt_pred_triples: List, calculate_on_pred: bool):
    start = time.time()

    global counter
    counter = 0
    #
    if len(gt_pred_triples) == 0:
        return None

    hash_ids, gt_triple_qids, gt_list, pred_list, gt_list_llm_asserted, _ = \
        map(list, zip(*gt_pred_triples))
    gold_graphs = [[x for x, flag in zip(sub1, sub2) if flag]
                   for sub1, sub2 in zip(gt_list, gt_list_llm_asserted)]
    filtered_triple_qids = [
        [qid for qid, flag in zip(qids, flags) if flag]
        for qids, flags in zip(gt_triple_qids, gt_list_llm_asserted)
    ]
    assert len(gold_graphs) == len(filtered_triple_qids)
    ####################################
    # Filter out instances where gold_graph is empty,
    # keeping all data structures aligned
    if calculate_on_pred:
        filtered = [
            (h, tq, g, p, a, gg)
            for h, tq, g, p, a, gg in zip(
                hash_ids,
                filtered_triple_qids,
                gt_list,
                pred_list,
                gt_list_llm_asserted,
                gold_graphs,
            )
            if len(p) > 0
        ]
        if len(filtered) == 0:
            return None
    else:
        filtered = [
            (h, tq, g, p, a, gg)
            for h, tq, g, p, a, gg in zip(
                hash_ids,
                filtered_triple_qids,
                gt_list,
                pred_list,
                gt_list_llm_asserted,
                gold_graphs,
            )
            if sum(a) > 0
        ]
        if len(filtered) == 0:
            return None

    # Unpack the filtered results back into separate lists
    hash_ids, gt_triple_qids, gt_list, pred_list, gt_list_llm_asserted, gold_graphs = \
        map(list, zip(*filtered))

    end = time.time()
    elapsed = end - start

    logger.debug(f'elapsed_time_1: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')

    ####################################

    gold_edges = split_to_edges(gold_graphs)
    end = time.time()
    elapsed = end - start
    logger.debug(f'elapsed_time_2: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')

    gold_graphs = [
        [list(t) for t in triples]
        for triples in gold_graphs
    ]

    if calculate_on_pred:
        pred_graphs = [
            [list(t) for t in triples]
            for triples in pred_list
        ]
        pred_edges = split_to_edges(pred_graphs)
        gold_tokens = get_tokens(gold_edges)
        pred_tokens = get_tokens(pred_edges)

        end = time.time()
        elapsed = end - start
        logger.debug(f'elapsed_time_3: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')

        assert len(gold_tokens) == len(pred_tokens) == len(gold_edges) == len(pred_edges) == \
               len(gold_graphs) == len(hash_ids)
        #
        to_ret = [
            (hi, gt, pt, ge, pe, pg, tq)
            for hi, gt, pt, ge, pe, pg, tq in zip(
                hash_ids,
                gold_tokens,
                pred_tokens,
                gold_edges,
                pred_edges,
                pred_graphs,
                gt_triple_qids
            )
        ]

        return to_ret
    else:
        gold_tokens = get_tokens(gold_edges)

        end = time.time()
        elapsed = end - start
        logger.debug(f'elapsed_time_3: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')

        assert len(gold_tokens) == len(gold_edges) == len(gold_graphs) == len(hash_ids)
        #
        to_ret = [
            (hi, gt, pt, ge, pe, pg, tq)
            for hi, gt, pt, ge, pe, pg, tq in zip(
                hash_ids,
                gold_tokens,
                [None] * len(gold_edges),
                gold_edges,
                [None] * len(gold_edges),
                [None] * len(gold_edges),
                gt_triple_qids
            )
        ]

        return to_ret


def calculate_uniqueness_scores_sentence_transformer(
        batch,
        st_scorer: SentenceTransformer,
        st_scorer_batch_size: int,
        phi: float,
        calculate_on_pred: bool
):
    hash_ids, gold_tokens, pred_tokens, gold_edges, pred_edges, pred_graphs, gt_triple_qids = \
        map(list, zip(*batch))
    logger.debug('Get bert score...')
    start = time.time()

    if calculate_on_pred:
        uniqueness_scores = \
            get_uniqueness_sentence_transformer_score_fast(
                all_edges=pred_edges,
                scorer=st_scorer,
                st_batch_size=st_scorer_batch_size,
                phi=phi
            )
    else:
        uniqueness_scores = \
            get_uniqueness_sentence_transformer_score_fast(
                all_edges=gold_edges,
                scorer=st_scorer,
                st_batch_size=st_scorer_batch_size,
                phi=phi
            )

    #
    end = time.time()
    elapsed = end - start
    logger.debug(f'elapsed_time_get_bert_score: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')
    assert len(hash_ids) == len(uniqueness_scores) == len(pred_graphs)

    return {
        'hash_ids': hash_ids,
        'uniqueness_scores': uniqueness_scores
    }


def get_uniqueness_sentence_transformer_score_fast(
    all_edges,
    scorer,
    st_batch_size: int,
    phi: float
):
    import torch
    from sentence_transformers import util

    # ---- flatten all triples ----
    flat_edges = []
    offsets = []
    cur = 0
    for edges in all_edges:
        offsets.append((cur, cur + len(edges)))
        flat_edges.extend(edges)
        cur += len(edges)

    # ---- encode once (device follows scorer) ----
    all_emb = scorer.encode(
        flat_edges,
        batch_size=st_batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    uniqueness_scores = []

    for start, end in offsets:
        n = end - start
        if n < 2:
            uniqueness_scores.append(1.0)
            continue

        emb = all_emb[start:end]
        cos = util.cos_sim(emb, emb)  # (n, n) on same device

        # take ONLY i<j entries (upper triangle, excluding diagonal)
        mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=cos.device), diagonal=1)
        sims = cos[mask]  # shape: (n*(n-1)/2,)

        unique_pairs_half = (sims < phi).sum().item()
        total_pairs_half = n * (n - 1) // 2

        # old code counts both (i,j) and (j,i); symmetric => factor cancels
        uniqueness_scores.append(unique_pairs_half / total_pairs_half)

    return uniqueness_scores


def return_concatenation_all_model_scores(
        hash_ids: List[str],
        uniqueness_scores: List,
        model: str,
        tkgu_type: str,
        model_alias: str
):
    start = time.time()
    assert len(hash_ids) == len(uniqueness_scores)

    assert len(uniqueness_scores) == len(hash_ids)
    #
    end = time.time()
    elapsed = end - start
    logger.debug(f'elapsed_time_6: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')
    scores_per_instance = list()
    for curr_hash_id, curr_uniqueness_score in zip(hash_ids, uniqueness_scores):
        scores_per_instance.append(
            {
                'hash_id': curr_hash_id,
                'score': curr_uniqueness_score,
                'metric': f'uniqueness',
                'evaluator_model': f'{model_alias}',
                'model': model,
                'tkgu_type': tkgu_type
            }
        )
    return {
        'scores_per_instance': scores_per_instance
    }