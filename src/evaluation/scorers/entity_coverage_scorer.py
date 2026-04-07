#
import itertools
import logging
import math
import threading
from itertools import zip_longest
from typing import List, Any, Tuple
import time

from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .misc.graph_matching import \
    split_to_edges, get_tokens, get_bert_score_fast, \
    get_sentence_transformer_score_fast
from concurrent.futures import as_completed, ThreadPoolExecutor

logger = logging.getLogger(__name__)


def batch_iter(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def calculate_entity_coverage_model(
        gt_pred_triples: List,
        tkgu_type: str,
        model: str,
        model_scorers: List[Any],
        batches_inter: List,
        scorer_workers: int,
        model_alias: str,
        model_batch_size=64,
        shared_model_across_threads=False,
        implementation_class='bert_scorer',
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
    batch_size_model = math.ceil(len(batches_inter) / scorer_workers)
    batches_model = list(batch_iter(batches_inter, batch_size_model))
    logger.info(f'scorer_workers_inside_calculate_entity_coverage_model {scorer_workers} '
                f'len(batches_inter) = {len(batches_inter)}, '
                f'batch_size_model = {batch_size_model}')
    if scorer_workers > 1:
        logger.info(
            f'[INFO] entity_coverage_sorer_rocessing MODEL {len(gt_pred_triples)} '
            f'(batches_inter: {len(batches_inter)}) instances in '
            f'{len(batches_model)} batches of size {batch_size_model} using {scorer_workers} workers. '
            f'implementation_class: {implementation_class}')

        with ThreadPoolExecutor(max_workers=scorer_workers) as model_executor:
            if implementation_class == 'bert_scorer':
                model_futures = {
                    model_executor.submit(
                        calculate_entity_coverage_scores_bert, batch,
                        model_scorers[0] if shared_model_across_threads else model_scorers[idx],
                        model_batch_size,
                        # model_alias
                    ): idx
                    for idx, batch in enumerate(batches_model)
                }
            elif implementation_class == 'sentence_transformer':
                model_futures = {
                    model_executor.submit(
                        calculate_entity_coverage_scores_sentence_transformer, batch,
                        model_scorers[0] if shared_model_across_threads else model_scorers[idx],
                        model_batch_size,
                        # model_alias
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
                logger.error(f" BERT batch {batch_idx} failed: {e}")
                raise
    else:
        for batch_idx, curr_batch in tqdm(enumerate(batches_model),
                                          desc='entity_coverage_batches_processing',
                                          total=len(batches_model)):
            logger.debug('model_single_execution')
            if implementation_class == 'bert_scorer':
                curr_res = calculate_entity_coverage_scores_bert(
                    batch=curr_batch,
                    bert_scorer=model_scorers[0],
                    bert_scorer_batch_size=model_batch_size,
                    # model_alias=model_alias
                )
                results_per_batch_model.append((batch_idx, curr_res))
            elif implementation_class == 'sentence_transformer':
                curr_res = calculate_entity_coverage_scores_sentence_transformer(
                    batch=curr_batch,
                    st_scorer=model_scorers[0],
                    st_scorer_batch_size=model_batch_size,
                    # model_alias=model_alias
                )
                results_per_batch_model.append((batch_idx, curr_res))
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
                batch_res[key] + [None] * (len(batch_res['hash_ids']) - len(batch_res[key]))
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


def return_concatenation_all_model_scores(
        hash_ids: List[str],  # --> ok
        gold_graphs: List,  # --> ok
        precisions_MS: List,  # --> ok
        recalls_MS: List,  # --> ok
        f1s_MS: List,  # --> ok
        model: str,  # --> ok
        tkgu_type: str,  # --> ok
        gt_triple_qids: List,  # --> ok
        metric: List[str],
        model_alias: str,
        scores_per_gt_entity: List[List[Tuple[str, str, float]]]
):
    #### BEGIN examples or first values of the input parameters
    # (len: 72) scores_per_gt_entity --> [[('tennis', 'tennis', 1.0000003576278687),
    #                       ('Women's Tennis Association', 'WTA', 0.680737316608429)],
    #                       [('Ministry of Agriculture, ROC', 'Republic of China', 0.45354557037353516),
    #                       ('Taiwan', 'Chen Chi-chung', 0.4148215651512146)],
    #                       .....
    # (len: 216) hash_ids --> ['D2iMhF5oj6', '52frORJ_l4', 'H_Q07V7Z2e',
    # (len: 216) gold_graphs --> [[['Women's Tennis Association', 'sport', 'tennis']],
    #       [['Ministry of Agriculture, ROC', 'country', 'Taiwan']],
    #       [['Governor of South Dakota', 'has list', 'list of governors of South Dakota'],
    #       ['Dennis Daugaard', 'position held', 'Lieutenant Governor of South Dakota'],
    #       ['Governor of South Dakota', 'substitute/deputy/replacement of office/officeholder', 'Lieutenant Governor of South Dakota'],
    #       ...
    # (len: 216) precisions_MS --> [0.2401053820337568, 0.4341835677623749, 0.46866999400986564, ...
    # (len: 216) recalls_MS --> [0.8403688371181488, 0.4341835677623749, 0.8436059892177582, ...
    # (len: 216) f1s_MS --> [0.37349726094105373, 0.4341835677618749, 0.6025757065836538, ...
    # model --> 'relik-oie'
    # tkgu_type --> 'x-triples'
    # (len: 216) gt_triple_qids --> [[('Q948442', 'P641', 'Q847')],
    #       [('Q11626125', 'P17', 'Q865')],
    #       [('Q2626318', 'P2354', 'Q878640'),
    #       ('Q881199', 'P39', 'Q1853549'),
    #       ('Q2626318', 'P2098', 'Q1853549'),
    #       ('Q881199', 'P102', 'Q29468')],
    #       ...
    # (len: 216) metric --> ['ent-coverage-all', 'ent-coverage-all', 'ent-coverage-all', ...
    # model_alias --> 'all-mpnet-base-v2'
    #### END examples or first values of the input parameters
    start = time.time()
    assert len(hash_ids) >= len(scores_per_gt_entity)
    assert len(hash_ids) == len(gold_graphs) == len(metric) == len(gt_triple_qids) \
           == len(precisions_MS) == len(recalls_MS) == len(f1s_MS)
    to_ret_scores_per_triple = list()
    to_ret_scores_per_instance = list()
    for idx_data, (curr_hash_id, curr_gold_triples, curr_metric, curr_gt_triple_qids, \
                   curr_precision_MS, curr_recall_MS, curr_f1_MS, curr_scores_per_gt_entity) in \
            enumerate(zip_longest(hash_ids, gold_graphs, metric, gt_triple_qids,
                                  precisions_MS, recalls_MS, f1s_MS,
                                  scores_per_gt_entity, fillvalue=None)):
        assert len(curr_gold_triples) == len(curr_gt_triple_qids)
        if curr_scores_per_gt_entity is not None:
            # logger.info('curr_scores_per_gt_entity_not_none')
            gt_entity_label_to_score = dict()
            for idx_entity, curr_gt_entity_score in enumerate(curr_scores_per_gt_entity):
                curr_gt_entity_label = curr_gt_entity_score[0]
                curr_pred_entity_label = curr_gt_entity_score[1]
                curr_gt_entity_score_value = curr_gt_entity_score[2]
                assert curr_gt_entity_label not in gt_entity_label_to_score
                gt_entity_label_to_score[curr_gt_entity_label] = curr_gt_entity_score_value

            assert len(curr_gt_triple_qids) == len(curr_gold_triples)
            for idx_triple, (triple_labels, triple_qids) in \
                    enumerate(zip(curr_gold_triples, curr_gt_triple_qids)):
                assert triple_labels[0] in gt_entity_label_to_score
                assert triple_labels[2] in gt_entity_label_to_score

                to_ret_scores_per_triple.append(
                    {
                        'hash_id': curr_hash_id,
                        'triple_head': triple_qids[0],
                        'triple_relation': triple_qids[1],
                        'triple_tail': triple_qids[2],
                        # 'triple_head_label': curr_gold_triple[0],
                        # 'triple_relation_label': curr_gold_triple[1],
                        # 'triple_tail_label': curr_gold_triple[2],
                        'metric': f'entity_coverage',
                        'model': model,
                        'score': gt_entity_label_to_score[triple_labels[0]],
                        'tkgu_type': tkgu_type,
                        'evaluator_model': model_alias,
                        'granularity_level': 'entity-head'
                    }
                )
                to_ret_scores_per_triple.append(
                    {
                        'hash_id': curr_hash_id,
                        'triple_head': triple_qids[0],
                        'triple_relation': triple_qids[1],
                        'triple_tail': triple_qids[2],
                        # 'triple_head_label': curr_gold_triple[0],
                        # 'triple_relation_label': curr_gold_triple[1],
                        # 'triple_tail_label': curr_gold_triple[2],
                        'metric': f'entity_coverage',
                        'model': model,
                        'score': gt_entity_label_to_score[triple_labels[2]],
                        'tkgu_type': tkgu_type,
                        'evaluator_model': model_alias,
                        'granularity_level': 'entity-tail'
                    }
                )

        to_ret_scores_per_instance.append(
            {
                'hash_id': curr_hash_id,
                'tkgu_type': tkgu_type,
                'metric': f'{curr_metric}-precision',
                'model': model,
                'score': curr_precision_MS,
                'evaluator_model': model_alias,
                'granularity_level': 'instance'
            }
        )
        to_ret_scores_per_instance.append(
            {
                'hash_id': curr_hash_id,
                'tkgu_type': tkgu_type,
                'metric': f'{curr_metric}-recall',
                'model': model,
                'score': curr_recall_MS,
                'evaluator_model': model_alias,
                'granularity_level': 'instance'
            }
        )
        to_ret_scores_per_instance.append(
            {
                'hash_id': curr_hash_id,
                'tkgu_type': tkgu_type,
                'metric': f'{curr_metric}-f1',
                'model': model,
                'score': curr_f1_MS,
                'evaluator_model': model_alias,
                'granularity_level': 'instance'
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
        'scores_per_instance': to_ret_scores_per_instance,
        'scores_per_triple': to_ret_scores_per_triple
    }


def calculate_entity_coverage_scores_sentence_transformer(
        batch,
        st_scorer: SentenceTransformer,
        st_scorer_batch_size: int
):
    hash_ids, gold_tokens, pred_tokens, gold_edges, pred_edges, gold_graphs, gt_triple_qids, \
        gg_ent_all, gg_ent_exist, gg_ent_emerg, pred_ent_all = \
        map(list, zip(*batch))
    logger.debug('Get bert score...')
    start = time.time()

    precisions_TS, recalls_TS, f1s_TS, scores_per_entity, _, _ = \
        get_sentence_transformer_score_fast(
            all_gold_edges=gg_ent_all,
            all_pred_edges=pred_ent_all,
            scorer=st_scorer,
            st_batch_size=st_scorer_batch_size
        )
    assert len(hash_ids) == len(precisions_TS) == len(recalls_TS) == len(f1s_TS) == \
           len(gold_graphs) == len(gt_triple_qids) == len(gg_ent_all) == len(gg_ent_exist) == \
           len(gg_ent_emerg) == len(pred_ent_all)

    to_ret_hash_ids = hash_ids
    to_ret_metric = [f'ent-coverage-all'] * len(hash_ids)
    to_ret_precisions_MS = precisions_TS
    to_ret_recalls_MS = recalls_TS
    to_ret_f1s_MS = f1s_TS
    to_ret_gold_graphs = gold_graphs
    to_ret_gt_triple_qids = gt_triple_qids

    # we only need to calculate scores_per_entity once for all entities (above), do not
    # distinguish emerging, etc.... this can be done later filter out in metrics
    # pandas dataframe inside s14_experiments_stats_v13.ipynb for example....
    to_ret_scores_per_entity_all = scores_per_entity

    precisions_TS, recalls_TS, f1s_TS, _, _, _ = \
        get_sentence_transformer_score_fast(
            all_gold_edges=gg_ent_emerg,
            all_pred_edges=pred_ent_all,
            scorer=st_scorer,
            st_batch_size=st_scorer_batch_size
        )
    assert len(hash_ids) == len(precisions_TS) == len(recalls_TS) == len(f1s_TS) == \
           len(gold_graphs) == len(gt_triple_qids) == len(gg_ent_all) == len(gg_ent_exist) == \
           len(gg_ent_emerg) == len(pred_ent_all)

    to_ret_hash_ids = to_ret_hash_ids + hash_ids
    to_ret_metric = to_ret_metric + [f'ent-coverage-emerg'] * len(hash_ids)
    to_ret_precisions_MS = to_ret_precisions_MS + precisions_TS
    to_ret_recalls_MS = to_ret_recalls_MS + recalls_TS
    to_ret_f1s_MS = to_ret_f1s_MS + f1s_TS
    to_ret_gold_graphs = to_ret_gold_graphs + gold_graphs
    to_ret_gt_triple_qids = to_ret_gt_triple_qids + gt_triple_qids

    precisions_TS, recalls_TS, f1s_TS, _, _, _ = \
        get_sentence_transformer_score_fast(
            all_gold_edges=gg_ent_exist,
            all_pred_edges=pred_ent_all,
            scorer=st_scorer,
            st_batch_size=st_scorer_batch_size
        )
    assert len(hash_ids) == len(precisions_TS) == len(recalls_TS) == len(f1s_TS) == \
           len(gold_graphs) == len(gt_triple_qids) == len(gg_ent_all) == len(gg_ent_exist) == \
           len(gg_ent_emerg) == len(pred_ent_all)

    to_ret_hash_ids = to_ret_hash_ids + hash_ids
    to_ret_metric = to_ret_metric + [f'ent-coverage-exist'] * len(hash_ids)
    to_ret_precisions_MS = to_ret_precisions_MS + precisions_TS
    to_ret_recalls_MS = to_ret_recalls_MS + recalls_TS
    to_ret_f1s_MS = to_ret_f1s_MS + f1s_TS
    to_ret_gold_graphs = to_ret_gold_graphs + gold_graphs
    to_ret_gt_triple_qids = to_ret_gt_triple_qids + gt_triple_qids

    end = time.time()
    elapsed = end - start
    logger.debug(f'elapsed_time_get_bert_score: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')
    assert len(hash_ids) == len(precisions_TS) == len(recalls_TS) == len(f1s_TS) == \
           len(gold_graphs) == len(gt_triple_qids) == len(gg_ent_all) == len(gg_ent_exist) == \
           len(gg_ent_emerg) == len(pred_ent_all)

    assert len(to_ret_hash_ids) == len(to_ret_metric) == len(to_ret_precisions_MS) == len(to_ret_recalls_MS) \
           == len(to_ret_f1s_MS) == len(to_ret_gold_graphs) == len(to_ret_gt_triple_qids) \
           == len(to_ret_scores_per_entity_all) * 3  # this can be removed, in theory it should be * 3 though
    to_ret_ent_all = {
        'hash_ids': to_ret_hash_ids,
        'metric': to_ret_metric,
        'precisions_MS': to_ret_precisions_MS,
        'recalls_MS': to_ret_recalls_MS,
        'f1s_MS': to_ret_f1s_MS,
        'gold_graphs': to_ret_gold_graphs,
        'gt_triple_qids': to_ret_gt_triple_qids,
        'scores_per_gt_entity': to_ret_scores_per_entity_all
    }

    return to_ret_ent_all


# ===== CHANGES START: add a global counter + lock (module-level) =====
_CALC_ENTITY_COVERAGE_BERT_INSTANCES = 0
_CALC_ENTITY_COVERAGE_BERT_LOCK = threading.Lock()


# ===== CHANGES END =====


def calculate_entity_coverage_scores_bert(
        batch,
        bert_scorer: BERTScorer,
        bert_scorer_batch_size: int
):
    hash_ids, gold_tokens, pred_tokens, gold_edges, pred_edges, gold_graphs, gt_triple_qids, \
        gg_ent_all, gg_ent_exist, gg_ent_emerg, pred_ent_all = \
        map(list, zip(*batch))

    logger.debug('Get bert score...')
    start = time.time()

    precisions_BS, recalls_BS, f1s_BS, scores_per_entity, _, _ = get_bert_score_fast(
        all_gold_edges=gg_ent_all,
        all_pred_edges=pred_ent_all,
        scorer=bert_scorer,
        bert_scorer_batch_size=bert_scorer_batch_size
    )
    to_ret_scores_per_entity_all = scores_per_entity
    end = time.time()
    elapsed = end - start
    logger.debug(f'elapsed_time_get_bert_score: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')
    assert (len(hash_ids) == len(precisions_BS) == len(recalls_BS) == len(f1s_BS) ==
            len(pred_ent_all) == len(gg_ent_all) == len(gg_ent_exist) == len(gg_ent_emerg) ==
            len(gold_graphs) == len(gt_triple_qids))

    to_ret_hash_ids = hash_ids
    to_ret_metric = [f'ent-coverage-all'] * len(hash_ids)
    to_ret_precisions_MS = precisions_BS
    to_ret_recalls_MS = recalls_BS
    to_ret_f1s_MS = f1s_BS
    to_ret_gold_graphs = gold_graphs
    to_ret_gt_triple_qids = gt_triple_qids

    precisions_BS, recalls_BS, f1s_BS, _, _, _ = get_bert_score_fast(
        all_gold_edges=gg_ent_emerg,
        all_pred_edges=pred_ent_all,
        scorer=bert_scorer,
        bert_scorer_batch_size=bert_scorer_batch_size
    )
    assert (len(hash_ids) == len(precisions_BS) == len(recalls_BS) == len(f1s_BS) ==
            len(pred_ent_all) == len(gg_ent_all) == len(gg_ent_exist) == len(gg_ent_emerg) ==
            len(gold_graphs) == len(gt_triple_qids))

    to_ret_hash_ids = to_ret_hash_ids + hash_ids
    to_ret_metric = to_ret_metric + [f'ent-coverage-emerg'] * len(hash_ids)
    to_ret_precisions_MS = to_ret_precisions_MS + precisions_BS
    to_ret_recalls_MS = to_ret_recalls_MS + recalls_BS
    to_ret_f1s_MS = to_ret_f1s_MS + f1s_BS
    to_ret_gold_graphs = to_ret_gold_graphs + gold_graphs
    to_ret_gt_triple_qids = to_ret_gt_triple_qids + gt_triple_qids

    precisions_BS, recalls_BS, f1s_BS, _, _, _ = get_bert_score_fast(
        all_gold_edges=gg_ent_exist,
        all_pred_edges=pred_ent_all,
        scorer=bert_scorer,
        bert_scorer_batch_size=bert_scorer_batch_size
    )
    assert (len(hash_ids) == len(precisions_BS) == len(recalls_BS) == len(f1s_BS) ==
            len(pred_ent_all) == len(gg_ent_all) == len(gg_ent_exist) == len(gg_ent_emerg) ==
            len(gold_graphs) == len(gt_triple_qids))

    to_ret_hash_ids = to_ret_hash_ids + hash_ids
    to_ret_metric = to_ret_metric + [f'ent-coverage-exist'] * len(hash_ids)
    to_ret_precisions_MS = to_ret_precisions_MS + precisions_BS
    to_ret_recalls_MS = to_ret_recalls_MS + recalls_BS
    to_ret_f1s_MS = to_ret_f1s_MS + f1s_BS
    to_ret_gold_graphs = to_ret_gold_graphs + gold_graphs
    to_ret_gt_triple_qids = to_ret_gt_triple_qids + gt_triple_qids
    assert len(to_ret_hash_ids) == len(to_ret_metric) == len(to_ret_precisions_MS) == len(to_ret_recalls_MS) \
           == len(to_ret_f1s_MS) == len(to_ret_gold_graphs) == len(to_ret_gt_triple_qids) \
           == len(to_ret_scores_per_entity_all) * 3  # this can be removed, in theory it should be * 3 though
    to_ret_ent_all = {
        'hash_ids': to_ret_hash_ids,
        'metric': to_ret_metric,
        'precisions_MS': to_ret_precisions_MS,
        'recalls_MS': to_ret_recalls_MS,
        'f1s_MS': to_ret_f1s_MS,
        'gold_graphs': to_ret_gold_graphs,
        'gt_triple_qids': to_ret_gt_triple_qids,
        'scores_per_gt_entity': to_ret_scores_per_entity_all
    }

    # ===== CHANGES START: increment + log counter (thread-safe) =====
    global _CALC_ENTITY_COVERAGE_BERT_INSTANCES
    with _CALC_ENTITY_COVERAGE_BERT_LOCK:
        _CALC_ENTITY_COVERAGE_BERT_INSTANCES += len(to_ret_hash_ids)
        _call_idx = _CALC_ENTITY_COVERAGE_BERT_INSTANCES

    # tweak as you like; leaving as a small constant keeps changes minimal
    logger.info(f'nr_processed_coverage_entity_instances: '
                f'{_CALC_ENTITY_COVERAGE_BERT_INSTANCES}')
    # ===== CHANGES END =====

    return to_ret_ent_all
