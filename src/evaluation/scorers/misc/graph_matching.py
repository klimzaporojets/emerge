#
import os
from typing import List

from sentence_transformers import util, SentenceTransformer

import logging
import time

import numpy as np
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from scipy.optimize import linear_sum_assignment
from spacy.lang.en import English
import networkx as nx
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score

level_name = os.getenv('LOG_LEVEL', 'INFO').upper()
level = logging.getLevelName(level_name)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=level,
)
logger = logging.getLogger(__name__)
# Create once, at module import
nlp = English()
tokenizer = nlp.tokenizer


def get_tokens(edges):
    return [
        [[tok.text for tok in doc] for doc in tokenizer.pipe(graphs)]
        for graphs in edges
    ]


def split_to_edges(graphs):
    processed_graphs = []
    for graph in graphs:
        # print(graph)
        # processed_graphs.append([";".join(str(triple)).lower().strip() for triple in graph])
        # kzaporoj - add split
        processed_graphs.append([
            " ".join(" ".join(triple).lower().strip().split())
            for triple in graph
        ])
        # processed_graphs.append([" ".join([str(elt).lower().strip() for elt in triple]split()) for triple in graph])
    return processed_graphs

def prepare_input_to_calculate_graph_scorers(gt_pred_triples: List):
    start = time.time()

    global counter
    counter = 0
    #
    if len(gt_pred_triples) == 0:
        return None

    hash_ids, gt_triple_qids, gt_list, pred_list, gt_list_llm_asserted, _ = \
        map(list, zip(*gt_pred_triples))
    # gold_graphs = [[x for x, flag in zip(sub1, sub2) if flag]
    #                for sub1, sub2 in zip(gt_list, gt_list_llm_asserted)]
    gold_graphs = [
        [x for x, flag in zip(gt, flags) if flag]
        for gt, flags in zip(gt_list, gt_list_llm_asserted)
    ]

    filtered_triple_qids = [
        [qid for qid, flag in zip(qids, flags) if flag]
        for qids, flags in zip(gt_triple_qids, gt_list_llm_asserted)
    ]
    ####################################
    # Filter out instances where gold_graph is empty,
    # keeping all data structures aligned
    filtered = [
        (h, tq, g, p, a, gg)
        for h, tq, g, p, a, gg in zip(
            hash_ids,
            # gt_triple_qids,
            filtered_triple_qids,
            gt_list,
            pred_list,
            gt_list_llm_asserted,
            gold_graphs,
        )
        if len(gg) > 0
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

    pred_graphs = [
        [list(t) for t in triples]
        for triples in pred_list
    ]
    gold_graphs = [
        [list(t) for t in triples]
        for triples in gold_graphs
    ]
    #

    gold_edges = split_to_edges(gold_graphs)
    pred_edges = split_to_edges(pred_graphs)
    end = time.time()
    elapsed = end - start
    logger.debug(f'elapsed_time_2: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')
    #
    gold_tokens = get_tokens(gold_edges)
    pred_tokens = get_tokens(pred_edges)
    #
    end = time.time()
    elapsed = end - start
    logger.debug(f'elapsed_time_3: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')

    assert len(gold_tokens) == len(pred_tokens) == len(gold_edges) == len(pred_edges) == \
           len(gold_graphs) == len(hash_ids) == len(gt_triple_qids)

    to_ret = [
        (hi, gt, pt, ge, pe, gg, tq, pg)
        for hi, gt, pt, ge, pe, gg, tq, pg in zip(
            hash_ids,
            gold_tokens,
            pred_tokens,
            gold_edges,
            pred_edges,
            gold_graphs,
            gt_triple_qids,
            pred_graphs
        )
    ]

    return to_ret

def prepare_input_to_calculate_ent_coverage(gt_pred_triples: List):
    start = time.time()

    global counter
    counter = 0
    #
    if len(gt_pred_triples) == 0:
        return None

    hash_ids, gt_triple_qids, gt_list, pred_list, gt_list_llm_asserted, gold_graphs_ent_types = \
        map(list, zip(*gt_pred_triples))
    #
    gold_graphs = [
        [x for x, flag in zip(gt, flags) if flag]
        for gt, flags in zip(gt_list, gt_list_llm_asserted)
    ]

    filtered_triple_qids = [
        [qid for qid, flag in zip(qids, flags) if flag]
        for qids, flags in zip(gt_triple_qids, gt_list_llm_asserted)
    ]

    filtered_entity_types = [
        [etype for etype, flag in zip(etypes, flags) if flag]
        for etypes, flags in zip(gold_graphs_ent_types, gt_list_llm_asserted)
    ]

    gold_graphs_entities_existing = list()
    gold_graphs_entities_emerging = list()
    gold_graphs_entities_all = list()
    pred_triples_entities = list()

    for curr_instance_triples_pred, curr_instance_triples_gt, curr_triples_types in \
            zip(pred_list, gold_graphs, filtered_entity_types):
        curr_inst_gt_emerging_entities = set()
        curr_inst_gt_existing_entities = set()
        curr_inst_gt_all_entities = set()
        curr_inst_predicted_entities = set()
        for curr_triple_gt, curr_triple_gt_type in zip(curr_instance_triples_gt, curr_triples_types):
            curr_inst_gt_all_entities.add(curr_triple_gt[0])
            curr_inst_gt_all_entities.add(curr_triple_gt[2])
            if curr_triple_gt_type[0]:
                curr_inst_gt_emerging_entities.add(curr_triple_gt[0])
            else:
                curr_inst_gt_existing_entities.add(curr_triple_gt[0])
            if curr_triple_gt_type[1]:
                curr_inst_gt_emerging_entities.add(curr_triple_gt[2])
            else:
                curr_inst_gt_existing_entities.add(curr_triple_gt[2])
        for curr_pred_triple in curr_instance_triples_pred:
            if curr_pred_triple[0] is not None \
                    and curr_pred_triple[0].strip() != '':
                curr_inst_predicted_entities.add(curr_pred_triple[0])

            if curr_pred_triple[2] is not None \
                    and curr_pred_triple[2].strip() != '':
                curr_inst_predicted_entities.add(curr_pred_triple[2])
        #
        curr_inst_predicted_entities = list(curr_inst_predicted_entities)
        curr_inst_gt_all_entities = list(curr_inst_gt_all_entities)
        curr_inst_gt_emerging_entities = list(curr_inst_gt_emerging_entities)
        curr_inst_gt_existing_entities = list(curr_inst_gt_existing_entities)
        #
        gold_graphs_entities_existing.append(curr_inst_gt_existing_entities)
        gold_graphs_entities_emerging.append(curr_inst_gt_emerging_entities)
        gold_graphs_entities_all.append(curr_inst_gt_all_entities)
        pred_triples_entities.append(curr_inst_predicted_entities)
        # logger.info('processing_triples')

    ####################################
    # Filter out instances where gold_graph is empty,
    # keeping all data structures aligned
    filtered = [
        (h, tq, g, p, a, gg,
         gg_ent_all, gg_ent_exist, gg_ent_emerg, pred_ent_all)
        for h, tq, g, p, a, gg, gg_ent_all, gg_ent_exist, gg_ent_emerg, pred_ent_all in zip(
            hash_ids,
            filtered_triple_qids,
            gt_list,
            pred_list,
            gt_list_llm_asserted,
            gold_graphs,
            gold_graphs_entities_all,
            gold_graphs_entities_existing,
            gold_graphs_entities_emerging,
            pred_triples_entities
        )
        if len(gg) > 0
    ]
    if len(filtered) == 0:
        return None
    # Unpack the filtered results back into separate lists
    hash_ids, gt_triple_qids, gt_list, pred_list, gt_list_llm_asserted, gold_graphs, \
        gold_graphs_entities_all, gold_graphs_entities_existing, gold_graphs_entities_emerging, \
        pred_triples_entities = \
        map(list, zip(*filtered))

    end = time.time()
    elapsed = end - start

    logger.debug(f'elapsed_time_1: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')

    ####################################

    pred_graphs = [
        [list(t) for t in triples]
        for triples in pred_list
    ]
    gold_graphs = [
        [list(t) for t in triples]
        for triples in gold_graphs
    ]
    #

    gold_edges = split_to_edges(gold_graphs)
    pred_edges = split_to_edges(pred_graphs)
    end = time.time()
    elapsed = end - start
    logger.debug(f'elapsed_time_2: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')

    gold_tokens = get_tokens(gold_edges)
    pred_tokens = get_tokens(pred_edges)
    end = time.time()
    elapsed = end - start
    logger.debug(f'elapsed_time_3: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')

    assert len(gold_tokens) == len(pred_tokens) == len(gold_edges) == len(pred_edges) == \
           len(gold_graphs) == len(hash_ids) == len(gt_triple_qids)

    to_ret = [
        (hi, gt, pt, ge, pe, gg, tq, gg_ent_all, gg_ent_exist, gg_ent_emerg, pred_ent_all)
        for hi, gt, pt, ge, pe, gg, tq, gg_ent_all, gg_ent_exist, gg_ent_emerg, pred_ent_all
        in zip(
            hash_ids,
            gold_tokens,
            pred_tokens,
            gold_edges,
            pred_edges,
            gold_graphs,
            gt_triple_qids,
            gold_graphs_entities_all,
            gold_graphs_entities_existing,
            gold_graphs_entities_emerging,
            pred_triples_entities
        )
    ]

    return to_ret


def get_sentence_transformer_score_fast(all_gold_edges,
                                        all_pred_edges,
                                        scorer: SentenceTransformer,
                                        st_batch_size: int):
    start = time.time()

    # -------------------------------------------------------
    # 1. Build unique sets for gold and pred edges
    # -------------------------------------------------------
    unique_gold = {}
    unique_pred = {}

    gold_id = 0
    pred_id = 0

    for sample_idx in range(len(all_gold_edges)):
        gold_edges = all_gold_edges[sample_idx]
        pred_edges = all_pred_edges[sample_idx]

        for g in gold_edges:
            if g not in unique_gold:
                unique_gold[g] = gold_id
                gold_id += 1
        for p in pred_edges:
            if p not in unique_pred:
                unique_pred[p] = pred_id
                pred_id += 1

    gold_list = list(unique_gold.keys())
    pred_list = list(unique_pred.keys())

    # -------------------------------------------------------
    # 2. Encode ONLY ONCE
    # -------------------------------------------------------
    logger.debug(f'=======================')
    logger.debug(f'gold_list_length: ({len(gold_list)}) -- st_batch_size {st_batch_size}')
    logger.debug(f'-----------------------------')
    logger.debug(f'pred_list_length: ({len(pred_list)}) -- st_batch_size {st_batch_size}')
    logger.debug(f'=======================')
    # print(f"Encoding {len(gold_list)} unique gold edges")
    gold_emb = scorer.encode(
        gold_list,
        batch_size=st_batch_size,
        convert_to_tensor=True,
        show_progress_bar=False
    )

    # print(f"Encoding {len(pred_list)} unique pred edges")

    pred_emb = scorer.encode(
        pred_list,
        batch_size=st_batch_size, convert_to_tensor=True,
        show_progress_bar=False
    )

    # -------------------------------------------------------
    # 3. Compute similarity matrix (pred × gold)
    # -------------------------------------------------------
    if len(gold_list) > 0 and len(pred_list) > 0:
        full_cos = util.cos_sim(pred_emb, gold_emb)  # shape: (#pred, #gold)

    # -------------------------------------------------------
    # 4. Now compute per-sample matrices using lookup
    # -------------------------------------------------------
    precisions, recalls, f1s = [], [], []
    per_triple_scores = []
    # kzaporoj - these two return all best scores without linear sum assignment
    per_gt_triple_scores_all = []
    per_pred_triple_scores_all = []

    for sample_idx in range(len(all_gold_edges)):
        gold_edges = all_gold_edges[sample_idx]
        pred_edges = all_pred_edges[sample_idx]

        G = len(gold_edges)
        P = len(pred_edges)

        # If no predicted or gold edges, skip costly work and avoid Hungarian errors
        if G == 0 or P == 0:
            per_triple_scores.append([(g, None, 0.0) for g in gold_edges])
            # Use None as pred index to signal "no prediction matched"
            per_gt_triple_scores_all.append([(g, None, 0.0, None) for g in gold_edges])
            per_pred_triple_scores_all.append([])
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue

        score_matrix = np.zeros((G, P))

        for gi, g in enumerate(gold_edges):
            g_idx = unique_gold[g]
            for pj, p in enumerate(pred_edges):
                p_idx = unique_pred[p]
                score_matrix[gi, pj] = float(full_cos[p_idx, g_idx])

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

        # Build triple-level results
        matched = []
        for gi, pj in zip(row_ind, col_ind):
            matched.append((gi, gold_edges[gi], pred_edges[pj], score_matrix[gi, pj]))
        matched.sort(key=lambda x: x[0])

        matched_dict = {gi: (g, p, s) for gi, g, p, s in matched}

        full_scores = []
        full_all_scores_per_gt = []
        full_all_scores_per_pred = []

        if G == 0 or P == 0:
            # keep default entries for golds (if any); no argmax possible
            for gi in range(G):
                full_scores.append(matched_dict.get(gi, (gold_edges[gi], None, 0.0)))
        else:
            for gi in range(G):
                full_scores.append(matched_dict.get(gi, (gold_edges[gi], None, 0.0)))
                max_score_pred_pos_idx = score_matrix[gi].argmax()
                max_pos_score = score_matrix[gi, max_score_pred_pos_idx]
                full_all_scores_per_gt.append(
                    (gold_edges[gi], pred_edges[max_score_pred_pos_idx], max_pos_score, max_score_pred_pos_idx)
                )

            for pi in range(P):
                max_score_gold_pos_idx = score_matrix[:, pi].argmax()
                max_pos_score = score_matrix[max_score_gold_pos_idx, pi]
                full_all_scores_per_pred.append(
                    (pred_edges[pi], gold_edges[max_score_gold_pos_idx], max_pos_score, max_score_gold_pos_idx)
                )

        per_triple_scores.append(full_scores)
        per_gt_triple_scores_all.append(full_all_scores_per_gt)
        per_pred_triple_scores_all.append(full_all_scores_per_pred)

        ##############################

        assigned_sum = score_matrix[row_ind, col_ind].sum()
        precision = assigned_sum / P
        recall = assigned_sum / G

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall + 1e-12))
    end = time.time()
    elapsed = end - start
    logger.debug(f'elapsed_time_TOTAL_get_st_score: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)')

    return precisions, recalls, f1s, per_triple_scores, per_gt_triple_scores_all, per_pred_triple_scores_all



def get_bert_score_fast(all_gold_edges, all_pred_edges, scorer: BERTScorer, bert_scorer_batch_size: int):
    import time
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    start = time.time()

    # -------------------------------------------------------
    # STEP 1 — Deduplicate all (gold, pred) text pairs
    # -------------------------------------------------------
    unique_pairs = {}  # (gold, pred) -> compact index
    unique_refs = []
    unique_cands = []

    # For reconstruction later:
    sample_pair_maps = []  # list of lists of (gi, pj, unique_idx)

    for sample_idx in range(len(all_gold_edges)):
        gold_edges = all_gold_edges[sample_idx]
        pred_edges = all_pred_edges[sample_idx]

        this_sample = []

        for gi, g in enumerate(gold_edges):
            for pj, p in enumerate(pred_edges):
                key = (g, p)
                if key not in unique_pairs:
                    idx = len(unique_refs)
                    unique_pairs[key] = idx
                    unique_refs.append(g)
                    unique_cands.append(p)
                else:
                    idx = unique_pairs[key]

                this_sample.append((gi, pj, idx))

        sample_pair_maps.append(this_sample)

    logger.debug("========== DEDUP BERTScore ==========")
    logger.debug(f"Unique refs: {len(unique_refs)}")
    logger.debug(f"Unique cands: {len(unique_cands)}")
    logger.debug("=====================================")

    # -------------------------------------------------------
    # STEP 2 — Compute BERTScore ONLY on unique pairs
    # -------------------------------------------------------
    if len(unique_refs) == 0 or len(unique_cands) == 0:
        # No pairs to score — return zero scores with proper per-entry entries
        # so downstream code (entity_coverage, graph_judge) finds aligned lists.
        per_triple = [
            [(g, None, 0.0) for g in gold_edges]
            for gold_edges in all_gold_edges
        ]
        per_gt_all = [
            [(g, None, 0.0, None) for g in gold_edges]
            for gold_edges in all_gold_edges
        ]
        return ([0.0] * len(all_gold_edges),
                [0.0] * len(all_gold_edges),
                [0.0] * len(all_gold_edges),
                per_triple,
                per_gt_all,
                [[] for _ in all_gold_edges])

    # Hard assert: no empty/whitespace-only strings
    bad_cands = [i for i, c in enumerate(unique_cands) if c is None or not str(c).strip()]
    bad_refs = [i for i, r in enumerate(unique_refs) if r is None or not str(r).strip()]

    if bad_cands:
        logger.warning(
            f'empty_candidate(s) length {len(bad_cands)} at indices '
            f'{bad_cands[:20]} (showing up to 20). '
            f'examples: {[repr(unique_cands[i]) for i in bad_cands[:5]]} '
            f'refs: {[repr(unique_refs[i]) for i in bad_cands[:5]]}'
        )

    if bad_refs:
        logger.warning(
            f'empty_reference(s) length {len(bad_refs)} at indices '
            f'{bad_refs[:20]} (showing up to 20). '
            f'examples: {[repr(unique_refs[i]) for i in bad_refs[:5]]}'
        )

    _, _, unique_F1 = scorer.score(
        cands=unique_cands,
        refs=unique_refs,
        batch_size=bert_scorer_batch_size,
        verbose=False
    )
    unique_F1 = unique_F1.cpu().numpy()

    logger.debug(f"BERTScore computed for {len(unique_F1)} unique pairs.")

    # -------------------------------------------------------
    # STEP 3 — Reconstruct full score matrices per sample
    # -------------------------------------------------------
    precisions, recalls, f1s = [], [], []
    per_triple_scores = []

    # kzaporoj - these two return all best scores without linear sum assignment
    per_gt_triple_scores_all = []
    per_pred_triple_scores_all = []

    for sample_idx, gold_edges, pred_edges in zip(
            range(len(all_gold_edges)),
            all_gold_edges,
            all_pred_edges
    ):
        G = len(gold_edges)
        P = len(pred_edges)

        score_matrix = np.zeros((G, P))

        # Fill in matrix from deduped index → F1 value lookup
        for gi, pj, unique_idx in sample_pair_maps[sample_idx]:
            score_matrix[gi, pj] = unique_F1[unique_idx]

        # Hungarian alignment
        row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

        matched = []
        for gi, pj in zip(row_ind, col_ind):
            matched.append((gi, gold_edges[gi], pred_edges[pj], score_matrix[gi, pj]))

        matched.sort(key=lambda x: x[0])
        matched_dict = {gi: (g, p, s) for gi, g, p, s in matched}

        full_scores = []
        full_all_scores_per_gt = []
        full_all_scores_per_pred = []

        # If one side is empty, argmax would fail. Keep defaults and skip argmax loops.
        if G == 0 or P == 0:
            for gi in range(G):
                full_scores.append(matched_dict.get(gi, (gold_edges[gi], None, 0.0)))
                # Use None as pred index to signal "no prediction matched"
                full_all_scores_per_gt.append((gold_edges[gi], None, 0.0, None))
        else:
            for gi in range(G):
                full_scores.append(matched_dict.get(gi, (gold_edges[gi], None, 0.0)))
                max_score_pred_pos_idx = score_matrix[gi].argmax()
                max_pos_score = score_matrix[gi, max_score_pred_pos_idx]
                full_all_scores_per_gt.append(
                    (gold_edges[gi], pred_edges[max_score_pred_pos_idx], max_pos_score, max_score_pred_pos_idx)
                )

            for pi in range(P):
                max_score_gold_pos_idx = score_matrix[:, pi].argmax()
                max_pos_score = score_matrix[max_score_gold_pos_idx, pi]
                full_all_scores_per_pred.append(
                    (pred_edges[pi], gold_edges[max_score_gold_pos_idx], max_pos_score, max_score_gold_pos_idx)
                )

        per_triple_scores.append(full_scores)
        per_gt_triple_scores_all.append(full_all_scores_per_gt)
        per_pred_triple_scores_all.append(full_all_scores_per_pred)
        # Sample-level metrics (same as original code)
        assigned_sum = score_matrix[row_ind, col_ind].sum()
        precision = assigned_sum / P if P > 0 else 0.0
        recall = assigned_sum / G if G > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

        if precision + recall > 0:
            f1s.append(2 * precision * recall / (precision + recall))
        else:
            f1s.append(0.0)

    elapsed = time.time() - start
    logger.debug(f"[FAST] TOTAL BERTScore time: {elapsed:.2f} s ({elapsed / 60:.2f} min)")

    return precisions, recalls, f1s, per_triple_scores, per_gt_triple_scores_all, per_pred_triple_scores_all


# Note: These graph matching metrics are computed by considering each graph as a set of edges and each edge as a
# sentence
def get_bleu_rouge(gold_tokens, pred_tokens, gold_sent, pred_sent):
    scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)

    precisions_bleu = []
    recalls_bleu = []
    f1s_bleu = []

    precisions_rouge = []
    recalls_rouge = []
    f1s_rouge = []

    # NEW: store per-triple aligned scores (BLEU and ROUGE separately)
    per_triple_scores_bleu = []  # per graph: [(gold, pred, score), ...]
    per_triple_scores_rouge = []

    for graph_idx in range(len(gold_tokens)):
        score_bleu = np.zeros((len(pred_tokens[graph_idx]), len(gold_tokens[graph_idx])))
        score_rouge = np.zeros((len(pred_tokens[graph_idx]), len(gold_tokens[graph_idx])))

        # Handle empty gold or pred tokens explicitly to avoid linear_sum_assignment errors
        if score_bleu.shape[0] == 0 or score_bleu.shape[1] == 0:
            precisions_bleu.append(0.0)
            recalls_bleu.append(0.0)
            f1s_bleu.append(0.0)
            precisions_rouge.append(0.0)
            recalls_rouge.append(0.0)
            f1s_rouge.append(0.0)
            per_triple_scores_bleu.append([(gold_sent[graph_idx][gi], None, 0.0)
                                           for gi in range(len(gold_tokens[graph_idx]))])
            per_triple_scores_rouge.append([(gold_sent[graph_idx][gi], None, 0.0)
                                            for gi in range(len(gold_tokens[graph_idx]))])
            continue

        for p_idx in range(len(pred_tokens[graph_idx])):
            for g_idx in range(len(gold_tokens[graph_idx])):
                score_bleu[p_idx, g_idx] = sentence_bleu(
                    [gold_tokens[graph_idx][g_idx]],
                    pred_tokens[graph_idx][p_idx],
                    smoothing_function=SmoothingFunction().method1,
                    auto_reweigh=True
                )
                score_rouge[p_idx, g_idx] = \
                    scorer_rouge.score(
                        gold_sent[graph_idx][g_idx],
                        pred_sent[graph_idx][p_idx]
                    )['rouge2'].precision

        def _scores(cost_matrix):
            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            precision = cost_matrix[row_ind, col_ind].sum() / cost_matrix.shape[0]
            recall = cost_matrix[row_ind, col_ind].sum() / cost_matrix.shape[1]
            f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
            return precision, recall, f1, row_ind, col_ind

        # --------------------------------------------------------
        # Compute BLEU scores + alignment
        # --------------------------------------------------------
        precision_bleu, recall_bleu, f1_bleu, row_ind_bleu, col_ind_bleu = _scores(score_bleu)
        precisions_bleu.append(precision_bleu)
        recalls_bleu.append(recall_bleu)
        f1s_bleu.append(f1_bleu)

        # FIXED VERSION — correct orientation: ri=row=pred, ci=col=gold
        matched_bleu = []
        for ri, ci in zip(row_ind_bleu, col_ind_bleu):
            matched_bleu.append((
                ci,  # gold index
                gold_sent[graph_idx][ci],  # gold sentence
                pred_sent[graph_idx][ri],  # pred sentence
                score_bleu[ri, ci]  # score pred→gold
            ))

        matched_bleu.sort(key=lambda x: x[0])  # reorder by gold index

        # NEW: now produce a full list covering ALL gold triples
        matched_dict = {gi: (g, p, s) for gi, g, p, s in matched_bleu}

        full_bleu_scores = []
        for gi in range(len(gold_tokens[graph_idx])):
            if gi in matched_dict:
                full_bleu_scores.append(matched_dict[gi])
            else:
                full_bleu_scores.append((gold_sent[graph_idx][gi], None, 0.0))

        per_triple_scores_bleu.append(full_bleu_scores)

        # --------------------------------------------------------
        # Compute ROUGE scores + alignment
        # --------------------------------------------------------
        precision_rouge, recall_rouge, f1_rouge, row_ind_rg, col_ind_rg = _scores(score_rouge)
        precisions_rouge.append(precision_rouge)
        recalls_rouge.append(recall_rouge)
        f1s_rouge.append(f1_rouge)

        # FIXED VERSION — correct orientation for ROUGE too
        matched_rg = []
        for ri, ci in zip(row_ind_rg, col_ind_rg):
            matched_rg.append((
                ci,
                gold_sent[graph_idx][ci],
                pred_sent[graph_idx][ri],
                score_rouge[ri, ci]
            ))

        matched_rg.sort(key=lambda x: x[0])

        # NEW: now produce a full list covering ALL gold triples
        matched_dict_rg = {gi: (g, p, s) for gi, g, p, s in matched_rg}

        full_rouge_scores = []
        for gi in range(len(gold_tokens[graph_idx])):
            if gi in matched_dict_rg:
                full_rouge_scores.append(matched_dict_rg[gi])
            else:
                full_rouge_scores.append((gold_sent[graph_idx][gi], None, 0.0))

        per_triple_scores_rouge.append(full_rouge_scores)

    return (
        np.array(precisions_rouge),
        np.array(recalls_rouge),
        np.array(f1s_rouge),
        np.array(precisions_bleu),
        np.array(recalls_bleu),
        np.array(f1s_bleu),
        per_triple_scores_rouge,  # NEW
        per_triple_scores_bleu  # NEW
    )
