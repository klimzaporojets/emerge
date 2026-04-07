import asyncio
import logging
import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

# from transformers import PreTrainedTokenizerBase
from unified_llm_client import UnifiedLLMClient

from dataset.emerge.utils.constants import ACTION_CATEGORY_ASSERT, ACTION_CATEGORY_DEPRECATE
from evaluation.scorers.misc.llm_calls import call_llm_and_return_parsed_result, \
    parse_returned_from_llm_as_a_judge_json_text
from evaluation.misc.utils import divide_list

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Global counter for progress logging
# ---------------------------------------------------------------------
counter = 0
counter_lock: Optional[asyncio.Lock] = None


async def increase_counter(number_to_increase: int, start_time: float) -> None:
    """
    Async-safe counter increment used for lightweight throughput logging.
    """
    global counter, counter_lock
    if counter_lock is None:
        counter_lock = asyncio.Lock()
    async with counter_lock:
        counter += number_to_increase
        curr_time = time.time()
        diff_mins = max((curr_time - start_time) / 60.0, 1e-6)
        avg_per_min = counter / diff_mins
        logger.info(
            f'total processed: {counter} in {diff_mins:.2f} mins, '
            f'avg: {avg_per_min:.2f} instances per min'
        )


# ---------------------------------------------------------------------
# Core batch LLM scoring logic
# ---------------------------------------------------------------------
async def calculate_llm_general_score_v13(
        current_instance: Dict[str, Any],
        # tokenizer: PreTrainedTokenizerBase,
        client: UnifiedLLMClient,
        prompt_paths: Dict,
        llm_backend: str,
        model: str,
        triples_per_prompt: int = 20
) -> List[Dict[str, Any]]:
    field_passage_timestamp = current_instance['field_passage_timestamp']
    delta_start_date = current_instance['field_delta_start_date']
    results = []

    hash_id = current_instance['field_hash_id']
    chunk = current_instance['passage']
    action_type = 'multiple'

    logger.debug(f'Processing job hash_id={hash_id}')

    def chunkify(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    attributes_per_action_category = {
        ACTION_CATEGORY_ASSERT: {
            'support',
            'specificity',
            'completeness',
            'relevance',
            'faithfulness',
            'novelty',
            'explanation',
            'triple_id',
        },
        ACTION_CATEGORY_DEPRECATE: {
            'support',
            'specificity',
            'completeness',
            'relevance',
            'faithfulness',
            'deprecation',
            'explanation',
            'triple_id',
        },
    }

    # Ported from v11: default values for robustness
    DEFAULT_ATTRIBUTE_VALUES = {
        'support': math.nan,
        'specificity': math.nan,
        'completeness': math.nan,
        'relevance': math.nan,
        'faithfulness': math.nan,
        'novelty': math.nan,
        'deprecation': math.nan,
        'explanation': None,
        'triple_id': None,
    }

    # Two modes: ASSERT (novelty) and DEPRECATE (deprecation)
    for action_category in [ACTION_CATEGORY_ASSERT, ACTION_CATEGORY_DEPRECATE]:

        triples = list(current_instance[action_category])
        if not triples:
            continue

        triples = sorted(triples, key=lambda t: t[:3])

        triple_labels_to_triples: Dict[Tuple[str, str, str], List[Tuple]] = defaultdict(list)
        for curr_triple in triples:
            triple_labels_to_triples[(curr_triple[0], curr_triple[1], curr_triple[2])].append(curr_triple)

        logger.debug(
            f'Action category {action_category} has {len(triples)} triples'
        )

        for chunk_idx, triples_chunk in enumerate(chunkify(triples, triples_per_prompt)):

            logger.debug(
                f'Processing chunk {chunk_idx + 1} with {len(triples_chunk)} triples '
                f'out of {len(triples)} total'
            )

            triples_chunk_only_labels = [tc[:3] for tc in triples_chunk]

            numbered_triples = [
                f'{j + 1}. {item[:3]}' for j, item in enumerate(triples_chunk)
            ]
            triples_str = '\n'.join(numbered_triples)

            triples_dict = {
                j + 1: triples_chunk_only_labels[j]
                for j in range(len(triples_chunk_only_labels))
            }

            triples_chunk_set = set(triples_chunk_only_labels)

            # ---- Ask LLM with batch prompt ----
            output_text = await call_llm_and_return_parsed_result(
                chunk=chunk,
                triples_str=triples_str,
                action_type=action_type,
                # tokenizer=tokenizer,
                client=client,
                prompt_paths=prompt_paths,
                action_category=action_category,
                llm_backend=llm_backend,
                model=model,
                passage_timestamp=field_passage_timestamp,
                delta_start_date=delta_start_date
            )

            all_identified_triples: Set = set()
            llm_triples_to_answer: Dict = {}
            llm_triples_to_prompt_responses: Dict = {}

            # ---- First attempt: parse batch JSON ----
            try:
                (
                    llm_triples_to_answer,
                    all_identified_triples,
                    llm_triples_to_prompt_responses,
                ) = parse_returned_from_llm_as_a_judge_json_text(
                    is_single=False,
                    all_identified_triples=all_identified_triples,
                    output_text=output_text,
                    llm_triples_to_answer=llm_triples_to_answer,
                    llm_triples_to_prompt_responses=llm_triples_to_prompt_responses,
                    triples=triples_dict,
                )
            except Exception:
                logger.warning(
                    f'[{action_category}] Batch parse failed — falling back to single triple mode.'
                )
                all_identified_triples = set()

            expected_attrs = attributes_per_action_category[action_category]

            all_identified_triples_verified = set()
            for curr_triple, curr_attributes in llm_triples_to_answer.items():
                if set(curr_attributes) == expected_attrs:
                    all_identified_triples_verified.add(curr_triple)
                else:
                    logger.warning(
                        f'[{action_category}] Incomplete attributes for {curr_triple}: '
                        f'{set(curr_attributes)} vs expected {expected_attrs}'
                    )

            missing_triples = triples_chunk_set - all_identified_triples_verified

            if missing_triples:
                logger.warning(
                    f'[{action_category}] {len(missing_triples)} triples not matched in multi run. '
                    f'len(all_identified_triples_verified): {len(all_identified_triples_verified)} '
                    f'output_text: {output_text}. '
                    f'Falling back to single-triple evaluation.'
                )

            # ---- Fallback for each missing triple ----
            for t in missing_triples:
                single_triple_str = f'{t}'
                output_single = await call_llm_and_return_parsed_result(
                    chunk=chunk,
                    triples_str=single_triple_str,
                    action_type='single',
                    # tokenizer=tokenizer,
                    client=client,
                    prompt_paths=prompt_paths,
                    action_category=action_category,
                    llm_backend=llm_backend,
                    model=model,
                    passage_timestamp=field_passage_timestamp,
                    delta_start_date=delta_start_date
                )

                (
                    llm_triples_to_answer,
                    all_identified_triples,
                    llm_triples_to_prompt_responses,
                ) = parse_returned_from_llm_as_a_judge_json_text(
                    is_single=True,
                    all_identified_triples=all_identified_triples,
                    output_text=output_single,
                    llm_triples_to_answer=llm_triples_to_answer,
                    llm_triples_to_prompt_responses=llm_triples_to_prompt_responses,
                    triples=t,
                )

            # Ported from v13: fill missing / remove extra attributes
            for curr_triple, curr_attributes in llm_triples_to_answer.items():
                missing = expected_attrs - set(curr_attributes)
                extra = set(curr_attributes) - expected_attrs

                if missing:
                    logger.warning(
                        f'[{action_category}] Missing attributes {missing} for triple {curr_triple}. '
                        f'Filling with defaults.'
                    )
                    for m in missing:
                        curr_attributes[m] = DEFAULT_ATTRIBUTE_VALUES[m]

                if extra:
                    logger.warning(
                        f'[{action_category}] Extra attributes {extra} for triple {curr_triple}. Removing.'
                    )
                    for e in extra:
                        curr_attributes.pop(e)

                assert set(curr_attributes) == expected_attrs

            expanded_triples_to_eval = {}
            for label_key, value in llm_triples_to_answer.items():
                for concrete_triple in triple_labels_to_triples[label_key]:
                    expanded_triples_to_eval[concrete_triple] = value

            results.append(
                {
                    'hash_id': hash_id,
                    'triples_to_eval': expanded_triples_to_eval,
                    'action_category': action_category,
                }
            )

    return results


# ---------------------------------------------------------------------
# Thread worker
# ---------------------------------------------------------------------
async def process_jobs_async(
        jobs_chunk: List[Dict[str, Any]],
        # tokenizer: PreTrainedTokenizerBase,
        client: UnifiedLLMClient,
        prompt_paths: Dict,
        start_time: float,
        llm_backend: str,
        model: str,
        triples_per_prompt: int,
) -> List[Dict[str, Any]]:
    '''
    Process a chunk of jobs sequentially (for logging / control),
    while LLM calls inside are async and batchable at the backend.
    '''
    thread_results: List[Dict[str, Any]] = []

    try:
        for curr_instance in jobs_chunk:
            batch_results = await calculate_llm_general_score_v13(
                current_instance=curr_instance,
                # tokenizer=tokenizer,
                client=client,
                prompt_paths=prompt_paths,
                llm_backend=llm_backend,
                model=model,
                triples_per_prompt=triples_per_prompt
            )

            await increase_counter(1, start_time)
            thread_results.extend(batch_results)

        return thread_results

    except Exception:
        raise

    finally:
        # no per-task teardown (shared client / tokenizer)
        pass


# ---------------------------------------------------------------------
# Public multithreaded entry point
# ---------------------------------------------------------------------


async def _calculate_llm_general_multi_thread(
        triples_with_passages: List[Dict[str, Any]],
        # tokenizer,
        # client: AsyncLLMClient,
        client: UnifiedLLMClient,
        prompt_paths: Dict,
        llm_backend: str,
        model: str,
        max_workers: int,
        triples_per_prompt: int,
) -> List[Dict[str, Any]]:
    """
    Async worker that spawns per-chunk tasks. Caller is expected to manage the event loop.
    """
    logger.debug(
        f'calculate_llm_general_multi_thread_v13: '
        f'{len(triples_with_passages)} jobs, '
        f'max_workers={max_workers}, chunk_size={triples_per_prompt}'
    )

    if not triples_with_passages:
        return []

    worker_count = max(1, max_workers)
    start_time = time.time()

    global counter, counter_lock
    counter = 0
    counter_lock = asyncio.Lock()

    job_chunks = divide_list(triples_with_passages, worker_count)

    tasks = [
        asyncio.create_task(
            process_jobs_async(
                jobs_chunk=chunk,
                # tokenizer=tokenizer,
                client=client,
                prompt_paths=prompt_paths,
                start_time=start_time,
                llm_backend=llm_backend,
                model=model,
                triples_per_prompt=triples_per_prompt,
            )
        )
        for chunk in job_chunks
    ]

    results_nested = await asyncio.gather(*tasks)

    all_results: List[Dict[str, Any]] = []
    for chunk_results in results_nested:
        all_results.extend(chunk_results)

    elapsed = time.time() - start_time
    logger.info(
        f'Finished {len(triples_with_passages)} jobs '
        f'in {elapsed:.2f}s'
    )

    return all_results


def calculate_llm_general_multi_thread(
        triples_with_passages: List[Dict[str, Any]],
        # tokenizer,
        client: UnifiedLLMClient,
        prompt_paths: Dict,
        llm_backend: str,
        model: str,
        triples_per_prompt: int = 20,
        max_workers: int = 8,
) -> List[Dict[str, Any]]:
    """
    Synchronous entry point used by the loaders. Internally spins up an asyncio
    loop to run the async implementation. If an AsyncLLMClient is provided, it
    will be opened with its context manager for the duration of the run.
    """

    async def _runner():
        async with client:
            return await _calculate_llm_general_multi_thread(
                triples_with_passages=triples_with_passages,
                # tokenizer=tokenizer,
                client=client,
                prompt_paths=prompt_paths,
                llm_backend=llm_backend,
                model=model,
                max_workers=max_workers,
                triples_per_prompt=triples_per_prompt,
            )

    return asyncio.run(_runner())

def calculate_batched_llm_general_v13(
        triples_with_passages: List[Dict],
        triples_per_prompt: int,
        max_workers: int,
        prompt_paths: Dict[str, str],
        llm_backend: str,
        model: str,
        triples_with_qids: bool,
        max_records: int = -1,
        tokenizer=None,
        client: UnifiedLLMClient = None,
) -> Dict[str, List]:
    '''
    Calculate LLM general scores for predicted triples.

    Notes:
    - triples_per_prompt = number of triples per prompt (NOT LLM batch size)
    - max_workers = number of concurrent async job streams (NOT threads)
    - This is a synchronous wrapper around an async implementation.
    '''

    # ------------------------------------------------------------------
    # Defensive checks
    # ------------------------------------------------------------------
    if client is None:
        raise ValueError('client must be provided (AsyncLLMClient or compatible)')

    # Prevent accidental nested event loop usage
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError(
            'calculate_batched_llm_general_v13 cannot be called from an async context. '
            'Use the async API directly instead.'
        )

    # ------------------------------------------------------------------
    # Input trimming
    # ------------------------------------------------------------------
    if max_records > -1:
        triples_with_passages = triples_with_passages[:max_records]

    # ------------------------------------------------------------------
    # Run async pipeline
    # ------------------------------------------------------------------
    def _runner():
        return calculate_llm_general_multi_thread(
            triples_with_passages=triples_with_passages,
            # tokenizer=tokenizer,
            triples_per_prompt=triples_per_prompt,
            max_workers=max_workers,
            client=client,
            prompt_paths=prompt_paths,
            llm_backend=llm_backend,
            model=model,
        )

    # to_ret_tuples = asyncio.run(_runner())
    to_ret_tuples = _runner()

    # ------------------------------------------------------------------
    # Helper to build output rows (DRY)
    # ------------------------------------------------------------------
    def _base_score_row(curr_tuple, curr_triple, metric, score, explanation):
        row = {
            'hash_id': curr_tuple['hash_id'],
            'prompt_type': curr_tuple['action_category'],
            'triple_head_label': curr_triple[0],
            'triple_relation_label': curr_triple[1],
            'triple_tail_label': curr_triple[2],
            'metric': metric,
            'model': None,
            'tkgu_type': None,
            'granularity_level': 'triple',
            'score': score,
            'score_explanation': explanation,
            'evaluator_model': model,
        }

        if triples_with_qids:
            row.update({
                'triple_head': curr_triple[3],
                'triple_relation': curr_triple[4],
                'triple_tail': curr_triple[5],
            })

        return row

    # ------------------------------------------------------------------
    # Flatten results
    # ------------------------------------------------------------------
    scores_per_triple: List[Dict[str, Any]] = []

    for curr_tuple in to_ret_tuples:
        for curr_triple, curr_llm_assessment in curr_tuple['triples_to_eval'].items():

            if triples_with_qids:
                assert len(curr_triple) == 6
            else:
                assert len(curr_triple) == 3

            # Core metrics
            for metric in ['support', 'specificity', 'relevance', 'completeness', 'faithfulness']:
                scores_per_triple.append(
                    _base_score_row(
                        curr_tuple,
                        curr_triple,
                        metric,
                        curr_llm_assessment[metric],
                        curr_llm_assessment['explanation'],
                    )
                )

            # Optional metrics
            if curr_llm_assessment.get('novelty') is not None:
                scores_per_triple.append(
                    _base_score_row(
                        curr_tuple,
                        curr_triple,
                        'novelty',
                        curr_llm_assessment['novelty'],
                        curr_llm_assessment['explanation'],
                    )
                )

            if curr_llm_assessment.get('deprecation') is not None:
                scores_per_triple.append(
                    _base_score_row(
                        curr_tuple,
                        curr_triple,
                        'deprecation',
                        curr_llm_assessment['deprecation'],
                        curr_llm_assessment['explanation'],
                    )
                )

    return {
        'scores_per_triple': scores_per_triple
    }
