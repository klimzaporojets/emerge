import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Set

from unified_llm_client import UnifiedLLMClient

from dataset.emerge.utils.constants import ACTION_CATEGORY_ASSERT, ACTION_CATEGORY_DEPRECATE
from evaluation.scorers.misc.llm_calls import call_llm_and_return_parsed_result, \
    parse_returned_from_llm_as_a_judge_text
from evaluation.misc.utils import divide_list

logger = logging.getLogger(__name__)
import time

# ---------------------------------------------------------------------
# Global counter for progress logging (MATCHES reference)
# ---------------------------------------------------------------------
counter = 0
counter_lock: Optional[asyncio.Lock] = None


def calculate_batched_factualness(
        triples_with_passages: List[Dict],
        triples_per_prompt: int,
        max_workers: int,
        prompt_paths: Dict[str, str],
        llm_backend: str,
        model: str,
        triples_with_qids: bool,
        max_records: int = -1,
        # tokenizer=None,
        client: UnifiedLLMClient = None,
) -> Dict[str, List]:
    '''
    Calculate factualness scores for predicted triples.

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
            'calculate_batched_factualness_v13 cannot be called from an async context. '
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
        return calculate_factualness_multi_thread_v13(
            pred_triples_with_passages=triples_with_passages,
            # tokenizer=tokenizer,
            triples_per_prompt=triples_per_prompt,
            max_workers=max_workers,
            client=client,
            prompt_paths=prompt_paths,
            llm_backend=llm_backend,
            model=model,
        )

    to_ret_tuples = _runner()

    # ------------------------------------------------------------------
    # Helper to build output rows (DRY, mirrors general version)
    # ------------------------------------------------------------------
    def _base_score_row(curr_tuple, curr_triple, score, explanation):
        assert isinstance(score, (int, float)), (f'score must be numeric, got {type(score)} '
                                                 f'with the value of {score}')

        row = {
            'hash_id': curr_tuple['hash_id'],
            'prompt_type': curr_tuple['action_category'],
            'triple_head_label': curr_triple[0],
            'triple_relation_label': curr_triple[1],
            'triple_tail_label': curr_triple[2],
            'metric': 'factualness',
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
        for curr_triple, curr_factualness in curr_tuple['triples_to_eval'].items():

            if triples_with_qids:
                assert len(curr_triple) == 6
            else:
                assert len(curr_triple) == 3

            scores_per_triple.append(
                _base_score_row(
                    curr_tuple=curr_tuple,
                    curr_triple=curr_triple,
                    score=curr_factualness,
                    explanation=curr_tuple['triples_to_explain'][curr_triple],
                )
            )

    return {
        'scores_per_triple': scores_per_triple
    }


async def increase_counter(number_to_increase: int, start_time: float) -> None:
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
# Core factualness logic (ASYNC wrapper)
# ---------------------------------------------------------------------
async def calculate_factualness_score_v13(
        current_instance: Dict[str, Any],
        # tokenizer: PreTrainedTokenizerBase,
        client: UnifiedLLMClient,
        prompt_paths: Dict,
        llm_backend: str,
        model: str,
        triples_per_prompt: int,
) -> List[Dict[str, Any]]:
    results = []

    hash_id = current_instance['field_hash_id']
    chunk = current_instance['passage']
    action_type = 'multiple'

    logger.debug(f'Processing job hash_id={hash_id}')

    def chunkify(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    for action_category in [ACTION_CATEGORY_ASSERT, ACTION_CATEGORY_DEPRECATE]:

        triples = list(current_instance[action_category])
        if not triples:
            continue

        triples = sorted(triples, key=lambda t: t[:3])

        triple_labels_to_triples: Dict[Tuple[str, str, str], List[Tuple]] = defaultdict(list)
        for t in triples:
            triple_labels_to_triples[t[:3]].append(t)

        for chunk_idx, triples_chunk in enumerate(chunkify(triples, triples_per_prompt)):

            triples_chunk_labels = [t[:3] for t in triples_chunk]
            # triples_chunk_set = set(triples_chunk_labels)

            triples_str = '\n'.join(
                f'{i + 1}. {t}' for i, t in enumerate(triples_chunk_labels)
            )

            # ---------- MULTI-TRIPLE CALL ----------
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
                passage_timestamp=0,
                delta_start_date='none'
            )

            all_identified_triples: Set[Tuple[str, str, str]] = set()
            llm_triples_to_answer: Dict = {}
            llm_triples_to_prompt_responses: Dict = {}
            # triples_chunk_only_labels = [tc[:3] for tc in triples_chunk]
            # numbered_triples = [
            #     f'{j + 1}. {item[:3]}' for j, item in enumerate(triples_chunk)
            # ]
            triples_chunk_set = set(triples_chunk_labels)

            try:
                (
                    llm_triples_to_answer,
                    all_identified_triples,
                    llm_triples_to_prompt_responses,
                ) = parse_returned_from_llm_as_a_judge_text(
                    is_single=False,
                    all_identified_triples=all_identified_triples,
                    output_text=output_text,
                    llm_triples_to_answer=llm_triples_to_answer,
                    llm_triples_to_prompt_responses=llm_triples_to_prompt_responses,
                    triples=triples_str,
                    triples_we_are_looking_for=triples_chunk_set
                )

            except Exception:
                logger.warning(
                    f'batch_parse_failed: [{action_category}] Batch parse failed — falling back to singles. '
                    f'Raw LLM output:\n{output_text}'
                )
                all_identified_triples = set()

            # ---------- FALLBACK TO SINGLE ----------
            missing_triples = triples_chunk_set - all_identified_triples

            if missing_triples:
                logger.warning(
                    f'missing_triples: [{action_category}] {len(missing_triples)} triples missing in multi run. '
                    f'Missing: {missing_triples}. '
                    f'Raw LLM output:\n{output_text}'
                )

            for t in missing_triples:
                output_single = await call_llm_and_return_parsed_result(
                    chunk=chunk,
                    triples_str=str(t),
                    action_type='single',
                    # tokenizer=tokenizer,
                    client=client,
                    prompt_paths=prompt_paths,
                    action_category=action_category,
                    llm_backend=llm_backend,
                    model=model,
                    passage_timestamp=0,
                    delta_start_date='none'
                )

                (
                    llm_triples_to_answer,
                    all_identified_triples,
                    llm_triples_to_prompt_responses,
                ) = parse_returned_from_llm_as_a_judge_text(
                    is_single=True,
                    all_identified_triples=all_identified_triples,
                    output_text=output_single,
                    llm_triples_to_answer=llm_triples_to_answer,
                    llm_triples_to_prompt_responses=llm_triples_to_prompt_responses,
                    triples=t,
                    triples_we_are_looking_for=triples_chunk_set
                )

            missing_triples = triples_chunk_set - all_identified_triples
            if missing_triples:
                logger.warning(
                    f'still_missing_triples: [{action_category}] {len(missing_triples)} triples missing in multi run. '
                    f'Missing: {missing_triples}. '
                    f'Raw LLM output:\n{output_text}'
                )
            # ---------- EXPAND TO CONCRETE TRIPLES ----------
            expanded_eval = {}
            expanded_explain = {}

            for label, val in llm_triples_to_answer.items():
                for concrete in triple_labels_to_triples[label]:
                    expanded_eval[concrete] = val

            for label, exp in llm_triples_to_prompt_responses.items():
                for concrete in triple_labels_to_triples[label]:
                    expanded_explain[concrete] = exp

            results.append(
                {
                    'hash_id': hash_id,
                    'triples_to_eval': expanded_eval,
                    'triples_to_explain': expanded_explain,
                    'action_category': action_category,
                }
            )

    return results


# ---------------------------------------------------------------------
# Async multi-worker factualness runner (MATCHES reference)
# ---------------------------------------------------------------------
async def _calculate_factualness_multi_thread_v13(
        pred_triples_with_passages: List[Dict[str, Any]],
        # tokenizer: PreTrainedTokenizerBase,
        client: UnifiedLLMClient,
        prompt_paths: Dict,
        llm_backend: str,
        model: str,
        max_workers: int,
        triples_per_prompt: int,
) -> List[Dict[str, Any]]:
    logger.debug(
        f'calculate_factualness_multi_thread_v13: '
        f'{len(pred_triples_with_passages)} jobs, '
        f'max_workers={max_workers}'
    )

    start_time = time.time()
    global counter, counter_lock
    counter = 0
    counter_lock = asyncio.Lock()

    worker_count = max(1, max_workers)
    job_chunks = divide_list(pred_triples_with_passages, worker_count)

    async def _run_chunk(chunk):
        chunk_results = []
        for inst in chunk:
            r = await calculate_factualness_score_v13(
                current_instance=inst,
                # tokenizer=tokenizer,
                client=client,
                prompt_paths=prompt_paths,
                llm_backend=llm_backend,
                model=model,
                triples_per_prompt=triples_per_prompt,
            )
            await increase_counter(1, start_time)
            chunk_results.extend(r)
        return chunk_results

    tasks = [asyncio.create_task(_run_chunk(c)) for c in job_chunks]
    results_nested = await asyncio.gather(*tasks)

    all_results: List[Dict[str, Any]] = []
    for r in results_nested:
        all_results.extend(r)

    return all_results


# ---------------------------------------------------------------------
# Public sync entry point (IDENTICAL SHAPE)
# ---------------------------------------------------------------------
def calculate_factualness_multi_thread_v13(
        pred_triples_with_passages: List[Dict[str, Any]],
        # tokenizer: PreTrainedTokenizerBase,
        client: UnifiedLLMClient,
        prompt_paths: Dict,
        llm_backend: str,
        model: str,
        triples_per_prompt: int = 20,
        max_workers: int = 8,
) -> List[Dict[str, Any]]:
    if not pred_triples_with_passages:
        return []

    async def _runner():
        async with client:
            return await _calculate_factualness_multi_thread_v13(
                pred_triples_with_passages=pred_triples_with_passages,
                # tokenizer=tokenizer,
                client=client,
                prompt_paths=prompt_paths,
                llm_backend=llm_backend,
                model=model,
                max_workers=max_workers,
                triples_per_prompt=triples_per_prompt,
            )

    return asyncio.run(_runner())

# if __name__ == '__main__':
#     """
#     Minimal smoke test for calculate_factualness_multi_thread_v13
#     using UnifiedLLMClient (OpenAI / Azure / vLLM compatible).
#     """
#
#     import os
#     import time
#     from transformers import AutoTokenizer
#
#     # ------------------------------------------------------------
#     # Config
#     # ------------------------------------------------------------
#     config = {
#         'llm_backend': 'openai',  # PIPELINE logic (prompt selection etc.)
#         'model': 'gpt-4o-mini',
#         'base_url': 'https://ai-research-proxy.azurewebsites.net',
#         'api_key_env': 'OPENAI_API_KEY',
#         'max_tokens': 1024,
#         'temperature': 0.0,
#         'concurrency': 8,
#         'timeout': 300,
#         'triples_per_prompt': 5,
#         'max_workers': 2,
#         'prompt_paths': {
#             'prompt_single_assert': 'prompts/prompts_general_metric/prompt_single_assert_template.txt',
#             'prompt_multi_assert': 'prompts/prompts_general_metric/prompt_multi_assert_template.txt',
#             'prompt_single_deprecate': 'prompts/prompts_general_metric/prompt_single_deprecate_template.txt',
#             'prompt_multi_deprecate': 'prompts/prompts_general_metric/prompt_multi_deprecate_template.txt',
#         },
#     }
#
#     # ------------------------------------------------------------
#     # Load prompt templates (same convention as rest of codebase)
#     # ------------------------------------------------------------
#     for k, path in list(config['prompt_paths'].items()):
#         with open(path, 'r', encoding='utf-8') as f:
#             config['prompt_paths'][k] = f.read()
#
#     # ------------------------------------------------------------
#     # Tokenizer (optional – can be None if prompts are pure text)
#     # ------------------------------------------------------------
#     tokenizer = None
#     # tokenizer = AutoTokenizer.from_pretrained(
#     #     'meta-llama/Llama-3.1-70B-Instruct'
#     # )
#
#     # ------------------------------------------------------------
#     # Unified LLM Client
#     # ------------------------------------------------------------
#     from utils.llm_client.unified_llm_client import UnifiedLLMClient
#
#     client = UnifiedLLMClient(
#         base_url=config['base_url'],
#         model=config['model'],
#         api_key=os.getenv(config['api_key_env']),
#         max_tokens=config['max_tokens'],
#         temperature=config['temperature'],
#         concurrency=config['concurrency'],
#         timeout=config['timeout'],
#         backend='openai',  # IMPORTANT: transport backend
#     )
#
#     # ------------------------------------------------------------
#     # Minimal test input
#     # ------------------------------------------------------------
#     test_jobs = [
#         {
#             'field_hash_id': 'test-1',
#             'passage': (
#                 'Douglas Adams was an English writer and humorist, '
#                 'best known for The Hitchhiker’s Guide to the Galaxy.'
#             ),
#             ACTION_CATEGORY_ASSERT: [
#                 ('Douglas Adams', 'instance of', 'human'),
#             ],
#             ACTION_CATEGORY_DEPRECATE: [],
#         }
#     ]
#
#     # ------------------------------------------------------------
#     # Run factualness pipeline
#     # ------------------------------------------------------------
#     # from your_module_path import calculate_factualness_multi_thread_v13
#
#     results = calculate_factualness_multi_thread_v13(
#         pred_triples_with_passages=test_jobs,
#         tokenizer=tokenizer,
#         client=client,
#         prompt_paths=config['prompt_paths'],
#         llm_backend=config['llm_backend'],  # PIPELINE semantic flag
#         model=config['model'],
#         triples_per_prompt=config['triples_per_prompt'],
#         max_workers=config['max_workers'],
#     )
#
#     # ------------------------------------------------------------
#     # Print results
#     # ------------------------------------------------------------
#     print('\n===== RESULTS =====')
#     for r in results:
#         print(r)
