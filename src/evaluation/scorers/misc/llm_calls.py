import asyncio
import json
import logging
import re
import traceback
from typing import Dict, Set, Tuple, Any
from datetime import datetime, timezone

from unified_llm_client import UnifiedLLMClient
import misc.llms.llm_cache as llmc

from dataset.emerge.utils.constants import ACTION_CATEGORY_ADD, ACTION_CATEGORY_ASSERT, ACTION_CATEGORY_DEPRECATE
from misc.llms.llm_prompting import get_prompt_from_config

_PROMPT_CACHE: Dict[str, str] = {}

logger = logging.getLogger(__name__)
_llm_cache_lock = asyncio.Lock()

def _build_prompt(action_category, action_type,
                  chunk, triples_str, prompt_paths: Dict[str, str],
                  passage_timestamp: int,
                  delta_start_date: str
                  ):
    """Build the correct prompt based on type/category/action."""
    name_map = {
        ('multiple', ACTION_CATEGORY_ADD): 'prompt_multi_assert',
        ('single', ACTION_CATEGORY_ADD): 'prompt_single_assert',
        ('multiple', ACTION_CATEGORY_ASSERT): 'prompt_multi_assert',
        ('single', ACTION_CATEGORY_ASSERT): 'prompt_single_assert',
        ('multiple', ACTION_CATEGORY_DEPRECATE): 'prompt_multi_deprecate',
        ('single', ACTION_CATEGORY_DEPRECATE): 'prompt_single_deprecate',
    }

    key = (action_type, action_category)
    if key not in name_map:
        raise RuntimeError(f'Unrecognized action_type/category: {action_type}, {action_category}')

    # --- Load actual file contents here ---
    prompt_contents = {}

    for curr_key, path in prompt_paths.items():
        if path not in _PROMPT_CACHE:
            with open(path, 'r', encoding='utf-8') as f:
                _PROMPT_CACHE[path] = f.read()
        prompt_contents[curr_key] = _PROMPT_CACHE[path]

    def format_llm_date(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()

    text_creation_date = format_llm_date(passage_timestamp)
    #
    evaluation_window_start = delta_start_date

    to_ret = get_prompt_from_config(
        chunk=chunk,
        chunk_formatted_date='none',
        triples_string=triples_str,
        prompt_template_content_name=name_map[key],
        prompt_contents=prompt_contents,
        text_creation_date=text_creation_date,
        evaluation_window_start=evaluation_window_start
    )
    return to_ret

def extract_json_anywhere(text: str):
    """
    Extracts the first valid JSON array or object from LLM output.
    Handles:
      - prepended text
      - appended text
      - markdown
      - multiple objects
    Returns:
      dict or list
    """

    # 1) try whole text first
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) try JSON arrays
    array_matches = re.findall(r'\[\s*{[\s\S]*?}\s*\]', text)
    for candidate in array_matches:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 3) try JSON objects
    object_matches = re.findall(r'{[\s\S]*?}', text)
    for candidate in object_matches:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    logger.error(f'no_valid_json_found in LLM output: {text}.')
    raise ValueError(f'no_valid_json_found in LLM output: {text}.')

def parse_returned_from_llm_as_a_judge_json_text(
        is_single: bool,
        all_identified_triples: set,
        output_text: str,
        llm_triples_to_answer: dict,
        llm_triples_to_prompt_responses: dict,
        triples
):
    """
    JSON-based parser for LLM-as-a-judge evaluation.
    Supports:
      • batch mode: JSON array → multiple triples
      • single mode: JSON object → one triple

    triples:
      - batch mode  → dict: { triple_id: (head, relation, tail) }
      - single mode → tuple: (head, relation, tail)
    """

    # ----------------------------------------------------------
    # MULTI-TRIPLE MODE
    # ----------------------------------------------------------
    if not is_single:

        try:
            parsed = extract_json_anywhere(output_text)
        except Exception as e:
            # logger.error(f"Failed to parse batch JSON: {e}\nOutput:\n{output_text}")
            # raise
            logger.error(f'failed_to_parse_batch_json: {e}\nOutput:\n{output_text}')
            return (
                llm_triples_to_answer,
                all_identified_triples,
                llm_triples_to_prompt_responses,
            )

        if not isinstance(parsed, list):
            raise ValueError(f'Batch mode expected a list, got: {type(parsed)}')

        matches = []

        for obj in parsed:
            triple_id = obj.get('triple_id')
            if triple_id is None:
                logger.warning(f'Batch JSON object missing "triple_id": {obj}')

            if triple_id not in triples:
                logger.warning(f'triple_id {triple_id} not found in triples dict.')

            head, relation, tail = triples[triple_id]
            triple_key = (head, relation, tail)

            all_identified_triples.add(triple_key)

            # store full object
            llm_triples_to_answer[triple_key] = obj

            explanation = obj.get('explanation', '').strip()
            llm_triples_to_prompt_responses[triple_key] = explanation

            matches.append((triple_id, triple_key, obj))

    # ----------------------------------------------------------
    # SINGLE-TRIPLE MODE
    # ----------------------------------------------------------
    else:
        if not isinstance(triples, tuple):
            raise TypeError('In single mode, "triples" must be a tuple.')

        # head, relation, tail = triples

        try:
            parsed = extract_json_anywhere(output_text)
        except Exception as e:
            logger.error(f'failed_to_parse_single_json: {e}\nOutput:\n{output_text}')
            return (
                llm_triples_to_answer,
                all_identified_triples,
                llm_triples_to_prompt_responses,
            )

        if not isinstance(parsed, dict):
            # raise ValueError(f'Expected JSON object in single mode, got: {type(parsed)}')
            logger.error(f'expected_json_object_in_single_mode, got: {type(parsed)}')
            return (
                llm_triples_to_answer,
                all_identified_triples,
                llm_triples_to_prompt_responses,
            )

        triple_key = triples
        all_identified_triples.add(triple_key)

        llm_triples_to_answer[triple_key] = parsed
        llm_triples_to_prompt_responses[triple_key] = parsed.get('explanation', '').strip()

        matches = [(1, triple_key, parsed)]

    # ----------------------------------------------------------
    # LOGGING
    # ----------------------------------------------------------
    logger.debug('*****************************************')
    logger.debug(
        f'parse_returned_llm_json_v13\n'
        f'is_single: {is_single}\n'
        f'output_text:\n{output_text}\n'
        f'matches: {matches}\n'
        f'triples: {triples}\n'
        f'llm_triples_to_answer: {llm_triples_to_answer}\n'
        f'all_identified_triples: {all_identified_triples}\n'
        f'llm_triples_to_prompt_responses: {llm_triples_to_prompt_responses}\n'
    )
    logger.debug('*****************************************')

    return (
        llm_triples_to_answer,
        all_identified_triples,
        llm_triples_to_prompt_responses,
    )


def parse_returned_from_llm_as_a_judge_text(
        is_single: bool,
        all_identified_triples: Set,
        output_text: str,
        llm_triples_to_answer: Dict[Tuple, Any],
        llm_triples_to_prompt_responses: Dict[Tuple, Any],
        triples: str | Tuple[str, str, str],
        triples_we_are_looking_for: Set
):
    if not is_single:

        # --- Fully robust multi-triple regex supporting:
        # single quotes, double quotes, smart quotes,
        # mixed quote styles, apostrophes inside values,
        # YES/NO with ., :, **, dashes, whitespace,
        # and multi-line or same-line explanations.
        pattern = (
            r'\s*\d+\.\s*'  # item number
            r"\(\s*(['\"“”‘’])(.+?)\1"  # head opening quote + content + same quote
            r"\s*,\s*(['\"“”‘’])(.+?)\3"  # relation
            r"\s*,\s*(['\"“”‘’])(.+?)\5\s*\)"  # tail
            r"\s*[-–—]?\s*(?:\*\*)?(YES|NO)(?:\*\*)?\.?:?\s*"  # YES/NO variants
            r"(.*?)"  # explanation (greedy but bounded by lookahead)
            r"(?=(?:\n+\s*\d+\.\s*)|(?:\s*\d+\.\s*)|\Z)"  # next item or end
        )
        if not isinstance(output_text, str):
            logger.error(f'output_text_wrong_multi! {output_text} ')
            return (
                llm_triples_to_answer,
                all_identified_triples,
                llm_triples_to_prompt_responses,
            )
        matches = re.findall(pattern, output_text, re.S)

        for q1, head, q2, relation, q3, tail, yn, explanation in matches:

            triple_key = (head, relation, tail)
            if triple_key not in triples_we_are_looking_for:
                continue
            all_identified_triples.add(triple_key)

            llm_assessment_bool = (yn == 'YES')

            # Store first occurrence or overwrite if previous answer was False
            if triple_key not in llm_triples_to_answer or not llm_triples_to_answer[triple_key]:
                llm_triples_to_answer[triple_key] = llm_assessment_bool
                llm_triples_to_prompt_responses[triple_key] = explanation.strip()

    else:
        # --- SINGLE TRIPLE MODE ---
        # triples is a tuple: (head, relation, tail)
        assert isinstance(triples, tuple)
        head, relation, tail = triples

        if not isinstance(output_text, str):
            logger.error(f'output_text_wrong_is_single! {output_text} ')
            return (
                llm_triples_to_answer,
                all_identified_triples,
                llm_triples_to_prompt_responses,
            )

        # Extract YES/NO anywhere in the output
        yn_match = re.search(r'\b(YES|NO)\b', output_text)

        if yn_match:
            yn = yn_match.group(1)
        else:
            yn = 'NO'  # conservative fallback

        llm_assessment_bool = (yn == 'YES')

        # Explanation = everything after YES/NO
        explanation = ''
        post_idx = output_text.find(yn) + len(yn)
        if post_idx < len(output_text):
            explanation = output_text[post_idx:].strip()

        triple_key = triples
        all_identified_triples.add(triple_key)

        if triple_key not in llm_triples_to_answer or not llm_triples_to_answer[triple_key]:
            llm_triples_to_answer[triple_key] = llm_assessment_bool
            llm_triples_to_prompt_responses[triple_key] = explanation

        matches = [(head, relation, tail, yn, explanation)]

    # --- LOGGING ---
    logger.debug('*****************************************')
    logger.debug(
        f'parse_returned_llm_text_v13\n'
        f'is_single: {is_single}\n'
        f'output_text: {output_text}\n'
        f'matches: {matches}\n'
        f'triples: {triples}\n'
        f'llm_triples_to_answer: {llm_triples_to_answer}\n'
        f'all_identified_triples: {all_identified_triples}\n'
        f'llm_triples_to_prompt_responses: {llm_triples_to_prompt_responses}\n'
    )
    logger.debug('*****************************************')

    return (
        llm_triples_to_answer,
        all_identified_triples,
        llm_triples_to_prompt_responses,
    )

async def call_llm_and_return_parsed_result(
        chunk: str,
        triples_str: str,
        action_type: str,
        client: UnifiedLLMClient,
        prompt_paths: Dict,
        action_category: str,
        llm_backend: str,
        model: str,
        passage_timestamp: int,
        delta_start_date: str
):
    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------
    prompt = _build_prompt(
        action_category=action_category,
        action_type=action_type,
        chunk=chunk,
        triples_str=triples_str,
        prompt_paths=prompt_paths,
        passage_timestamp=passage_timestamp,
        delta_start_date=delta_start_date
    )
    #
    logger.debug(f'prompt_to_use_with_llm_is: {prompt}')
    # Check cache first
    # --------------------------------------------------------
    # Check cache (async-safe)
    # --------------------------------------------------------
    async with _llm_cache_lock:
        cached = llmc.llm_cache.get(llm_backend, model, prompt)

    if cached is not None:
        logger.info('cache_hit_v13 [CACHE_HIT_V13]')
        return cached

    logger.debug(f'Using backend: {llm_backend}')

    async def _dispatch_generation():
        if client is None:
            raise ValueError(f'client is required for llm_backend={llm_backend}')

        return await client.generate(prompt)

    try:
        output_text = await _dispatch_generation()
    except Exception as e:
        logger.error(
            f"{prompt_paths.get('api_llm_device')}:{prompt_paths.get('api_llm_port')} "
            f'GenerationError for: {e}. Prompt length={len(prompt.split())}, '
            f'prompt_content: {prompt}'
            f'chunk length={len(chunk.split())}'
        )
        logger.error(traceback.format_exc())
        return ''

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    logger.debug(f'LLM raw output: {output_text}')
    logger.debug('===================================')
    #
    async with _llm_cache_lock:
        llmc.llm_cache.set(llm_backend, model, prompt, output_text)
    return output_text
