"""KG_PROMPT wrapper — dedicated prompt-runner for the KG-conditioned baseline.

Unlike KGGen (which builds its prompt inside the DSPy `kg_gen` lib) and EDC+
(which builds its own from templates), this runner sends the fully-formed
`prompt` field produced by COMPOSE (`scripts/kg_context/compose_kg_prompt.py`)
to the LLM VERBATIM, parses the JSON-array response, and emits predictions in
the SAME schema KGGen emits (`predicted_triples` = list of
`{extracted_relation, triple_qids, triple_labels}`), so the existing scorer
works unchanged.

The LLM is called through `edc.utils.unified_llm_client.UnifiedLLMClient`
(the same async client EDC+ uses for gpt-5.1 via the Azure research proxy).
"""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import traceback
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from tqdm import tqdm

# general_io / prediction live one dir up (wrappers/), on PYTHONPATH via run.sh
from general_io import (
    obtain_already_predicted_hash_ids,
    obtain_instances_to_process,
    batch_iterable,
    save_batch_predictions,
)
from prediction import Prediction

# UnifiedLLMClient is bundled under the edc_plus tree; run.sh adds edc_tt2kg to PYTHONPATH
from edc.utils.unified_llm_client import UnifiedLLMClient

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('kg-prompt-wrapper')

_cache_write_lock = Lock()


# ------------------------------------------------------------------ caching
# (mirrors kggen/wrapper.py: sha256 over a canonical payload; JSONL cache file)
def _make_cache_key(
        *,
        model_name: str,
        temperature: float,
        base_url: Optional[str],
        prompt: str,
        max_tokens: int,
) -> str:
    payload = {
        'model': model_name,
        'temperature': temperature,
        'base_url': base_url,
        'prompt': prompt,
        'max_tokens': max_tokens,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
    ).hexdigest()


def _load_cache(cache_path: Path) -> Dict[str, Dict]:
    cache: Dict[str, Dict] = {}
    if not cache_path.exists():
        return cache
    with open(cache_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                cache[record['cache_key']] = record['result']
            except Exception:
                logger.exception('error_reading_into_cache: ')
    return cache


def _append_cache(cache_path: Path, cache_key: str, result: Dict):
    entry = {'cache_key': cache_key, 'result': result}
    with _cache_write_lock:
        with open(cache_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            f.flush()


# ------------------------------------------------- prediction (de)serialization
def prediction_to_dict(pred: Prediction) -> Dict:
    return {'predicted_triples': pred.predicted_triples}


def prediction_from_dict(data: Dict) -> Prediction:
    pred = Prediction()
    for triple in data.get('predicted_triples', []):
        pred.add_predicted_triple(
            predicted_relation=triple['extracted_relation'],
            predicted_triple_qids=triple['triple_qids'],
            predicted_triple_labels=triple['triple_labels'],
        )
    return pred


# ----------------------------------------------------------------- parse logic
def _norm_entity_id(value) -> Optional[str]:
    """Map a model-emitted entity id to the QID stored in triple_qids.

    A newly-minted entity carries the literal string "MINTED" (no QID) — those
    map to None, exactly like KGGen emits (None, None, None) for un-linked
    entities. Empty / missing ids also map to None. Real QIDs pass through.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped.upper() == 'MINTED':
        return None
    return stripped


def _extract_first_json_array(text: str):
    """Robustly extract the first complete top-level JSON array from `text`.

    Handles ```json fences and leading/trailing prose by scanning for the first
    '[' and matching its closing ']' with a string-aware bracket counter.
    Returns the parsed Python list, or None if none is found / it won't parse.
    """
    if not text or not text.strip():
        return None
    s = text.strip()
    start = s.find('[')
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                candidate = s[start:i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    logger.warning('found bracketed span but json.loads failed')
                    return None
    return None


def parse_response_to_prediction(text: str) -> Prediction:
    """Convert a raw LLM response into a Prediction (KGGen-schema triples).

    Each array element:
      {"operation": ..., "head": {"label","id"}, "relation": {"label","id"},
       "tail": {"label","id"}}
    -> add_predicted_triple(
           predicted_relation=<relation label>,
           predicted_triple_qids=[head_id_or_None, relation_pid, tail_id_or_None],
           predicted_triple_labels=[head_label, relation_label, tail_label])

    Unparseable / malformed responses yield an EMPTY Prediction (warn, no crash).
    """
    prediction = Prediction()
    arr = _extract_first_json_array(text)
    if arr is None:
        logger.warning('unparseable_llm_response — emitting empty predicted_triples')
        return prediction
    if not isinstance(arr, list):
        logger.warning('parsed_json_not_a_list — emitting empty predicted_triples')
        return prediction

    for element in arr:
        if not isinstance(element, dict):
            logger.warning(f'skipping non-dict array element: {element!r}')
            continue
        head = element.get('head') or {}
        relation = element.get('relation') or {}
        tail = element.get('tail') or {}
        if not isinstance(head, dict) or not isinstance(relation, dict) \
                or not isinstance(tail, dict):
            logger.warning(f'skipping element with non-dict h/r/t: {element!r}')
            continue

        head_label = head.get('label')
        relation_label = relation.get('label')
        tail_label = tail.get('label')

        head_id = _norm_entity_id(head.get('id'))
        relation_pid = relation.get('id')
        tail_id = _norm_entity_id(tail.get('id'))

        prediction.add_predicted_triple(
            predicted_relation=relation_label,
            predicted_triple_qids=[head_id, relation_pid, tail_id],
            predicted_triple_labels=[head_label, relation_label, tail_label],
        )
    return prediction


# ------------------------------------------------------------------- execution
async def _process_batch(
        *,
        client: UnifiedLLMClient,
        batch_instances: List[Dict],
        model_name: str,
        temperature: float,
        base_url: Optional[str],
        max_tokens: int,
        cache: Dict[str, Dict],
        cache_path: Path,
) -> List:
    async def process_one(instance_id: int, instance: Dict):
        parsed = instance['parsed_instance']
        # HARD FAIL if `prompt` missing: COMPOSE was not run with the KG-aware
        # templates. This propagates and stops the batch on purpose.
        if 'prompt' not in parsed or parsed['prompt'] is None:
            raise KeyError(
                f"instance {parsed.get('hash_id')} has no 'prompt' field — "
                f"re-run COMPOSE_KG_PROMPT with the KG-aware templates"
            )
        prompt = parsed['prompt']

        cache_key = _make_cache_key(
            model_name=model_name,
            temperature=temperature,
            base_url=base_url,
            prompt=prompt,
            max_tokens=max_tokens,
        )

        if cache_key in cache:
            logger.info(f'[CACHE HIT] kg_prompt instance {instance_id}')
            cached_dict = cache[cache_key]
            cached = deepcopy(cached_dict)
            if cached.get('success'):
                cached['prediction'] = prediction_from_dict(cached_dict['prediction'])
                return instance_id, cached

        try:
            response_text = await client.generate(
                prompt,
                temperature=temperature,
                max_new_tokens=int(max_tokens),
            )
            # parse_response_to_prediction never raises on bad content — it
            # returns an empty Prediction and logs a warning.
            prediction = parse_response_to_prediction(response_text)

            result = {'success': True, 'prediction': prediction_to_dict(prediction)}
            cache[cache_key] = result
            _append_cache(cache_path, cache_key, result)

            return instance_id, {'success': True, 'prediction': prediction}
        except Exception as e:
            # network / client errors (NOT parse errors) — fail this instance
            # softly so the batch continues.
            logger.error(f'Error in instance {instance_id}: {e}')
            logger.error(traceback.format_exc())
            return instance_id, {'success': False, 'error': str(e), 'triples': []}

    tasks = [process_one(i, inst) for i, inst in enumerate(batch_instances)]
    results: List = [None] * len(batch_instances)
    # UnifiedLLMClient.generate limits real concurrency via its internal
    # asyncio.Semaphore(concurrency); gather just schedules all of them.
    for coro in tqdm(
            asyncio.as_completed(tasks),
            desc='processing_instances',
            total=len(batch_instances),
    ):
        idx, res = await coro
        results[idx] = res
    return results


async def _run_all(args, cache: Dict[str, Dict], cache_path: Path):
    already_predicted_hash_ids = obtain_already_predicted_hash_ids(output_dir=args.output)
    instances_to_process: List = obtain_instances_to_process(
        already_predicted_hash_ids=already_predicted_hash_ids,
        input_dataset_path=args.input,
    )
    logger.info(f'len(instances_to_process): {len(instances_to_process)}')

    batched_instances_to_process = batch_iterable(
        items=instances_to_process,
        batch_size=args.batch_size,
    )
    logger.info(f'len(batched_instances_to_process): {len(batched_instances_to_process)}')

    concurrency = args.max_workers if args.max_workers and args.max_workers > 0 else 16
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    model_max_context = (
        int(args.model_max_context)
        if args.model_max_context not in (None, '', 'None', 'null')
        else None
    )
    # run_benchmarks passes every arg as a string, so a "not set" value arrives
    # as the literal "None"/"null" — normalize those back to real None.
    authorization_header_type = args.authorization_header_type
    if authorization_header_type in (None, '', 'None', 'null'):
        authorization_header_type = None
    api_version = args.api_version if args.api_version not in ('None', 'null') else None

    logger.info(
        'Constructing UnifiedLLMClient: '
        f'base_url={args.base_url} model={args.model} backend={args.backend} '
        f'mode={args.mode} api_version={args.api_version} '
        f'model_max_context={model_max_context} concurrency={concurrency} '
        f'api_key_env={args.api_key_env} (key_present={api_key is not None}) '
        f'authorization_header_type={authorization_header_type}'
    )

    client = UnifiedLLMClient(
        base_url=args.base_url,
        model=args.model,
        model_max_context=model_max_context,
        api_key=api_key,
        concurrency=concurrency,
        timeout=300,
        backend=args.backend,
        api_version=api_version,
        mode=args.mode,
        authorization_header_type=authorization_header_type,
    )

    opened_files: Dict[str, object] = {}
    try:
        async with client:
            for idx_batch, curr_batch in enumerate(batched_instances_to_process):
                logger.info(f'********* kg_prompt processing_idx_batch {idx_batch} **********')
                logger.info(f'curr_batch[0]: {curr_batch[0]}')
                results = await _process_batch(
                    client=client,
                    batch_instances=curr_batch,
                    model_name=args.model,
                    temperature=args.temperature,
                    base_url=args.base_url,
                    max_tokens=int(args.max_tokens),
                    cache=cache,
                    cache_path=cache_path,
                )
                save_batch_predictions(
                    batch_instances=curr_batch,
                    results=results,
                    output_dir=args.output,
                    model_name=f'kg-prompt-{args.model}',
                    model_config_name=f'kg-prompt-{args.model}',
                    opened_files=opened_files,
                )
    finally:
        for f in opened_files.values():
            f.close()


def main():
    parser = argparse.ArgumentParser()
    # ---- mirror kggen ----
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--max-workers', type=int, default=None)
    parser.add_argument('--base-url', type=str, default=None)
    parser.add_argument('--cache-path', type=str, required=True)
    parser.add_argument('--max-tokens', type=str, required=True)
    # ---- UnifiedLLMClient connection params (for gpt-5.1 via Azure proxy) ----
    parser.add_argument('--backend', type=str, default='http')
    parser.add_argument('--mode', type=str, default='completion')
    parser.add_argument('--api-version', type=str, default='v1')
    parser.add_argument('--api-key-env', type=str, default='OPENAI_API_KEY')
    parser.add_argument('--model-max-context', type=str, default=None)
    parser.add_argument('--authorization-header-type', type=str, default=None)

    args = parser.parse_args()

    logger.info('============BEGIN WRAPPER PARAMETERS')
    logger.info(f'args.input: {args.input}')
    logger.info(f'args.output: {args.output}')
    logger.info(f'args.model: {args.model}')
    logger.info(f'args.temperature: {args.temperature}')
    logger.info(f'args.max-workers: {args.max_workers}')
    logger.info(f'args.batch-size: {args.batch_size}')
    logger.info(f'args.base-url: {args.base_url}')
    logger.info(f'args.cache-path: {args.cache_path}')
    logger.info(f'args.max-tokens: {args.max_tokens}')
    logger.info(f'args.backend: {args.backend}')
    logger.info(f'args.mode: {args.mode}')
    logger.info(f'args.api-version: {args.api_version}')
    logger.info(f'args.api-key-env: {args.api_key_env}')
    logger.info(f'args.model-max-context: {args.model_max_context}')
    logger.info(f'args.authorization-header-type: {args.authorization_header_type}')
    logger.info('============END WRAPPER PARAMETERS')

    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_path)
    logger.info(f'Loaded {len(cache)} cache entries')

    os.makedirs(args.output, exist_ok=True)

    asyncio.run(_run_all(args, cache, cache_path))


if __name__ == '__main__':
    main()
