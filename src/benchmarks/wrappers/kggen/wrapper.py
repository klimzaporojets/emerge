import argparse
import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import List, Dict
# ===================== BEGIN DSPy vLLM COMPAT PATCH =====================
#
# # ---- DSPy JSONAdapter → ChatAdapter shim (vLLM-compatible) ----
# import dspy.adapters.json_adapter as ja
# from dspy.adapters.chat_adapter import ChatAdapter
#
# class JSONAdapterShim(ChatAdapter):
#     """
#     Drop-in replacement for DSPy JSONAdapter that
#     uses Chat Completions instead of OpenAI Responses API.
#     """
#     pass
#
# ja.JSONAdapter = JSONAdapterShim
#
# # ---- HARD DISABLE TOOL / FUNCTION CALL PROMPTING ----
# # vLLM does NOT support OpenAI tool calling.
# # This forces DSPy to request plain JSON instead of tool-shaped outputs.
#
# def _no_tools_prepare_messages(self, signature, demos, inputs):
#     messages = ChatAdapter.prepare_messages(self, signature, demos, inputs)
#
#     for m in messages:
#         m.pop("tools", None)
#         m.pop("tool_calls", None)
#         m.pop("functions", None)
#         m.pop("function_call", None)
#
#     return messages
#
# ChatAdapter.prepare_messages = _no_tools_prepare_messages

# ====================== END DSPy vLLM COMPAT PATCH ======================
from kg_gen import KGGen
from tqdm import tqdm

from general_io import obtain_already_predicted_hash_ids, obtain_instances_to_process, batch_iterable, \
    save_batch_predictions
from prediction import Prediction

import json
import hashlib
from pathlib import Path
from threading import Lock

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('kggen-wrapper')

UNKNOWN_QID = 'NIL'
UNKNOWN_RELATION_ID = 'NAN'

_cache_write_lock = Lock()


def _make_cache_key(
        *,
        model_name: str,
        temperature: float,
        chunk_size: int | None,
        base_url: str | None,
        text: str,
        max_tokens: int
) -> str:
    payload = {
        'model': model_name,
        'temperature': temperature,
        'chunk_size': chunk_size,
        'base_url': base_url,
        'text': text,
        'max_tokens': max_tokens
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
    ).hexdigest()


def prediction_to_dict(pred: Prediction) -> Dict:
    """
    Convert Prediction object to JSON-serializable dict.
    """
    return {
        'predicted_triples': pred.predicted_triples,
    }


def prediction_from_dict(data: Dict) -> Prediction:
    """
    Reconstruct Prediction object from dict.
    """
    pred = Prediction()
    for triple in data.get('predicted_triples', []):
        pred.add_predicted_triple(
            predicted_relation=triple['extracted_relation'],
            predicted_triple_qids=triple['triple_qids'],
            predicted_triple_labels=triple['triple_labels']
        )
    return pred


#### END change ####


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
    entry = {
        'cache_key': cache_key,
        'result': result,
    }
    with _cache_write_lock:
        with open(cache_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            f.flush()


def run_kggen(
        instance_to_process: List[Dict],
        model_name: str,
        temperature: float,
        max_workers: int | None,
        chunk_size: int | None,
        base_url: str | None,
        max_tokens: int,
        cache: Dict[str, Dict],
        cache_path: Path,
) -> List:
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('KG_GEN_API_KEY')

    kg_init_params = {
        'model': model_name,
        'temperature': temperature,
        'api_key': api_key,
        'api_base': base_url,
        'max_tokens': int(max_tokens)
    }

    logger.info(
        f'Initializing KGGen with params kg_init_params: '
        f'{ {k: ("***" if k == "api_key" else v) for k, v in kg_init_params.items()} }'
    )
    kg = KGGen(
        **kg_init_params
    )

    def process_one(instance_id: int, instance: Dict):
        text = instance['parsed_instance']['passage']

        cache_key = _make_cache_key(
            model_name=model_name,
            temperature=temperature,
            chunk_size=chunk_size,
            base_url=base_url,
            text=text,
            max_tokens=max_tokens
        )

        if cache_key in cache:
            logger.info(f'[CACHE HIT] KGGen instance {instance_id}')
            cached_dict = cache[cache_key]
            cached = deepcopy(cached_dict)
            if cached.get('success'):
                cached['prediction'] = prediction_from_dict(
                    cached_dict['prediction']
                )
                return instance_id, cached

        try:
            generate_params = {'input_data': text}
            if chunk_size:
                generate_params['chunk_size'] = chunk_size

            logger.debug(f'*********querying_kg_gen_with: {generate_params}')

            graph = kg.generate(**generate_params)
            logger.debug(f'*********kg_gen_returned_graph: {graph}')
            prediction = Prediction()

            if hasattr(graph, 'relations') and graph.relations:
                for h, r, t in graph.relations:
                    prediction.add_predicted_triple(
                        predicted_relation=(h, r, t),
                        predicted_triple_qids=(None, None, None),
                        predicted_triple_labels=(None, None, None)
                    )

            result = {
                'success': True,
                'prediction': prediction_to_dict(prediction),
            }

            cache[cache_key] = result
            _append_cache(cache_path, cache_key, result)

            # Return runtime object
            return instance_id, {
                'success': True,
                'prediction': prediction,
            }
        except Exception as e:
            logger.error(f'Error in instance {instance_id}: {e}')
            logger.error(traceback.format_exc())

            result = {
                'success': False,
                'error': str(e),
                'triples': [],
            }

            return instance_id, result

    results = [None] * len(instance_to_process)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_one, i, text)
            for i, text in enumerate(instance_to_process)
        ]
        for f in tqdm(as_completed(futures), desc='processing_instances',
                      total=len(instance_to_process)):
            idx, res = f.result()
            results[idx] = res

    return results


def main():
    logging.getLogger('model2vec.hf_utils').setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--max-workers', type=int, default=None)
    parser.add_argument('--chunk-size', type=int, default=None)
    parser.add_argument('--base-url', type=str, default=None)
    parser.add_argument('--cache-path', type=str, required=True)
    parser.add_argument('--max-tokens', type=str, required=True)

    args = parser.parse_args()

    logger.info('============BEGIN WRAPPER PARAMETERS')
    logger.info(f'args.input: {args.input}')
    logger.info(f'args.output: {args.output}')
    logger.info(f'args.model: {args.model}')
    logger.info(f'args.temperature: {args.temperature}')
    logger.info(f'args.max-workers: {args.max_workers}')
    logger.info(f'args.chunk-size: {args.chunk_size}')
    logger.info(f'args.batch-size: {args.batch_size}')
    logger.info(f'args.base-url: {args.base_url}')
    logger.info(f'args.cache-path: {args.cache_path}')
    logger.info(f'args.max-tokens: {args.max_tokens}')
    logger.info('============END WRAPPER PARAMETERS')

    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_path)
    logger.info(f'Loaded {len(cache)} cache entries')

    os.makedirs(args.output, exist_ok=True)
    already_predicted_hash_ids = obtain_already_predicted_hash_ids(
        output_dir=args.output
    )
    instances_to_process: List = obtain_instances_to_process(
        already_predicted_hash_ids=already_predicted_hash_ids,
        input_dataset_path=args.input
    )

    logger.info(f'len(instances_to_process): {len(instances_to_process)}')

    batched_instances_to_process = batch_iterable(
        items=instances_to_process,
        batch_size=args.batch_size
    )

    logger.info(f'len(batched_instances_to_process): {len(batched_instances_to_process)}')
    opened_files = {}
    try:
        for idx_batch, curr_batched_instance in enumerate(batched_instances_to_process):
            logger.info(f'********* kg_gen processing_idx_batch {idx_batch} **********')
            logger.info(f'curr_batched_instance[0]: {curr_batched_instance[0]}')
            results = run_kggen(
                instance_to_process=curr_batched_instance,
                model_name=args.model,
                temperature=args.temperature,
                max_workers=args.max_workers,
                chunk_size=args.chunk_size,
                base_url=args.base_url,
                cache=cache,
                cache_path=cache_path,
                max_tokens=args.max_tokens
            )
            save_batch_predictions(
                batch_instances=curr_batched_instance,
                results=results,
                output_dir=args.output,
                model_name=f'kg-gen-{args.model}',
                model_config_name=f'kg-gen-{args.model}',
                opened_files=opened_files
            )
    finally:
        for f in opened_files.values():
            f.close()


if __name__ == '__main__':
    main()
