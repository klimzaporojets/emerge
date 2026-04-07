import argparse
import logging
import os
import time
import traceback
import json
import hashlib
from copy import deepcopy
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from tempfile import TemporaryDirectory
from typing import List, Dict

from src.textPrcess import TextProcessor
from src.kgAgent import NER_Agent
import src.config as rakg_config

from general_io import (
    obtain_already_predicted_hash_ids,
    obtain_instances_to_process,
    batch_iterable,
    save_batch_predictions,
)
from prediction import Prediction

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('rakg-wrapper')

UNKNOWN_QID = 'NIL'
UNKNOWN_RELATION_ID = 'NAN'

_cache_write_lock = Lock()

_progress_lock = Lock()
_processed_count = 0


# ---------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------

def _make_cache_key(
        *,
        text: str,
        use_similarity: bool,
        openai_model: str | None,
        openai_embedding_model: str | None,
        openai_similarity_model: str | None,
        base_url: str | None,
        base_url_embedding_model: str | None,
) -> str:
    payload = {
        'text': text,
        'use_similarity': use_similarity,
        'openai_model': openai_model,
        'openai_embedding_model': openai_embedding_model,
        'openai_similarity_model': openai_similarity_model,
        'base_url': base_url,
        'base_url_embedding_model': base_url_embedding_model,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
    ).hexdigest()


def prediction_to_dict(pred: Prediction) -> Dict:
    return {
        'predicted_triples': pred.predicted_triples,
    }


def prediction_from_dict(data: Dict) -> Prediction:
    pred = Prediction()
    for triple in data.get('predicted_triples', []):
        pred.add_predicted_triple(
            predicted_relation=triple['extracted_relation'],
            predicted_triple_qids=triple['triple_qids'],
            predicted_triple_labels=triple['triple_labels'],
        )
    return pred


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
                continue
    return cache


def _append_cache(cache_path: Path, cache_key: str, result: Dict):
    entry = {
        'cache_key': cache_key,
        'result': result,
    }
    with _cache_write_lock:
        with open(cache_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# ---------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------

def run_rakg(
        instance_to_process: List[Dict],
        max_workers: int | None,
        use_similarity: bool,
        openai_model: str | None,
        openai_embedding_model: str | None,
        openai_similarity_model: str | None,
        base_url: str | None,
        base_url_embedding_model:str|None,
        openai_api_key: str | None,
        cache: Dict[str, Dict],
        cache_path: Path,
) -> List:
    # ---- configure RAKG LLM backend (global config) ----
    rakg_config.USE_OPENAI = openai_model is not None
    rakg_config.OPENAI_MODEL = openai_model
    rakg_config.OPENAI_EMBEDDING_MODEL = openai_embedding_model
    rakg_config.OPENAI_SIMILARITY_MODEL = openai_similarity_model
    rakg_config.base_url = base_url
    rakg_config.base_url_embedding_model = base_url_embedding_model
    rakg_config.OPENAI_API_KEY = openai_api_key

    logger.info(
        f'RAKG config: USE_OPENAI={rakg_config.USE_OPENAI}, '
        f'MODEL={rakg_config.OPENAI_MODEL}, '
        f'EMBEDDING={rakg_config.OPENAI_EMBEDDING_MODEL}, '
        f'SIMILARITY={rakg_config.OPENAI_SIMILARITY_MODEL}, '
        f'BASE_URL={rakg_config.base_url}, '
        f'BASE_URL={rakg_config.base_url_embedding_model}'
    )

    ner_agent = NER_Agent()

    def process_one(instance_id: int, instance: Dict):
        global _processed_count

        text = instance['parsed_instance']['passage']
        topic = instance['parsed_instance'].get('title', '')

        cache_key = _make_cache_key(
            text=text,
            use_similarity=use_similarity,
            openai_model=openai_model,
            openai_embedding_model=openai_embedding_model,
            openai_similarity_model=openai_similarity_model,
            base_url=base_url,
            base_url_embedding_model=base_url_embedding_model
        )

        if cache_key in cache:
            logger.info(f'[CACHE HIT] RAKG instance {instance_id}')
            cached_dict = cache[cache_key]
            cached = deepcopy(cached_dict)
            if cached.get('success'):
                cached['prediction'] = prediction_from_dict(
                    cached_dict['prediction']
                )
                with _progress_lock:
                    _processed_count += 1
                    do_log = (_processed_count % 1 == 0)

                if do_log:
                    logger.info(
                        f'Processed {_processed_count} instances (including previous runs)'
                    )

                return instance_id, cached

        try:
            max_retries = 15
            converted_kg = None
            for attempt in range(max_retries):
                try:
                    processor = TextProcessor(text, topic)
                    text_split = processor.process()

                    with TemporaryDirectory() as tmpdir:
                        ner_output = os.path.join(tmpdir, 'ner.jsonl')

                        ner_result = ner_agent.extract_from_text_multiply(
                            text_split['sentences'],
                            text_split['sentence_to_id'],
                            output_file=ner_output,
                        )

                        if use_similarity:
                            sim = ner_agent.similartiy_result(ner_result)
                            entity_list_process = ner_agent.entity_Disambiguation(
                                ner_result, sim
                            )
                        else:
                            entity_list_process = ner_result

                        kg_output = os.path.join(tmpdir, 'kg.jsonl')
                        kg_result = ner_agent.get_target_kg_all(
                            entity_list_process,
                            text_split['id_to_sentence'],
                            text_split['sentences'],
                            text_split['sentence_to_id'],
                            text_split['vectors'],
                            output_file=kg_output,
                        )

                    converted_kg = ner_agent.convert_knowledge_graph(kg_result)
                    break

                except Exception as e:
                    if 'RateLimitError' in str(e) and attempt < max_retries - 1:
                        sleep_s = 1 + attempt
                        logger.warning(
                            f'rate_limit_hit_in_instance {instance_id}, '
                            f'retrying_in {sleep_s}s '
                            f'(attempt {attempt + 1}/{max_retries})'
                        )
                        time.sleep(sleep_s)
                    else:
                        raise

            prediction = Prediction()
            if not converted_kg:
                logging.warning(f'no_converted_kg for instance: {text}')

            for idx, triple in enumerate(converted_kg['relations']):
                h, r, t = triple[:3]
                prediction.add_predicted_triple(
                    predicted_relation=(h, r, t),
                    predicted_triple_qids=(None, None, None),
                    predicted_triple_labels=(None, None, None),
                )

            result = {
                'success': True,
                'prediction': prediction_to_dict(prediction),
            }

            cache[cache_key] = result
            _append_cache(cache_path, cache_key, result)

            with _progress_lock:
                _processed_count += 1
                do_log = (_processed_count % 1 == 0)

            if do_log:
                logger.info(
                    f'Processed {_processed_count} instances (including previous runs)'
                )

            return instance_id, {
                'success': True,
                'prediction': prediction,
            }

        except Exception as e:
            logger.error(f'Error in instance {instance_id}: {e}')
            logger.error(traceback.format_exc())

            with _progress_lock:
                _processed_count += 1
                do_log = (_processed_count % 1 == 0)

            if do_log:
                logger.info(
                    f'Processed {_processed_count} instances (including previous runs)'
                )

            return instance_id, {
                'success': False,
                'error': str(e),
                'triples': [],
            }

    results = [None] * len(instance_to_process)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_one, i, inst)
            for i, inst in enumerate(instance_to_process)
        ]
        for f in as_completed(futures):
            idx, res = f.result()
            results[idx] = res

    return results


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    #
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--max-workers', type=int, default=None)
    parser.add_argument('--cache-path', type=str, required=True)
    parser.add_argument('--openai-model', type=str, default=None)
    parser.add_argument('--openai-embedding-model', type=str, default=None)
    parser.add_argument('--openai-similarity-model', type=str, default=None)
    parser.add_argument('--base-url', type=str, default=None)
    parser.add_argument('--base-url-embedding-model', type=str, default=None)
    parser.add_argument('--openai-api-key', type=str, default=None)
    #

    parser.add_argument(
        '--use-similarity',
        action='store_true',
        help='Enable similarity-based entity disambiguation',
    )

    args = parser.parse_args()

    if args.openai_api_key is None:
        args.openai_api_key = os.getenv('OPENAI_API_KEY')

    logger.info('============BEGIN WRAPPER PARAMETERS')
    for k, v in vars(args).items():
        logger.info(f'{k}: {v}')
    logger.info('============END WRAPPER PARAMETERS')

    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_path)

    os.makedirs(args.output, exist_ok=True)

    already_predicted_hash_ids = obtain_already_predicted_hash_ids(
        output_dir=args.output
    )
    global _processed_count
    _processed_count = len(already_predicted_hash_ids)

    instances_to_process = obtain_instances_to_process(
        already_predicted_hash_ids=already_predicted_hash_ids,
        input_dataset_path=args.input,
    )

    batched_instances = batch_iterable(
        items=instances_to_process,
        batch_size=args.batch_size,
    )

    opened_files = {}
    try:
        for batch in batched_instances:
            results = run_rakg(
                instance_to_process=batch,
                max_workers=args.max_workers,
                use_similarity=args.use_similarity,
                openai_model=args.openai_model,
                openai_embedding_model=args.openai_embedding_model,
                openai_similarity_model=args.openai_similarity_model,
                base_url=args.base_url,
                base_url_embedding_model=args.base_url_embedding_model,
                openai_api_key=args.openai_api_key,
                cache=cache,
                cache_path=cache_path,
            )
            save_batch_predictions(
                batch_instances=batch,
                results=results,
                output_dir=args.output,
                model_name=f'rakg-{args.openai_model}'
                           f'-{args.openai_embedding_model}'
                           f'-{args.openai_similarity_model}'
                           f'-{args.use_similarity}',
                model_config_name=f'rakg-{args.openai_model}'
                                  f'-{args.openai_embedding_model}'
                                  f'-{args.openai_similarity_model}'
                                  f'-{args.use_similarity}',
                opened_files=opened_files,
            )
    finally:
        for f in opened_files.values():
            f.close()


if __name__ == '__main__':
    main()
