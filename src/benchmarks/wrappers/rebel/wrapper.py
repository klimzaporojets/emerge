import argparse
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import List, Dict

import json
import hashlib
from pathlib import Path
from threading import Lock

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerBase, PreTrainedModel

from general_io import (
    obtain_already_predicted_hash_ids,
    obtain_instances_to_process,
    batch_iterable,
    save_batch_predictions,
)
from prediction import Prediction

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('rebel-wrapper')

_cache_write_lock = Lock()


# -------------------------
# Cache key (REBEL-only)
# -------------------------

def _make_cache_key(
        *,
        model_name: str,
        device: str,
        max_length: int,
        num_beams: int,
        text: str,
) -> str:
    payload = {
        'model': model_name,
        'device': device,
        'max_length': max_length,
        'num_beams': num_beams,
        'text': text,
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


# -------------------------
# REBEL parsing (validated)
# -------------------------

def extract_triplets(decoded_text: str):
    triplets = []
    subject, object_, relation = '', '', ''
    state = None

    tokens = (
        decoded_text
        .replace('<s>', '')
        .replace('</s>', '')
        .strip()
        .split()
    )

    for token in tokens:
        if token == '<triplet>':
            if subject and relation and object_:
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            subject, object_, relation = '', '', ''
            state = 'subject'

        elif token == '<subj>':
            state = 'object'

        elif token == '<obj>':
            state = 'relation'

        else:
            if state == 'subject':
                subject += ' ' + token
            elif state == 'object':
                object_ += ' ' + token
            elif state == 'relation':
                relation += ' ' + token

    if subject and relation and object_:
        triplets.append((subject.strip(), relation.strip(), object_.strip()))

    return triplets


# ============================================================
# >>> CHANGE START: helper for token-based text splitting <<<
# ============================================================

def _split_text_into_token_chunks(
        text: str,
        tokenizer: PreTrainedTokenizerBase,
        chunk_size: int = 256,
) -> List[str]:
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        chunk_ids = token_ids[i:i + chunk_size]
        if not chunk_ids:
            break
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))

    return chunks

# ============================================================
# >>> CHANGE END <<<
# ============================================================


# -------------------------
# Core runner
# -------------------------

def run_rebel(
        instance_to_process: List[Dict],
        model_name: str,
        device: str,
        max_length: int,
        num_beams: int,
        max_workers: int | None,
        cache: Dict[str, Dict],
        cache_path: Path,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase
) -> List:

    def process_one(instance_id: int, instance: Dict):
        text = instance['parsed_instance']['passage']

        cache_key = _make_cache_key(
            model_name=model_name,
            device=device,
            max_length=max_length,
            num_beams=num_beams,
            text=text,
        )

        if cache_key in cache:
            logger.info(f'REBEL CACHE HIT instance {instance_id}')
            cached_dict = cache[cache_key]
            cached = deepcopy(cached_dict)
            if cached.get('success'):
                cached['prediction'] = prediction_from_dict(
                    cached_dict['prediction']
                )
                return instance_id, cached

        try:
            # ============================================================
            # >>> CHANGE START: split long text into 256-token chunks <<<
            # ============================================================
            chunks = _split_text_into_token_chunks(
                text=text,
                tokenizer=tokenizer,
                chunk_size=max_length,
            )
            # ============================================================
            # >>> CHANGE END <<<
            # ============================================================

            prediction = Prediction()

            for chunk in chunks:
                inputs = tokenizer(
                    chunk,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors='pt',
                )

                with torch.no_grad():
                    generated = model.generate(
                        inputs['input_ids'].to(device),
                        attention_mask=inputs['attention_mask'].to(device),
                        max_length=512,
                        num_beams=num_beams
                    )

                decoded = tokenizer.batch_decode(
                    generated, skip_special_tokens=False
                )

                for sent in decoded:
                    for h, r, t in extract_triplets(sent):
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

            return instance_id, {
                'success': True,
                'prediction': prediction,
            }

        except Exception as e:
            logger.error(f'Error in instance {instance_id}: {e}')
            logger.error(traceback.format_exc())
            return instance_id, {
                'success': False,
                'error': str(e),
                'triples': [],
            }

    results = [None] * len(instance_to_process)

    # ============================================================
    # >>> CHANGE START: disable threading when using CUDA <<<
    # ============================================================
    if device.startswith("cuda"):
        # CUDA + multithreading is unsafe → run sequentially
        for i, inst in tqdm(enumerate(instance_to_process), desc='processing_instances',
                            total=len(instance_to_process)):
            idx, res = process_one(i, inst)
            results[idx] = res
    else:
        # CPU path unchanged
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_one, i, inst)
                for i, inst in enumerate(instance_to_process)
            ]
            for f in as_completed(futures):
                idx, res = f.result()
                results[idx] = res
    # ============================================================
    # >>> CHANGE END <<<
    # ============================================================

    return results


# -------------------------
# CLI (REBEL-only)
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--max-workers', type=int, default=None)
    parser.add_argument('--cache-path', type=str, required=True)

    # REBEL params
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--num-beams', type=int, default=3)

    args = parser.parse_args()

    logger.info('********* REBEL wrapper input parameters *********')
    for k, v in vars(args).items():
        logger.info(f'{k}: {v}')
    logger.info('**********************')

    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_path)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    already_predicted_hash_ids = obtain_already_predicted_hash_ids(args.output)
    instances_to_process = obtain_instances_to_process(
        already_predicted_hash_ids=already_predicted_hash_ids,
        input_dataset_path=args.input,
    )

    batched_instances = batch_iterable(
        items=instances_to_process,
        batch_size=args.batch_size,
    )

    logger.info(f'Loading REBEL model once: {args.model} on {args.device}')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device)
    model.eval()

    opened_files = {}
    try:
        for batch in batched_instances:
            results = run_rebel(
                instance_to_process=batch,
                model_name=args.model,
                device=args.device,
                max_length=args.max_length,
                num_beams=args.num_beams,
                max_workers=args.max_workers,
                cache=cache,
                cache_path=cache_path,
                model=model,
                tokenizer=tokenizer
            )
            save_batch_predictions(
                batch_instances=batch,
                results=results,
                output_dir=args.output,
                model_name=f'rebel-{args.model}-{args.max_length}-{args.num_beams}',
                model_config_name=f'rebel-{args.model}-{args.max_length}-{args.num_beams}',
                opened_files=opened_files,
            )
    finally:
        for f in opened_files.values():
            f.close()


if __name__ == '__main__':
    main()
