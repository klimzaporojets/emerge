import json
import logging
import os
from typing import Set, List, Dict

from tqdm import tqdm
from typing import Iterable, List, TypeVar

T = TypeVar('T')

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('general-io')


def save_batch_predictions(
    *,
    batch_instances: List[Dict],
    results: List,
    output_dir: str,
    model_name: str,
    model_config_name: str,
    opened_files: Dict[str, object],
):
    '''
    Attach predictions to instances and append them to output JSONL files.

    Assumes:
      - batch_instances[i] corresponds to results[i]
      - each batch_instances[i] has:
            - parsed_instance
            - rel_dir
            - input_file
    '''

    assert len(batch_instances) == len(results)

    predicted_instances_to_save = []

    for idx, result in enumerate(results):
        instance = batch_instances[idx]

        if 'predictions' not in instance['parsed_instance']:
            instance['parsed_instance']['predictions'] = []

        if not result['success']:
            continue

        prediction_json = {
            'predicted_triples': list(
                result['prediction'].predicted_triples
            ),
            'model': model_name,
            'model_config_name': model_config_name,
        }

        instance['parsed_instance']['predictions'].append(prediction_json)
        predicted_instances_to_save.append(instance)

    logger.info(
        f'Saving {len(predicted_instances_to_save)} predicted instances'
    )

    # opened_files: Dict[str, object] = {}

    # try:
    for instance in predicted_instances_to_save:
        rel_dir = instance['rel_dir']
        filename = instance['input_file']

        curr_output_dir = os.path.join(output_dir, rel_dir)
        os.makedirs(curr_output_dir, exist_ok=True)

        curr_output_file = os.path.join(curr_output_dir, filename)

        if curr_output_file not in opened_files:
            opened_files[curr_output_file] = open(
                curr_output_file, 'a', encoding='utf-8'
            )

        opened_files[curr_output_file].write(
            json.dumps(
                instance['parsed_instance'],
                ensure_ascii=False,
            )
            + '\n'
        )

    # finally:
    for f in opened_files.values():
        f.flush()

def obtain_already_predicted_hash_ids(output_dir: str) -> Set[str]:
    '''
    Walk output_dir, read all JSONL files, and collect hash_ids
    of instances for which predictions already exist.

    Returns:
        Set[str]: hash_ids that have already been processed
    '''
    already_predicted: Set[str] = set()

    if not os.path.exists(output_dir):
        logger.info(f'Output directory does not exist yet: {output_dir}')
        return already_predicted

    for dirpath, _, filenames in os.walk(output_dir):
        for filename in filenames:
            if not filename.endswith('.jsonl'):
                continue

            file_path = os.path.join(dirpath, filename)
            logger.info(f'Scanning predictions file: {file_path}')

            try:
                with open(file_path, 'rt', encoding='utf-8') as f:
                    for line_num, line in enumerate(
                        tqdm(f, desc=f'reading {filename}')
                    ):
                        line = line.strip()
                        if not line:
                            continue

                        record = json.loads(line)

                        hash_id = record['hash_id']

                        already_predicted.add(hash_id)

            except Exception as e:
                logger.error(f'Failed to read {file_path}: {e}')

    logger.info(f'Found {len(already_predicted)} already-predicted instances')
    return already_predicted


def batch_iterable(items: List[T], batch_size: int) -> List[List[T]]:
    '''
    Split a list into batches of size batch_size.
    '''
    if batch_size <= 0:
        raise ValueError('batch_size must be > 0')

    return [
        items[i: i + batch_size]
        for i in range(0, len(items), batch_size)
    ]


def obtain_instances_to_process(
        already_predicted_hash_ids: Set[str],
        input_dataset_path: str,
) -> List[Dict]:
    '''
    Walk input_dataset_path, read JSONL files, and return
    a flat list of instances that still need processing.
    '''
    texts_to_process: List[Dict] = []

    for dirpath, _, filenames in os.walk(input_dataset_path):
        for curr_filename in filenames:
            file_path = os.path.join(dirpath, curr_filename)
            rel_dir = os.path.relpath(dirpath, input_dataset_path)
            if rel_dir == '.':
                rel_dir = ''

            logger.info(f'Scanning file: {file_path}')

            with open(file_path, 'rt', encoding='utf-8') as file:
                for line_str in tqdm(file, desc=f'reading {curr_filename}'):
                    parsed_json = json.loads(line_str)

                    hash_id = parsed_json.get('hash_id')
                    if hash_id in already_predicted_hash_ids:
                        continue

                    texts_to_process.append(
                        {
                            # 'hash_id': hash_id,
                            'parsed_instance': parsed_json,
                            'rel_dir': rel_dir,
                            'input_file': curr_filename,
                        }
                    )

    logger.info(f'Collected {len(texts_to_process)} unprocessed instances')
    return texts_to_process
