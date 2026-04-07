"""Merge multiple prediction datasets into a single main dataset.

Reads a main dataset (JSONL files), merges assessments (human/LLM) and
model predictions (EDC, RELIK, KG-Gen, RAKG, REBEL, tool-based) into it,
then writes the merged result to an output directory preserving the
original file structure.
"""
import argparse
import json
import logging
import os
from typing import Dict, List, Tuple

from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Annotation merging
# ---------------------------------------------------------------------------

def merge_annotations(instance1: Dict, instance2: Dict) -> Dict:
    """Merge assessment annotations from instance2 into instance1.

    Idempotent: skips annotations that already exist (by annotator+prompt key).
    """
    assert len(instance1['tkgu_triples']) == len(instance2['tkgu_triples'])

    for triple_inst1, triple_inst2 in zip(
        instance1['tkgu_triples'],
        instance2['tkgu_triples'],
    ):
        assert triple_inst1['triple'] == triple_inst2['triple']
        assert triple_inst1['tkgu_operations'] == triple_inst2['tkgu_operations']

        if 'human_assessment' in triple_inst2 and len(triple_inst2['human_assessment']) > 0:
            if 'human_assessment' not in triple_inst1 or len(triple_inst1['human_assessment']) == 0:
                triple_inst1['human_assessment'] = triple_inst2['human_assessment']
            else:
                existing = set()
                for a in triple_inst1['human_assessment']:
                    existing.add((a['annotator_name'], a['prompt_type']))
                for a in triple_inst2['human_assessment']:
                    if (a['annotator_name'], a['prompt_type']) not in existing:
                        triple_inst1['human_assessment'].append(a)

        if 'llm_assessment' in triple_inst2 and len(triple_inst2['llm_assessment']) > 0:
            if 'llm_assessment' not in triple_inst1 or len(triple_inst1['llm_assessment']) == 0:
                triple_inst1['llm_assessment'] = triple_inst2['llm_assessment']
            else:
                existing = set()
                for a in triple_inst1['llm_assessment']:
                    existing.add((a['llm_name'], a['llm_prompt_type']))
                for a in triple_inst2['llm_assessment']:
                    if (a['llm_name'], a['llm_prompt_type']) not in existing:
                        triple_inst1['llm_assessment'].append(a)

    return instance1


# ---------------------------------------------------------------------------
# EDC prediction merging
# ---------------------------------------------------------------------------

def merge_edc_predictions(
    instance_dataset: Dict,
    predictions: Dict,
    model_name: str,
    model_type: str,
    hash_id: str,
    canonicalize: bool,
    snapshot_to_property_id_to_label: Dict[str, Dict[str, str]],
) -> Dict:
    """Merge EDC-style predictions into the dataset instance."""
    if 'predictions' not in instance_dataset:
        instance_dataset['predictions'] = {}

    if model_name in instance_dataset['predictions']:
        logger.info(f'SKIP_ALREADY_MERGED ({hash_id}) model={model_name} — predictions already present')
        return instance_dataset

    predicted_triples = []
    predicted_triples_oie_deprecate = []
    predicted_triples_oie_add = []
    predicted_triples_oie_not_in_text = []
    predicted_triples_entities_to_kg = []

    predicted_triples_set = set()
    predicted_triples_oie_deprecate_set = set()
    predicted_triples_oie_add_set = set()
    predicted_triples_oie_not_in_text_set = set()
    predicted_triples_entities_to_kg_set = set()

    for triple in predictions['schema_canonicalization']:
        if triple is None:
            continue
        if len(triple) < 4:
            continue
        if str(triple[3]).lower() not in {'add', 'deprecate'}:
            continue
        triple_data = {
            'action': triple[3],
            'extracted_relation': triple[:3],
            'triple_qids': ['--NME--', triple[1], '--NME--'],
            'triple_labels': ['--NME--', triple[1], '--NME--'],
        }
        signature = (
            triple_data['action'],
            tuple(triple_data['extracted_relation']),
            tuple(triple_data['triple_qids']),
            tuple(triple_data['triple_labels']),
        )
        if signature not in predicted_triples_set:
            predicted_triples.append(triple_data)
            predicted_triples_set.add(signature)

    for triple in predictions['schema_canonicalization_not_in_text']:
        if triple is None:
            continue
        if len(triple) < 3:
            continue
        triple_data = {
            'extracted_relation': triple,
            'triple_qids': ['--NME--', triple[1], '--NME--'],
            'triple_labels': ['--NME--', triple[1], '--NME--'],
        }
        signature = (
            tuple(triple_data['extracted_relation']),
            tuple(triple_data['triple_qids']),
            tuple(triple_data['triple_labels']),
        )
        if signature not in predicted_triples_entities_to_kg_set:
            predicted_triples_entities_to_kg.append(triple_data)
            predicted_triples_entities_to_kg_set.add(signature)

    for idx_prediction, curr_prediction_oie in enumerate(predictions['oie']):
        curr_prediction_oie = list(curr_prediction_oie)
        if len(curr_prediction_oie) < 3:
            continue
        if len(curr_prediction_oie) > 3 and str(curr_prediction_oie[3]).lower() not in {'add', 'deprecate'}:
            continue

        if canonicalize:
            canonicalized_triple = predictions['schema_canonicalization'][idx_prediction]
            curr_snapshot_date = instance_dataset['delta_dates'][0]
            curr_property_id_to_label = snapshot_to_property_id_to_label[curr_snapshot_date]
            if canonicalized_triple is not None:
                if canonicalized_triple[1] in curr_property_id_to_label:
                    curr_prediction_oie[1] = curr_property_id_to_label[canonicalized_triple[1]]

        if len(curr_prediction_oie) > 3 and curr_prediction_oie[3].lower() == 'add':
            if tuple(curr_prediction_oie[:3]) not in predicted_triples_oie_add_set:
                predicted_triples_oie_add.append(curr_prediction_oie[:3])
                predicted_triples_oie_add_set.add(tuple(curr_prediction_oie[:3]))
        elif len(curr_prediction_oie) > 3 and curr_prediction_oie[3].lower() == 'deprecate':
            if tuple(curr_prediction_oie[:3]) not in predicted_triples_oie_deprecate_set:
                predicted_triples_oie_deprecate.append(curr_prediction_oie[:3])
                predicted_triples_oie_deprecate_set.add(tuple(curr_prediction_oie[:3]))
        elif len(curr_prediction_oie) == 3 and str(curr_prediction_oie[2]).lower() not in {'add', 'deprecate'}:
            if tuple(curr_prediction_oie[:3]) not in predicted_triples_oie_add_set:
                predicted_triples_oie_add.append(curr_prediction_oie[:3])
                predicted_triples_oie_add_set.add(tuple(curr_prediction_oie[:3]))

    for idx, curr_prediction_oie_not_in_text in enumerate(predictions['oie_not_in_text']):
        if len(curr_prediction_oie_not_in_text) < 3:
            continue
        curr_prediction_oie_not_in_text = list(curr_prediction_oie_not_in_text)
        if canonicalize:
            canonicalized_triple = predictions['schema_canonicalization_not_in_text'][idx]
            curr_snapshot_date = instance_dataset['delta_dates'][0]
            curr_property_id_to_label = snapshot_to_property_id_to_label[curr_snapshot_date]
            if canonicalized_triple is not None:
                if canonicalized_triple[1] in curr_property_id_to_label:
                    curr_prediction_oie_not_in_text[1] = curr_property_id_to_label[canonicalized_triple[1]]

        if tuple(curr_prediction_oie_not_in_text[:3]) not in predicted_triples_oie_not_in_text_set:
            predicted_triples_oie_not_in_text.append(curr_prediction_oie_not_in_text[:3])
            predicted_triples_oie_not_in_text_set.add(tuple(curr_prediction_oie_not_in_text[:3]))

    if len(predicted_triples_oie_deprecate + predicted_triples_oie_add + predicted_triples_oie_not_in_text) == 0:
        logger.warning(f'({hash_id}) empty predictions for {model_name}: {predictions}')

    instance_dataset['predictions'][model_name] = {
        'predicted_triples': predicted_triples,
        'predicted_triples_oie': {
            'oie_add': predicted_triples_oie_add,
            'oie_deprecate': predicted_triples_oie_deprecate,
            'oie_not_in_text': predicted_triples_oie_not_in_text,
        },
        'predicted_triples_entities_to_kg': predicted_triples_entities_to_kg,
        'model': model_name,
        'model_type': model_type,
    }

    return instance_dataset


# ---------------------------------------------------------------------------
# Main dataset loading
# ---------------------------------------------------------------------------

def load_main_dataset(input_main_dataset_path: str) -> Tuple[List[Dict], Dict[str, int]]:
    """Load the main dataset that other datasets will be merged into."""
    logger.info(f'loading main dataset from {input_main_dataset_path}')

    loaded_main_dataset: List[Dict] = []
    main_hash_id_to_line_idx: Dict[str, int] = {}

    for dirpath, _, filenames in os.walk(input_main_dataset_path):
        for fname in sorted(filenames):
            if not fname.endswith('.jsonl'):
                continue
            fpath = os.path.join(dirpath, fname)
            with open(fpath, 'rt', encoding='utf-8') as f:
                for line in tqdm(f, desc=f'loading {fname}'):
                    parsed_line = json.loads(line)
                    hash_id = parsed_line['hash_id']
                    main_hash_id_to_line_idx[hash_id] = len(loaded_main_dataset)
                    loaded_main_dataset.append(parsed_line)

    logger.info(f'loaded {len(loaded_main_dataset)} instances from main dataset')
    return loaded_main_dataset, main_hash_id_to_line_idx


# ---------------------------------------------------------------------------
# Assessment merging
# ---------------------------------------------------------------------------

def merge_assessments_into_main_dataset(
    loaded_main_dataset: List[Dict],
    main_hash_id_to_line_idx: Dict[str, int],
    curr_dataset_to_merge: Dict,
) -> None:
    """Merge a single annotation-style dataset into the main dataset.

    Idempotent: merge_annotations deduplicates by (annotator, prompt_type) key.
    """
    input_path = curr_dataset_to_merge['input_dataset_path']
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f'assessment input_dataset_path does not exist: {input_path} '
            f'(dataset_title={curr_dataset_to_merge["dataset_title"]})'
        )

    logger.info(f'merging assessments: {curr_dataset_to_merge["dataset_title"]}')

    for dirpath, _, filenames in os.walk(input_path):
        for fname in sorted(filenames):
            if not fname.endswith('.jsonl'):
                continue
            fpath = os.path.join(dirpath, fname)
            with open(fpath, 'rt', encoding='utf-8') as f:
                for line in tqdm(f, desc=f'merging assessments from {fname}'):
                    parsed_line = json.loads(line)
                    hash_id = parsed_line['hash_id']
                    if hash_id in main_hash_id_to_line_idx:
                        idx = main_hash_id_to_line_idx[hash_id]
                        loaded_main_dataset[idx] = merge_annotations(
                            instance1=loaded_main_dataset[idx],
                            instance2=parsed_line,
                        )


# ---------------------------------------------------------------------------
# Prediction merging
# ---------------------------------------------------------------------------

def merge_predictions_into_main_dataset(
    loaded_main_dataset: List[Dict],
    main_hash_id_to_line_idx: Dict[str, int],
    curr_dataset_to_merge: Dict,
    snapshot_to_property_id_to_label: Dict[str, Dict[str, str]],
) -> None:
    """Merge a single prediction-style dataset into the main dataset."""
    input_path = curr_dataset_to_merge['input_dataset_path']
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f'prediction input_dataset_path does not exist: {input_path} '
            f'(predictions_model_name={curr_dataset_to_merge["predictions_model_name"]})'
        )

    model_name = curr_dataset_to_merge['predictions_model_name']
    model_type = curr_dataset_to_merge['predictions_model_type']
    logger.info(f'merging predictions: {model_name} (type={model_type})')

    if model_type in {'edc', 'edc-original'}:
        canonicalize = curr_dataset_to_merge['canonicalize']
        _merge_edc_type(loaded_main_dataset, main_hash_id_to_line_idx,
                        curr_dataset_to_merge, snapshot_to_property_id_to_label,
                        canonicalize)

    elif model_type in {'relik', 'kg-gen', 'rakg', 'rebel', 'tool'}:
        _merge_jsonl_type(loaded_main_dataset, main_hash_id_to_line_idx,
                          input_path, model_name, model_type)

    else:
        raise RuntimeError(f'unrecognized predictions_model_type: {model_type}')


def _merge_edc_type(
    loaded_main_dataset: List[Dict],
    main_hash_id_to_line_idx: Dict[str, int],
    curr_dataset_to_merge: Dict,
    snapshot_to_property_id_to_label: Dict[str, Dict[str, str]],
    canonicalize: bool,
) -> None:
    """Merge EDC-style predictions (JSON files with _last_processed_results_list suffix)."""
    input_path = curr_dataset_to_merge['input_dataset_path']
    model_name = curr_dataset_to_merge['predictions_model_name']
    model_type = curr_dataset_to_merge['predictions_model_type']

    for dirpath, _, filenames in os.walk(input_path):
        for fname in sorted(filenames):
            if not fname.endswith('_last_processed_results_list.json'):
                continue
            fpath = os.path.join(dirpath, fname)
            logger.info(f'processing EDC file: {fname}')
            with open(fpath, 'rt', encoding='utf-8') as f:
                edc_predictions_content = json.load(f)
                for curr_prediction in edc_predictions_content:
                    hash_id = curr_prediction['hash_id']
                    if hash_id in main_hash_id_to_line_idx:
                        idx = main_hash_id_to_line_idx[hash_id]
                        loaded_main_dataset[idx] = merge_edc_predictions(
                            instance_dataset=loaded_main_dataset[idx],
                            predictions=curr_prediction,
                            model_name=model_name,
                            model_type=model_type,
                            hash_id=hash_id,
                            canonicalize=canonicalize,
                            snapshot_to_property_id_to_label=snapshot_to_property_id_to_label,
                        )


def _merge_jsonl_type(
    loaded_main_dataset: List[Dict],
    main_hash_id_to_line_idx: Dict[str, int],
    input_path: str,
    model_name: str,
    model_type: str,
) -> None:
    """Merge JSONL-style predictions (relik, kg-gen, rakg, rebel, tool).

    Idempotent: skips instances where model_name is already in predictions.
    Handles both list and dict prediction formats.
    Only merges the specific model_name — ignores other models in the file.
    """
    for dirpath, _, filenames in os.walk(input_path):
        for fname in sorted(filenames):
            if not fname.endswith('.jsonl'):
                continue
            fpath = os.path.join(dirpath, fname)
            logger.info(f'processing {model_type} file: {fname}')
            with open(fpath, 'rt', encoding='utf-8') as f:
                for line in f:
                    parsed_line = json.loads(line)
                    hash_id = parsed_line['hash_id']
                    if hash_id not in main_hash_id_to_line_idx:
                        logger.warning(f'hash_id not found in main dataset: {hash_id}')
                        continue

                    idx = main_hash_id_to_line_idx[hash_id]
                    main_parsed_line = loaded_main_dataset[idx]
                    if 'predictions' not in main_parsed_line:
                        main_parsed_line['predictions'] = {}

                    if model_name in main_parsed_line['predictions']:
                        continue

                    source_preds = parsed_line['predictions']
                    if isinstance(source_preds, dict):
                        if model_name not in source_preds:
                            raise KeyError(
                                f'model_name={model_name} not found in predictions dict '
                                f'keys={list(source_preds.keys())} for hash_id={hash_id}'
                            )
                        pred = source_preds[model_name]
                    elif isinstance(source_preds, list):
                        if len(source_preds) != 1:
                            raise RuntimeError(
                                f'expected exactly 1 prediction in list, got {len(source_preds)} '
                                f'for hash_id={hash_id} model_name={model_name}'
                            )
                        pred = source_preds[0]
                        pred['model_type'] = model_type
                    else:
                        raise RuntimeError(
                            f'unexpected predictions type: {type(source_preds)} '
                            f'for hash_id={hash_id}'
                        )

                    main_parsed_line['predictions'][model_name] = pred
                    loaded_main_dataset[idx] = main_parsed_line


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def merge_all_datasets(
    loaded_main_dataset: List[Dict],
    main_hash_id_to_line_idx: Dict[str, int],
    input_other_datasets_paths: List[Dict],
    snapshot_to_property_id_to_label: Dict[str, Dict[str, str]],
) -> None:
    """Merge all other datasets (assessments and predictions) into the main dataset."""
    for curr_dataset_to_merge in input_other_datasets_paths:
        if curr_dataset_to_merge['merge_predictions']:
            merge_predictions_into_main_dataset(
                loaded_main_dataset=loaded_main_dataset,
                main_hash_id_to_line_idx=main_hash_id_to_line_idx,
                curr_dataset_to_merge=curr_dataset_to_merge,
                snapshot_to_property_id_to_label=snapshot_to_property_id_to_label,
            )
        else:
            merge_assessments_into_main_dataset(
                loaded_main_dataset=loaded_main_dataset,
                main_hash_id_to_line_idx=main_hash_id_to_line_idx,
                curr_dataset_to_merge=curr_dataset_to_merge,
            )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_merged_dataset(
    loaded_main_dataset: List[Dict],
    main_hash_id_to_line_idx: Dict[str, int],
    input_main_dataset_path: str,
    output_merged_dataset_path: str,
) -> None:
    """Save the merged dataset preserving the original file structure."""
    logger.info(f'saving merged dataset to {output_merged_dataset_path}')

    for dirpath, _, filenames in os.walk(input_main_dataset_path):
        for fname in sorted(filenames):
            if not fname.endswith('.jsonl'):
                continue
            input_file_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(input_file_path, input_main_dataset_path)
            output_file_path = os.path.join(output_merged_dataset_path, rel_path)

            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, 'wt', encoding='utf-8') as outfile:
                with open(input_file_path, 'rt', encoding='utf-8') as f:
                    for line in tqdm(f, desc=f'writing {fname}'):
                        parsed_line = json.loads(line)
                        hash_id = parsed_line['hash_id']
                        merged_content = loaded_main_dataset[main_hash_id_to_line_idx[hash_id]]
                        outfile.write(json.dumps(merged_content, ensure_ascii=False) + '\n')

    logger.info('merged dataset saved')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Merge predictions into a main dataset.')
    parser.add_argument('--config_file', required=True, type=str, help='Path to config JSON.')
    args = parser.parse_args()

    with open(args.config_file, 'rt', encoding='utf-8') as f:
        config = json.load(f)

    loaded_main_dataset, main_hash_id_to_line_idx = load_main_dataset(
        config['input_main_dataset_path'],
    )

    snapshot_to_property_id_to_label: Dict[str, Dict[str, str]] = {}
    if 'snapshots_to_schema' in config:
        base_target_schema_dir = config['base_target_schema_dir']
        for curr_schema_snapshot in config['snapshots_to_schema']:
            curr_snapshot_date = curr_schema_snapshot['snapshot_date']
            target_schema_path = curr_schema_snapshot['target_schema_path']
            schema_file_path = os.path.join(base_target_schema_dir, target_schema_path)
            rel_id_to_label = {}
            with open(schema_file_path, 'rt', encoding='utf-8') as schema_f:
                for curr_line in tqdm(schema_f, desc=f'loading schema {target_schema_path}'):
                    curr_parsed_line = json.loads(curr_line)
                    rel_id_to_label[curr_parsed_line['metadata']['property']] = curr_parsed_line['text']
            snapshot_to_property_id_to_label[curr_snapshot_date] = rel_id_to_label

    merge_all_datasets(
        loaded_main_dataset=loaded_main_dataset,
        main_hash_id_to_line_idx=main_hash_id_to_line_idx,
        input_other_datasets_paths=config['input_other_datasets_paths'],
        snapshot_to_property_id_to_label=snapshot_to_property_id_to_label,
    )

    save_merged_dataset(
        loaded_main_dataset=loaded_main_dataset,
        main_hash_id_to_line_idx=main_hash_id_to_line_idx,
        input_main_dataset_path=config['input_main_dataset_path'],
        output_merged_dataset_path=config['output_merged_dataset_path'],
    )


if __name__ == '__main__':
    main()
