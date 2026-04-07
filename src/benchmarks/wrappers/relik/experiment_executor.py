import json
import logging
import os
from typing import Dict, List

from tqdm import tqdm

from benchmark_model import BenchmarkModel
from prediction import Prediction


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


# def join(*parts) -> str:
#     return str(os.path.join(*(str(p) for p in parts)))

# def join(*parts: str) -> str:
#     parts = [str(p) for p in parts]   # convert all to strings
#     return str(os.path.join(*parts))

def join(*parts: str) -> str:
    path = ""
    for p in parts:
        path = os.path.join(path, str(p)) if path else str(p)
    return path


class ExperimentExecutor():
    def __init__(self, config: Dict,
                 config_benchmark: Dict,
                 benchmark_model: BenchmarkModel,
                 snapshot_idx: int):

        self.output_results_dir = join(
            config['output_base_dir'],
            config_benchmark['output_dir']
        )

        self.last_processed_dir = join(
            config['last_processed_base_dir'],
            config_benchmark['last_processed_dir']
        )
        self.input_dataset_path: str = config['input_dataset_base_path']
        self.model_name = config_benchmark['name']
        self.display_score_every = config['display_score_every']
        self.model_config_name = config_benchmark['config_name']
        self.dataset_alias = ''
        self.entity_index_alias = ''
        self.relation_index_alias = ''

        if snapshot_idx > -1:

            snap = config_benchmark['snapshots'][snapshot_idx]

            self.dataset_alias = snap['dataset_alias']
            self.entity_index_alias = snap['entity_index_alias']
            self.relation_index_alias = snap['relation_index_alias']

            suffix = (
                f"{self.dataset_alias}_"
                f"ent_idx_{self.entity_index_alias}_"
                f"rel_idx_{self.relation_index_alias}"
            )

            self.output_results_dir = join(self.output_results_dir, suffix)
            self.last_processed_dir = join(self.last_processed_dir, suffix)
            self.input_dataset_path: str = join(
                self.input_dataset_path,
                config_benchmark['snapshots'][snapshot_idx]['dataset_path']
            )

        self.batch_size = config_benchmark['batch_size']

        os.makedirs(self.output_results_dir, exist_ok=True)
        os.makedirs(self.last_processed_dir, exist_ok=True)

        self.benchmark_model: BenchmarkModel = benchmark_model
        self.snapshot_idx = snapshot_idx

    def process_batch(self, texts_to_process: List, output_file, last_processed_file, last_successfully_processed):
        text_chunks: List[str]
        text_chunks = [tp['passage'] for tp in texts_to_process]
        mentions_to_qids: List[Dict[str, str]]  # we are here, complete this!!
        # timestamps = [int(tp['passage_timestamp']) for tp in texts_to_process]
        timestamps = [int(tp['revision_timestamp']) for tp in texts_to_process]
        mentions_to_qids = [{k: v for d in tp['mentions'] for k, v in d.items()} for tp in
                            texts_to_process]
        predictions: List[Prediction] = \
            self.benchmark_model.run(text_chunks, timestamps,
                                     mentions_to_qids=mentions_to_qids)
        for idx2, curr_prediction in enumerate(predictions):
            if 'predictions' not in texts_to_process[idx2]:
                texts_to_process[idx2]['predictions'] = list()

            curr_prediction_json = {
                'predicted_triples': list(curr_prediction.predicted_triples),
                'model': self.model_name,
                'model_config_name': self.model_config_name,
                'dataset_alias': self.dataset_alias,
                'entity_index_alias': self.entity_index_alias,
                'relation_index_alias': self.relation_index_alias,
                'dataset_path': self.input_dataset_path,
                'entity_index_path': self.benchmark_model.entity_index_path,
                'relation_index_path': self.benchmark_model.relation_index_path
            }

            texts_to_process[idx2]['predictions'].append(curr_prediction_json)
            output_file.write(json.dumps(texts_to_process[idx2], ensure_ascii=False) + '\n')
            output_file.flush()

        os.makedirs(os.path.dirname(last_processed_file), exist_ok=True)
        with open(last_processed_file, 'w', encoding='utf-8') as file:
            file.write(
                str(last_successfully_processed))  # Convert number to string and write to file
            file.flush()
        # texts_to_process = list()

    def run(self):
        logger.info(f'Running {self.benchmark_model} on '
                    f'files inside: {self.input_dataset_path}, '
                    f'the results will be saved to {self.output_results_dir}')

        texts_to_process: List[Dict] = list()
        for dirpath, dirnames, filenames in os.walk(self.input_dataset_path):
            for curr_filename in filenames:
                rel_dir = os.path.relpath(dirpath, self.input_dataset_path)
                if rel_dir == ".":
                    rel_dir = ""
                rel_complete_dir = os.path.join(self.output_results_dir, rel_dir)
                os.makedirs(rel_complete_dir, exist_ok=True)
                output_file_path = os.path.join(rel_complete_dir, curr_filename)
                # output_file = open(output_file_path, 'wt', encoding='utf-8')
                output_file = open(output_file_path, 'a', encoding='utf-8')
                last_processed_file = os.path.join(self.last_processed_dir, rel_dir, curr_filename)
                last_successfully_processed = -1

                if os.path.exists(last_processed_file):
                    with open(last_processed_file, 'r') as file:
                        last_successfully_processed = int(
                            file.read())  # Read the string and convert it back to an integer

                file_path = os.path.join(dirpath, curr_filename)  # Get the full file path
                logger.info(f'processing file_path: {file_path} and saving output to {output_file_path}')
                #
                line_count = sum(1 for line in open(file_path, 'rt'))
                outputted = False
                with open(file_path, 'rt', encoding='utf-8') as file:
                    for idx_line, line_str in enumerate(
                            tqdm(file, desc=f'predicting for: {curr_filename}', total=line_count)):
                        if idx_line <= last_successfully_processed:
                            # was already successfully processed
                            logger.debug(f'detected that the line {idx_line} was successfully processed in some '
                                         f'execution that happened before, so not processing it and ignoring')
                            continue
                        if idx_line > last_successfully_processed and not outputted:
                            logger.info(f'starting processing from the idx_line {idx_line}')
                            outputted = True

                        parsed_json = json.loads(line_str)
                        texts_to_process.append(parsed_json)
                        if len(texts_to_process) >= self.batch_size:
                            last_successfully_processed = idx_line
                            self.process_batch(texts_to_process=texts_to_process,
                                               output_file=output_file,
                                               last_processed_file=last_processed_file,
                                               last_successfully_processed=last_successfully_processed)
                            texts_to_process = list()
                    if len(texts_to_process) > 0:
                        logger.info(f'finishing the last batch of size {len(texts_to_process)}')
                        last_successfully_processed = idx_line
                        self.process_batch(texts_to_process=texts_to_process,
                                           output_file=output_file,
                                           last_processed_file=last_processed_file,
                                           last_successfully_processed=last_successfully_processed)
                        texts_to_process = list()
