# unlike _v7,
# _v8 uses and produces the final format introduced by s06b_refactor_final_format.

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import logging
import os
import random
import subprocess
import time
import traceback
from typing import Tuple, Dict, Set

import requests
import tiktoken
from huggingface_hub.errors import GenerationError
from tqdm import tqdm

import os
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from huggingface_hub import InferenceClient

import nltk
from nltk.corpus import words
import subprocess

from dataset.emerge.utils.s05_generate_dataset_utils_v8 import generate_instances_v8

# Download the necessary NLTK data files
nltk.download('words')
nltk.download('punkt')

# Load the list of English words
english_words = set(words.words())

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)


def count_lines(filename):
    return int(subprocess.check_output(['wc', '-l', filename]).split()[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/s05_generate_dataset_with_llm/20241106/'
                                's05_generate_dataset_with_llm.json',
                        help='The config file that contains all the parameters')

    parser.add_argument('--api_llm_port',
                        type=int,
                        required=True,
                        default=8080,
                        help='The port where the LLM API is deployed.')

    parser.add_argument('--max_workers',
                        type=int,
                        # default='./output/tmp/last_processed.json',
                        default=32,
                        help='Max threads to call TGI api.')

    parser.add_argument('--api_llm_device',
                        type=int,
                        required=False,
                        # required=True,
                        default=0,
                        help='The device where the LLM API is deployed.')

    parser.add_argument('--shuffle_input_file',
                        help='Shuffles the input file passed in --input_file parameter.',
                        action='store_true')

    # parser.add_argument('--restart_apptainer_script_name',
    #                     help='The name of the script to restart the apptainer.',
    #                     default='s05_restart_apptainer_v3.sh'
    #                     )

    parser.add_argument('--wait_for_restart_time',
                        type=int,
                        help='Time in seconds to wait for TGI apptainer restart script.',
                        default=60
                        )

    parser.add_argument('--llm_assessor_name',
                        help='The name of llm used as evaluator.',
                        default='Meta-Llama-3.1-8B'
                        )

    parser.add_argument('--dry_run',
                        help='Dry run, do not without invoking llms.',
                        action='store_true')

    tokenizer_splitter = tiktoken.get_encoding("cl100k_base")

    args = parser.parse_args()
    config = json.load(open(args.config_file, 'rt'))
    dry_run = args.dry_run
    # input_file = args.input_file
    input_files_to_process = set(config['input_files_to_process'])

    input_dir_joined_snippets_triples = config['input_dir_joined_snippets_triples']
    # prompt_type = config['prompt_type']
    llm_tokenizer = config['llm_tokenizer']
    llm_client_url = config['llm_client_url']
    output_dir = config['output_dir']
    last_processed_dir = config['last_processed_dir']
    config['dry_run'] = dry_run
    config['shuffle_input_file'] = args.shuffle_input_file
    config['wait_for_restart_time'] = args.wait_for_restart_time
    config['api_llm_port'] = args.api_llm_port
    config['api_llm_device'] = args.api_llm_device
    config['llm_assessor_name'] = args.llm_assessor_name
    # config['input_file'] = args.input_file
    # config['restart_apptainer_script_name'] = args.restart_apptainer_script_name
    config['max_workers'] = args.max_workers
    # config['only_removed_triples'] = args.only_removed_triples

    max_triples_batch_size = config['max_triples_batch_size']

    logger.info(f'the following config is being processed: {config}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(last_processed_dir, exist_ok=True)
    # last_processed_file = os.path.join(last_processed_dir, input_file)

    #
    input_files = os.listdir(input_dir_joined_snippets_triples)
    # removed_triple_actions = {'qualifier_removed_edge', 'removed_edge'}
    # addition_triple_actions = {'added_edge', 'qualifier_added_edge'}

    continue_looping = True
    tokenizer: PreTrainedTokenizerBase = None
    client: InferenceClient = None
    while continue_looping:
        try:
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(llm_tokenizer)
            llm_client_url_with_port = f'{llm_client_url}:{args.api_llm_port}'
            logger.info(f'starting llm_client_url_with_port in {llm_client_url_with_port}')
            client: InferenceClient = InferenceClient(llm_client_url_with_port)
            continue_looping = False
        except Exception as e:
            # Print the stack trace
            traceback.print_exc()
            logger.error('error above occurred, sleeping 10 secs and continuing')
            # Continue execution
            time.sleep(10)

    nr_triples_ignored = 0

    parsed_lines_to_process = list()
    logger.info(f'inside python script passed parameters: '
                f'dry_run: {dry_run}, config_file: {args.config_file}')

    for dirpath, dirnames, filenames in os.walk(config['input_dir_joined_snippets_triples']):
        for input_file in filenames:
            last_processed_file = os.path.join(last_processed_dir, f'{input_file}.txt')

            if not input_file.endswith('.jsonl'):
                continue

            parsed_lines_to_process = list()

            if len(input_files_to_process) > 0 and input_file not in input_files_to_process:
                logger.info(f'ignoring_the_following_file: {input_file}')
                continue

            output_file_path = os.path.join(output_dir, input_file)
            # outfile = open(output_file_path, 'wt', encoding='utf-8')
            input_file_path = os.path.join(dirpath, input_file)
            logger.info(f'starting processing {input_file_path}')
            nr_lines_in_infile = count_lines(input_file_path)
            from threading import Lock

            counter_lock = Lock()
            counter = {'count': 0}

            ##
            if 'prompt_single_assert_path' in config:
                with open(config['prompt_single_assert_path'], 'r', encoding='utf-8') as f:
                    config['assert_single_prompt_template_content'] = f.read()
            if 'prompt_multi_assert_path' in config:
                with open(config['prompt_multi_assert_path'], 'r', encoding='utf-8') as f:
                    config['assert_multi_prompt_template_content'] = f.read()
            if 'prompt_single_deprecate_path' in config:
                with open(config['prompt_single_deprecate_path'], 'r', encoding='utf-8') as f:
                    config['deprecate_single_prompt_template_content'] = f.read()
            if 'prompt_multi_deprecate_path' in config:
                with open(config['prompt_multi_deprecate_path'], 'r', encoding='utf-8') as f:
                    config['deprecate_multi_prompt_template_content'] = f.read()
            ##

            generate_instances_v8(
                config=config,
                tokenizer=tokenizer,
                client=client,
                tokenizer_splitter=tokenizer_splitter,
                output_file_path=output_file_path,
                input_file_path=input_file_path,
                counter=counter,
                counter_lock=counter_lock,
                start_time_all=time.time(),
                last_processed_file_path=last_processed_file
            )
    logger.info(f'FINALLY_FINISHED_BYE_BYE, YOU CAN KILL ME')
