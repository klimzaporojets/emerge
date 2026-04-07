# unlike _v8 uses and produces the final format introduced by s06b_refactor_final_format.
import os
import traceback
import subprocess
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import json
import logging
import time
from huggingface_hub.errors import GenerationError

from typing import Dict, Tuple, Set, List, Any

from huggingface_hub import InferenceClient
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from typing_extensions import deprecated

import os
from dataset.emerge.utils.constants import ACTION_CATEGORY_DEPRECATE, ACTION_CATEGORY_ASSERT
from dataset.emerge.utils.s03_utils import split_text

from nltk.tokenize import word_tokenize
from nltk.corpus import words

from dataset.emerge.utils.s05_prompt_llm_utils_v8 import call_llm_and_return_parsed_result_v8, parse_returned_llm_text_v8

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)

# Load the list of English words
english_words = set(words.words())
import threading
sync_once_event = threading.Event()

#
# # define a global lock
# sync_lock = threading.Lock()


def calculate_english_word_percentage(text):
    # Tokenize the input text
    tokens = word_tokenize(text)

    # Filter out non-English words
    english_word_count = sum(1 for word in tokens if word.lower() in english_words)

    # Calculate the percentage
    total_words = len(tokens)
    percentage = (english_word_count / total_words) * 100 if total_words > 0 else 0

    return percentage


def is_script_running(script_name):
    try:
        # Use pgrep to search for the script name
        result = subprocess.run(['pgrep', '-f', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            logger.info(f"Script '{script_name}' is_running_with_pid(s): {result.stdout.decode().strip()}")
            return True
    except Exception as e:
        logger.info(f"Error occurred: {e}")
    return False


def split_into_batches(lst: List,
                       batch_size: int,
                       parsed_line_idx: int,
                       chunks: List[str],
                       # max_nr_tokens_in_chunk: int,
                       # tokenizer_splitter,
                       chunk_timestamp: int,
                       action: str) -> List[Dict[str, Any]]:
    batches = []
    for curr_chunk in chunks:
        for i in range(0, len(lst), batch_size):
            batch = lst[i:i + batch_size]
            # batch_act = [tuple(trpl + [action]) for trpl in batch]
            # in batch_act we are only interested in qids, not labels, which is a mess of decoding
            batch_act = [tuple([trpl[0][:trpl[0].index('(') - 1],
                                trpl[1][:trpl[1].index('(') - 1],
                                trpl[2][:trpl[2].index('(') - 1]
                                ] + [action]) for trpl in batch]
            logger.debug(f'batch_is: {batch} '
                        f'and batch_act is: {batch_act}')
            batch_str = [str(instance_in_batch) for instance_in_batch in batch]
            batches.append(
                {
                    'line_idx': parsed_line_idx,
                    'labeled_batch': batch_str,
                    'triples_w_actions_batch': batch_act,
                    # 'triples_batch_str': batch_str,
                    'chunk': curr_chunk,
                    'chunk_timestamp': chunk_timestamp,
                    'action': action
                }
            )
    return batches


# def obtain_curr_parsed_triple(curr_triple, qid_to_mentions, config):
#     pass

def obtain_curr_parsed_triple(curr_triple, qid_to_mentions, config):
    # triple_head = ''
    # triple_tail = ''
    triple_entry_name = 'triple'
    if 'triple_qids' in curr_triple:
        triple_entry_name = 'triple_qids'

    if curr_triple[triple_entry_name][0] in qid_to_mentions:
        triple_head = f'{" / ".join(set(qid_to_mentions[curr_triple[triple_entry_name][0]]))}'
    else:
        triple_head = curr_triple['triple_labels'][0]

    if curr_triple[triple_entry_name][2] in qid_to_mentions:
        triple_tail = f'{" / ".join(set(qid_to_mentions[curr_triple[triple_entry_name][2]]))}'
    else:
        triple_tail = curr_triple['triple_labels'][2]

    if not config['triples_only_labels']:
        to_ret_curr_parsed_triple = \
            [
                f'{curr_triple[triple_entry_name][0]} '
                f'({triple_head})',
                f'{curr_triple[triple_entry_name][1]} '
                f'({curr_triple["triple_labels"][1]})',
                f'{curr_triple[triple_entry_name][2]} '
                f'({triple_tail})'
            ]
    else:
        to_ret_curr_parsed_triple = \
            [
                f'{triple_head}',
                f'{curr_triple["triple_labels"][1]}',
                f'{triple_tail}'
            ]

    return to_ret_curr_parsed_triple


def process_batch_of_lines(line_idx,
                           chunk,
                           chunk_timestamp: int,
                           to_query_triple_batch,  # aka "labeled_batch"
                           action_category,
                           file_name,
                           config,
                           prompt_type,
                           tokenizer,
                           client):
    # here multiprocessing and returns LLM results, but no further processing
    continue_looping = True
    logger.debug('post 1')
    already_numbered = False
    nr_errors = 0
    while continue_looping and nr_errors < 2:
        try:
            if len(to_query_triple_batch) > 1:
                action_type = 'multiple'
                if not already_numbered:
                    to_query_triple_batch = \
                        [f"{j + 1}. {item}" for j, item in enumerate(to_query_triple_batch)]
                    already_numbered = True
                added_triples_str = '\n'.join(to_query_triple_batch)
                # logger.info(f'process_batch_of_lines_multiple to '
                #             f'{to_query_triple_batch} --> '
                #             f'{added_triples_str}')
            else:
                action_type = 'single'
                # added_triples_str = str(list(to_query_triple_batch[0]))
                assert len(to_query_triple_batch) == 1
                added_triples_str = to_query_triple_batch[0]
                # logger.info(f'process_batch_of_lines_single to '
                #             f'{to_query_triple_batch} --> '
                #             f'{added_triples_str}')
            logger.debug(f'line_idx {line_idx},'
                        f' to_query_triple_str_is {added_triples_str} '
                        f' to_query_triple_batch_is {to_query_triple_batch}')
            str_response = \
                call_llm_and_return_parsed_result_v8(
                    prompt_type=prompt_type,
                    chunk=chunk,
                    chunk_timestamp=chunk_timestamp,
                    triples_str=added_triples_str,
                    action_type=action_type,
                    tokenizer=tokenizer,
                    client=client,
                    config=config,
                    action_category=action_category
                )
            #
            to_ret = {
                'line_idx': line_idx,
                'str_response': str_response,
                'action_category': action_category,
                'action_type': action_type,
                'triples_str': added_triples_str
            }
            continue_looping = False
            # sync_once_event.clear()
            return to_ret
        except GenerationError as e:
            nr_errors += 1
            script_name = config['restart_apptainer_script_name']
            error_message = traceback.format_exc()
            logger.error(f'GenerationError stack is as follows: '
                         f'{error_message}')
            #
            if 'Server error: error trying to connect: No such file or directory' in error_message:
                logger.error(f'{config["api_llm_device"]}:{config["api_llm_port"]}'
                             f' {file_name}:{line_idx} error '
                             f'above occurred, this is generation_error '
                             'yet something related to file or '
                             'directory, so NOT IGNORING ignore the following chunk: '
                             f'{chunk} by setting continue_looping to True'
                             f', sleeping 10 secs and continuing and also restarting the LLM')
            else:
                # continue_looping = False
                logger.error(f'{config["api_llm_device"]}:{config["api_llm_port"]} '
                             f' {file_name}:{line_idx} '
                             'error above occurred, this is generation_error '
                             'so i just ignore the following chunk: '
                             f'{chunk} by setting continue_looping to True'
                             f', sleeping 10 secs and continuing and also restarting the LLM')
            # Continue execution
            logfile = f'run_{random.randint(1000, 9999)}.log'
            command = f'bash scripts/slurm/{config["restart_apptainer_script_name"]} 2>&1 | tee {logfile}'

            # command = (f'bash scripts/slurm/{config["restart_apptainer_script_name"]} '
            #            f'{config["api_llm_device"]} -- {config["api_llm_port"]}')
            if not is_script_running(script_name):
                if not sync_once_event.is_set():  # check if already executed
                    # only one thread can enter here
                    with threading.Lock():
                        if not sync_once_event.is_set():
                            # <<< this runs only once >>>
                            sync_once_event.set()  # mark as done

                            logger.info(f'about to run: {command}')
                            result = subprocess.run(command, shell=True, text=True, capture_output=True)
                            logger.info(f'just have run: {command} ; '
                                        f'standard output: {result.stdout}; '
                                        f'standard error: {result.stderr}; '
                                        f'return code: {result.returncode}')
                            sync_once_event.clear()
                        else:
                            logger.info(f'do_not_entering_synchronized1, sleeping: {config["wait_for_restart_time"]}')
                            time.sleep(config['wait_for_restart_time'])
                else:
                    logger.info(f'do_not_entering_synchronized2, sleeping: {config["wait_for_restart_time"]}')
                    time.sleep(config['wait_for_restart_time'])
                # random_milliseconds = random.randint(5000, 15000)
                # random_seconds = random_milliseconds / 1000
                # time.sleep(random_seconds)
            else:
                logger.info(f'script_already_running1: {command} '
                            f'sleeping for {config["wait_for_restart_time"]} secs')
                time.sleep(config['wait_for_restart_time'])


        except requests.exceptions.ConnectionError as e:

            nr_errors += 1
            # Example usage
            script_name = config['restart_apptainer_script_name']

            if not is_script_running(script_name):

                if not sync_once_event.is_set():  # check if already executed
                    # only one thread can enter here
                    with threading.Lock():
                        if not sync_once_event.is_set():
                            # <<< this runs only once >>>
                            sync_once_event.set()  # mark as done

                            logger.error(f'connection_error_ocurred, executing restarting'
                                         f'script {script_name}')
                            # Continue execution
                            logfile = f'run_{random.randint(1000, 9999)}.log'
                            command = f'bash scripts/slurm/{config["restart_apptainer_script_name"]} 2>&1 | tee {logfile}'

                            logger.info(f'about to run: {command}')
                            result = subprocess.run(command, shell=True, text=True, capture_output=True)
                            logger.info(f'just have run: {command} ; '
                                        f'standard output: {result.stdout}; '
                                        f'standard error: {result.stderr}; '
                                        f'return code: {result.returncode}')
                            sync_once_event.clear()
                        else:
                            logger.info(
                                f'do_not_entering_synchronized1b, sleeping: {config["wait_for_restart_time"]}')
                            time.sleep(config['wait_for_restart_time'])
                else:
                    logger.info(
                        f'do_not_entering_synchronized2b, sleeping: {config["wait_for_restart_time"]}')
                    time.sleep(config['wait_for_restart_time'])

            else:
                traceback.print_exc()
                # logger.error(f'{config["api_llm_device"]}:{config["api_llm_port"]} - '
                logger.error(f'script_already_running_error2 ConnectionError2 occurred, yet '
                             f'{script_name} is running, so sleeping for '
                             f'{config["wait_for_restart_time"]} secs before'
                             f'continuing')
                # time.sleep(210)
                time.sleep(config['wait_for_restart_time'])


        except Exception as e:
            nr_errors += 1
            # Print the stack trace
            logger.error('other_exception_occurred, '
                         'sleeping around a min secs and continuing')
            traceback.print_exc()
            # Generate a random number of milliseconds between 5000 and 15000
            random_milliseconds = random.randint(60000, 80000)

            # Convert milliseconds to seconds
            random_seconds = random_milliseconds / 1000
            # Sleep for the random duration
            time.sleep(random_seconds)
            # time.sleep(config['wait_for_restart_time'])
            #
    logger.error('extreme_error_returning_none '
                 f'following_input: '
                 f'line_idx={line_idx},'
                f'chunk={chunk},' 
                f'chunk_timestamp={chunk_timestamp},' 
                f'to_query_triple_batch={to_query_triple_batch},'
                f'action_category={action_category},'
                f'file_name={file_name},'
                f'config={config},'
                f'prompt_type={prompt_type}')
    return None


def throw_threads(config,
                  to_query_triples,
                  curr_input_file,
                  tokenizer,
                  client,
                  llm_triples_to_answer,
                  all_identified_triples,
                  llm_triples_to_prompt_responses):
    #
    with (ThreadPoolExecutor(max_workers=config['max_workers']) as executor):
        futures = []
        for curr_triples in tqdm(to_query_triples,
                                 desc=f'iterating over {curr_input_file}'):
            futures.append(executor.submit(
                process_batch_of_lines,
                curr_triples['line_idx'],
                curr_triples['chunk'],
                curr_triples['chunk_timestamp'],
                curr_triples['labeled_batch'],
                curr_triples['action'],
                curr_input_file,
                config,
                config['prompt_type'],
                tokenizer,
                client
            ))

            # for curr_future in tqdm(as_completed(futures),
            #                         desc=f'completed futures {curr_input_file}',
            #                         total=len(to_query_triples)):
        for curr_future in tqdm(as_completed(futures),
                                desc=f'processing {curr_input_file}',
                                total=len(to_query_triples),
                                smoothing=0):
            logger.debug(f'curr_future is: {curr_future}')
            curr_future_result = curr_future.result()
            logger.debug(f'curr_future_result is: {curr_future_result}')
            if curr_future_result is None:
                logger.error('curr_future_result_in_none')
                return
            curr_line_idx = curr_future_result['line_idx']
            str_response = curr_future_result['str_response']
            action_category = curr_future_result['action_category']
            action_type = curr_future_result['action_type']
            triples_str = curr_future_result['triples_str']

            assert action_type in {'single', 'multiple'}
            # llm_triples_to_answer, all_identified_triples,
            #             # output_text_all,
            #             llm_triples_to_prompt_responses
            (llm_triples_to_answer[curr_line_idx],
             all_identified_triples[curr_line_idx],
             llm_triples_to_prompt_responses[curr_line_idx]) = \
                parse_returned_llm_text_v8(
                    is_single=(action_type == 'single'),
                    all_identified_triples=all_identified_triples[curr_line_idx],
                    output_text=str_response,
                    action_category=action_category,
                    config=config,
                    llm_triples_to_answer=llm_triples_to_answer[curr_line_idx],
                    llm_triples_to_prompt_responses=llm_triples_to_prompt_responses[curr_line_idx],
                    triples_str=triples_str
                )


def read_last_processed(path):
    if os.path.exists(path):
        with open(path) as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return -1
    return -1


def save_last_processed(path, number):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wt') as f:
        f.write(str(number))


def save_checkpoint(input_file_path,
                    config,
                    to_query_triples,
                    tokenizer,
                    client,
                    llm_triples_to_answer,
                    all_identified_triples,
                    llm_triples_to_prompt_responses,
                    line_idx_to_parsed_line,
                    checkpoint_line_idx,
                    last_processed_file_path,
                    output_file):
    # if len(to_query_triples) == 0:
    #     logger.warning('to_query_triples_in_zero')
    #     return
    deprecated_triple_tkgus = set(config['deprecated_triple_tkgus'])
    addition_triple_tkgus = set(config['addition_triple_tkgus'])
    tkgu_types = set(config['tkgu_types'])
    curr_input_file = os.path.basename(input_file_path)
    logger.debug('PASS 1 throwing threads')
    throw_threads(config,
                  to_query_triples,
                  curr_input_file,
                  tokenizer,
                  client,
                  llm_triples_to_answer,
                  all_identified_triples,
                  llm_triples_to_prompt_responses)
    logger.info('now checking the triples that have not been extracted')

    to_query_failed_triples_in_batch = list()
    for curr_to_query_triple_batch in to_query_triples:
        curr_q_line_idx = curr_to_query_triple_batch['line_idx']
        # triples_batch = curr_to_query_triple_batch['triples_batch']
        triples_w_actions_batch = curr_to_query_triple_batch['triples_w_actions_batch']

        for idx_triple, curr_triple in enumerate(triples_w_actions_batch):
            if curr_triple not in all_identified_triples[curr_q_line_idx]:
                to_add_labeled_batch = curr_to_query_triple_batch['labeled_batch'][idx_triple]
                logger.warning(f'something_wrong_with_curr_triple {curr_triple} '
                               f'all_identified_triples[curr_q_line_idx]: '
                               f'{all_identified_triples[curr_q_line_idx]} '
                               f'for line_idx {curr_q_line_idx}, adding to reprocess '
                               f'the following labeled batch: {to_add_labeled_batch}')

                to_query_failed_triples_in_batch.append(
                    {
                        # 'labeled_batch': [str(list(curr_triple[:3]))],
                        'labeled_batch': [to_add_labeled_batch],
                        'triples_w_actions_batch': [curr_triple],
                        'chunk': curr_to_query_triple_batch['chunk'],
                        'chunk_timestamp': curr_to_query_triple_batch['chunk_timestamp'],
                        'line_idx': curr_to_query_triple_batch['line_idx'],
                        'action': curr_to_query_triple_batch['action']
                    }
                )

    logger.info('PASS 2 throwing threads')
    throw_threads(config,
                  to_query_failed_triples_in_batch,
                  curr_input_file,
                  tokenizer,
                  client,
                  llm_triples_to_answer,
                  all_identified_triples,
                  llm_triples_to_prompt_responses)

    logger.info('impacting assessment results into the dataset')
    #
    for curr_line_idx, curr_parsed_line in line_idx_to_parsed_line.items():

        qid_to_mentions = dict()
        for curr_mention in curr_parsed_line['mentions']:
            if curr_mention['qid'] not in qid_to_mentions:
                qid_to_mentions[curr_mention['qid']] = list()
            qid_to_mentions[curr_mention['qid']].append(curr_mention['mention_text'])

        for curr_tkgu_triple in curr_parsed_line['tkgu_triples']:

            curr_parsed_triple = \
                obtain_curr_parsed_triple(curr_tkgu_triple, qid_to_mentions, config)

            curr_parsed_triple = [
                curr_parsed_triple[0][:curr_parsed_triple[0].index('(') - 1],
                curr_parsed_triple[1][:curr_parsed_triple[1].index('(') - 1],
                curr_parsed_triple[2][:curr_parsed_triple[2].index('(') - 1]
            ]

            curr_action_categories = set()
            triple_tkgu_operations = set(curr_tkgu_triple['tkgu_operations'])
            #
            intersection_deprecation_tkgus = \
                triple_tkgu_operations \
                    .intersection(deprecated_triple_tkgus) \
                    .intersection(tkgu_types)
            #
            intersection_addition_tkgus = \
                triple_tkgu_operations \
                    .intersection(addition_triple_tkgus) \
                    .intersection(tkgu_types)
            #
            if len(intersection_deprecation_tkgus) > 0:
                curr_action_categories.add(ACTION_CATEGORY_DEPRECATE)

            if len(intersection_addition_tkgus) > 0:
                curr_action_categories.add(ACTION_CATEGORY_ASSERT)

            for curr_action_category in curr_action_categories:
                already_assessed = False
                for curr_llm_assessment in curr_tkgu_triple['llm_assessment']:
                    if curr_llm_assessment['llm_name'] == config['llm_assessor_name'] \
                            and curr_llm_assessment['llm_prompt_type'] == curr_action_category:
                        already_assessed = True
                if already_assessed:
                    continue

                curr_parsed_triple_categorized = tuple(curr_parsed_triple + \
                                                       [curr_action_category])

                if curr_parsed_triple_categorized in llm_triples_to_answer[curr_line_idx]:
                    curr_llm_assessment = dict()
                    curr_llm_assessment['llm_name'] = config['llm_assessor_name']
                    curr_llm_assessment['llm_assessment'] = \
                        llm_triples_to_answer[curr_line_idx][curr_parsed_triple_categorized]
                    curr_llm_assessment['llm_prompt_type'] = curr_action_category
                    if config['log_prompt_per_triple']:
                        # curr_llm_assessment[f'{curr_action_category}_prompt'] = \
                        curr_llm_assessment[f'llm_prompt'] = \
                            llm_triples_to_prompt_responses[curr_line_idx][curr_parsed_triple_categorized]
                    curr_tkgu_triple['llm_assessment'].append(curr_llm_assessment)
                else:
                    logger.warning('!!!triple_not_assessed!!!')
                    # logger.warning('!!!triple_not_assessed: setting false by default !!! '
                    #                f'{curr_parsed_triple_categorized}')
                    # curr_llm_assessment = dict()
                    # curr_llm_assessment['llm_name'] = config['llm_assessor_name']
                    # curr_llm_assessment[curr_action_category] = False
                    # if config['log_prompt_per_triple']:
                    #     curr_llm_assessment[f'llm_prompt'] = \
                    #         'NOT_ASSESSED'
                    # curr_tkgu_triple['llm_assessment'].append(curr_llm_assessment)

        output_file.write(json.dumps(curr_parsed_line, ensure_ascii=False) + '\n')
        output_file.flush()
    save_last_processed(path=last_processed_file_path,
                        number=checkpoint_line_idx)


def generate_instances_v8(
        config,
        tokenizer: PreTrainedTokenizerBase,
        client: InferenceClient,
        tokenizer_splitter,
        output_file_path,
        last_processed_file_path,
        input_file_path,
        counter,
        counter_lock,
        start_time_all
):
    last_processed_line = read_last_processed(last_processed_file_path)
    max_triples_batch_size = config['max_triples_batch_size']

    deprecated_triple_tkgus = set(config['deprecated_triple_tkgus'])
    addition_triple_tkgus = set(config['addition_triple_tkgus'])
    tkgu_types = set(config['tkgu_types'])

    line_idx_to_parsed_line = dict()

    checkpoint_save_interval = config['checkpoint_save_interval']
    llm_triples_to_answer: Dict[int, Dict[Tuple, bool]] = dict()
    llm_triples_to_prompt_responses: Dict[int, Dict[Tuple, str]] = dict()
    all_identified_triples: Dict[int, Set] = dict()
    # curr_line_idx_batch: Set = set()
    to_query_triples = list()
    mode = 'wt' if last_processed_line <= 0 else 'at'

    output_file = open(output_file_path, mode, encoding='utf-8')
    max_line_idx = last_processed_line
    for line_idx, curr_line in enumerate(open(input_file_path, 'rt', encoding='utf-8')):
        max_line_idx = line_idx
        if line_idx <= last_processed_line:
            logger.info(f'from {last_processed_file_path}, '
                        f'line_idx {line_idx} was already processed, ignoring')
            continue
        parsed_line = json.loads(curr_line)

        qid_to_mentions = dict()
        for curr_mention in parsed_line['mentions']:
            if curr_mention['qid'] not in qid_to_mentions:
                qid_to_mentions[curr_mention['qid']] = list()
            qid_to_mentions[curr_mention['qid']].append(curr_mention['mention_text'])

        passage = parsed_line['passage']

        percentage_english_words = calculate_english_word_percentage(passage.lower())
        if percentage_english_words < config['min_percentage_english_words']:
            logger.info(f'ignoring_chunk_not_english '
                        f'percentage: {percentage_english_words}, '
                        f'chunk: {passage}')
            continue

        line_idx_to_parsed_line[line_idx] = parsed_line
        start_time = time.time()
        field_tkgu_triples = parsed_line['tkgu_triples']
        to_query_added_triples = list()
        to_query_removed_triples = list()
        counter_all_added_triples = 0
        counter_all_deprecated_triples = 0

        # field_tkgu_triples_to_assess = list()

        for curr_tkgu_triple in field_tkgu_triples:
            if 'llm_assessment' not in curr_tkgu_triple:
                curr_tkgu_triple['llm_assessment'] = list()
            triple_tkgu_operations = set(curr_tkgu_triple['tkgu_operations'])

            assessed_prompt_types = set()

            for curr_llm_assessment in curr_tkgu_triple['llm_assessment']:
                if curr_llm_assessment['llm_name'] == config['llm_assessor_name']:
                    assessed_prompt_types.add(curr_llm_assessment['llm_prompt_type'])

            #
            intersection_deprecation_tkgus = \
                triple_tkgu_operations \
                    .intersection(deprecated_triple_tkgus) \
                    .intersection(tkgu_types)
            #
            intersection_addition_tkgus = \
                triple_tkgu_operations \
                    .intersection(addition_triple_tkgus) \
                    .intersection(tkgu_types)

            to_query_prompt_types = set()
            if len(intersection_deprecation_tkgus) > 0:
                to_query_prompt_types.add(ACTION_CATEGORY_DEPRECATE)

            if len(intersection_addition_tkgus) > 0:
                to_query_prompt_types.add(ACTION_CATEGORY_ASSERT)

            # prompt_types_to_assess = set(config['prompt_types_to_assess'])

            if len(to_query_prompt_types.difference(assessed_prompt_types)) == 0:
                logger.debug(f'triple was already assessed using the {config["llm_assessor_name"]} '
                             f'model for types {assessed_prompt_types}')
                continue
            # else:
            #     logger.info()
            to_query_prompt_types = to_query_prompt_types.difference(assessed_prompt_types)
            curr_parsed_triple = \
                obtain_curr_parsed_triple(curr_tkgu_triple, qid_to_mentions, config)
            logger.debug(f'{parsed_line["hash_id"]} curr_parsed_triple is: {curr_parsed_triple} '
                         f'to_query_prompt_types: {to_query_prompt_types} '
                         f'assessor_name: {config["llm_assessor_name"]}')

            #
            if ACTION_CATEGORY_DEPRECATE in to_query_prompt_types:
                counter_all_deprecated_triples += 1
                to_query_removed_triples.append(curr_parsed_triple)

            if ACTION_CATEGORY_ASSERT in to_query_prompt_types:
                counter_all_added_triples += 1
                to_query_added_triples.append(curr_parsed_triple)

        #
        if len(passage.split(' ')) > config['max_nr_tokens_in_chunk']:
            logger.info('max_nr_tokens_in_chunk exceeded, splitting')
            chunks = split_text(text=passage, tokenizer=tokenizer_splitter,
                                max_tokens=config['max_nr_tokens_in_chunk'])
            logger.info(f'original_chunk: {passage}')
            logger.info(f'splitted_chunks: {chunks}')
        else:
            chunks = [passage]

        to_query_added_triples_batches = \
            split_into_batches(
                to_query_added_triples,
                batch_size=max_triples_batch_size,
                parsed_line_idx=line_idx,
                chunks=chunks,
                chunk_timestamp=parsed_line['revision_timestamp'],
                action=ACTION_CATEGORY_ASSERT
            )

        to_query_removed_triples_batches = \
            split_into_batches(
                to_query_removed_triples,
                batch_size=max_triples_batch_size,
                parsed_line_idx=line_idx,
                chunks=chunks,
                chunk_timestamp=parsed_line['revision_timestamp'],
                action=ACTION_CATEGORY_DEPRECATE
            )
        if len(to_query_added_triples) > max_triples_batch_size:
            logger.debug('******************************************')
            logger.debug(f'produced_to_query_added_triples_batches: {to_query_added_triples_batches}; '
                         f'\nfrom_to_query_added_triples: {to_query_added_triples}')
            logger.debug('******************************************')
        else:
            logger.debug('******************************************')
            logger.debug(f'single_to_query_added_triples_batches: {to_query_added_triples_batches}; '
                         f'\nfrom_to_query_added_triples: {to_query_added_triples}')
            logger.debug('******************************************')

        to_query_triples = to_query_triples + \
                           to_query_added_triples_batches + to_query_removed_triples_batches

        ########
        llm_triples_to_answer[line_idx] = dict()
        llm_triples_to_prompt_responses[line_idx] = dict()
        all_identified_triples[line_idx] = set()
        ########

        # prompt_type = config['prompt_type']

        logger.debug(f'len(to_query_added_triples_batches): '
                    f'{len(to_query_added_triples_batches)}')
        if len(to_query_added_triples_batches) > 1:
            logger.debug(f'to_query_added_triples_batches: {to_query_added_triples_batches}')

        if (line_idx + 1) % checkpoint_save_interval == 0:
            save_checkpoint(input_file_path=input_file_path,
                            config=config,
                            to_query_triples=to_query_triples,
                            tokenizer=tokenizer,
                            client=client,
                            llm_triples_to_answer=llm_triples_to_answer,
                            all_identified_triples=all_identified_triples,
                            llm_triples_to_prompt_responses=llm_triples_to_prompt_responses,
                            line_idx_to_parsed_line=line_idx_to_parsed_line,
                            checkpoint_line_idx=line_idx,
                            last_processed_file_path=last_processed_file_path,
                            output_file=output_file)

            line_idx_to_parsed_line = dict()
            llm_triples_to_answer = dict()
            llm_triples_to_prompt_responses = dict()
            all_identified_triples = dict()
            to_query_triples = list()

        with counter_lock:
            counter['count'] += 1
        curr_time = time.time()
        per_sec = counter['count'] / (curr_time - start_time_all)
        logger.debug(f'finished, counter is now {counter["count"]}, per second: '
              f'{per_sec} per_day: {per_sec * 86400}')

    save_checkpoint(
        input_file_path=input_file_path,
        config=config,
        to_query_triples=to_query_triples,
        tokenizer=tokenizer,
        client=client,
        llm_triples_to_answer=llm_triples_to_answer,
        all_identified_triples=all_identified_triples,
        llm_triples_to_prompt_responses=llm_triples_to_prompt_responses,
        line_idx_to_parsed_line=line_idx_to_parsed_line,
        checkpoint_line_idx=max_line_idx,
        last_processed_file_path=last_processed_file_path,
        output_file=output_file
    )
