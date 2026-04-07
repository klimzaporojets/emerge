import random
import subprocess
import threading
import time
import traceback
from typing import List, Tuple

import requests

import edc.utils.llm_utils as llm_utils
import logging

logger = logging.getLogger(__name__)


class SchemaDefinerTgiApi:
    def __init__(self, tgi_url:str,
                 max_workers: int,
                 cuda_device_apptainer: int,
                 wait_for_restart_time: int,
                 restart_apptainer_script_name: str
                 ) -> None:
        self._sync_once_event = threading.Event()
        self._sync_once_lock = threading.Lock()
        #
        self.max_workers = max_workers
        self.cuda_device_apptainer = cuda_device_apptainer
        self.wait_for_restart_time = wait_for_restart_time
        self.restart_apptainer_script_name = restart_apptainer_script_name
        #
        self.tgi_url = tgi_url

    def define_schema_v2(
            self,
            input_text_str: str,
            extracted_triplets_list: List[str],
            few_shot_examples_str: str,
            prompt_template_str: str,
            idx_it:int
    ) -> Tuple[List[List[str]], int]:
        # Given a piece of text and a list of triplets extracted from it, define each of the relation present

        relations_present = set()
        for t in extracted_triplets_list:
            relations_present.add(t[1])

        filled_prompt = prompt_template_str.format_map(
            {
                "text": input_text_str,
                "few_shot_examples": few_shot_examples_str,
                "relations": relations_present,
                "triples": extracted_triplets_list,
            }
        )
        filled_prompt = filled_prompt + '\nAnswer: '
        payload = {
            "inputs": filled_prompt,
            "parameters": {
                # "max_new_tokens": 1024,
                "max_new_tokens": 2048,
                "temperature": 0.7,
                "repetition_penalty": 1.0,
                "stop": ["\n\n"]
            }
        }

        should_loop = True
        response_json = None
        while should_loop:
            try:
                response = requests.post(self.tgi_url, json=payload)
                response.raise_for_status()
                response_json = response.json()
                assert len(response_json) == 1
                response_json = response_json[0]
                should_loop = False
            except Exception as e:
                script_name = self.restart_apptainer_script_name
                error_message = traceback.format_exc()
                logger.error(f'GenerationError stack is as follows: '
                             f'{error_message}')
                logfile = f'run_{random.randint(1000, 9999)}.log'
                command = (f'bash jobs/s01_run_v2/{self.restart_apptainer_script_name} '
                           f'{self.cuda_device_apptainer} '
                           f'{self.max_workers} 2>&1 | tee {logfile}')

                if not llm_utils.is_script_running(script_name):
                    if not self._sync_once_event.is_set():  # check if already executed
                        # only one thread can enter here
                        with self._sync_once_lock:
                            if not self._sync_once_event.is_set():
                                # <<< this runs only once >>>
                                self._sync_once_event.set()  # mark as done

                                logger.info(f'about to run: {command}')
                                result = subprocess.run(command, shell=True, text=True, capture_output=True)
                                logger.info(f'just have run: {command} ; '
                                            f'standard output: {result.stdout}; '
                                            f'standard error: {result.stderr}; '
                                            f'return code: {result.returncode}')
                                self._sync_once_event.clear()
                            else:
                                logger.info(f'do_not_entering_synchronized1, sleeping: {self.wait_for_restart_time}')
                                time.sleep(self.wait_for_restart_time)
                    else:
                        logger.info(f'do_not_entering_synchronized2, sleeping: {self.wait_for_restart_time}')
                        time.sleep(self.wait_for_restart_time)
                else:
                    logger.info(f'script_already_running1: {command} '
                                f'sleeping for {self.wait_for_restart_time} secs')
                    time.sleep(self.wait_for_restart_time)

        # completion = response.json()['generated_text']
        completion = response_json['generated_text']
        assert 'Answer: ' in completion
        last_index = completion.rfind('Answer: ')

        # Extract everything after it (if found)
        if last_index != -1:
            result = completion[last_index + len('Answer: '):]
        else:
            result = ''  # Or handle as needed

        # print(f'***** schema_definition_tgi_api: schema_definition_prompt: {payload}\n\n')
        completion = result
        completion = completion.replace("\\_", " ")
        completion = completion.replace("_", " ")
        # print(f'***** schema_definition_tgi_api: schema_definition_answer: {completion}\n\n')
        relation_definition_dict = llm_utils.parse_relation_definition(completion)
        # print(f'***** schema_definition_tgi_api: relation_definition_dict: {relation_definition_dict}\n\n')
        logger.debug(f'***************************************************************************\n'
              f'schema_definition_tgi_api completion after extracting only Answer: \n'
              f'*-*-*-*-* extracted_triplets_list: {extracted_triplets_list} \n'
              f'*-*-*-*-* completion: {completion}\n'
              f'*-*-*-*-* relation_definition_dict: {relation_definition_dict}\n'
              f'****************************************************************************\n')


        missing_relations = [rel for rel in relations_present if rel not in relation_definition_dict]
        if len(missing_relations) != 0:
            # logger.debug(f'Relations {missing_relations} are missing from the relation definition!')
            # logger.warning(f'warning_missing_relations Relations {missing_relations} '
            logger.debug(f'warning_missing_relations Relations {missing_relations} '
                           f'are missing from the relation definition!: \n'
                           f'***************************************************************************\n'
              f'schema_definition_tgi_api completion after extracting only Answer: \n'
              f'*-*-*-*-* extracted_triplets_list: {extracted_triplets_list} \n'
              f'*-*-*-*-* completion: {completion}\n'
              f'*-*-*-*-* relation_definition_dict: {relation_definition_dict}\n'
              f'****************************************************************************\n')
        return relation_definition_dict, idx_it


    def define_schema(
            self,
            input_text_str: str,
            extracted_triplets_list: List[str],
            few_shot_examples_str: str,
            prompt_template_str: str,
            idx_it:int
    ) -> Tuple[List[List[str]], int]:
        # Given a piece of text and a list of triplets extracted from it, define each of the relation present

        relations_present = set()
        for t in extracted_triplets_list:
            relations_present.add(t[1])

        filled_prompt = prompt_template_str.format_map(
            {
                "text": input_text_str,
                "few_shot_examples": few_shot_examples_str,
                "relations": relations_present,
                "triples": extracted_triplets_list,
            }
        )
        filled_prompt = filled_prompt + '\nAnswer: '
        payload = {
            "inputs": filled_prompt,
            "parameters": {
                # "max_new_tokens": 1024,
                "max_new_tokens": 2048,
                "temperature": 0.7,
                "repetition_penalty": 1.0,
                "stop": ["\n\n"]
            }
        }

        response = requests.post(self.tgi_url, json=payload)

        try:
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f'extract_emerge_tgi_api: ERROR with following payload: {payload}')
            traceback.print_exc()
            if e.response is not None:
                print(f'Server responded with: {e.response.text}')
            return None


        response_json = response.json()
        assert len(response_json) == 1
        response_json = response_json[0]
        completion = response_json['generated_text']
        assert 'Answer: ' in completion
        last_index = completion.rfind('Answer: ')

        # Extract everything after it (if found)
        if last_index != -1:
            result = completion[last_index + len('Answer: '):]
        else:
            result = ''  # Or handle as needed

        completion = result
        completion = completion.replace("\\_", " ")
        completion = completion.replace("_", " ")
        relation_definition_dict = llm_utils.parse_relation_definition(completion)
        logger.debug(f'***************************************************************************\n'
              f'schema_definition_tgi_api completion after extracting only Answer: \n'
              f'*-*-*-*-* extracted_triplets_list: {extracted_triplets_list} \n'
              f'*-*-*-*-* completion: {completion}\n'
              f'*-*-*-*-* relation_definition_dict: {relation_definition_dict}\n'
              f'****************************************************************************\n')


        missing_relations = [rel for rel in relations_present if rel not in relation_definition_dict]
        if len(missing_relations) != 0:
            logger.debug(f'warning_missing_relations Relations {missing_relations} '
                           f'are missing from the relation definition!: \n'
                           f'***************************************************************************\n'
              f'schema_definition_tgi_api completion after extracting only Answer: \n'
              f'*-*-*-*-* extracted_triplets_list: {extracted_triplets_list} \n'
              f'*-*-*-*-* completion: {completion}\n'
              f'*-*-*-*-* relation_definition_dict: {relation_definition_dict}\n'
              f'****************************************************************************\n')
        return relation_definition_dict, idx_it
