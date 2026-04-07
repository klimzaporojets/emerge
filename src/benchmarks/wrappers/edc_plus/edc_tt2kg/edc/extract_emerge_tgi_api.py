import threading
import traceback

import edc.utils.llm_utils as llm_utils

from importlib import reload
import requests
from typing import Dict, List, Tuple
import logging
import time
import random

reload(logging)

import subprocess

logger = logging.getLogger(__name__)

class ExtractorEmergeTgiApi:
    def __init__(self, tgi_url: str,
                 max_workers: int,
                 cuda_device_apptainer: int,
                 restart_apptainer_script_name: str,
                 wait_for_restart_time: int
                 ) -> None:
        self.tgi_url = tgi_url.rstrip("/") + "/generate"
        self._sync_once_event = threading.Event()
        self._sync_once_lock = threading.Lock()
        self.max_workers = max_workers
        self.cuda_device_apptainer = cuda_device_apptainer
        self.restart_apptainer_script_name = restart_apptainer_script_name
        self.wait_for_restart_time = wait_for_restart_time

    def extract_v2(
            self,
            input_text_str: str,
            few_shot_examples_str: str,
            prompt_template_str: str,
            entities_hint: str = None,
            relations_hint: str = None,
            id_txt: int = None
    ) -> Tuple[Dict[str, List[List[str]]], int]:
        """
        v2 is error-proof, able to restart the server
        Args:
            input_text_str:
            few_shot_examples_str:
            prompt_template_str:
            entities_hint:
            relations_hint:
            id_txt:

        Returns:

        """
        assert (entities_hint is None and relations_hint is None) or (
                relations_hint is not None and relations_hint is not None
        )

        assert (entities_hint is None and relations_hint is None) or (
                relations_hint is not None and relations_hint is not None
        )
        filled_prompt = prompt_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
                "input_text": input_text_str,
                "entities_hint": entities_hint,
                "relations_hint": relations_hint,
            }
        )

        filled_prompt = filled_prompt + "\nTriples in text:"
        payload = {
            "inputs": filled_prompt,
            "parameters": {
                "max_new_tokens": 2048,
                "temperature": 0.2,
                "repetition_penalty": 1.0,
                "stop": ["\n\n"]
            }
        }

        should_loop = True
        completion = None
        while should_loop:
            try:
                response = requests.post(self.tgi_url, json=payload)
                response.raise_for_status()
                completion = response.json()["generated_text"]
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

                                print(f'about_to_run: {command}')
                                result = subprocess.run(command, shell=True, text=True, capture_output=True)
                                print(f'just_have_run: {command} ; '
                                            f'standard output: {result.stdout}; '
                                            f'standard error: {result.stderr}; '
                                            f'return code: {result.returncode}')
                                self._sync_once_event.clear()
                            else:
                                print(f'do_not_entering_synchronized1, sleeping: ' f'{self.wait_for_restart_time}')
                                time.sleep(self.wait_for_restart_time)
                    else:
                        print(f'do_not_entering_synchronized2, sleeping: {self.wait_for_restart_time}')
                        time.sleep(self.wait_for_restart_time)
                else:
                    print(f'script_already_running1: {command} '
                                f'sleeping for {self.wait_for_restart_time} secs')
                    time.sleep(self.wait_for_restart_time)

        completion_triples_in_text = completion.strip()
        completion_triples_not_in_text = ""
        if "Triples not in text:" in completion:
            completion_triples_in_text = completion[:completion.index("Triples not in text:")].strip()
            completion_triples_not_in_text = completion[completion.index("Triples not in text:") +
                                                        len("Triples not in text:"):].strip()

        extracted_triplets_list: List[List[str]] = llm_utils.parse_raw_triplets_emerge_v3(completion_triples_in_text)
        extracted_triplets_not_in_text_list: List[List[str]] = llm_utils.parse_raw_triplets_emerge_v3(
            completion_triples_not_in_text)

        if len(extracted_triplets_list) == 0 and len(extracted_triplets_not_in_text_list) == 0:
            print(f'**************************************************')
            print(f'no_triples_detected_at_all, payload: {payload} '
                  f'\n \n ****** GENERATED_TEXT: {completion}')
            print(f'**************************************************')

        return {
            "extracted_triplets_list": extracted_triplets_list,
            "extracted_triplets_not_in_text_list": extracted_triplets_not_in_text_list,
            "input_text_str": input_text_str
        }, id_txt

    def extract(
            self,
            input_text_str: str,
            few_shot_examples_str: str,
            prompt_template_str: str,
            entities_hint: str = None,
            relations_hint: str = None,
            id_txt: int = None,
    ) -> Tuple[Dict[str, List[List[str]]], int]:
        assert (entities_hint is None and relations_hint is None) or (
                relations_hint is not None and relations_hint is not None
        )

        assert (entities_hint is None and relations_hint is None) or (
                relations_hint is not None and relations_hint is not None
        )
        filled_prompt = prompt_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
                "input_text": input_text_str,
                "entities_hint": entities_hint,
                "relations_hint": relations_hint,
            }
        )

        filled_prompt = filled_prompt + "\nTriples in text:"
        payload = {
            "inputs": filled_prompt,
            "parameters": {
                "max_new_tokens": 2048,
                "temperature": 0.2,
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
            return {
                'extracted_triplets_list': [],
                'extracted_triplets_not_in_text_list': []
            }, id_txt

        completion = response.json()["generated_text"]

        completion_triples_in_text = completion.strip()
        completion_triples_not_in_text = ""
        if "Triples not in text:" in completion:
            completion_triples_in_text = completion[:completion.index("Triples not in text:")].strip()
            completion_triples_not_in_text = completion[completion.index("Triples not in text:") +
                                                        len("Triples not in text:"):].strip()

        extracted_triplets_list: List[List[str]] = llm_utils.parse_raw_triplets_emerge_v3(completion_triples_in_text)
        extracted_triplets_not_in_text_list: List[List[str]] = llm_utils.parse_raw_triplets_emerge_v3(
            completion_triples_not_in_text)

        if len(extracted_triplets_list) == 0 and len(extracted_triplets_not_in_text_list) == 0:
            print(f'**************************************************')
            print(f'no_triples_detected_at_all, payload: {payload} '
                  f'\n \n ****** GENERATED_TEXT: {completion}')
            print(f'**************************************************')

        return {
            "extracted_triplets_list": extracted_triplets_list,
            "extracted_triplets_not_in_text_list": extracted_triplets_not_in_text_list,
            "input_text_str": input_text_str
        }, id_txt
