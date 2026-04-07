import pdb
import pickle
import random
import subprocess
import threading
import time
import traceback
from typing import List, Dict
import os

import requests

import edc.utils.llm_utils as llm_utils
import re
import numpy as np
import copy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def get_first_letter_after_answer(text):
    match = re.search(r'Answer:\s*([^\s])', text)
    if match:
        return match.group(1)
    return None


class SchemaCanonicalizerEmergeTgiApi:
    # The class to handle the last stage: Schema Canonicalization
    def __init__(
            self,
            target_schema_dict: dict,
            embedder: SentenceTransformer,
            relations_cache_path: str,
            tgi_url_verifier: str,
            max_workers: int,
            cuda_device_apptainer:int,
            wait_for_restart_time:int,
            restart_apptainer_script_name:str,
            use_restart_script:bool
    ) -> None:
        self.use_restart_script = use_restart_script
        self._sync_once_event = threading.Event()
        self._sync_once_lock = threading.Lock()

        self.cuda_device_apptainer = cuda_device_apptainer
        self.wait_for_restart_time = wait_for_restart_time
        self.restart_apptainer_script_name = restart_apptainer_script_name
        # The canonicalizer uses an embedding model to first fetch candidates from the target schema, then uses a verifier schema to decide which one to canonicalize to or not
        # canonoicalize at all.

        self.schema_dict = target_schema_dict

        self.tgi_url_verifier = tgi_url_verifier
        self.max_workers = max_workers

        self.embedder = embedder

        if not os.path.exists(relations_cache_path):
            os.makedirs(os.path.dirname(relations_cache_path), exist_ok=True)

            # Embed the target schema
            self.schema_embedding_dict = {}

            print("Embedding target schema...")
            for relation, relation_definition in tqdm(target_schema_dict.items()):
                embedding = self.embedder.encode(relation_definition)
                self.schema_embedding_dict[relation] = embedding
            pickle.dump(self.schema_embedding_dict, open(relations_cache_path, 'wb'))
        else:
            print('loading_schema_from_cache !! ')
            self.schema_embedding_dict = pickle.load(open(relations_cache_path, 'rb'))

    def retrieve_similar_relations(self, query_relation_definitions: List[str], top_k=5):
        target_relation_list = list(self.schema_embedding_dict.keys())
        target_relation_embedding_list = list(self.schema_embedding_dict.values())

        if "sts_query" in self.embedder.prompts:
            queries_embeddings = self.embedder.encode(query_relation_definitions, prompt_name="sts_query")
        else:
            queries_embeddings = self.embedder.encode(query_relation_definitions)

        result_dicts = []
        result_scores = []

        if queries_embeddings is None or len(queries_embeddings) == 0:
            logger.error("queries_embeddings is empty!")
            return result_dicts, result_scores

        if not target_relation_embedding_list or len(target_relation_embedding_list) == 0:
            logger.error("target_relation_embedding_list is empty!")
            return result_dicts, result_scores

        try:
            scores = queries_embeddings @ np.array(
                target_relation_embedding_list).T  # shape: (num_queries, num_targets)
        except ValueError as e:
            # pdb.set_trace()  # enter debugger here
            raise
        # For each query, find the top_k indices with highest scores
        top_k = min(top_k, scores.shape[1])

        # argsort descending along axis=1 for each query
        highest_score_indices = np.argsort(-scores, axis=1)[:, :top_k]



        for query_idx in range(len(query_relation_definitions)):
            idxs = highest_score_indices[query_idx]
            scores_for_query = scores[query_idx, idxs]
            dict_for_query = {
                target_relation_list[idx]: self.schema_dict[target_relation_list[idx]]
                for idx in idxs
            }
            result_dicts.append(dict_for_query)
            result_scores.append(list(scores_for_query))

        return result_dicts, result_scores

    def llm_verify_v2(
            self,
            input_text_str: str,
            query_triplet: List[str],
            query_relation_definition: str,
            prompt_template_str: str,
            candidate_relation_definition_dict: dict,
            relation_example_dict: dict = None,
            idx_triple=None,
    ):
        canonicalized_triplet = copy.deepcopy(query_triplet)
        query_triplet_check = copy.deepcopy(query_triplet)
        if isinstance(query_triplet_check[0], str):
            query_triplet_check[0] = query_triplet_check[0].replace('_', ' ')
        if isinstance(query_triplet_check[2], str):
            query_triplet_check[2] = query_triplet_check[2].replace('_', ' ')
        query_triplet_check = query_triplet_check[:3]
        logger.debug(f'query_triplet_check is: {query_triplet_check} from query_triplet {query_triplet}')
        choice_letters_list = []
        choices = ""
        candidate_relations = list(candidate_relation_definition_dict.keys())
        candidate_relation_descriptions = list(candidate_relation_definition_dict.values())
        for idx, rel in enumerate(candidate_relations):
            choice_letter = chr(ord("@") + idx + 1)
            choice_letters_list.append(choice_letter)
            choices += f"{choice_letter}. '{rel}': {candidate_relation_descriptions[idx]}\n"
            if relation_example_dict is not None:
                choices += f"Example: '{relation_example_dict[candidate_relations[idx]]['triple']}' can be extracted from '{candidate_relations[idx]['sentence']}'\n"
        choices += f"{chr(ord('@') + idx + 2)}. None of the above.\n"

        verification_prompt = prompt_template_str.format_map(
            {
                "input_text": input_text_str,
                "query_triplet": query_triplet_check,
                "query_relation": query_triplet_check[1],
                "query_relation_definition": query_relation_definition,
                "choices": choices,
            }
        )
        verification_prompt += '\nAnswer: '

        payload = {
            "inputs": verification_prompt,
            "parameters": {
                "max_new_tokens": 6,
                "temperature": 0.1,
                "repetition_penalty": 1.0,
                "stop": ["\n\n\n"]
            }
        }

        verification_result = None
        should_loop = True
        while should_loop:
            try:
                response = requests.post(self.tgi_url_verifier, json=payload)
                response.raise_for_status()
                response_json = response.json()
                assert len(response_json) == 1
                verification_result = response_json[0]
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



        curr_answer = get_first_letter_after_answer(verification_result['generated_text'])
        if curr_answer in choice_letters_list:
            canonicalized_triplet[1] = candidate_relations[choice_letters_list.index(curr_answer)]
            logger.debug(f'****** yeah_in_choice_letters_list: letter '
                         f'{curr_answer} from {verification_result} in '
                         f'{choice_letters_list}')
        else:
            logger.debug(f'****** not_in_choice_letters_list: letter '
                         f'{curr_answer} from {verification_result} not_in '
                         f'{choice_letters_list}')
            return None, idx_triple

        return canonicalized_triplet, idx_triple


    def llm_verify(
            self,
            input_text_str: str,
            query_triplet: List[str],
            query_relation_definition: str,
            prompt_template_str: str,
            candidate_relation_definition_dict: dict,
            relation_example_dict: dict = None,
            idx_triple=None,
    ):
        canonicalized_triplet = copy.deepcopy(query_triplet)
        query_triplet_check = copy.deepcopy(query_triplet)
        if isinstance(query_triplet_check[0], str):
            query_triplet_check[0] = query_triplet_check[0].replace('_', ' ')
        if isinstance(query_triplet_check[2], str):
            query_triplet_check[2] = query_triplet_check[2].replace('_', ' ')
        query_triplet_check = query_triplet_check[:3]
        logger.debug(f'query_triplet_check is: {query_triplet_check} from query_triplet {query_triplet}')
        choice_letters_list = []
        choices = ""
        candidate_relations = list(candidate_relation_definition_dict.keys())
        candidate_relation_descriptions = list(candidate_relation_definition_dict.values())
        for idx, rel in enumerate(candidate_relations):
            choice_letter = chr(ord("@") + idx + 1)
            choice_letters_list.append(choice_letter)
            choices += f"{choice_letter}. '{rel}': {candidate_relation_descriptions[idx]}\n"
            if relation_example_dict is not None:
                choices += f"Example: '{relation_example_dict[candidate_relations[idx]]['triple']}' can be extracted from '{candidate_relations[idx]['sentence']}'\n"
        choices += f"{chr(ord('@') + idx + 2)}. None of the above.\n"

        verification_prompt = prompt_template_str.format_map(
            {
                "input_text": input_text_str,
                "query_triplet": query_triplet_check,
                "query_relation": query_triplet_check[1],
                "query_relation_definition": query_relation_definition,
                "choices": choices,
            }
        )
        verification_prompt += '\nAnswer: '

        payload = {
            "inputs": verification_prompt,
            "parameters": {
                "max_new_tokens": 6,
                "temperature": 0.1,
                "repetition_penalty": 1.0,
                "stop": ["\n\n\n"]
            }
        }

        response = requests.post(self.tgi_url_verifier, json=payload)

        try:
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f'schema_canonicalization_emerge_tgi_api: ERROR with following payload: {payload}')
            traceback.print_exc()
            if e.response is not None:
                print(f'Server responded with: {e.response.text}')
            return None, idx_triple

        response_json = response.json()
        assert len(response_json) == 1
        verification_result = response_json[0]

        curr_answer = get_first_letter_after_answer(verification_result['generated_text'])
        if curr_answer in choice_letters_list:
            canonicalized_triplet[1] = candidate_relations[choice_letters_list.index(curr_answer)]
            logger.debug(f'****** yeah_in_choice_letters_list: letter '
                         f'{curr_answer} from {verification_result} in '
                         f'{choice_letters_list}')
        else:
            logger.debug(f'****** not_in_choice_letters_list: letter '
                         f'{curr_answer} from {verification_result} not_in '
                         f'{choice_letters_list}')
            return None, idx_triple

        return canonicalized_triplet, idx_triple

    def canonicalize(
            self,
            input_text_list: List[Dict[str,str]],
            input_text_idxs: List[int],
            c_oie_triplets: List,
            open_relation_definition_dicts: List[dict],
            verify_prompt_template: str,
            enrich=False,
    ):
        idx_triple_to_canonicalized_triple = dict()
        relations_triples = list()
        idx_triples_to_check = list()
        for idx_triple, curr_open_triple in enumerate(c_oie_triplets):
            idx_text = input_text_idxs[idx_triple]
            curr_open_relation = curr_open_triple[1]

            if curr_open_relation in self.schema_dict:
                idx_triple_to_canonicalized_triple[idx_triple] = (curr_open_triple, {})


            if len(self.schema_dict) != 0:
                if curr_open_relation not in open_relation_definition_dicts[idx_text]:
                    idx_triple_to_canonicalized_triple[idx_triple] = None
                else:
                    relations_triples.append(open_relation_definition_dicts[idx_text][curr_open_relation])
                    idx_triples_to_check.append(idx_triple)
            else:
                idx_triple_to_canonicalized_triple[idx_triple] = None

        candidate_relations, candidate_scores = self.retrieve_similar_relations(
            relations_triples
        )
        assert len(candidate_relations) == len(idx_triples_to_check)
        idx_triple_to_cand_relations = dict()
        idx_triple_to_cand_scores = dict()
        for idx_tr, curr_idx_triple in enumerate(idx_triples_to_check):
            idx_triple_to_cand_relations[curr_idx_triple] = candidate_relations[idx_tr]
            idx_triple_to_cand_scores[curr_idx_triple] = candidate_scores[idx_tr]
        #
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for idx_cand_relation, curr_cand_relations in enumerate(candidate_relations):
                if self.use_restart_script:
                    futures.append(executor.submit(
                        self.llm_verify_v2,
                        input_text_list[input_text_idxs[idx_triples_to_check[idx_cand_relation]]]['passage'],
                        c_oie_triplets[idx_triples_to_check[idx_cand_relation]],
                        open_relation_definition_dicts[input_text_idxs[idx_triples_to_check[idx_cand_relation]]] \
                            [c_oie_triplets[idx_triples_to_check[idx_cand_relation]][1]],
                        verify_prompt_template,
                        curr_cand_relations,
                        None,
                        idx_triples_to_check[idx_cand_relation]
                    ))
                else:
                    futures.append(executor.submit(
                        self.llm_verify,
                        input_text_list[input_text_idxs[idx_triples_to_check[idx_cand_relation]]]['passage'],
                        c_oie_triplets[idx_triples_to_check[idx_cand_relation]],
                        open_relation_definition_dicts[input_text_idxs[idx_triples_to_check[idx_cand_relation]]] \
                            [c_oie_triplets[idx_triples_to_check[idx_cand_relation]][1]],
                        verify_prompt_template,
                        curr_cand_relations,
                        None,
                        idx_triples_to_check[idx_cand_relation]
                    ))

            results = [future.result() for future in tqdm(as_completed(futures),
                                                          desc='relation verification',
                                                          total=len(candidate_relations))]
        for curr_result in results:
            curr_idx_triple = curr_result[1]
            curr_canonicalized_triple = curr_result[0]
            idx_triple_to_canonicalized_triple[curr_idx_triple] = curr_canonicalized_triple

        print(f'enrich_is_in: {enrich}')
        for curr_idx_triple, curr_canonicalized_triple in copy.deepcopy(idx_triple_to_canonicalized_triple).items():
            if curr_canonicalized_triple is None:
                # Cannot be canonicalized
                if enrich:
                    idx_text = input_text_idxs[curr_idx_triple]
                    curr_open_relation = c_oie_triplets[curr_idx_triple][1]
                    self.schema_dict[curr_open_relation] = open_relation_definition_dicts[idx_text][curr_open_relation]
                    if 'sts_query' in self.embedder.prompts:
                        embedding = self.embedder.encode(
                            open_relation_definition_dicts[idx_text][curr_open_relation], prompt_name="sts_query"
                        )
                    else:
                        embedding = self.embedder.encode(open_relation_definition_dicts[idx_text][curr_open_relation])
                    self.schema_embedding_dict[curr_open_relation] = embedding
                    canonicalized_triplet = c_oie_triplets[curr_idx_triple]
                    idx_triple_to_canonicalized_triple[curr_idx_triple] = canonicalized_triplet

        idx_text_to_res = dict()
        for curr_idx_triple in range(len(c_oie_triplets)):
            curr_idx_text = input_text_idxs[curr_idx_triple]
            if curr_idx_text not in idx_text_to_res:
                idx_text_to_res[curr_idx_text] = ([], [])
            curr_candidate_relations = list()
            curr_candidate_scores = list()
            idx_text_to_res[curr_idx_text][0].append(idx_triple_to_canonicalized_triple[curr_idx_triple])
            if curr_idx_triple in idx_triple_to_cand_relations:
                curr_candidate_scores = idx_triple_to_cand_scores[curr_idx_triple]
                curr_candidate_relations = idx_triple_to_cand_relations[curr_idx_triple]
            idx_text_to_res[curr_idx_text][1].append(dict(zip(curr_candidate_relations, curr_candidate_scores)))
        canonicalized_triplets = []
        canon_candidate_dict_list = []
        for curr_idx_text in range(len(input_text_list)):
            if curr_idx_text not in idx_text_to_res:
                canonicalized_triplets.append([])
                canon_candidate_dict_list.append([])
            else:
                canonicalized_triplets.append(idx_text_to_res[curr_idx_text][0])
                canon_candidate_dict_list.append(idx_text_to_res[curr_idx_text][1])
        return canonicalized_triplets, canon_candidate_dict_list
