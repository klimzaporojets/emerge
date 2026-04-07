import asyncio
import copy
import logging
import os
import pdb
import pickle
import re
import traceback
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from edc.utils.unified_llm_client import UnifiedLLMClient

logger = logging.getLogger(__name__)

def get_first_letter_after_answer(text: str) -> Optional[str]:
    text = text.strip()

    # Remove common markdown emphasis (**bold**, *italic*)
    text = re.sub(r'(\*\*|\*)', '', text)

    if 'Answer:' in text:
        match = re.search(r'Answer:\s*([A-Z])\b', text)
        if match:
            return match.group(1)
        return None
    else:
        match = re.match(r'^\s*([A-Z])\b(?:\.)?', text)
        if match:
            return match.group(1)
        return None

# def get_first_letter_after_answer(text: str) -> Optional[str]:
#     text = text.strip()
#     if 'Answer:' in text:
#         match = re.search(r'Answer:\s*([^\s])', text)
#         if match:
#             return match.group(1)
#         return None
#     else:
#         match = re.match(r'^([A-Z])(?:\.)?', text)
#         if match:
#             letter = match.group(1)
#             return letter
#         return None

class SchemaCanonicalizerEmergeLLMApi:
    '''
    Schema canonicalization using:
      - embeddings to retrieve candidate canonical relations
      - an LLM verifier (UnifiedLLMClient-compatible) to pick A/B/C/.../None

    llm_client must expose:
        await generate(prompt: str) -> str

    Important: this class is asyncio-based (no threads for LLM calls).
    '''

    def __init__(
            self,
            target_schema_dict: dict,
            embedder: SentenceTransformer,
            relations_cache_path: str,
            llm_client: UnifiedLLMClient,
            sc_verify_llm_generation_profile: Dict[str, Any],
            max_workers: int = 64,
    ) -> None:
        self.schema_dict = target_schema_dict
        self.embedder = embedder
        self.llm = llm_client
        self.sc_verify_llm_generation_profile = sc_verify_llm_generation_profile

        # Keep max_workers name for compatibility with old config,
        # but it is used only as a concurrency hint for our asyncio gather batching.
        self.max_workers = max_workers

        # Cache embeddings for the target schema
        if not os.path.exists(relations_cache_path):
            os.makedirs(os.path.dirname(relations_cache_path), exist_ok=True)

            self.schema_embedding_dict = {}

            logger.info('Embedding target schema...')
            for relation, relation_definition in tqdm(target_schema_dict.items(),
                                                      desc='embedding_target_schema_dict'):
                embedding = self.embedder.encode(relation_definition, show_progress_bar=False)
                self.schema_embedding_dict[relation] = embedding

            logger.info('dumping_target_schema...')
            pickle.dump(self.schema_embedding_dict, open(relations_cache_path, 'wb'))
        else:
            print('start_loading_schema_from_cache !! ')
            self.schema_embedding_dict = pickle.load(open(relations_cache_path, 'rb'))
            print('end_loading_schema_from_cache !! ')

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve_similar_relations(self, query_relation_definitions: List[str], top_k: int = 5):
        target_relation_list = list(self.schema_embedding_dict.keys())
        target_relation_embedding_list = list(self.schema_embedding_dict.values())

        if 'sts_query' in getattr(self.embedder, 'prompts', {}):
            queries_embeddings = self.embedder.encode(query_relation_definitions, prompt_name='sts_query')
        else:
            queries_embeddings = self.embedder.encode(query_relation_definitions)

        result_dicts = []
        result_scores = []

        if queries_embeddings is None or len(queries_embeddings) == 0:
            # pdb.set_trace()
            logger.error('queries_embeddings is empty!')
            return result_dicts, result_scores

        if not target_relation_embedding_list or len(target_relation_embedding_list) == 0:
            logger.error('target_relation_embedding_list is empty!')
            return result_dicts, result_scores

        scores = queries_embeddings @ np.array(target_relation_embedding_list).T
        top_k = min(top_k, scores.shape[1])

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

    # ------------------------------------------------------------------
    # LLM verifier (async)
    # ------------------------------------------------------------------
    async def llm_verify(
            self,
            input_text_str: str,
            query_triplet: List[Any],
            query_relation_definition: str,
            prompt_template_str: str,
            candidate_relation_definition_dict: dict,
            relation_example_dict: Optional[dict] = None,
            idx_triple: Optional[int] = None,
    ) -> Tuple[Optional[List[Any]], Optional[int]]:
        try:
            canonicalized_triplet = copy.deepcopy(query_triplet)

            query_triplet_check = copy.deepcopy(query_triplet)
            if isinstance(query_triplet_check[0], str):
                query_triplet_check[0] = query_triplet_check[0].replace('_', ' ')
            if isinstance(query_triplet_check[2], str):
                query_triplet_check[2] = query_triplet_check[2].replace('_', ' ')
            query_triplet_check = query_triplet_check[:3]

            logger.debug(f'query_triplet_check is: {query_triplet_check} from query_triplet {query_triplet}')

            candidate_relations = list(candidate_relation_definition_dict.keys())
            candidate_relation_descriptions = list(candidate_relation_definition_dict.values())

            choice_letters_list = []
            choices = ''

            for idx, rel in enumerate(candidate_relations):
                choice_letter = chr(ord('@') + idx + 1)
                choice_letters_list.append(choice_letter)
                choices += f"{choice_letter}. '{rel}': {candidate_relation_descriptions[idx]}\n"

                # Keep optional example behavior, but do it safely (the original code had indexing bugs)
                if relation_example_dict is not None:
                    try:
                        ex = relation_example_dict.get(rel, None)
                        if ex and 'triple' in ex and 'sentence' in ex:
                            choices += f"Example: '{ex['triple']}' can be extracted from '{ex['sentence']}'\n"
                    except Exception:
                        pass

            # None-of-the-above option
            none_letter = chr(ord('@') + len(candidate_relations) + 1)
            choices += f'{none_letter}. None of the above.\n'

            verification_prompt = prompt_template_str.format_map(
                {
                    'input_text': input_text_str,
                    'query_triplet': query_triplet_check,
                    'query_relation': query_triplet_check[1],
                    'query_relation_definition': query_relation_definition,
                    'choices': choices,
                }
            )
            verification_prompt += '\nAnswer: '

            # LLM call
            # pdb.set_trace()
            logger.debug(f'BEFORE llm_verify_with_self.llm.generate: {verification_prompt} '
                        f'  ************ {self.sc_verify_llm_generation_profile}')
            completion = await self.llm.generate(prompt=verification_prompt,
                                                 **self.sc_verify_llm_generation_profile)
            logger.debug(f'AFTER llm_verify_with_self.llm.generate: {verification_prompt} '
                        f'  ************ {self.sc_verify_llm_generation_profile} '
                        f'***** {completion}')

            # Parse answer letter.
            # Original expected 'Answer: X' to appear in output. We keep that,
            # but add a robust fallback if the model returns just 'A' or similar.
            # pdb.set_trace()

            curr_answer = get_first_letter_after_answer(completion)
            logger.info(f'schema_canonicalization_emerge_llm_api_completion_is: '
                        f'{completion} curr_answer_in: {curr_answer}')
            if curr_answer is None:
                # fallback: first non-space character
                stripped = completion.strip()
                curr_answer = stripped[0] if stripped else None

            if curr_answer in choice_letters_list:
                canonicalized_triplet[1] = candidate_relations[choice_letters_list.index(curr_answer)]
                logger.info(
                    f'****** yeah_in_choice_letters_list: letter {curr_answer} from completion: {completion}'
                )
                return canonicalized_triplet, idx_triple

            logger.info(
                f'****** not_in_choice_letters_list: letter {curr_answer} from completion: "{completion.strip()}"'
            )
            return None, idx_triple
        except Exception as e:
            # logger.error(e)
            logger.error(f'schema_canonicalization_emerge_llm_api_llm_verify error '
                         f'{e}')
            traceback.print_exc()
            # pdb.set_trace()
            raise

    # ------------------------------------------------------------------
    # Canonicalize (async)
    # ------------------------------------------------------------------
    async def canonicalize_async(
            self,
            input_text_list: List[Dict[str, str]],
            input_text_idxs: List[int],
            c_oie_triplets: List,
            open_relation_definition_dicts: List[dict],
            verify_prompt_template: str,
            enrich: bool = False,
            top_k: int = 5,
    ):
        idx_triple_to_canonicalized_triple: Dict[int, Any] = {}
        relations_triples = []
        idx_triples_to_check = []

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
        # pdb.set_trace()
        logger.info('start_retrieve_similar_relations')
        candidate_relations, candidate_scores = self.retrieve_similar_relations(relations_triples, top_k=top_k)
        logger.info('end_retrieve_similar_relations')
        assert len(candidate_relations) == len(idx_triples_to_check)

        idx_triple_to_cand_relations = {}
        idx_triple_to_cand_scores = {}
        for idx_tr, curr_idx_triple in enumerate(idx_triples_to_check):
            idx_triple_to_cand_relations[curr_idx_triple] = candidate_relations[idx_tr]
            idx_triple_to_cand_scores[curr_idx_triple] = candidate_scores[idx_tr]

        # Build tasks
        tasks = []
        for idx_cand_relation, curr_cand_relations in enumerate(candidate_relations):
            idx_triple = idx_triples_to_check[idx_cand_relation]
            idx_text = input_text_idxs[idx_triple]

            ## BEGIN CHANGE
            tasks.append(
                {
                    'input_text_str': input_text_list[idx_text]['passage'],
                    'query_triplet': c_oie_triplets[idx_triple],
                    'query_relation_definition': open_relation_definition_dicts[idx_text][
                        c_oie_triplets[idx_triple][1]
                    ],
                    'candidate_relation_definition_dict': curr_cand_relations,
                    'idx_triple': idx_triple,
                }
            )
            ## END CHANGE

        # Run with bounded fan-out to avoid huge task bursts
        results: List[Tuple[Optional[List[Any]], Optional[int]]] = []
        if not tasks:
            results = []
        else:
            chunk_size = max(1, int(self.max_workers))

            for i in tqdm(range(0, len(tasks), chunk_size), desc='relation_verification'):
                chunk = tasks[i:i + chunk_size]

                ## BEGIN CHANGE
                # Create fresh coroutines per chunk (coroutines are single-use!)
                chunk_coroutines = [
                    self.llm_verify(
                        input_text_str=task['input_text_str'],
                        query_triplet=task['query_triplet'],
                        query_relation_definition=task['query_relation_definition'],
                        prompt_template_str=verify_prompt_template,
                        candidate_relation_definition_dict=task['candidate_relation_definition_dict'],
                        relation_example_dict=None,
                        idx_triple=task['idx_triple'],
                    )
                    for task in chunk
                ]
                # chunk_results = await asyncio.gather(*chunk_coroutines)
                ## BEGIN CHANGE
                chunk_results = await asyncio.gather(*chunk_coroutines, return_exceptions=True)
                first_exception = None
                for r in chunk_results:
                    if isinstance(r, Exception) and first_exception is None:
                        # pdb.set_trace()
                        first_exception = r

                if first_exception is not None:
                    # pdb.set_trace()
                    raise first_exception
                ## END CHANGE

                results.extend(chunk_results)

        for canonicalized_triple, idx_triple in results:
            idx_triple_to_canonicalized_triple[idx_triple] = canonicalized_triple

        print(f'enrich_is_in: {enrich}')

        # Enrichment logic (preserved)
        for curr_idx_triple, curr_canonicalized_triple in copy.deepcopy(idx_triple_to_canonicalized_triple).items():
            if curr_canonicalized_triple is None:
                if enrich:
                    idx_text = input_text_idxs[curr_idx_triple]
                    curr_open_relation = c_oie_triplets[curr_idx_triple][1]

                    self.schema_dict[curr_open_relation] = open_relation_definition_dicts[idx_text][curr_open_relation]

                    if 'sts_query' in getattr(self.embedder, 'prompts', {}):
                        embedding = self.embedder.encode(
                            open_relation_definition_dicts[idx_text][curr_open_relation],
                            prompt_name='sts_query',
                        )
                    else:
                        embedding = self.embedder.encode(open_relation_definition_dicts[idx_text][curr_open_relation])

                    self.schema_embedding_dict[curr_open_relation] = embedding

                    canonicalized_triplet = c_oie_triplets[curr_idx_triple]
                    idx_triple_to_canonicalized_triple[curr_idx_triple] = canonicalized_triplet

        # Pack outputs (preserved)
        idx_text_to_res = {}
        for curr_idx_triple in range(len(c_oie_triplets)):
            curr_idx_text = input_text_idxs[curr_idx_triple]
            if curr_idx_text not in idx_text_to_res:
                idx_text_to_res[curr_idx_text] = ([], [])
            curr_candidate_relations = []
            curr_candidate_scores = []
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