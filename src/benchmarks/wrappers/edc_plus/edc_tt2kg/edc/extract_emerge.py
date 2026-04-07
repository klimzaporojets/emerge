import logging
import pdb
from typing import List, Dict
import os
from pathlib import Path
import edc.utils.llm_utils as llm_utils
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

from importlib import reload

reload(logging)

logger = logging.getLogger(__name__)


class ExtractorEmerge:
    # The class to handle the first stage: Open Information Extraction
    def __init__(self, model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None, openai_model=None) -> None:
        assert openai_model is not None or (model is not None and tokenizer is not None)
        self.model = model
        self.tokenizer = tokenizer
        self.openai_model = openai_model

    def extract(
            self,
            input_text_str: str,
            few_shot_examples_str: str,
            prompt_template_str: str,
            entities_hint: str = None,
            relations_hint: str = None,
    ) -> Dict[str, List[List[str]]]:
    # ) -> List[List[str]]:
        assert (entities_hint is None and relations_hint is None) or (
                relations_hint is not None and relations_hint is not None
        )
        # logger.info('inside_extract_emerge.extract about to call prompt_template_str.format_map')
        # print('inside_extract_emerge.extract about to call prompt_template_str.format_map')
        filled_prompt = prompt_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
                "input_text": input_text_str,
                "entities_hint": entities_hint,
                "relations_hint": relations_hint,
            }
        )
        # logger.info(f'filled_prompt_is: {filled_prompt}')
        # print(f'filled_prompt_is: {filled_prompt}')
        messages = [{"role": "user", "content": filled_prompt}]

        # print(f'kzaporoj: messages are {messages}')

        if self.openai_model is None:
            # llm_utils.generate_completion_transformers([messages], self.model, self.tokenizer, device=self.device)
            # print('invoking_llm_utils.generate_completion_transformers with the following messages: '
            #       f'{messages}')
            print('invoking_llm_utils.generate_completion_transformers')
            completion = llm_utils.generate_completion_transformers(
                messages, self.model, self.tokenizer, answer_prepend="Triples in text: ", max_new_token=2048
            )
        else:
            print('invoking_llm_utils.openai_chat_completion')
            completion = llm_utils.openai_chat_completion(self.openai_model, None, messages)
        # logger.info(f'completion_is: {completion}')
        print(f'completion_is: {completion}\n\n')
        #
        completion_triples_in_text = completion.strip()
        completion_triples_not_in_text = ''
        if 'Triples not in text:' in completion:
            completion_triples_in_text = completion[:completion.index('Triples not in text:')].strip()
            completion_triples_not_in_text = \
                completion[completion.index('Triples not in text:') +
                           len('Triples not in text:'):].strip()
        # extracted_triplets_list = llm_utils.parse_raw_triplets_emerge(completion_triples_in_text)
        # extracted_triplets_not_in_text_list = llm_utils.parse_raw_triplets_emerge(completion_triples_not_in_text)
        extracted_triplets_list = llm_utils.parse_raw_triplets_emerge_v3(completion_triples_in_text)
        extracted_triplets_not_in_text_list = llm_utils.parse_raw_triplets_emerge_v3(completion_triples_not_in_text)
        print(f'extracted_triplets_list: {extracted_triplets_list}')
        print(f'extracted_triplets_not_in_text_list: {extracted_triplets_not_in_text_list}')
        # pdb.set_trace()
        #
        return {
            'extracted_triplets_list': extracted_triplets_list,
            'extracted_triplets_not_in_text_list': extracted_triplets_not_in_text_list
        }
