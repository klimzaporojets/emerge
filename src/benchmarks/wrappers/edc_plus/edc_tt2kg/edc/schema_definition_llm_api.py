# unlike schema_definition_tgi_api, this is more generic and will work with any llm
# either executed as tgi or llm api
import logging
import pdb
from typing import List, Tuple, Dict, Any

import edc.utils.llm_utils as llm_utils
from edc.utils.unified_llm_client import UnifiedLLMClient

logger = logging.getLogger(__name__)


class SchemaDefinerLLMApi:
    '''
    Backend-agnostic schema definer using a UnifiedLLMClient-compatible interface.

    The client must expose:
        await generate(prompt: str) -> str
    '''

    def __init__(self,
                 llm_client: UnifiedLLMClient,
                 sd_llm_generation_profile: Dict[str, Any]
                 ):
        self.llm: UnifiedLLMClient = llm_client
        self.sd_llm_generation_profile = sd_llm_generation_profile

    async def define_schema(
            self,
            input_text_str: str,
            extracted_triplets_list: List[List[str]],
            few_shot_examples_str: str,
            prompt_template_str: str,
            idx_it: int
    ) -> Tuple[Dict[str, Any], int]:
        # --------------------------------------------------------------
        # Collect relations (identical to original)
        # --------------------------------------------------------------
        relations_present = {t[1] for t in extracted_triplets_list}

        # --------------------------------------------------------------
        # Build prompt (identical to original)
        # --------------------------------------------------------------
        filled_prompt = prompt_template_str.format_map(
            {
                'text': input_text_str,
                'few_shot_examples': few_shot_examples_str,
                'relations': relations_present,
                'triples': extracted_triplets_list,
            }
        )
        filled_prompt += '\nAnswer: '

        # --------------------------------------------------------------
        # LLM call (replaces TGI HTTP call)
        # --------------------------------------------------------------
        completion = await self.llm.generate(filled_prompt,
                                             **self.sd_llm_generation_profile)

        # --------------------------------------------------------------
        # Extract answer section (identical logic)
        # --------------------------------------------------------------
        # pdb.set_trace()
        # assert 'Answer: ' in completion

        completion = self._extract_answer(completion)

        # Normalize underscores (legacy behavior preserved)
        completion = completion.replace('\\_', ' ')
        completion = completion.replace('_', ' ')

        # --------------------------------------------------------------
        # Parse relation definitions
        # --------------------------------------------------------------
        relation_definition_dict = llm_utils.parse_relation_definition(completion)

        # --------------------------------------------------------------
        # Debug logging (same content as original)
        # --------------------------------------------------------------
        logger.debug(
            '***************************************************************************\n'
            'schema_definer_llm_api completion after extracting only Answer:\n'
            f'*-*-*-*-* extracted_triplets_list: {extracted_triplets_list}\n'
            f'*-*-*-*-* completion: {completion}\n'
            f'*-*-*-*-* relation_definition_dict: {relation_definition_dict}\n'
            '****************************************************************************\n'
        )

        # --------------------------------------------------------------
        # Missing relations warning (identical semantics)
        # --------------------------------------------------------------
        missing_relations = [
            rel for rel in relations_present
            if rel not in relation_definition_dict
        ]

        if missing_relations:
            logger.debug(
                'warning_missing_relations Relations %s are missing from the relation definition!\n'
                '***************************************************************************\n'
                'schema_definer_llm_api completion after extracting only Answer:\n'
                f'*-*-*-*-* extracted_triplets_list: {extracted_triplets_list}\n'
                f'*-*-*-*-* completion: {completion}\n'
                f'*-*-*-*-* relation_definition_dict: {relation_definition_dict}\n'
                '****************************************************************************\n',
                missing_relations,
            )

        return relation_definition_dict, idx_it

    @staticmethod
    def _extract_answer(completion: str) -> str:
        marker = 'Answer: '
        # last_index = 0
        to_ret_completion = completion.strip()
        if 'Answer: ' in completion:
            last_index = completion.rfind(marker)
            to_ret_completion = completion[last_index + len(marker):].strip()
        # if last_index == -1:
        #     return ''

        return to_ret_completion
