import asyncio
import logging
import pdb
from typing import Dict, List, Tuple, Optional, Any

import edc.utils.llm_utils as llm_utils
from edc.utils.unified_llm_client import UnifiedLLMClient

logger = logging.getLogger(__name__)


class ExtractorEmergeLLMApi:
    '''
    Backend-agnostic extractor using a UnifiedLLMClient-compatible interface.

    llm_client must expose:
        await generate(prompt: str) -> str
    '''

    def __init__(self, llm_client: UnifiedLLMClient,
                 oie_llm_generation_profile: Dict[str, Any]):
        self.llm = llm_client
        self.oie_llm_generation_profile = oie_llm_generation_profile

    # ------------------------------------------------------------------
    # Async core
    # ------------------------------------------------------------------
    async def extract_async(
            self,
            input_text_str: str,
            few_shot_examples_str: str,
            prompt_template_str: str,
            entities_hint: Optional[str] = None,
            relations_hint: Optional[str] = None,
            id_txt: Optional[int] = None,
    ) -> Tuple[Dict[str, List[List[str]]], Optional[int]]:
        # Same assertions as original
        assert (entities_hint is None and relations_hint is None) or (
                entities_hint is not None and relations_hint is not None
        )

        # Build prompt (identical)
        filled_prompt = prompt_template_str.format_map(
            {
                'few_shot_examples': few_shot_examples_str,
                'input_text': input_text_str,
                'entities_hint': entities_hint,
                'relations_hint': relations_hint,
            }
        )
        filled_prompt += '\nTriples in text:'

        # LLM call
        completion = await self.llm.generate(
            prompt=filled_prompt,
            **self.oie_llm_generation_profile
        )

        # --------------------------------------------------------------
        # Split output (identical logic)
        # --------------------------------------------------------------
        if completion is None:
            logger.error('********************************************** \n'
                         'completion_in_none for some reason for '
                         f'the following prompt: \n'
                         f'{filled_prompt}\n and '
                         f'self.oie_llm_generation_profile: {self.oie_llm_generation_profile}'
                         f'\n **********************************************')
            completion = ''
        completion_triples_in_text = completion.strip()
        completion_triples_not_in_text = ''

        if 'Triples not in text:' in completion:
            completion_triples_in_text = completion[
                : completion.index('Triples not in text:')
            ].strip()
            completion_triples_not_in_text = completion[
                completion.index('Triples not in text:') + len('Triples not in text:'):
            ].strip()

        # --------------------------------------------------------------
        # Parse triples (identical)
        # --------------------------------------------------------------
        extracted_triplets_list: List[List[str]] = (
            llm_utils.parse_raw_triplets_emerge_v3(completion_triples_in_text)
        )
        # extracted_triplets_list = list(set(extracted_triplets_list))
        extracted_triplets_list = [
            list(t) for t in set(tuple(t) for t in extracted_triplets_list)
        ]
        #
        extracted_triplets_not_in_text_list: List[List[str]] = (
            llm_utils.parse_raw_triplets_emerge_v3(completion_triples_not_in_text)
        )
        extracted_triplets_not_in_text_list = [
            list(t) for t in set(tuple(t) for t in extracted_triplets_not_in_text_list)
        ]
        #
        logger.info(f'=====================================================')
        logger.info(f'=====================================================')
        logger.debug(f'****************** filled_prompt: {filled_prompt.strip()}')
        logger.info(f'****************** COMPLETION: {completion.strip()}')
        logger.info(f'**********************************************************\n'
                    f'completion_triples_in_text: "{completion_triples_in_text}" \n'
                    f'****** \n'
                    f'extracted_triplets_list: {extracted_triplets_list} \n'
                    f'*********************************************************\n')
        logger.info(f'**********************************************************\n'
                    f'completion_triples_not_in_text: "{completion_triples_not_in_text}" \n'
                    f'****** \n'
                    f'extracted_triplets_not_in_text_list: {extracted_triplets_not_in_text_list} \n'
                    f'*********************************************************\n')
        logger.info(f'=====================================================')
        logger.info(f'=====================================================')

        if len(extracted_triplets_list) == 0 and len(extracted_triplets_not_in_text_list) == 0:
            logger.warning(
                '**************************************************\n'
                'no_triples_detected_at_all\n'
                f'****** GENERATED_TEXT:\n{completion}\n '
                f'****** triples_in_text:\n{completion_triples_in_text}\n'
                f'****** triples_not_in_text:\n{completion_triples_not_in_text}\n'
                '**************************************************'
            )

        return {
            'extracted_triplets_list': extracted_triplets_list,
            'extracted_triplets_not_in_text_list': extracted_triplets_not_in_text_list,
            'input_text_str': input_text_str,
        }, id_txt

    # ------------------------------------------------------------------
    # Sync wrapper (for legacy call sites)
    # ------------------------------------------------------------------
    # def extract(
    #         self,
    #         input_text_str: str,
    #         few_shot_examples_str: str,
    #         prompt_template_str: str,
    #         entities_hint: Optional[str] = None,
    #         relations_hint: Optional[str] = None,
    #         id_txt: Optional[int] = None,
    # ) -> Tuple[Dict[str, List[List[str]]], Optional[int]]:
    #     '''
    #     Synchronous wrapper.
    #
    #     If you are already inside an event loop, call extract_async instead.
    #     '''
    #
    #     assert (entities_hint is None and relations_hint is None) or (
    #             relations_hint is not None and relations_hint is not None
    #     )
    #
    #     assert (entities_hint is None and relations_hint is None) or (
    #             relations_hint is not None and relations_hint is not None
    #     )
    #
    #     try:
    #         loop = asyncio.get_running_loop()
    #     except RuntimeError:
    #         loop = None
    #
    #     if loop and loop.is_running():
    #         raise RuntimeError(
    #             'extract() was called inside a running event loop. '
    #             'Use await extract_async(...) instead.'
    #         )
    #
    #     return asyncio.run(
    #         self.extract_async(
    #             input_text_str=input_text_str,
    #             few_shot_examples_str=few_shot_examples_str,
    #             prompt_template_str=prompt_template_str,
    #             entities_hint=entities_hint,
    #             relations_hint=relations_hint,
    #             id_txt=id_txt,
    #         )
    #     )
