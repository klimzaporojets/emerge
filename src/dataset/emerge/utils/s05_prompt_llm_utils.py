import logging
import re
import traceback
from typing import Dict, Set, List, Tuple, Any
from datetime import datetime

from huggingface_hub import InferenceClient
from huggingface_hub.errors import GenerationError
from transformers import PreTrainedTokenizerBase

import os
from dataset.emerge.utils.constants import ACTION_CATEGORY_ADD, ACTION_CATEGORY_ASSERT, ACTION_CATEGORY_DEPRECATE

from dataset.emerge.utils.s05_prompts import \
    get_prompt_removed_single_triple_v5, get_prompt_removed_triples_v5, get_prompt_added_single_triple_v4, \
    get_prompt_added_triples_v4, get_prompt_from_config

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)


def parse_returned_llm_text_v8(is_single: bool,
                               all_identified_triples: Set,
                               output_text: str,
                               action_category: str,
                               config,
                               llm_triples_to_answer: Dict[Tuple, Any],
                               llm_triples_to_prompt_responses: Dict[Tuple, Any],
                               triples_str: str
                               ):
    prompt_and_output_log = output_text
    if not is_single:
        # THE PATTERN BELOW IS GOOD, BUT MISSES THE NOT CLOSED WITH ) entities in TRIPLES
        # pattern = (r"\[[\"'](Q\d+)\s*\((.+?)\)[\"'],\s*[\"'](P\d+)\s*\((.+?)\)[\"'],\s*[\"'](Q\d+)\s*\((.+?)\)[\"']\].+?(YES|NO)(.*?)(?=\n|$)"
        # )
        # THE PATTERN BELOW IS slightly better because it should not miss
        # THE NOT CLOSED WITH ) entities in TRIPLES, but if it gives problems change
        # to the old one above
        # pattern = r'\[\s*(?:"|\')(Q\d+)\s*\((.+?)\)?(?:"|\')\s*,\s*(?:"|\')(P\d+)\s*\((.+?)\)?(?:"|\')\s*,\s*(?:"|\')(Q\d+)\s*\((.+?)\)?(?:"|\')\s*\].+?(YES|NO)(.*?)(?=\n|$)'
        # pattern = r'\[\s*(?:"|\')(Q\d+)\s*\((.+?)\)?(?:"|\')\s*,\s*(?:"|\')(P\d+)\s*\((.+?)\)?(?:"|\')\s*,\s*(?:"|\')(Q\d+)\s*\((.+?)\)?(?:"|\')\s*\].+?(YES|NO)(.*?)(?=\d.|\n(\[\s*(?:"|\')(Q\d+))|$)'
        # pattern = r'\[\s*(?:"|\')(Q\d+)\s*\((.+?)\)?(?:"|\')\s*,\s*(?:"|\')(P\d+)\s*\((.+?)\)?(?:"|\')\s*,\s*(?:"|\')(Q\d+)\s*\((.+?)\)?(?:"|\')\s*\].+?(YES|NO)(.*?)(?=\n\d\.|\n(\[\s*(?:"|\')(Q\d+))|$)'
        # pattern = r'\[\s*(?:"|\')(Q\d+)\s*\((.+?)\)?(?:"|\')\s*,\s*(?:"|\')(P\d+)\s*\((.+?)\)?(?:"|\')\s*,\s*(?:"|\')(Q\d+)\s*\((.+?)\)?(?:"|\')\s*\].+?(YES|NO)(.*?)(?=(\n\d*\.)|(\[\s*["\']Q\d+)|$)'
        pattern = r'\[\s*(?:"|\')(Q\d+)\s*\((.+?)\)?(?:"|\')\s*,\s*(?:"|\')(P\d+)\s*\((.+?)\)?(?:"|\')\s*,\s*(?:"|\')(Q\d+)\s*\((.+?)\)?(?:"|\')\s*\].+?(YES|NO)(.*?)(?=(\n\d*\.)|(\n\s*\[\s*["\']Q\d+)|$)'
        #
        # Find all matches in the text
        matches = re.findall(pattern, output_text, re.S)

        for match in matches:
            # head, relation, tail, llm_assessment, explanation = match
            head_id, head_label, relation_id, relation_label, tail_id, tail_label, \
                llm_assessment, explanation, _, _ = match

            ###### BEGIN (2025.08.30): trying to fix encoding problem
            # if '\\' in head:
            #     head = head.encode('utf-8').decode('unicode_escape')
            # if '\\' in relation:
            #     relation = relation.encode('utf-8').decode('unicode_escape')
            # if '\\' in tail:
            #     tail = tail.encode('utf-8').decode('unicode_escape')
            ###### END
            all_identified_triples.add((head_id, relation_id, tail_id, action_category))
            assert llm_assessment in {'YES', 'NO'}
            llm_assessment_bool = False
            if llm_assessment == 'YES':
                llm_assessment_bool = True
            # assigns if it is False or was not assigned, if it is true, then no assignment
            if ((head_id, relation_id, tail_id, action_category) not in llm_triples_to_answer) or \
                    not llm_triples_to_answer[(head_id, relation_id, tail_id, action_category)]:
                # the if above makes sure that it adds the first occurrence of a particular triple only,
                # the following ones are probably hallucinations
                # and should be ignored
                llm_triples_to_answer[(head_id, relation_id, tail_id, action_category)] = \
                    llm_assessment_bool
                if config['log_prompt_per_triple']:
                    llm_triples_to_prompt_responses \
                        [(head_id, relation_id, tail_id, action_category)] = explanation
                    # llm_triples_to_prompt_responses[(head, relation, tail, action_category)] = \
                    #     prompt_and_output_log

        # output_text_all = output_text_all + ' ' + output_text
    else:
        # pattern = r"\[[\"'](Q\d+ \([^\]]+?\))[\"'], [\"'](P\d+ \([^\]]+?\))[\"'], [\"'](Q\d+ \([^\]]+?\))[\"']\]"
        # pattern = r"\[[\"'](Q\d+ \((.*?)\))[\"'], [\"'](P\d+ \((.*?)?\))[\"'], [\"'](Q\d+ \((.*?)?\))[\"']\]"
        # pattern = r"\[[\"'](Q\d+ \((.*?)\))[\"'], [\"'](P\d+ \((.*?)?\))[\"'], [\"'](Q\d+ \((.*?)?\))[\"']\]"
        pattern = (r"\[[\"'](Q\d+)\s*\((.*?)\)[\"'],"
                   r"\s*[\"'](P\d+)\s*\((.*?)\)?[\"'],"
                   r"\s*[\"'](Q\d+)\s*\((.*?)\)?[\"']\]")
        logger.debug(f'triples_str_is: {triples_str}')
        matches = re.findall(pattern, triples_str, re.S)
        # matches = re.findall(pattern, output_text, re.S)

        if 'YES' in output_text:
            logger.debug(f'single_output_YES: {output_text}')
            llm_assessment_bool = True
        else:
            logger.debug(f'single_output_NO: {output_text}')
            llm_assessment_bool = False

        nr_matches = 0
        for match in matches:
            nr_matches += 1
            # head, _, relation, _, tail, _ = match
            head_id, head_label, relation_id, relation_label, tail_id, tail_label = match
            ###### BEGIN (2025.08.30): trying to fix encoding problem
            # if '\\' in head:
            #     head = head.encode('utf-8').decode('unicode_escape')
            # if '\\' in relation:
            #     relation = relation.encode('utf-8').decode('unicode_escape')
            # if '\\' in tail:
            #     tail = tail.encode('utf-8').decode('unicode_escape')
            ###### END
            logger.debug(f'match_is: {match}, head,relation,tail is: '
                         f'{head_id} ({head_label}), '
                         f'{relation_id} ({relation_label}),'
                         f'{tail_id} ({tail_label})')
            all_identified_triples.add((head_id, relation_id, tail_id, action_category))
            # if (head, relation, tail, action_category) not in llm_triples_to_answer:
            if ((head_id, relation_id, tail_id, action_category) not in llm_triples_to_answer) or \
                    not llm_triples_to_answer[(head_id, relation_id, tail_id, action_category)]:
                # the if above makes sure that it adds the first occurrence of a particular triple only,
                # the following ones are probably hallucinations
                # and should be ignored
                llm_triples_to_answer[(head_id, relation_id, tail_id, action_category)] = \
                    llm_assessment_bool
                if config['log_prompt_per_triple']:
                    llm_triples_to_prompt_responses \
                        [(head_id, relation_id, tail_id, action_category)] = \
                        prompt_and_output_log

            assert nr_matches < 2  # this is a single case, only one triple should be there
        if nr_matches == 0:
            logger.warning(f'nr_matches_in_zero! for triples_str in {triples_str} '
                           f'and output_text in {output_text}')
        # output_text_all = output_text_all + ' ' + output_text
    logger.debug(f'*****************************************')
    logger.debug(f'parse_returned_llm_text_v8 \n'
                f'is_single : {is_single} \n'
                f'output_text : {output_text} \n'
                f'matches: {matches} \n'
                f'triples_str: {triples_str} \n'
                f'llm_triples_to_answer: {llm_triples_to_answer} \n'
                f'all_identified_triples: {all_identified_triples} \n'
                f'llm_triples_to_prompt_responses: {llm_triples_to_prompt_responses} \n')
    logger.debug(f'*****************************************')
    return (llm_triples_to_answer, all_identified_triples,
            # output_text_all,
            llm_triples_to_prompt_responses)


def call_llm_and_return_parsed_result_v8(
        prompt_type: str,
        chunk: str,
        chunk_timestamp: int,
        triples_str: str,
        action_type: str,
        tokenizer: PreTrainedTokenizerBase,
        client: InferenceClient,
        config: Dict,
        action_category: str
):
    if prompt_type == 'prompt1':
        if action_category in {ACTION_CATEGORY_ADD, ACTION_CATEGORY_ASSERT} \
                and action_type == 'multiple':
            prompt = get_prompt_added_triples_v4(chunk=chunk, triples_string=triples_str)
        elif action_category in {ACTION_CATEGORY_ADD, ACTION_CATEGORY_ASSERT} \
                and action_type == 'single':
            prompt = get_prompt_added_single_triple_v4(chunk=chunk, triples_string=triples_str)
        elif action_category == ACTION_CATEGORY_DEPRECATE and action_type == 'multiple':
            prompt = get_prompt_removed_triples_v5(chunk=chunk, triples_string=triples_str)
        elif action_category == ACTION_CATEGORY_DEPRECATE and action_type == 'single':
            prompt = get_prompt_removed_single_triple_v5(chunk=chunk, triples_string=triples_str)
        else:
            raise RuntimeError(f'Prompt action_type not recognized: {action_type} or category: {action_category}')
    elif prompt_type == 'prompt2':
        chunk_formatted_date = datetime.fromtimestamp(chunk_timestamp).strftime("%Y-%m-%d")
        if action_category in {ACTION_CATEGORY_ADD, ACTION_CATEGORY_ASSERT} \
                and action_type == 'multiple':
            prompt = get_prompt_from_config(
                chunk=chunk,
                chunk_formatted_date=chunk_formatted_date,
                triples_string=triples_str,
                prompt_template_content_name='assert_multi_prompt_template_content',
                prompt_contents=config
            )
        elif action_category in {ACTION_CATEGORY_ADD, ACTION_CATEGORY_ASSERT} \
                and action_type == 'single':
            prompt = get_prompt_from_config(
                chunk=chunk,
                chunk_formatted_date=chunk_formatted_date,
                triples_string=triples_str,
                prompt_template_content_name='assert_single_prompt_template_content',
                prompt_contents=config
            )
        elif action_category == ACTION_CATEGORY_DEPRECATE and action_type == 'multiple':
            prompt = get_prompt_from_config(
                chunk=chunk,
                chunk_formatted_date=chunk_formatted_date,
                triples_string=triples_str,
                prompt_template_content_name='deprecate_multi_prompt_template_content',
                prompt_contents=config
            )
        elif action_category == ACTION_CATEGORY_DEPRECATE and action_type == 'single':
            prompt = get_prompt_from_config(
                chunk=chunk,
                chunk_formatted_date=chunk_formatted_date,
                triples_string=triples_str,
                prompt_template_content_name='deprecate_single_prompt_template_content',
                prompt_contents=config
            )
        else:
            raise RuntimeError(f'Prompt action_type not recognized: {action_type} or category: {action_category}')
    else:
        raise RuntimeError(f'Prompt not recognized: {prompt_type}')

    #
    messages = [{'role': 'user', 'content': prompt}]
    logger.debug(f'*** {action_type} --- {action_category} ---  '
                 f'following_messages_sent_to_llm_server: {messages}')
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        return_dict=False,
        return_tensors='pt'
    )
    logger.debug(f'inputs_length_is: {len(inputs)}')
    outputs = []
    try:
        generated = client.text_generation(inputs, stream=True, max_new_tokens=2048)
        for t in generated:
            outputs.append(t)
    except GenerationError as e:
        logger.error(f'{config["api_llm_device"]}:{config["api_llm_port"]} '
                     f'GenerationError_for {e} the following prompt of length '
                     f'{len(prompt.split(" "))} '
                     f'length of the chunk: {len(chunk.split(" "))} '
                     f'{messages}')
        stack_trace = traceback.format_exc()
        logger.error(f'{config["api_llm_device"]}:{config["api_llm_port"]} '
                     f'the complete stack is: {stack_trace}')
        raise

    output_text = ''.join(outputs)

    logger.debug(f'*** (GENERATED) {action_type} --- {action_category} ---  '
                 f'following_messages_sent_to_llm_server: {output_text}')

    logger.debug('---------------------------------------')
    logger.debug('---------------------------------------')
    prompt_and_output_log = f'the_prompt_itself_is: {prompt} \n AND ' \
                            f'output_text_of_prompt: {output_text}'
    logger.debug(prompt_and_output_log)
    logger.debug('---------------------------------------')
    logger.debug('---------------------------------------')

    return output_text
