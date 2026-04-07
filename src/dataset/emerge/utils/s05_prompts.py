from typing import Dict


def get_prompt_added_triples(chunk, triples_string):
    prompt = f"""You are given the following text:
{chunk}

Does the text contain the information represented in the following list of triples?
{triples_string}

Give a short explanation, and then write a numbered list for each of the triples with YES if the triple
is represented in the text, and NO otherwise.
            """
    return prompt


def get_prompt_added_single_triple(chunk, triples_string):
    prompt = f"""You are given the following text:
{chunk}

Does the text contain the information represented in the following triple?
{triples_string}

Give a short explanation, and then write the triple above followed by YES if the triple
is represented in the text, and NO otherwise.
            """
    return prompt


def get_prompt_added_single_triple_v2(chunk, triples_string):
    prompt = f"""You are given the following text:
{chunk}

Does the text contain information that explicitly or implicitly represents the following triple?
{triples_string}

Answer "YES" if the text above explicitly or implicitly states or implies the given triple, or 
answer "NO" otherwise. Follow your "YES" or "NO" answer with a brief explanation.
            """

    return prompt


def get_prompt_added_triples_v2(chunk, triples_string):
    prompt = f"""You are given the following text:
{chunk}

Does the text contain the information represented in the following list of triples?
{triples_string}

Write a numbered list with the triples above, where each of the triples is followed by YES if the triple is represented in the text, and NO otherwise. Follow your "YES" or "NO" answer for each of the triples with a brief explanation. 
            """
    return prompt


def get_prompt_added_triples_v3(chunk, triples_string):
    prompt = f"""You are given the following text:
{chunk}

Can the following triples be directly or indirectly (the text provides some hints) inferred from the text, use common sense?
{triples_string}

Write a numbered list with the triples above, where each of the triples is followed by YES if the triple is represented in the text, and NO otherwise. Follow your "YES" or "NO" answer for each of the triples with a brief explanation. 
            """
    return prompt


def get_prompt_added_triples_v4(chunk, triples_string):
    prompt = f"""You are given the following text:
{chunk}

Can the following triples be directly or indirectly (the text provides some hints) inferred from the text? Use common sense but not the knowledge that can not be inferred from the text above.
{triples_string}

Write a numbered list with the triples above, where each of the triples is followed by YES if the triple is represented in the text, and NO otherwise. Follow your "YES" or "NO" answer for each of the triples with a brief explanation. 
            """
    return prompt


# def get_prompt_from_config_v12(chunk: str,
#                            chunk_formatted_date: str,
#                            triples_string: str,
#                            prompt_template_content_name: str,
#                            prompt_contents: Dict):
#     prompt_template_content = prompt_contents[prompt_template_content_name]
#     assert '##chunk##' in prompt_template_content
#     assert '##triples_string##' in prompt_template_content
#     prompt = f"{prompt_template_content}".replace("##chunk##", chunk).replace("##triples_string##", triples_string)
#     if '##date_text##' in prompt:
#         prompt = f"{prompt}".replace("##date_text##", chunk_formatted_date)
#     return prompt

def get_prompt_from_config(
        chunk: str,
        chunk_formatted_date: str,
        triples_string: str,
        prompt_template_content_name: str,
        text_creation_date: str,
        evaluation_window_start:str,
        prompt_contents: Dict
):
    prompt_template_content = prompt_contents[prompt_template_content_name]
    assert '##chunk##' in prompt_template_content
    assert '##triples_string##' in prompt_template_content
    prompt = f"{prompt_template_content}".replace("##chunk##", chunk).replace("##triples_string##", triples_string)
    # if '##date_text##' in prompt:
    #     prompt = f"{prompt}".replace("##date_text##", chunk_formatted_date)
    # Legacy / optional date
    prompt = prompt.replace("##date_text##", chunk_formatted_date)

    # New temporal placeholders (safe even if absent)
    prompt = prompt.replace("##text_creation_date##", text_creation_date)
    prompt = prompt.replace("##evaluation_window_start##", evaluation_window_start)

    return prompt


def get_prompt_added_single_triple_v3(chunk, triples_string):
    prompt = f"""You are given the following text:
{chunk}

Does the text contain information that explicitly or implicitly (the text provides some hints), directly or indirectly represents the following triple, use common sense?
{triples_string}

Answer "YES" if the text above explicitly or implicitly states or implies the given triple, or 
answer "NO" otherwise. Follow your "YES" or "NO" answer with a brief explanation.
            """

    return prompt


def get_prompt_added_single_triple_v4(chunk, triples_string):
    prompt = f"""You are given the following text:
{chunk}

Does the text contain information that explicitly or implicitly (the text provides some hints), directly or indirectly represents the following triple? Use common sense but not the knowledge that can not be inferred from the text above.
{triples_string}

Answer "YES" if the text above explicitly or implicitly states or implies the given triple, or 
answer "NO" otherwise. Follow your "YES" or "NO" answer with a brief explanation.
            """

    return prompt


def get_prompt_removed_triples(chunk, triples_string):
    prompt = f"""You are given the following text:
{chunk}

Does the knowledge contained in text explicitly states or implies that the following triples have to be deprecated or removed from the knowledge graph?
{triples_string}

Give a short explanation, and then write a numbered list for each of the triples with YES if the text explicitly states or implies the deprecation of the triple, and NO if there is not enough information in text to remove or deprecate the triple.
            """
    return prompt


def get_prompt_removed_triples_v2(chunk, triples_string):
    prompt = f"""You are given the following text:
{chunk}

Does the knowledge contained in text explicitly states or implies that the following triples have to be deprecated or removed from the knowledge graph?
{triples_string}

Write a numbered list for each of the triples with YES if the text explicitly states or implies that the triple has to be deprecated or removed from the knowledge graph, and NO if there is not enough information in text to remove or deprecate the triple. Follow your "YES" or "NO" answer for each of the triples with a brief explanation.
            """
    return prompt


def get_prompt_removed_triples_v3(chunk, triples_string):
    prompt = f"""You are given the following text:
{chunk}

Does this text imply that the following triples must be deprecated or removed from the knowledge graph?
{triples_string}

Write a numbered list with the triples above, where each of the triples is followed by YES if the text implies that the triples must be deprecated or removed from the knowledge graph, and NO otherwise. Follow your "YES" or "NO" answer for each of the triples with a brief explanation.
            """
    return prompt


def get_prompt_removed_triples_v4(chunk, triples_string):
    prompt = f"""You are given the following text:
{chunk}

Does this text imply that the following triples are no longer valid or do not hold?
{triples_string}

Write a numbered list with the triples above, where each of the triples is followed by YES if the text implies that the triples are no longer valid or do not hold, and NO otherwise. Follow your "YES" or "NO" answer for each of the triples with a brief explanation.
            """
    return prompt


def get_prompt_removed_triples_v5(chunk, triples_string):
    prompt = f"""You are given the following text:
{chunk}

Does this text imply that the following triples are no longer valid or do not hold at the moment the text was written?
{triples_string}

Write a numbered list with the triples above, where each of the triples is followed by YES if the text implies that the triple is no longer valid or do not hold, and NO otherwise. Follow your "YES" or "NO" answer for each of the triples with a brief explanation.
            """
    return prompt


def get_prompt_removed_single_triple(chunk, triples_string) -> str:
    prompt = f"""You are given the following text:
{chunk}

Does the knowledge contained in text explicitly states or implies that the following triple has to be deprecated or removed from the knowledge graph? 
{triples_string}

Give a short explanation, and then write the triple above followed by YES if the text
explicitly states or implies the deprecation of the triple, and NO if there is not enough information in text to remove or deprecate the triple.
            """
    return prompt


def get_prompt_removed_single_triple_v2(chunk, triples_string) -> str:
    prompt = f"""You are given the following text:
{chunk}

Does the knowledge contained in text explicitly or implicitly states or implies that the following triple has to be deprecated or removed from the knowledge graph?   
{triples_string}

Answer "YES" if the text explicitly or implicitly states or implies the deprecation of the triple, or answer "NO" if there is not enough information in text to remove or deprecate the triple. Follow your "YES" or "NO" answer with brief explanation. """
    return prompt


def get_prompt_removed_single_triple_v3(chunk, triples_string) -> str:
    prompt = f"""You are given the following text:
{chunk}

Does this text imply that the following triple must be deprecated or removed from the knowledge graph?
{triples_string}

Answer "YES" if the text implies that the triple must be deprecated or removed from the knowledge graph, or answer "NO" otherwise. Follow your "YES" or "NO" answer with brief explanation. """
    return prompt


def get_prompt_removed_single_triple_v4(chunk, triples_string) -> str:
    prompt = f"""You are given the following text:
{chunk}

Does this text imply that the following triple is no longer valid or does not hold?
{triples_string}

Answer "YES" if the text implies that the triple is no longer valid or does not hold, or answer "NO" otherwise. Follow your "YES" or "NO" answer with brief explanation. """
    return prompt


def get_prompt_removed_single_triple_v5(chunk, triples_string) -> str:
    prompt = f"""You are given the following text:
{chunk}

Does this text imply that the following triple is no longer valid or does not hold at the moment the text was written?
{triples_string}

Answer "YES" if the text implies that the triple is no longer valid or does not hold, or answer "NO" otherwise. Follow your "YES" or "NO" answer with brief explanation. """
    return prompt


def get_prompt1_old(chunk, triples_string):
    prompt1 = f"""You are given the following text:
{chunk}

Does the text contain the information represented in the following list of triples?
{triples_string}

Give a short explanation, and then write a numbered list for each of the triples with YES if the triple
is represented in the text, and NO otherwise.

1. YES
2. NO
3. YES
            """
    return prompt1
