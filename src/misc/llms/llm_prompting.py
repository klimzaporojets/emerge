from typing import Dict


def get_prompt_from_config(
        chunk: str,
        chunk_formatted_date: str,
        triples_string: str,
        prompt_template_content_name: str,
        text_creation_date: str,
        evaluation_window_start: str,
        prompt_contents: Dict
):
    prompt_template_content = prompt_contents[prompt_template_content_name]
    assert '##chunk##' in prompt_template_content
    assert '##triples_string##' in prompt_template_content
    prompt = f'{prompt_template_content}'.replace('##chunk##', chunk). \
        replace('##triples_string##', triples_string)
    #
    # Legacy / optional date
    prompt = prompt.replace('##date_text##', chunk_formatted_date)

    # New temporal placeholders (safe even if absent)
    prompt = prompt.replace('##text_creation_date##', text_creation_date)
    prompt = prompt.replace('##evaluation_window_start##', evaluation_window_start)

    return prompt
