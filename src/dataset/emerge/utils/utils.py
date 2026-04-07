from typing import Dict


def get_llm_assessment(
        triple: Dict,
        llm_assessor_name: str,
        llm_prompt_type: str,
        hash_id: str
) -> bool:
    """
    Extract LLM assessment from a triple for a specific assessor and prompt type.

    :param triple: Triple dictionary containing llm_assessment list
    :param llm_assessor_name: Name of the LLM assessor
    :param llm_prompt_type: 'triple_deprecation' or 'triple_assessment'
    :param hash_id: Hash ID for error reporting
    :return: The assessment value (bool) or False if not found
    """
    assessment = [
        ct for ct in triple['llm_assessment']
        if ct['llm_name'] == llm_assessor_name
           and ct['llm_prompt_type'] == llm_prompt_type
    ]
    if assessment:
        return assessment[0]['llm_assessment']
    else:
        print(f'ERROR for triple {triple} can not find '
              f'llm_assessor_name {llm_assessor_name} and '
              f'llm_prompt_type {llm_prompt_type} '
              f'hash_id {hash_id}')
        return False
