"""Functional smoke test for `call_llm_and_return_parsed_result_v8`: drives all
8 (prompt1/prompt2 × multi/single × assert/deprecate) prompt-building branches
with a mock `InferenceClient` + mock tokenizer. Catches signature drifts in
prompt-builder helpers, missing template-content keys in the config, and
incompatible kwargs.

Runs in <1 sec on CPU. No HF download, no GPU.
"""
from unittest.mock import MagicMock

import pytest

from dataset.emerge.utils.constants import (
    ACTION_CATEGORY_ADD,
    ACTION_CATEGORY_ASSERT,
    ACTION_CATEGORY_DEPRECATE,
)
from dataset.emerge.utils.s05_prompt_llm_utils_v8 import (
    call_llm_and_return_parsed_result_v8,
)


@pytest.fixture
def fake_config():
    """A config that mimics the on-disk JSON used by the v9 405B run.
    All four prompt2 template names are populated so every action_type ×
    action_category branch can resolve."""
    template = (
        "C=##chunk## T=##triples_string## D=##date_text## "
        "TC=##text_creation_date## EWS=##evaluation_window_start##"
    )
    return {
        "log_prompt_per_triple": True,
        "api_llm_device": "127.0.0.2",
        "api_llm_port": 8083,
        "assert_multi_prompt_template_content": template,
        "assert_single_prompt_template_content": template,
        "deprecate_multi_prompt_template_content": template,
        "deprecate_single_prompt_template_content": template,
    }


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.apply_chat_template.return_value = "tokenized-prompt"
    return tok


@pytest.fixture
def mock_client():
    cli = MagicMock()
    # text_generation(stream=True) returns an iterable of token strings
    cli.text_generation.return_value = iter(
        ["[\"Q1 (foo)\", \"P2 (bar)\", \"Q3 (baz)\"] YES"]
    )
    return cli


def _call(prompt_type, action_category, action_type, fake_config, mock_tokenizer, mock_client):
    return call_llm_and_return_parsed_result_v8(
        prompt_type=prompt_type,
        chunk="The text passage about Albert Einstein.",
        chunk_timestamp=1672531200,  # 2023-01-01 UTC
        triples_str='["Q937 (Einstein)", "P19 (born in)", "Q3012 (Ulm)"]',
        action_type=action_type,
        tokenizer=mock_tokenizer,
        client=mock_client,
        config=fake_config,
        action_category=action_category,
    )


PROMPT2_CASES = [
    (ACTION_CATEGORY_ADD,        "multiple"),
    (ACTION_CATEGORY_ADD,        "single"),
    (ACTION_CATEGORY_ASSERT,     "multiple"),
    (ACTION_CATEGORY_ASSERT,     "single"),
    (ACTION_CATEGORY_DEPRECATE,  "multiple"),
    (ACTION_CATEGORY_DEPRECATE,  "single"),
]


@pytest.mark.parametrize("action_category, action_type", PROMPT2_CASES)
def test_prompt2_branches_run(
    action_category, action_type, fake_config, mock_tokenizer, mock_client
):
    # Reset the mock generator for each parametrised case
    mock_client.text_generation.return_value = iter(["YES ok"])
    out = _call(
        "prompt2", action_category, action_type, fake_config, mock_tokenizer, mock_client
    )
    assert "YES" in out
    # Tokenizer was called once with apply_chat_template
    assert mock_tokenizer.apply_chat_template.called
    # Client was called once
    assert mock_client.text_generation.called


PROMPT1_CASES = [
    (ACTION_CATEGORY_ADD,        "multiple"),
    (ACTION_CATEGORY_ADD,        "single"),
    (ACTION_CATEGORY_DEPRECATE,  "multiple"),
    (ACTION_CATEGORY_DEPRECATE,  "single"),
]


@pytest.mark.parametrize("action_category, action_type", PROMPT1_CASES)
def test_prompt1_branches_run(
    action_category, action_type, fake_config, mock_tokenizer, mock_client
):
    mock_client.text_generation.return_value = iter(["YES ok"])
    out = _call(
        "prompt1", action_category, action_type, fake_config, mock_tokenizer, mock_client
    )
    assert "YES" in out


def test_unknown_prompt_type_raises(fake_config, mock_tokenizer, mock_client):
    with pytest.raises(RuntimeError, match="Prompt not recognized"):
        _call(
            "prompt9000",
            ACTION_CATEGORY_ADD,
            "single",
            fake_config,
            mock_tokenizer,
            mock_client,
        )


def test_unknown_action_category_raises(fake_config, mock_tokenizer, mock_client):
    with pytest.raises(RuntimeError, match="Prompt action_type not recognized"):
        _call(
            "prompt2",
            "BOGUS_CATEGORY",
            "single",
            fake_config,
            mock_tokenizer,
            mock_client,
        )
