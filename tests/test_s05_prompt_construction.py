"""Runtime tests for `get_prompt_from_config`: substitution, missing
placeholders, signature regression.

Lightweight (no GPU, no LLM, no transformers download).
"""
import pytest

from dataset.emerge.utils.s05_prompts import get_prompt_from_config


def test_substitutes_chunk_and_triples():
    config = {"tpl": "C=##chunk## | T=##triples_string##"}
    out = get_prompt_from_config(
        chunk="ch",
        chunk_formatted_date="2023-01-01",
        triples_string="ts",
        prompt_template_content_name="tpl",
        text_creation_date="2023-02-02",
        evaluation_window_start="2023-03-03",
        prompt_contents=config,
    )
    assert out == "C=ch | T=ts"


def test_substitutes_all_placeholders():
    config = {
        "tpl": (
            "C=##chunk## | T=##triples_string## | "
            "D=##date_text## | TC=##text_creation_date## | "
            "EWS=##evaluation_window_start##"
        )
    }
    out = get_prompt_from_config(
        chunk="ch",
        chunk_formatted_date="cf",
        triples_string="ts",
        prompt_template_content_name="tpl",
        text_creation_date="tc",
        evaluation_window_start="ews",
        prompt_contents=config,
    )
    assert "C=ch" in out
    assert "T=ts" in out
    assert "D=cf" in out
    assert "TC=tc" in out
    assert "EWS=ews" in out


def test_template_without_optional_placeholders_is_noop():
    """A legacy template with only ##chunk## / ##triples_string## must still work."""
    config = {"tpl": "##chunk##/##triples_string##"}
    out = get_prompt_from_config(
        chunk="A",
        chunk_formatted_date="X",
        triples_string="B",
        prompt_template_content_name="tpl",
        text_creation_date="Y",
        evaluation_window_start="Z",
        prompt_contents=config,
    )
    assert out == "A/B"


def test_required_placeholders_assertion():
    """The function asserts that ##chunk## and ##triples_string## are in the
    template. A bad template should fail loudly, not silently produce garbage."""
    config = {"tpl": "no chunk here"}
    with pytest.raises(AssertionError):
        get_prompt_from_config(
            chunk="A",
            chunk_formatted_date="X",
            triples_string="B",
            prompt_template_content_name="tpl",
            text_creation_date="Y",
            evaluation_window_start="Z",
            prompt_contents=config,
        )


def test_regression_5arg_call_succeeds_with_defaults():
    """Job-22422141 regression: the legacy 5-arg call signature must keep
    working as long as `text_creation_date` / `evaluation_window_start` have
    sensible defaults (empty string → placeholder substitution is a no-op).

    Backstory: in 2026-05-02 these two params were added without defaults,
    breaking all 4 call sites in `s05_prompt_llm_utils_v8.py` and burning
    1 H100-hour on TypeErrors. This test guards both directions:
      - keeps callers happy without forcing them to thread the new args
      - if someone makes the params required again, this test fails loudly
    """
    config = {"tpl": "##chunk##/##triples_string##"}
    out = get_prompt_from_config(
        chunk="A",
        chunk_formatted_date="X",
        triples_string="B",
        prompt_template_content_name="tpl",
        prompt_contents=config,
    )
    assert out == "A/B"
