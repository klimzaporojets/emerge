from dataclasses import dataclass
from typing import Optional


@dataclass
class KGPromptConfig:
    """Config for the KG-conditioned prompt-runner baseline (kg_prompt).

    Mirrors KGGenConfig, plus the UnifiedLLMClient connection params needed to
    call gpt-5.1 (Azure research proxy) directly. The wrapper sends the composed
    `prompt` field VERBATIM to the LLM; there is no chunk_size (no local
    chunking) — the whole prompt is one request.
    """
    model: str
    cache_path: str
    max_tokens: int
    output_path: str
    temperature: float = 0.0
    max_workers: Optional[int] = None
    base_url: Optional[str] = None
    batch_size: int = 100
    # ---- UnifiedLLMClient connection params ----
    backend: str = 'http'
    mode: str = 'completion'
    api_version: Optional[str] = 'v1'
    api_key_env: str = 'OPENAI_API_KEY'
    model_max_context: Optional[int] = None
    authorization_header_type: Optional[str] = None
    name: str = 'kg_prompt'
