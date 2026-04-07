from dataclasses import dataclass
from typing import Optional


@dataclass
class RAKGConfig:
    batch_size: int
    max_workers: int
    cache_path: str
    openai_model: str
    openai_embedding_model: str
    openai_similarity_model: str
    base_url: str
    base_url_embedding_model: str
    use_similarity: bool
    output_path: str
    name: str = 'rakg'
