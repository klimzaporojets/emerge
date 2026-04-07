from dataclasses import dataclass
from typing import Optional


@dataclass
class KGGenConfig:
    model: str
    cache_path: str
    max_tokens: int
    output_path: str
    temperature: float = 0.0
    max_workers: Optional[int] = None
    chunk_size: Optional[int] = None
    base_url: Optional[str] = None
    batch_size: int = 100
    name:str='kggen'
