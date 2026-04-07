from dataclasses import dataclass
from typing import Optional


@dataclass
class REBELConfig:
    model: str
    batch_size: int
    max_workers: int
    cache_path: str
    device: str
    max_length: int
    num_beams: int
    output_path: str
    name:str = 'rebel'
