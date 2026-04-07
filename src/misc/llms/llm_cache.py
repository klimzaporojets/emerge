# cache.py
from .json_llm_cache import JsonLLMCache

llm_cache:JsonLLMCache = None   # will be initialized later

def init_llm_cache(path: str, flush_every: int = 50):
    """
    Initialize the global cache instance with runtime parameters.
    Call this ONCE from your __main__.
    """
    global llm_cache
    llm_cache = JsonLLMCache(path, flush_every)
