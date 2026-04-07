import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class JsonLLMCache:
    def __init__(self, path='llm_cache.json', flush_every=50):
        self.path = Path(path)
        self.flush_every = flush_every
        self.lock = threading.Lock()

        # Stats
        self.hits = 0
        self.misses = 0

        if self.path.exists():
            try:
                self.store = json.loads(self.path.read_text())
            except Exception:
                print('Warning: cache file is corrupted → starting fresh')
                self.store = {}
        else:
            self.store = {}

        self._pending = 0

    def _make_key(self, backend, model, prompt):
        return f"{backend}::{model}::{prompt}"

    def get(self, backend, model, prompt):
        key = self._make_key(backend, model, prompt)
        ##
        ## kzaporoj - begin trying this because if it exists in None, means some error, probably permission
        if key not in self.store:
            value = None
        else:
            value = self.store.get(key)
            if value is None:
                value = ''
        ## kzaporoj - end trying this because if it exists in None, means some error, probably permission
        if value is not None:
            self.hits += 1
        else:
            self.misses += 1

        return value

    def set(self, backend, model, prompt, value):
        logger.debug(f'gonna_save flush_every {self.flush_every}')
        key = self._make_key(backend, model, prompt)
        logger.debug(f'just_made_key')
        with self.lock:
            logger.debug('entering_first_lock_1')
            self.store[key] = value
            logger.debug('entering_first_lock_2')
            self._pending += 1
            logger.debug(f'entering_first_lock_3 _pending in {self._pending} and flush_every {self.flush_every}')

            if self._pending >= self.flush_every:
                logger.debug('gonna_flush')
                self.flush()

    def flush(self):
        logger.debug('starting_flushing')
        self.path.write_text(json.dumps(self.store, ensure_ascii=False))
        self._pending = 0
        logger.debug('just_flushed')
        logger.info(f'just_saved_cache_with {len(self.store)} entries')

    def stats(self):
        total = self.hits + self.misses
        if total == 0:
            pct = 0.0
        else:
            pct = (self.hits / total) * 100.0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate_pct': pct
        }
