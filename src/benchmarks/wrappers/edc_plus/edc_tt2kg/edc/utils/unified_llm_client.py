import asyncio
import json
import logging
import random

import aiohttp
from typing import Optional, Literal, Dict, Any

try:
    from openai import AsyncOpenAI, BadRequestError
except ImportError:
    AsyncOpenAI = None

from transformers import AutoTokenizer

BackendType = Literal['auto', 'openai', 'http']
ModeType = Literal['chat', 'completion', 'responses']


logger = logging.getLogger(__name__)


class UnifiedLLMClient:
    '''
    Unified async LLM client supporting:
    - vLLM (local, OpenAI-compatible)
    - Azure OpenAI
    - OpenAI-compatible proxies
    - TGI (OpenAI-compatible mode)
    - Raw HTTP fallback (aiohttp)

    Usage everywhere else:
        text = await client.generate(prompt, temperature=..., max_tokens=..., stop=...)
    '''

    def __init__(
            self,
            base_url: str,
            model: str,
            model_max_context: int,
            api_key: Optional[str] = None,
            concurrency: int = 64,
            timeout: int = 300,
            backend: BackendType = 'auto',
            api_version: Optional[str] = None,
            mode: ModeType = 'chat',  # <<< ADDED
            authorization_header_type=None
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        #### START fix semaphore
        self._concurrency = concurrency
        #### END fix semaphore
        self.sem = asyncio.Semaphore(concurrency)
        self.authorization_header_type = authorization_header_type
        self.backend = backend
        self.api_version = api_version
        self.api_key = api_key or 'dummy'
        self.mode = mode  # <<< ADDED
        self.model_max_context = model_max_context
        self._session: Optional[aiohttp.ClientSession] = None
        self._openai_client: Optional[AsyncOpenAI] = None
        ### NEW: load tokenizer ONCE (fast, reused)
        if self.model_max_context is not None and self.model_max_context > 0:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model,
                use_fast=True,
            )
        else:
            self._tokenizer = None
        self._decide_backend()

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------
    def _decide_backend(self) -> None:
        if self.backend == 'auto':
            if AsyncOpenAI is not None:
                self.backend = 'openai'
            else:
                self.backend = 'http'

        if self.backend == 'openai' and AsyncOpenAI is None:
            raise RuntimeError('AsyncOpenAI requested but openai package not installed')

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    async def __aenter__(self):
        #### START fix semaphore
        self.sem = asyncio.Semaphore(self._concurrency)
        #### END fix semaphore
        if self.backend == 'http':
            logger.info('self.backend_initialization to http')
            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                if self.authorization_header_type is None:
                    headers['Authorization'] = f'Bearer {self.api_key}'
                elif self.authorization_header_type == 'WILLMA_SURF':
                    headers['X-API-KEY'] = f'{self.api_key}'
                else:
                    raise RuntimeError(f'non_recognized_authorization_header_type: '
                                       f'{self.authorization_header_type}')
            logger.info(f'headers_in_aenter: {headers}')
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=headers,
            )
        elif self.backend == 'openai':
            logger.info('self.backend_initialization to openai')
            kwargs = {
                'api_key': self.api_key,
                'base_url': self._get_base_url_w_version(),
            }

            # if self.api_version:
            #     kwargs['default_query'] = {'api-version': self.api_version}

            self._openai_client = AsyncOpenAI(**kwargs)
        else:
            raise RuntimeError(f'self.backend_not_recognized: {self.backend}')

        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate(self, prompt: str, **gen_params) -> str:
        '''
        Generate text for a single prompt.

        All generation parameters MUST be provided per call.
        Unsupported parameters will raise.
        # '''
        # if self._session is None or self._session.closed:
        #     raise RuntimeError("LLM session is closed at generate() "
        #                        "llm_session_is_closed_at_generate")
        if self.backend == 'http':
            if self._session is None or self._session.closed:
                raise RuntimeError(
                    "HTTP LLM session is closed at generate()"
                )
        elif self.backend == 'openai':
            if self._openai_client is None:
                raise RuntimeError(
                    "OpenAI client not initialized at generate()"
                )
        if self.backend == 'http' and self._session is None:
            raise RuntimeError(
                'UnifiedLLMClient must be used as an async context manager:\n'
                'async with UnifiedLLMClient(...) as client:'
            )

        assert 'temperature' in gen_params
        assert 'max_new_tokens' in gen_params

        # max_tokens_name = 'max_new_tokens' if 'max_new_tokens' in gen_params else 'max_completion_tokens'

        async with self.sem:
            ### NEW: trim prompt based on max_new_tokens + model context
            prompt = self._trim_prompt_if_needed(
                prompt,
                gen_params['max_new_tokens'],
            )

            normalized = self._normalize_gen_params(gen_params)

            if self.backend == 'openai':
                result = await self._generate_openai(
                    prompt=prompt,
                    params=normalized
                )
            else:
                result = await self._generate_http(
                    prompt=prompt,
                    params=normalized
                )

            if result is None:  # <<< CHANGED: explicit None check
                logger.error(
                    'llm_returned_none: backend=%s model=%s '
                    'prompt_len=%d '
                    'prompt=*********************************'
                    '\n %r \n*******************************'
                    'params=%r',
                    self.backend,
                    self.model,
                    len(prompt),
                    prompt,
                    normalized,
                )
                return ''  # <<< CHANGED: normalize None → empty string

            return result
    # ------------------------------------------------------------------
    # OpenAI-compatible path (vLLM, Azure, TGI, proxy)
    # ------------------------------------------------------------------
    def _get_base_url_w_version(self) -> str:
        if '/v1' in self.base_url:
            return self.base_url

        if self.api_version:
            return f'{self.base_url}/{self.api_version}'

        return f'{self.base_url}/v1'

    # ------------------------------------------------------------------
    # Parameter normalization & validation
    # ------------------------------------------------------------------
    def _normalize_gen_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        allowed_common = {'temperature', 'stop', 'max_length', 'truncate_prompt_tokens'}
        #       "max_length": 14048,
        #       "truncate_prompt_tokens": 12000,
        allowed_openai = allowed_common | {'max_new_tokens', 'repetition_penalty'}
        allowed_http = allowed_common | {
            'max_new_tokens',
            'repetition_penalty'
        }

        if self.backend == 'openai':
            allowed = allowed_openai
        else:
            allowed = allowed_http

        for key in params:
            if key not in allowed:
                raise ValueError(
                    f'Unsupported generation parameter "{key}" for backend "{self.backend}"'
                )

        if 'max_new_tokens' not in params:
            raise ValueError('max_new_tokens must be provided per generation call')

        normalized = dict(params)

        max_new_tokens = normalized.pop('max_new_tokens')

        normalized.update(self._token_param(max_new_tokens))

        return normalized

    def _token_param(self, max_new_tokens: int) -> dict:
        # Responses API uses max_output_tokens
        if self.mode == 'responses':
            return {'max_output_tokens': max_new_tokens}

        # Chat/completions
        if self.backend == 'openai':
            if self.model.lower().startswith('gpt-5') or 'gpt-5' in self.model.lower():
                return {'max_completion_tokens': max_new_tokens}
            return {'max_tokens': max_new_tokens}
        else:
            if self.model.lower().startswith('gpt-5') or 'gpt-5' in self.model.lower():
                return {'max_completion_tokens': max_new_tokens}
            return {'max_tokens': max_new_tokens}
    def _trim_prompt_if_needed(self, prompt: str, max_new_tokens: int) -> str:
        if self.model_max_context is None or self.model_max_context <= 0:
            return prompt

        max_input_tokens = self.model_max_context - max_new_tokens
        if max_input_tokens <= 0:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} exceeds model context={self.model_max_context}"
            )

        tokens = self._tokenizer(
            prompt,
            truncation=True,
            max_length=max_input_tokens,
        )
        return self._tokenizer.decode(
            tokens["input_ids"],
            skip_special_tokens=True,
        )

    async def _generate_openai(
            self,
            prompt: str,
            params: Dict[str, Any],
    ) -> str:
        try:
            if self.mode == 'chat':
                response = await self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    **params,
                )
                return response.choices[0].message.content

            elif self.mode == 'completion':
                response = await self._openai_client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    **params,
                )
                return response.choices[0].text
            elif self.mode == 'responses':
                response = await self._openai_client.responses.create(
                    model=self.model,
                    input=prompt,
                    **params,
                )
                # SDK response shape: easiest is output_text helper if available,
                # otherwise extract from output blocks.
                if hasattr(response, "output_text"):
                    return response.output_text
                # fallback:
                texts = []
                for item in getattr(response, "output", []) or []:
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", None) == "output_text":
                            texts.append(getattr(c, "text", ""))
                return "".join(texts)
            else:
                raise RuntimeError(f'mode_not_recognized: {self.mode}')

        except BadRequestError as e:
            msg = str(e)

            # Azure OpenAI content policy hit → log and IGNORE
            if (
                    "content management policy" in msg
                    or "response was filtered" in msg
            ):
                logger.error(
                    "OpenAI content policy triggered — skipping generation. "
                    "Model=%s PromptLen=%d",
                    self.model,
                    len(prompt),
                )
                return None  # <<< IMPORTANT: safe empty result

            # otherwise: real bug → re-raise
            raise

    # ------------------------------------------------------------------
    # Raw HTTP fallback (aiohttp)
    # ------------------------------------------------------------------
    async def _generate_http(
            self,
            prompt: str,
            params: Dict[str, Any],
            retries: int = 10
            # retry_delay: float = 0.5,
    ) -> str:
        base_url = self._get_base_url_w_version()
        if self.mode == 'chat':
            url = f'{base_url}/chat/completions'
            payload = {
                'model': self.model,
                'messages': [{"role": "user", "content": prompt}],
                **params,
            }
        elif self.mode == 'completion':
            url = f'{base_url}/completions'
            payload = {
                'model': self.model,
                'prompt': prompt,
                **params,
            }
        elif self.mode == 'responses':
            url = f'{base_url}/responses'
            payload = {
                'model': self.model,
                'input': prompt,
                **params,
            }
        else:
            raise RuntimeError(f'mode_not_recognized: {self.mode}')

        for attempt in range(1, retries + 1):
            try:
                logger.debug(f'about_to_session_post_to: {url} ****** payload: {payload}')

                async with self._session.post(url, json=payload) as r:
                    r.raise_for_status()
                    j = await r.json()
                    try:
                        if self.mode == 'chat':
                            return j['choices'][0]['message']['content']
                        elif self.mode == 'completion':
                            return j['choices'][0]['text']
                        else:  # responses
                            if 'output_text' in j and isinstance(j['output_text'], str):
                                return j['output_text']
                            texts = []
                            for item in (j.get('output') or []):
                                for c in (item.get('content') or []):
                                    if c.get('type') == 'output_text':
                                        texts.append(c.get('text', ''))
                            return ''.join(texts)

                    except Exception:
                        # Log and do NOT fail the job
                        logger.exception(
                            "LLM response parsing failed (mode=%s). Returning empty string. "
                            "Top-level keys=%s; response=%s",
                            self.mode,
                            list(j.keys()) if isinstance(j, dict) else type(j),
                            json.dumps(j, ensure_ascii=False)[:5000],  # cap to avoid huge logs
                        )
                        # Optional: also log minimal request context (don’t dump secrets)
                        logger.error(
                            "LLM request context: url=%s payload_keys=%s",
                            url,
                            list(payload.keys()) if isinstance(payload, dict) else type(payload),
                        )
                        return ""
            except aiohttp.ClientResponseError as e:  # <<< CHANGED: catch HTTP status errors (e.g., 429)
                # Retry only on rate limit / transient server errors
                logger.exception('llm_generation_failed: %s', e)

                if e.status == 400:
                    logger.error(
                        'http_generate_bad_request_400: '
                        'model=%s mode=%s prompt_len=%d params=%s '
                        f'prompt=***********\n {prompt} \n************',
                        self.model,
                        self.mode,
                        len(prompt),
                        params,
                    )
                    return ''  # <<< fail soft, no retry

                if e.status not in (429, 500, 502, 503, 504, 499):
                    raise

                logger.warning(
                    'http_generate_failure: HTTP %s (attempt %d/%d): %s',
                    e.status,
                    attempt,
                    retries,
                    repr(e),
                )

                if attempt == retries:
                    # sometimes there are some entries the model for some reason just fails (500)
                    return ''

                # await asyncio.sleep(retry_delay)
                await asyncio.sleep(random.uniform(1.0, 2.0))  # <<< CHANGED

            except (
                    aiohttp.ClientOSError,
                    aiohttp.ServerDisconnectedError,
                    asyncio.TimeoutError,
            ) as e:
                logger.exception('llm_generation_failed_2: %s', e)

                logger.warning(
                    'http_generate_failure: HTTP generate failed (attempt %d/%d): %s',
                    attempt,
                    retries,
                    repr(e),
                )

                if attempt == retries:
                    raise

                await asyncio.sleep(random.uniform(1.0, 2.0))  # <<< CHANGED

        raise RuntimeError('nothing_returned_http')
