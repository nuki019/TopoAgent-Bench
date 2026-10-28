"""
Hybrid LLM Router with Concurrency Control and Model Tier Resolution.
"""

import os
import json
import time
import asyncio
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, NamedTuple, Callable


def load_dotenv(env_path: Optional[str] = None) -> Dict[str, str]:
    """
    Manually parses key=value pairs from .env without requiring python-dotenv.
    Populates os.environ for loaded keys that are not already set.
    """
    candidate_paths = []
    if env_path:
        candidate_paths.append(Path(env_path))
    
    # Priority candidate paths
    candidate_paths.extend([
        Path(r"C:\Users\wfy\.gemini\antigravity\scratch\TopoAgent-Bench\.env"),
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env",
    ])

    loaded = {}
    for path in candidate_paths:
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for raw_line in f:
                        line = raw_line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        loaded[key] = val
                        # Always set into os.environ if not already present or override
                        os.environ[key] = val
                break
            except Exception:
                pass
    return loaded


# Automatically load env on import
load_dotenv()


class LLMResponse:
    """
    Execution output and telemetry metrics for an LLM call.
    Unpacks cleanly as a 4-tuple: (content, prompt_tokens, completion_tokens, latency_ms)
    while supporting direct attribute access (res.content, res.latency_ms, res.model).
    """

    def __init__(
        self,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        model: str = "",
        tier: str = "",
    ):
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.latency_ms = latency_ms
        self.model = model
        self.tier = tier

    def __iter__(self):
        yield self.content
        yield self.prompt_tokens
        yield self.completion_tokens
        yield self.latency_ms

    def __getitem__(self, idx):
        return (self.content, self.prompt_tokens, self.completion_tokens, self.latency_ms)[idx]

    def __len__(self):
        return 4

    def __repr__(self) -> str:
        return (
            f"LLMResponse(content={self.content[:30]!r}..., "
            f"prompt_tokens={self.prompt_tokens}, completion_tokens={self.completion_tokens}, "
            f"latency_ms={self.latency_ms:.2f}, model={self.model!r}, tier={self.tier!r})"
        )


class HybridLLMRouter:
    """
    High-performance LLM Router enforcing strict concurrency,
    zero external HTTP dependencies, and model tier resolution.
    """

    # Global semaphore container shared across all instances
    _global_semaphore: Optional[asyncio.Semaphore] = None
    _max_concurrency: int = 5

    def __init__(
        self,
        max_concurrency: Optional[int] = None,
        mock_mode: bool = False,
        timeout: float = 60.0,
    ):
        load_dotenv()
        env_limit = os.getenv("MAX_CONCURRENCY")
        if max_concurrency is not None:
            self.max_concurrency = max_concurrency
        elif env_limit and env_limit.isdigit():
            self.max_concurrency = int(env_limit)
        else:
            self.max_concurrency = 5

        HybridLLMRouter._max_concurrency = self.max_concurrency
        self.mock_mode = mock_mode
        self.timeout = timeout
        self.mock_handler: Optional[Callable[[str, str, str], str]] = None

    @classmethod
    def get_semaphore(cls) -> asyncio.Semaphore:
        """Lazily initialize global semaphore in the running event loop."""
        if cls._global_semaphore is None:
            cls._global_semaphore = asyncio.Semaphore(cls._max_concurrency)
        return cls._global_semaphore

    def resolve_model(self, tier: str) -> Tuple[str, str, str, bool]:
        """
        Resolves model tier to (model_name, endpoint_url, api_key, is_local_vllm).
        Tiers:
          - 'flagship' -> os.getenv('FLAGSHIP_MODEL', 'kimi-k3')
          - 'fast'     -> os.getenv('FAST_MODEL', 'qwen3.8-27b')
          - 'alt_fast' -> os.getenv('ALT_FAST_MODEL', 'deepseek-v4.1-flash')
          - 'local_vllm' -> os.getenv('LOCAL_VLLM_URL', 'http://127.0.0.1:8000/v1')
        """
        bailian_base = os.getenv(
            "BAILIAN_BASE_URL",
            "https://llm-sn32yenb08wvkx41.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        ).rstrip("/")
        bailian_key = os.getenv("BAILIAN_API_KEY", "")

        tier_lower = tier.lower()
        if tier_lower == "flagship":
            return os.getenv("FLAGSHIP_MODEL", "kimi-k3"), bailian_base, bailian_key, False
        elif tier_lower == "fast":
            return os.getenv("FAST_MODEL", "qwen3.8-27b"), bailian_base, bailian_key, False
        elif tier_lower == "alt_fast":
            return os.getenv("ALT_FAST_MODEL", "deepseek-v4.1-flash"), bailian_base, bailian_key, False
        elif tier_lower == "local_vllm":
            vllm_base = os.getenv("LOCAL_VLLM_URL", "http://127.0.0.1:8000/v1").rstrip("/")
            vllm_model = os.getenv("LOCAL_VLLM_MODEL", "default")
            return vllm_model, vllm_base, "EMPTY", True
        else:
            # Direct model name fallback
            return tier, bailian_base, bailian_key, False

    def _sync_http_request(
        self,
        url: str,
        api_key: str,
        payload: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        """Synchronous HTTP POST using urllib.request."""
        body_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            resp_bytes = response.read()
            return json.loads(resp_bytes.decode("utf-8"))

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        model_tier: str = "fast",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        mock: Optional[bool] = None,
    ) -> LLMResponse:
        """
        Asynchronously generates response with global semaphore rate-limiting,
        urllib async-wrap, and disabled thinking mode for low latency.
        """
        use_mock = self.mock_mode if mock is None else mock

        if use_mock:
            t0 = time.perf_counter()
            # Simulate quick execution
            await asyncio.sleep(0.01)
            if self.mock_handler:
                content = self.mock_handler(prompt, system_prompt, model_tier)
            else:
                content = f"[Mock {model_tier}] Response for: {prompt[:40]}"
            latency_ms = (time.perf_counter() - t0) * 1000.0
            p_tokens = len(prompt.split()) + len(system_prompt.split()) + 10
            c_tokens = len(content.split()) + 5
            return LLMResponse(
                content=content,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                latency_ms=latency_ms,
                model=f"mock-{model_tier}",
                tier=model_tier,
            )

        sem = self.get_semaphore()
        async with sem:
            model_name, base_url, api_key, is_local_vllm = self.resolve_model(model_tier)
            url = f"{base_url}/chat/completions"

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "enable_thinking": False,  # Strict constraint: ~500-900ms low latency
            }

            t0 = time.perf_counter()
            try:
                raw_resp = await asyncio.to_thread(
                    self._sync_http_request,
                    url=url,
                    api_key=api_key,
                    payload=payload,
                    timeout=self.timeout,
                )
            except Exception as e:
                # If local vLLM fails, fallback immediately to fast tier via Bailian
                if is_local_vllm:
                    fallback_model, fallback_base, fallback_key, _ = self.resolve_model("fast")
                    fallback_url = f"{fallback_base}/chat/completions"
                    payload["model"] = fallback_model
                    raw_resp = await asyncio.to_thread(
                        self._sync_http_request,
                        url=fallback_url,
                        api_key=fallback_key,
                        payload=payload,
                        timeout=self.timeout,
                    )
                    model_name = fallback_model
                else:
                    raise e

            latency_ms = (time.perf_counter() - t0) * 1000.0
            content = raw_resp["choices"][0]["message"]["content"]
            usage = raw_resp.get("usage", {})
            p_tokens = usage.get("prompt_tokens", 0)
            c_tokens = usage.get("completion_tokens", 0)

            return LLMResponse(
                content=content,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                latency_ms=latency_ms,
                model=model_name,
                tier=model_tier,
            )
