"""LLM provider abstraction with multi-model Groq quota pooling."""
import os
import warnings
import logging
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass
from enum import Enum

from .rate_limits import RateLimitStore, classify_rate_limit

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore[assignment, misc]

try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore[assignment]

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


class ProviderType(Enum):
    GROQ = "groq"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    tokens_used: int = 0


class GroqRateLimitError(Exception):
    """Raised when a Groq model returns a 429 rate-limit response."""

    def __init__(self, model: str, kind: str, message: str = ""):
        super().__init__(message)
        self.model = model
        self.kind = kind  # "rpm" or "rpd"
        self.message = message


class LLMProvider:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def generate(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:  # noqa: ARG002
        raise NotImplementedError


class GroqProvider(LLMProvider):
    """Single Groq model provider. Raises GroqRateLimitError on 429."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        key = (api_key or os.getenv("GROQ_API_KEY") or "").strip()
        super().__init__(key)
        if Groq is None:
            raise ImportError("groq package required. Install with: pip install groq")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found or empty")
        self.client = Groq(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                temperature=0.7,
                top_p=1,
                stream=True,
            )
            full_response = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
            return LLMResponse(
                text=full_response,
                provider="groq",
                model=self.model,
                tokens_used=0,
            )
        except Exception as e:
            err = str(e)
            if (
                "429" in err
                or "rate_limit" in err.lower()
                or "rate limit" in err.lower()
            ):
                kind = classify_rate_limit(err)
                raise GroqRateLimitError(self.model, kind, err)
            raise RuntimeError(f"Groq API error ({self.model}): {e}")


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None):
        key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        super().__init__(key)
        if genai is None:
            raise ImportError(
                "google-generativeai package required. "
                "Install with: pip install google-generativeai"
            )
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found or empty")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7,
                ),
            )
            return LLMResponse(
                text=response.text,
                provider="gemini",
                model="gemini-1.5-flash",
                tokens_used=0,
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")


class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None):
        key = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
        super().__init__(key)
        if httpx is None:
            raise ImportError("httpx package required. Install with: pip install httpx")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found or empty")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "meta-llama/llama-3.1-8b-instruct:free"

    def generate(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                return LLMResponse(
                    text=data["choices"][0]["message"]["content"],
                    provider="openrouter",
                    model=self.model,
                    tokens_used=data.get("usage", {}).get("total_tokens", 0),
                )
        except Exception as e:
            raise RuntimeError(f"OpenRouter API error: {e}")


class OllamaProvider(LLMProvider):
    def __init__(self):
        super().__init__(api_key=None)
        if httpx is None:
            raise ImportError("httpx package required. Install with: pip install httpx")
        self.host = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.model = (os.getenv("OLLAMA_MODEL") or "llama3.2").strip()
        try:
            with httpx.Client() as client:
                client.get(f"{self.host}/api/tags", timeout=3.0).raise_for_status()
        except Exception as e:
            raise ValueError(f"Ollama server not reachable at {self.host}: {e}")

    def generate(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {"num_predict": max_tokens, "temperature": 0.7},
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                return LLMResponse(
                    text=data["message"]["content"],
                    provider="ollama",
                    model=self.model,
                    tokens_used=data.get("eval_count", 0),
                )
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")


class LLMRouter:
    """Routes Groq requests across models by tier, with rate-limit fallback."""

    # Heavy tier: comparison + literature review queries get gpt-oss first
    HEAVY_MODELS: List[str] = [
        "openai/gpt-oss-120b",
        "compound-beta",
        "compound-beta-mini",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ]
    # Light tier: everything else skips gpt-oss to preserve its quota
    LIGHT_MODELS: List[str] = [
        "compound-beta",
        "compound-beta-mini",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ]

    def __init__(
        self,
        rate_store: Optional[RateLimitStore] = None,
        cache_enabled: bool = True,
        providers: Optional[List[ProviderType]] = None,  # legacy, unused
    ):
        self.rate_store = rate_store or RateLimitStore()
        self.cache_enabled = cache_enabled
        self._groq_key: Optional[str] = None
        self._non_groq_providers: Dict[ProviderType, Optional[LLMProvider]] = {}

    def _groq_api_key(self) -> Optional[str]:
        if self._groq_key is None:
            self._groq_key = (os.getenv("GROQ_API_KEY") or "").strip() or None
        return self._groq_key

    def _get_non_groq_provider(self, pt: ProviderType) -> Optional[LLMProvider]:
        if pt in self._non_groq_providers:
            return self._non_groq_providers[pt]
        try:
            p: Optional[LLMProvider] = None
            if pt == ProviderType.GEMINI:
                p = GeminiProvider()
            elif pt == ProviderType.OPENROUTER:
                p = OpenRouterProvider()
            elif pt == ProviderType.OLLAMA:
                p = OllamaProvider()
            self._non_groq_providers[pt] = p
            return p
        except (ImportError, ValueError):
            self._non_groq_providers[pt] = None
            return None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        tier: str = "light",
        cache_lookup: Optional[Callable] = None,
        cache_store: Optional[Callable] = None,
    ) -> Optional[LLMResponse]:
        # Check cache first
        if self.cache_enabled and cache_lookup:
            cached = cache_lookup(prompt)
            if cached:
                return LLMResponse(
                    text=cached, provider="cache", model="cached", tokens_used=0
                )

        models = self.HEAVY_MODELS if tier == "heavy" else self.LIGHT_MODELS
        groq_key = self._groq_api_key()

        # Try Groq models in tier order
        all_groq_exhausted = True
        if groq_key:
            for model in models:
                if not self.rate_store.is_available(model):
                    logger.debug(f"Skipping {model} (rate limited)")
                    continue
                all_groq_exhausted = False
                try:
                    response = GroqProvider(model=model, api_key=groq_key).generate(
                        prompt, max_tokens=max_tokens
                    )
                    if self.cache_enabled and cache_store and response:
                        cache_store(prompt, response)
                    logger.info(f"Used Groq model: {model}")
                    return response
                except GroqRateLimitError as e:
                    logger.warning(f"{model} rate limited ({e.kind}), trying next")
                    if e.kind == "rpd":
                        self.rate_store.mark_rpd_limited(model)
                    else:
                        self.rate_store.mark_rpm_limited(model, e.message)
                    continue
                except Exception as e:
                    logger.warning(f"{model} failed: {str(e)[:100]}")
                    continue

        if groq_key and all_groq_exhausted:
            raise RuntimeError(
                "All Groq models are rate limited. Run /model to see cooldown times."
            )

        # Fall back to non-Groq providers
        for pt in [ProviderType.GEMINI, ProviderType.OPENROUTER, ProviderType.OLLAMA]:
            provider = self._get_non_groq_provider(pt)
            if provider is None:
                continue
            try:
                response = provider.generate(prompt, max_tokens=max_tokens)
                if self.cache_enabled and cache_store and response:
                    cache_store(prompt, response)
                return response
            except Exception as e:
                logger.warning(f"{pt.value} failed: {str(e)[:100]}")
                continue

        raise RuntimeError(
            "No LLM providers available. Please configure GROQ_API_KEY, GEMINI_API_KEY, "
            "or OPENROUTER_API_KEY — or start a local Ollama server (ollama serve)"
        )

    def is_available(self) -> bool:
        if self._groq_api_key():
            return True
        for pt in [ProviderType.GEMINI, ProviderType.OPENROUTER, ProviderType.OLLAMA]:
            if self._get_non_groq_provider(pt) is not None:
                return True
        return False
