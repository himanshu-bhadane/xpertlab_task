"""A client for any OpenAI-compatible chat API, standard library only.

Reads LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL from environment.
Defaults:
  LLM_BASE_URL: https://openrouter.ai/api/v1
  LLM_MODEL:    google/gemini-3.8-flash

Features:
- Thread-safe usage logging to usage.jsonl
- Disk cache (SHA-256 of prompt + model + temp) to avoid redundant spend and rate limits
- Automatic fallback/retry without response_format if HTTP 400 is returned (FAQ recommendation)
- Exponential backoff retry on HTTP 429 (rate limits) and 5xx errors
"""
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

_USAGE_LOCK = threading.Lock()
CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", ".cache/llm"))


def _cache_key(model, messages, temperature):
    raw = f"{model}:{temperature}:{json.dumps(messages, sort_keys=True, ensure_ascii=True)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def chat(
    messages,
    model=None,
    temperature=0.0,
    max_tokens=8192,
    response_format=None,
    timeout=120,
    max_retries=4,
    use_cache=True
):
    base = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    api_key = os.environ.get("LLM_API_KEY", "")
    target_model = model or os.environ.get("LLM_MODEL", "google/gemini-3.8-flash")

    if not api_key:
        raise ValueError("LLM_API_KEY is not set in environment.")

    # 1. Check disk cache if enabled
    if use_cache:
        key = _cache_key(target_model, messages, temperature)
        cache_file = CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            try:
                cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
                return cached_data.get("content", "")
            except Exception:
                pass

    current_response_format = response_format
    last_error = None

    for attempt in range(max_retries + 1):
        body = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        if current_response_format:
            body["response_format"] = current_response_format

        req_data = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/xpert-cold-start",
            "X-Title": "Xpert Cold Start Challenge",
        }

        try:
            req = urllib.request.Request(
                f"{base}/chat/completions",
                data=req_data,
                headers=headers,
                method="POST"
            )
            started = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}

            # Thread-safe write to usage.jsonl
            with _USAGE_LOCK:
                with open("usage.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "model": target_model,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "seconds": round(time.time() - started, 2),
                    }) + "\n")

            # Save to disk cache
            if use_cache:
                try:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cache_file = CACHE_DIR / f"{key}.json"
                    cache_file.write_text(json.dumps({
                        "model": target_model,
                        "content": content,
                        "cached_at": time.time()
                    }, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass

            return content

        except urllib.error.HTTPError as e:
            last_error = e
            status = e.code
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                pass

            # If HTTP 400 and response_format was used, retry without response_format immediately
            if status == 400 and current_response_format is not None:
                # Some models/providers on OpenRouter reject response_format: {"type": "json_object"}
                current_response_format = None
                continue

            # Exponential backoff on rate limits (429) or transient 5xx server errors
            if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait_sec = (2 ** attempt) * 2
                time.sleep(wait_sec)
                continue

            raise RuntimeError(f"HTTP {status} from LLM provider: {error_body}") from e

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep((2 ** attempt) * 2)
                continue
            raise

    raise last_error
