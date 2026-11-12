"""A client for any OpenAI-compatible chat API, standard library only.

Reads LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL from environment.
Defaults:
  LLM_BASE_URL: https://openrouter.ai/api/v1
  LLM_MODEL:    google/gemini-3.8-flash

Every call appends its token counts to usage.jsonl, so you can see what a run costs.
Implements exponential backoff retry on HTTP 429 (rate limits) or 5xx server errors.
"""
import json
import os
import time
import urllib.error
import urllib.request


def chat(
    messages,
    model=None,
    temperature=0.0,
    max_tokens=None,
    response_format=None,
    timeout=120,
    max_retries=3
):
    base = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    api_key = os.environ.get("LLM_API_KEY", "")
    target_model = model or os.environ.get("LLM_MODEL", "google/gemini-3.8-flash")

    if not api_key:
        raise ValueError("LLM_API_KEY is not set in environment.")

    body = {
        "model": target_model,
        "messages": messages,
        "temperature": temperature
    }
    if max_tokens:
        body["max_tokens"] = max_tokens
    if response_format:
        body["response_format"] = response_format

    req_data = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/xpert-cold-start",
        "X-Title": "Xpert Cold Start Challenge"
    }

    last_error = None
    for attempt in range(max_retries + 1):
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

            usage = data.get("usage") or {}
            with open("usage.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "model": target_model,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "seconds": round(time.time() - started, 2),
                }) + "\n")

            return data["choices"][0]["message"]["content"]

        except urllib.error.HTTPError as e:
            last_error = e
            status = e.code
            # Retry on rate limit (429) or transient 5xx errors
            if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait_sec = (2 ** attempt) * 2
                time.sleep(wait_sec)
                continue
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
            raise RuntimeError(f"HTTP {status} from LLM provider: {error_body}") from e
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep((2 ** attempt) * 2)
                continue
            raise

    raise last_error
