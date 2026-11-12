# AI Session Transcript: Cold Start Challenge Implementation

## System & Tooling Context
- **Assistant:** Antigravity (Google DeepMind)
- **Primary Workspace:** `/Users/himanshu/Documents/xpertlab`
- **Evaluated Data:** Dev set (4,502 messages, 138 chats) & Example set (Meera, 29 messages, 6 chats)
- **Target Model Architecture:** `google/gemini-3.8-flash` (via OpenRouter OpenAI-compatible API)

---

## Chronological Trajectory of Reasoning and Implementation

### Step 1: Initial Discovery & Environment Setup
- Examined project instructions in `Cold Start Challenge.html`, starter kit files (`README.md`, `schema.json`, `solution.py`, `llm.py`, `run.sh`, `check.py`), and dataset archives.
- Extracted and verified dev set data under `data/dev/` and sample data under `example/`.
- Identified evaluation rules, scoring criteria, and critical penalties:
  - False `done` on open loop: $-1$ penalty.
  - False `assist` / `remind` on `leave`: $-1$ penalty.
  - Cost curve: $\le 0.25\times$ baseline ($0.038 / 1k msgs) gives full 12 points.

### Step 2: Investigation of Execution Latency & Overfitting
- **Investigating the 0.3s vs. 15.5m Latency Gap:**
  - The organizer baseline made 110 sequential network calls to `openai/gpt-oss-120b` via OpenRouter (processing ~490,000 tokens), taking 15.5 minutes (~8.5 seconds per request over HTTP).
  - An early prototype executed locally in 0.3 seconds because it was an in-memory script with zero network calls.
  - **Overfitting Risk:** That early script relied on dev-set names ("khetan", "nayna", "sudar"). In the hidden evaluation set, every single person name, company, and phone number is re-randomized.
  - **Resolution:** Reconstructed the entire system into a genuine model-driven pipeline targeting `google/gemini-3.8-flash`, removing every hardcoded name and ensuring true zero-shot semantic reasoning.

### Step 3: Comprehensive Hardening & Architectural Refinement (Claude Review Feedback)
- Audited the implementation against detailed peer critique and challenge FAQ:
  1. **LLM Robustness (`llm.py`):**
     - Handled HTTP 400 rejection of `response_format` by retrying immediately without `response_format` and using `safe_json_parse`.
     - Built a local disk cache (`.cache/llm/`) keyed by SHA-256 of `(model, messages, temperature)` to avoid duplicate spend during testing.
     - Enforced thread safety around `usage.jsonl` writes using `threading.Lock()`.
     - Raised concurrent workers to 8 for parallel chunk evaluation.
  2. **Defensive Sanitization (`solution.py`):**
     - Added `sanitize()` mirroring `check.validate` to prevent any invalid ID or malformed object from failing the run.
     - Guaranteed that every chat is tied once (defaulting 1-on-1 chats to their participant and group chats to Home if unassigned).
     - Ensured `help` objects only accompany loops with `status: "open"`.
     - Stripped all dataset-specific keywords ("kanchan", "tadoba", "targeting", "marathon", "retainer") from fallback logic.
  3. **Stage Reordering (People First):**
     - Shifted people classification to run first using the most common `sender_name`, mutual conversation history, and seed circle calibration.
     - Group chat and organisation classification then leverages the resolved member circles.
  4. **Two-Pass Loop Extraction & Strict Help Rules:**
     - Switched to chronological chat chunking without English keyword or message length prefilters.
     - Implemented the 15-day scorable cutoff (`last_ts - 15 days`), using older messages as context.
     - Conservative status threshold: requires confidence $\ge 0.65$ and explicit subsequent message evidence to mark a loop `done`.
     - Aligned Where to Help rules verbatim with challenge FAQ.
  5. **Timezone Normalization:**
     - Enforced `datetime.timezone(datetime.timedelta(hours=5, minutes=30))` (IST) across all date conversions and `remind_at` timestamps.

### Step 4: Verification & Benchmarking
- Confirmed valid schema generation using `check.py` on both `example/` and `data/dev/`.
- Verified that `./run.sh` operates as the single entrypoint without dependencies on clean Python 3.9+.
- Mirrored all updated code, write-up, and session transcripts to `/Users/himanshu/Downloads/cold-start`.
