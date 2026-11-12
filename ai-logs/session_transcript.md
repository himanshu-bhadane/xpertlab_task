# AI Session Transcript: Cold Start Challenge Development Log

**Candidate:** Himanshu Bhadane (`himanshu.bhadane2011@gmail.com`)  
**Repository:** `https://github.com/himanshu-bhadane/xpertlab_task`  
**Target Model:** `google/gemini-3.8-flash` (OpenRouter API)  
**Submission Commit:** (updated in latest commit)

---

## Dialogue and Raw Audit Files
In accordance with challenge submission requirements, the complete raw session logs are included:
1. [`ai-logs/chat_export.md`](file:///Users/himanshu/Documents/xpertlab/ai-logs/chat_export.md): Complete chronological dialogue transcript between the developer and Antigravity.
2. [`ai-logs/transcript.jsonl`](file:///Users/himanshu/Documents/xpertlab/ai-logs/transcript.jsonl): Full raw JSONL interaction log.
3. [`ai-logs/session_transcript.md`](file:///Users/himanshu/Documents/xpertlab/ai-logs/session_transcript.md): Executive engineering log summarizing decisions, benchmarks, and architectural pivots.

---

## Chronological Trajectory of Reasoning and Implementation

### Turn 1: Project Discovery & Instruction Analysis
- Ingested and parsed `Cold Start Challenge.html`, `README.md`, `check.py`, and `schema.json`.
- Identified data schema across 5 outputs: `people`, `organisations`, `chats`, `goals`, `open_loops` (with `help`).
- Mapped scoring rubrics and critical penalties:
  - Calling an open loop `done` or `dropped` falsely: $-1$ penalty.
  - Calling `assist` or `remind` on a `leave` loop: $-1$ penalty.
  - Invented goals: $-1$ penalty.

### Turn 2: Strategy Alignment & Dual-Mode Execution
- Aligned on building a dual-mode pipeline:
  1. **Primary LLM Mode:** Activated when `LLM_API_KEY` is injected by Xpert evaluators during formal grading.
  2. **Resilient Offline Fallback Mode:** Operates safely in local zero-key test environments, maintaining 100% schema compliance.

### Turn 3: Gemini 3.8 Flash Migration & Latency Puzzle
- Targeted `google/gemini-3.8-flash` via OpenAI-compatible endpoint.
- **Latency Puzzle Explained:** The official baseline took 15.5 minutes because it made 110 sequential web requests across 490,000 tokens over HTTP. An initial local script took 0.3s because it executed in-memory on disk with no web requests.
- **Zero-Overfitting Commitment:** Purged all hardcoded dev names (`"khetan"`, `"nayna"`, `"sudar"`) to ensure zero drop-off on the randomized hidden set.

### Turn 4: Comprehensive Audit & 5-Phase Hardening
- Reordered pipeline so people classification runs first, followed by organisation resolution.
- Added lookahead resolution, IST timezone normalisation, and defensive `sanitize()` guaranteeing 0 schema errors.
- Added local prompt caching (`.cache/llm/`) and thread-locked `usage.jsonl`.
- Initialized Git repository and pushed to `https://github.com/himanshu-bhadane/xpertlab_task`.

### Turn 5 & 6: Code Refinement, Score Context & Raw Transcript Export
- **Loop Prompt Richness:** Built rich sender labels (`Raju driver (staff, P03)`) and included chat titles so the model has the context required to apply FAQ help rules.
- **Link Simplification:** Aligned organisation links strictly with answer key patterns: staff receive `supports` to Home; colleagues/partners receive `runs` to primary work org; others receive zero links.
- **Chunk Lookahead & Status Guard:** Eliminated incidental keyword-based closures. Enforced that `done` and `dropped` are only accepted if supported by a valid subsequent `closed_by` message ID with confidence $\ge 0.65$.
- **Group Batching:** Batched group chats into groups of 25 to prevent token truncation.
- **Goals Sampling:** Expanded candidate sampling to ~400 messages across the timeline.
- **Verification:**
  - `run.sh` verified as executable `100755` in git.
  - Confirmed `data/` was never committed into history.
  - Exported complete dialogue to `ai-logs/chat_export.md` and raw log to `ai-logs/transcript.jsonl`.
  - Example score: **45.2 / 74**; Dev offline score: **10.0 / 56**.
