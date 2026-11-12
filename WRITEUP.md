# Write-up: Cold Start Challenge

*Xpert Winter Internship 2026*

## What I focused on, and why

A personal assistant that lives in WhatsApp succeeds or fails on trust and boundaries. If an assistant pokes the user about private spouse communications, hallucinates completed actions, or costs a fortune to run, the user uninstalls it immediately.

I focused on three core engineering priorities:
1. **Zero-Overfitting & Generalization Across Unseen Lives:** In the hidden evaluation set, every name, phone number, company name, and location is re-randomized. Systems that rely on hardcoded personal names or entity keywords fail completely. I designed a generic, model-driven architecture that extracts organizations, circles, loops, and goals using semantic comprehension and structural signals.
2. **Actionable Loop Discrimination & Safe Status Resolution:** WhatsApp conversations are noisy. Distinguishing actionable commitments (*"I'll book the tickets tomorrow"*) from casual conversational chatter (*"let's talk later"*) is critical. Because marking an active loop as `done` carries a $-1$ penalty (hiding real work from the user), the system defaults unresolved loops to `open` unless explicit subsequent confirmation of completion is present.
3. **Execution Latency & Cost Optimization:** Scoring limits runs to 30 minutes and evaluates cost against a $0.038 / 1,000 messages baseline. Rather than firing hundreds of slow, unindexed sequential LLM requests across thousands of noisy messages, our pipeline batches filtered conversational contexts into parallel worker pools using `google/gemini-3.8-flash`. It finishes in **~1 to 2 minutes** total while keeping costs well under **$0.005 / 1,000 messages** (qualifying for the maximum 12 cost points).

---

## My approach

The system runs via `run.sh` in standard Python 3.9+ without external package dependencies, orchestrated across 4 structured stages:

1. **Stage 1: Organisation & Chat Topology Resolution:**
   - Dynamically ingests `seed.json` organizations.
   - Instantiates the universal `Home` residence (`type: residence, side: home, role: owner`).
   - Group chats are analyzed to identify company entities, secondary properties (workspaces, vacation homes), and community groups.
   - Every chat is mapped to exactly one organization or person. Direct 1:1 messages belong to the individual contact by default.
2. **Stage 2: People Circles & Links Classification:**
   - Ingests `seed.json` baseline circles.
   - Distinguishes `spouse` via high 1:1 message volume, daily frequency, and domestic coordination.
   - Evaluates remaining people across the 12 circles (`spouse`, `child`, `parent`, `in-law`, `sibling`, `relative`, `friend`, `colleague`, `client`, `vendor`, `partner`, `staff`).
   - Assigns organizational links: colleagues and partners receive `runs` on their associated companies; domestic and office staff receive `supports`.
3. **Stage 3: Open Loop Mining & Help Decision Engine:**
   - Scans message threads for actionable triggers in English and Hinglish (commitments, deadlines, unanswered questions, followups).
   - Extracts loop attributes: `title`, `type` (`commitment`, `question`, `deadline`, `followup`), `owner` (`ME` vs contact ID), and parses relative dates into normalized `YYYY-MM-DD` deadlines.
   - **Where to Help:**
     - **`leave`**: When owned by someone else who is handling it (staff, colleague, spouse, vendor), or personal family plans.
     - **`assist`**: When owned by the user and actionable via reversible digital preparation (finding links, drafting orders, comparing options).
     - **`remind`**: When it requires the user's personal decision, attendance, or direct commitment.
4. **Stage 4: Goals Discovery:**
   - Detects explicit multi-month intentions across health, work, finances, family, and experiences, requiring at least one verifiable evidence message ID.

---

## What it costs

- **Model:** `google/gemini-3.8-flash` (via OpenRouter OpenAI-compatible API).
- **Cost Metrics on Dev Set (4,502 messages):**
  - **Inference Volume:** ~80,000 prompt tokens + ~6,000 completion tokens across batched thread contexts.
  - **OpenRouter Pricing:** At $0.075 / 1M prompt tokens and $0.30 / 1M completion tokens, the full run costs **~$0.0078 total**, or **~$0.0017 per 1,000 messages**.
  - **Comparison with Baseline:** The baseline (`openai/gpt-oss-120b`) costs **$0.038 per 1,000 messages**. Our solution is more than **20× cheaper** than baseline, well within the $\le 0.25\times$ threshold required to receive the full **12 cost points**.
- **Where the money goes:** 100% of token expenditure is directed to reasoning over dense conversational threads for loop extraction, circle disambiguation, and goal verification. Zero tokens are wasted on raw uncaptioned media or broadcast announcements.

---

## Where it's wrong

Expected failure modes and estimated error rates on the hidden set:

1. **Informal Offline Task Closures (~25–35% error rate):**
   - WhatsApp tasks are frequently resolved offline (paying via cash, verbal confirmation on a phone call, or sharing an uncaptioned receipt image). Because the system strictly avoids the $-1$ false `done` penalty, unresolved tasks remain `open`, reducing recall on completed loops.
2. **Friend vs. Colleague Boundary in Small Ventures (~15–20% error rate):**
   - In entrepreneurial settings, close friends frequently discuss business introductions, and colleagues converse in informal social or sports groups. Without historical organizational hierarchy, some friends who discuss work tasks may be classified as colleagues or vice versa.
3. **Implicit Deadlines Without Explicit Date Tokens (~15% error rate):**
   - Relative deadlines lacking explicit temporal keywords (e.g. *"before the weekend"* or conversational turns like *"see you at dinner"*) will have their due date recorded as `null`.
4. **Conservative Goal Recall (~30% error rate):**
   - To avoid penalties from hallucinating ungrounded goals, the system only outputs goals backed by concrete evidence message IDs. Vaguely expressed aspirations without direct evidence are omitted.
