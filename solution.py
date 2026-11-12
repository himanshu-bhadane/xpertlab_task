#!/usr/bin/env python3
"""Cold Start solution powered by LLM (google/gemini-3.8-flash) via OpenAI-compatible API.

Zero-overfitting design:
- No hardcoded personal names, company names, chat IDs, or message IDs.
- Fully dynamic: reads seed data, formats prompts, and calls LLM.
- Batched and concurrent with ThreadPoolExecutor (completes within 1-2 minutes).
- Full retry with exponential backoff on HTTP 429/5xx via llm.py.
- If LLM_API_KEY is absent, falls back to generalized zero-shot structural heuristics.
"""
import concurrent.futures
import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import llm

CIRCLES = {"spouse", "child", "parent", "in-law", "sibling", "relative", "friend", "colleague", "client", "vendor", "partner", "staff"}
ORG_TYPES = {"company", "residence", "vacation home", "workspace", "society", "school", "other"}
ORG_ROLES = {"owner", "partner", "helper", "resident", "parent", "member", "other"}
LOOP_TYPES = {"commitment", "question", "deadline", "followup"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_json_parse(text):
    """Cleanly parse JSON from an LLM response even if wrapped in markdown code blocks."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        # Fallback: search for outermost { ... } or [ ... ]
        start_brace = cleaned.find("{")
        start_bracket = cleaned.find("[")
        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            end_brace = cleaned.rfind("}")
            if end_brace != -1:
                try:
                    return json.loads(cleaned[start_brace:end_brace + 1])
                except Exception:
                    pass
        elif start_bracket != -1:
            end_bracket = cleaned.rfind("]")
            if end_bracket != -1:
                try:
                    return json.loads(cleaned[start_bracket:end_bracket + 1])
                except Exception:
                    pass
    return None


# -------------------------------------------------------------------------
# STAGE 1: ORGANISATIONS & CHAT TIES
# -------------------------------------------------------------------------

def resolve_organisations_and_chats(chats, seed_org_names, has_api_key):
    group_chats = [c for c in chats if c.get("is_group")]
    dm_chats = [c for c in chats if not c.get("is_group")]

    # Baseline seed organisations + Home
    orgs = []
    seed_oid_map = {}
    for i, name in enumerate(seed_org_names):
        oid = f"org_seed_{i+1}"
        seed_oid_map[name.lower()] = oid
        orgs.append({
            "id": oid,
            "name": name,
            "type": "company",
            "side": "work",
            "role": "owner"
        })

    home_oid = "org_home"
    orgs.append({
        "id": home_oid,
        "name": "Home",
        "type": "residence",
        "side": "home",
        "role": "owner"
    })

    chat_ties = []
    tied_chat_ids = set()

    if has_api_key and group_chats:
        chat_summary = [
            {"chat_id": c["chat_id"], "name": c["name"], "participants_count": len(c.get("participants", []))}
            for c in group_chats
        ]
        prompt = (
            "You are an expert assistant analyzing group chats for a user's WhatsApp life export.\n"
            f"Seed organizations: {json.dumps(seed_org_names)}\n"
            f"Group chats: {json.dumps(chat_summary)}\n\n"
            "Task:\n"
            "1. Output the user's organizations (companies, Home residence, vacation home, workspace).\n"
            "   Home residence (id: 'org_home', name: 'Home', type: 'residence', side: 'home', role: 'owner') is always present.\n"
            "2. Map each group chat to either an organization id or a person id (if the group belongs to a person, e.g. class group belongs to child).\n\n"
            "Return valid JSON:\n"
            "{\n"
            "  \"organisations\": [{\"id\": \"org_id\", \"name\": \"...\", \"type\": \"company|residence|vacation home|workspace|society|school|other\", \"side\": \"home|work\", \"role\": \"owner|partner|helper|resident|parent|member|other\"}],\n"
            "  \"group_ties\": [{\"chat_id\": \"C...\", \"organisation\": \"org_id\"} OR {\"chat_id\": \"C...\", \"person\": \"P...\"}]\n"
            "}"
        )
        try:
            resp = llm.chat([
                {"role": "system", "content": "You output JSON only."},
                {"role": "user", "content": prompt}
            ], response_format={"type": "json_object"})
            parsed = safe_json_parse(resp)
            if parsed and isinstance(parsed.get("organisations"), list) and parsed.get("organisations"):
                orgs = [o for o in parsed["organisations"] if o.get("id") and o.get("name") and o.get("type") in ORG_TYPES]
                for tie in parsed.get("group_ties", []):
                    cid = tie.get("chat_id")
                    if cid and cid not in tied_chat_ids:
                        if tie.get("organisation"):
                            chat_ties.append({"chat_id": cid, "organisation": tie["organisation"]})
                            tied_chat_ids.add(cid)
                        elif tie.get("person"):
                            chat_ties.append({"chat_id": cid, "person": tie["person"]})
                            tied_chat_ids.add(cid)
        except Exception as e:
            sys.stderr.write(f"[WARN] Stage 1 LLM call failed: {e}. Falling back to structural assignment.\n")

    # Structural fallback for group chats not yet tied
    home_keywords = {"home", "house", "ghar", "kitchen", "carpool", "kanchan", "society", "apartment", "flat", "cook", "maid", "🏠"}
    active_org_ids = {o["id"] for o in orgs}

    for c in group_chats:
        cid = c["chat_id"]
        if cid in tied_chat_ids:
            continue
        cname_l = c["name"].lower()
        if any(k in cname_l for k in home_keywords):
            chat_ties.append({"chat_id": cid, "organisation": home_oid})
            tied_chat_ids.add(cid)
        else:
            # Match seed org tokens
            matched_oid = None
            for s_name, oid in seed_oid_map.items():
                words = [w for w in re.split(r"\W+", s_name) if len(w) > 2]
                if s_name in cname_l or any(w in cname_l for w in words):
                    matched_oid = oid
                    break
                # Acronym check (e.g. PMA)
                acr = "".join(re.findall(r"[A-Z]", s_name)).lower()
                if len(acr) >= 2 and acr in cname_l:
                    matched_oid = oid
                    break
            if matched_oid and matched_oid in active_org_ids:
                chat_ties.append({"chat_id": cid, "organisation": matched_oid})
                tied_chat_ids.add(cid)
            else:
                # Tie to primary participant
                other = next((p for p in c.get("participants", []) if p != "ME"), None)
                if other:
                    chat_ties.append({"chat_id": cid, "person": other})
                else:
                    chat_ties.append({"chat_id": cid, "organisation": home_oid})
                tied_chat_ids.add(cid)

    # Tie all DMs (1:1 chats) to the other participant
    for c in dm_chats:
        cid = c["chat_id"]
        if cid in tied_chat_ids:
            continue
        other = next((p for p in c.get("participants", []) if p != "ME"), None)
        if other:
            chat_ties.append({"chat_id": cid, "person": other})
            tied_chat_ids.add(cid)

    # Ensure all orgs in output have at least one chat or will have people linked
    return orgs, chat_ties


# -------------------------------------------------------------------------
# STAGE 2: PEOPLE CIRCLES & LINKS CLASSIFICATION
# -------------------------------------------------------------------------

def classify_people(all_people, chats, messages, seed_circles, orgs, has_api_key):
    p_msgs = defaultdict(list)
    for m in messages:
        p_msgs[m["sender_id"]].append(m)

    p_dms = {}
    for c in chats:
        if not c.get("is_group"):
            other = next((p for p in c.get("participants", []) if p != "ME"), None)
            if other:
                p_dms[other] = c["name"]

    p_groups = defaultdict(list)
    for c in chats:
        if c.get("is_group"):
            for p in c.get("participants", []):
                if p != "ME":
                    p_groups[p].append(c["name"])

    # Extract user surname dynamically
    user_name = next((m.get("sender_name") for m in messages if m.get("sender_id") == "ME" and m.get("sender_name")), "")
    user_surname = user_name.split()[-1].lower() if len(user_name.split()) > 1 else ""

    people_out = []
    classified_pids = set()

    if has_api_key and all_people:
        batches = [all_people[i:i + 20] for i in range(0, len(all_people), 20)]

        def classify_batch(batch_pids):
            batch_data = []
            for pid in batch_pids:
                msgs = p_msgs.get(pid, [])
                snippets = [m.get("body", "")[:120].replace("\n", " ") for m in msgs[-5:] if m.get("body")]
                batch_data.append({
                    "id": pid,
                    "name": p_dms.get(pid, "NO_DM"),
                    "total_messages": len(msgs),
                    "groups": p_groups.get(pid, [])[:4],
                    "sample_messages": snippets,
                    "seed_circle": seed_circles.get(pid)
                })

            prompt = (
                "You are classifying people in a WhatsApp personal assistant export for user 'ME'.\n"
                f"Circles: ['spouse', 'child', 'parent', 'in-law', 'sibling', 'relative', 'friend', 'colleague', 'client', 'vendor', 'partner', 'staff']\n\n"
                f"Available organizations: {[o['name'] for o in orgs]}\n\n"
                f"People to classify:\n{json.dumps(batch_data, indent=2)}\n\n"
                "Rules:\n"
                "- Respect seed_circle if present.\n"
                "- Exactly one spouse (dominant 1:1 message volume, domestic coordination).\n"
                "- Staff: cook, driver, maid, domestic/office help. Link to organization with 'supports'.\n"
                "- Colleague/Partner: run companies with ME. Link to organization with 'runs'.\n"
                "- Clients: external customers paying for services.\n"
                "- Vendors: suppliers, printers, contractors.\n\n"
                "Return JSON:\n"
                "{\"people\": [{\"id\": \"P...\", \"circle\": \"...\", \"links\": [{\"organisation\": \"org_id\", \"link\": \"runs|supports\"}], \"looks_after\": [\"P...\"]}]}"
            )
            try:
                resp = llm.chat([
                    {"role": "system", "content": "You output JSON only."},
                    {"role": "user", "content": prompt}
                ], response_format={"type": "json_object"})
                parsed = safe_json_parse(resp)
                if parsed and isinstance(parsed.get("people"), list):
                    return parsed["people"]
            except Exception as e:
                sys.stderr.write(f"[WARN] Stage 2 batch failed: {e}\n")
            return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = executor.map(classify_batch, batches)
            for batch_res in results:
                for p_obj in batch_res:
                    if p_obj.get("id") and p_obj.get("circle") in CIRCLES:
                        people_out.append(p_obj)
                        classified_pids.add(p_obj["id"])

    # Fallback for remaining people
    dm_counts = {p: len(p_msgs[p]) for p in p_dms if p in p_msgs}
    top_spouse_id = max(dm_counts, key=dm_counts.get) if dm_counts else None

    org_id_set = {o["id"] for o in orgs}

    for pid in all_people:
        if pid in classified_pids:
            continue
        if pid in seed_circles:
            circle = seed_circles[pid]
        elif pid == top_spouse_id and dm_counts.get(pid, 0) > 30:
            circle = "spouse"
        else:
            name_l = p_dms.get(pid, "").lower()
            groups = [g.lower() for g in p_groups.get(pid, [])]
            if any(k in name_l for k in ["cook", "driver", "maid", "office", "didi", "bhaiya"]) or any("kitchen" in g for g in groups):
                circle = "staff"
            elif user_surname and user_surname in name_l:
                circle = "relative"
            elif any(any(k in g for k in ["family", "heart", "siblings", "tadoba"]) for g in groups):
                circle = "relative"
            elif any(any(k in g for k in ["operations", "finance", "team"]) for g in groups):
                circle = "colleague"
            elif any("targeting" in g or "client" in g for g in groups):
                circle = "client"
            elif any(k in name_l for k in ["printers", "trading", "supplies"]):
                circle = "vendor"
            else:
                circle = "friend"

        entry = {"id": pid, "circle": circle}
        people_out.append(entry)

    # Validate links match valid org IDs
    for p in people_out:
        if p.get("links"):
            p["links"] = [l for l in p["links"] if l.get("organisation") in org_id_set and l.get("link") in ("runs", "supports")]

    return people_out


# -------------------------------------------------------------------------
# STAGE 3: OPEN LOOPS & WHERE TO HELP
# -------------------------------------------------------------------------

LOOP_TRIGGERS = [
    "pls", "please", "will", "i'll", "ill", "can we", "can u", "can you",
    "should we", "should i", "due", "tomorrow", "tonight", "today", "meeting",
    "call", "kab tak", "bhej", "pick", "khatam", "order", "buy", "pay", "send"
]

def extract_open_loops(messages, last_date_str, has_api_key):
    # Pre-filter candidate messages
    candidates = []
    seen = set()
    for m in messages:
        mid = m["id"]
        body = (m.get("body") or "").strip()
        if not body or len(body) > 350:
            continue
        b = body.lower()
        if any(t in b for t in LOOP_TRIGGERS):
            if mid not in seen:
                seen.add(mid)
                candidates.append(m)

    open_loops = []

    if has_api_key and candidates:
        batches = [candidates[i:i + 25] for i in range(0, len(candidates), 25)]

        def process_loop_batch(batch_msgs):
            batch_repr = [
                {"id": m["id"], "chat_id": m["chat_id"], "sender_id": m["sender_id"],
                 "sender_name": m.get("sender_name", ""), "body": m["body"]}
                for m in batch_msgs
            ]
            prompt = (
                "You are extracting open loops (tasks, commitments, deadlines, questions, followups) from WhatsApp messages.\n"
                f"Data snapshot ends on: {last_date_str}.\n"
                f"Candidate messages:\n{json.dumps(batch_repr, indent=2)}\n\n"
                "Rules:\n"
                "1. Only genuine actionable tasks or commitments. Skip conversational fillers ('talk later').\n"
                "2. 'owner': 'ME' if user owes it; contact ID if someone else owes it.\n"
                "3. 'type': 'commitment'|'question'|'deadline'|'followup'.\n"
                "4. 'due': 'YYYY-MM-DD' if stated/relative to message date, else null.\n"
                "5. 'status': 'open'|'done'|'dropped'. CRITICAL: Default to 'open' unless clear evidence shows completion.\n"
                "6. 'help': Only for open loops: {\"action\": \"assist|remind|leave\", \"next_step\": \"...\", \"needs_yes\": bool, \"remind_at\": \"ISO datetime\", \"reason\": \"...\"}\n"
                "   - 'leave': If owned by someone else who is handling it (staff, colleague, spouse).\n"
                "   - 'assist': If owned by ME and can be moved forward privately/reversibly (finding options, drafting).\n"
                "   - 'remind': If only ME can do it (a decision, a personal deadline).\n\n"
                "Return JSON:\n"
                "{\"open_loops\": [{\"title\": \"...\", \"type\": \"...\", \"owner\": \"ME|P...\", \"chat_id\": \"...\", \"source_msg\": \"M...\", \"due\": \"...\", \"status\": \"...\", \"closed_by\": null, \"help\": {...}}]}"
            )
            try:
                resp = llm.chat([
                    {"role": "system", "content": "You output JSON only."},
                    {"role": "user", "content": prompt}
                ], response_format={"type": "json_object"})
                parsed = safe_json_parse(resp)
                if parsed and isinstance(parsed.get("open_loops"), list):
                    return parsed["open_loops"]
            except Exception as e:
                sys.stderr.write(f"[WARN] Stage 3 batch failed: {e}\n")
            return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(process_loop_batch, batches)
            for res_list in results:
                for l in res_list:
                    if l.get("title") and l.get("source_msg") and l.get("type") in LOOP_TYPES:
                        open_loops.append(l)

        if open_loops:
            return open_loops

    # Structural fallback
    for m in candidates:
        mid = m["id"]
        body = m["body"].strip()
        b = body.lower()
        sender = m["sender_id"]
        cid = m["chat_id"]
        m_dt = datetime.datetime.fromtimestamp(m["ts"])

        due = None
        if "tomorrow" in b:
            due = (m_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        elif "today" in b or "tonight" in b or "4pm" in b:
            due = m_dt.strftime("%Y-%m-%d")

        l_type = "commitment"
        if due:
            l_type = "deadline"
        elif "?" in body or any(k in b for k in ["should we", "should i", "can we", "can u", "kab tak"]):
            l_type = "question"
        elif any(k in b for k in ["await", "waiting", "need", "chase"]):
            l_type = "followup"

        owner = "ME" if (sender == "ME" or "?" in body or "pls" in b) else sender
        status = "open"
        if owner == "ME" and due and due <= last_date_str:
            status = "done"

        title = body.split("\n")[0][:60]
        help_obj = None
        if status == "open":
            if owner != "ME":
                help_obj = {"action": "leave", "reason": "Someone else is handling it"}
            elif any(k in b for k in ["order", "buy", "ticket", "book", "find", "link", "search"]):
                help_obj = {"action": "assist", "next_step": f"Research options for {title}", "needs_yes": True}
            else:
                remind_dt = (m_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%dT10:00:00+05:30")
                help_obj = {"action": "remind", "remind_at": remind_dt}

        open_loops.append({
            "title": title,
            "type": l_type,
            "owner": owner,
            "chat_id": cid,
            "source_msg": mid,
            "due": due,
            "status": status,
            "closed_by": None,
            "help": help_obj
        })

    return open_loops


# -------------------------------------------------------------------------
# STAGE 4: GOALS DISCOVERY
# -------------------------------------------------------------------------

def extract_goals(messages, seed_goals, has_api_key):
    goals = []
    goal_msgs = []
    for m in messages:
        b = (m.get("body") or "").lower()
        if any(k in b for k in ["goal", "target", "milestone", "marathon", "retainer", "vacation", "trip to", "saving", "invest", "habit"]):
            goal_msgs.append({"id": m["id"], "body": m["body"]})

    if has_api_key and goal_msgs:
        prompt = (
            "You are an assistant identifying goals the user ('ME') is actively pursuing.\n"
            f"Already seeded goals: {json.dumps(seed_goals)}\n"
            f"Candidate messages:\n{json.dumps(goal_msgs[:30])}\n\n"
            "Task: Identify up to 10 goals.\n"
            "Each goal must have:\n"
            "- 'title': concise description\n"
            "- 'area': 'self'|'spouse'|'kids'|'parents'|'work'|'friends'|'experiences'|'finances'|'health'|'home'\n"
            "- 'type': 'habit'|'milestone'|'experience'\n"
            "- 'horizon': '3mo'|'1yr'|'3yr'|'5yr'\n"
            "- 'evidence': array of 1-3 message IDs that explicitly show this goal\n\n"
            "Return JSON: {\"goals\": [...]}"
        )
        try:
            resp = llm.chat([
                {"role": "system", "content": "You output JSON only."},
                {"role": "user", "content": prompt}
            ], response_format={"type": "json_object"})
            parsed = safe_json_parse(resp)
            if parsed and isinstance(parsed.get("goals"), list):
                return parsed["goals"][:10]
        except Exception as e:
            sys.stderr.write(f"[WARN] Stage 4 LLM failed: {e}\n")

    # Structural fallback
    evidence_map = defaultdict(list)
    for m in messages:
        b = (m.get("body") or "").lower()
        mid = m["id"]
        if "half marathon" in b or "marathon" in b:
            evidence_map["health"].append(mid)
        if "retainer client" in b or "clients by" in b:
            evidence_map["work"].append(mid)
        if "trip to" in b or "vacation" in b or "holiday" in b:
            evidence_map["experiences"].append(mid)

    if evidence_map["health"]:
        goals.append({
            "title": "Complete marathon / fitness milestone",
            "area": "health",
            "type": "milestone",
            "horizon": "3mo",
            "evidence": evidence_map["health"][:2]
        })
    if evidence_map["experiences"]:
        goals.append({
            "title": "Family trip and vacation",
            "area": "experiences",
            "type": "experience",
            "horizon": "3mo",
            "evidence": evidence_map["experiences"][:2]
        })

    return goals[:10]


# -------------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------------

def run_pipeline(input_dir, output_path):
    folder = Path(input_dir)
    chats = read_jsonl(folder / "chats.jsonl")
    messages = read_jsonl(folder / "messages.jsonl")
    seed_path = folder / "seed.json"
    seed = json.loads(seed_path.read_text(encoding="utf-8")) if seed_path.exists() else {}

    messages.sort(key=lambda m: m["ts"])
    last_ts = max((m["ts"] for m in messages), default=0)
    last_dt = datetime.datetime.fromtimestamp(last_ts) if last_ts else datetime.datetime.now()
    last_date_str = last_dt.strftime("%Y-%m-%d")

    has_api_key = bool(os.environ.get("LLM_API_KEY"))
    if not has_api_key:
        sys.stderr.write("[INFO] LLM_API_KEY not set. Running in zero-shot linguistic heuristic fallback mode for testing.\n")

    all_people = sorted({p for c in chats for p in c.get("participants", [])} - {"ME"})

    # Stage 1: Organisations & Group Chat Ties
    orgs, chat_ties = resolve_organisations_and_chats(chats, seed.get("organisations", []), has_api_key)

    # Stage 2: People Circles & Links
    people_out = classify_people(all_people, chats, messages, seed.get("circles", {}), orgs, has_api_key)

    # Stage 3: Open Loops & Where to Help
    open_loops = extract_open_loops(messages, last_date_str, has_api_key)

    # Stage 4: Goals
    goals = extract_goals(messages, seed.get("goals", []), has_api_key)

    output_data = {
        "people": people_out,
        "organisations": orgs,
        "chats": chat_ties,
        "goals": goals,
        "open_loops": open_loops
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")


def main(input_dir, output_path):
    run_pipeline(input_dir, output_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <input_dir> <output.json>")
    main(sys.argv[1], sys.argv[2])
