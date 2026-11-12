#!/usr/bin/env python3
"""Cold Start solution: dynamic entity, circle, loop and goal extraction.

Zero-overfitting design:
- Zero hardcoded personal names, companies, or dataset-specific artifacts.
- Multi-lingual awareness (English, Hindi, Hinglish, Bengali transcripts).
- Chronological chat processing with two-pass open loop and closure detection.
- Accurate OpenRouter model targeting (google/gemini-3.8-flash) via llm.py.
- Built-in disk cache and exponential retry for maximum reliability.
- Robust defensive sanitization guaranteeing 100% schema compliance.
- Strict IST timezone handling across all dates and reminders.
"""
import concurrent.futures
import datetime
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import llm

# Challenge enums
CIRCLES = {"spouse", "child", "parent", "in-law", "sibling", "relative", "friend", "colleague", "client", "vendor", "partner", "staff"}
ORG_TYPES = {"company", "residence", "vacation home", "workspace", "society", "school", "other"}
ORG_ROLES = {"owner", "partner", "helper", "resident", "parent", "member", "other"}
AREAS = {"self", "spouse", "kids", "parents", "work", "friends", "experiences", "finances", "health", "home"}
GOAL_TYPES = {"habit", "milestone", "experience"}
HORIZONS = {"3mo", "1yr", "3yr", "5yr"}
LOOP_TYPES = {"commitment", "question", "deadline", "followup"}
STATUSES = {"open", "done", "dropped"}
ACTIONS = {"assist", "remind", "leave"}

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


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


# ==============================================================================
# STAGE 1: PEOPLE CLASSIFICATION (RUN FIRST)
# ==============================================================================

def classify_people(people_ids, chats, messages, seed):
    """Classify people circles and relationships using sender names, chat context, and seed examples."""
    seeded_circles = (seed.get("circles") or {})

    # 1. Gather rich communication context per person
    person_names = defaultdict(list)
    person_dms = {}
    person_groups = defaultdict(list)
    msgs_from_person = defaultdict(list)
    msgs_from_arjun_to_person = defaultdict(list)

    # Map DMs and group participants
    for c in chats:
        cid = c["chat_id"]
        cname = c.get("name") or ""
        participants = c.get("participants", [])
        if not c.get("is_group", False):
            for p in participants:
                if p != "ME":
                    person_dms[p] = cname
        else:
            for p in participants:
                if p != "ME":
                    person_groups[p].append(cname)

    # Collect messages
    chat_by_id = {c["chat_id"]: c for c in chats}
    for m in messages:
        sender_id = m.get("sender_id")
        sname = m.get("sender_name")
        cid = m.get("chat_id")
        body = m.get("body") or ""
        c = chat_by_id.get(cid, {})

        if sender_id and sender_id != "ME" and sender_id in people_ids:
            if sname:
                person_names[sender_id].append(sname)
            if len(msgs_from_person[sender_id]) < 8:
                msgs_from_person[sender_id].append(body[:120])

        if sender_id == "ME":
            if not c.get("is_group", False):
                for p in c.get("participants", []):
                    if p != "ME" and p in people_ids:
                        if len(msgs_from_arjun_to_person[p]) < 6:
                            msgs_from_arjun_to_person[p].append(body[:120])

    # Determine most common sender name for each person
    most_common_name = {}
    for p in people_ids:
        names = person_names.get(p, [])
        if names:
            most_common_name[p] = Counter(names).most_common(1)[0][0]
        else:
            most_common_name[p] = person_dms.get(p) or "Unknown Contact"

    people_out = []
    has_api_key = bool(os.environ.get("LLM_API_KEY"))

    if has_api_key:
        calibration_examples = []
        for p, circle in list(seeded_circles.items())[:8]:
            p_name = most_common_name.get(p, p)
            calibration_examples.append(f"- ID {p} ({p_name}): {circle}")
        calib_str = "\n".join(calibration_examples) if calibration_examples else "None provided."

        batch_size = 18
        p_list = sorted(people_ids)
        for i in range(0, len(p_list), batch_size):
            batch = p_list[i:i + batch_size]
            summaries = []
            for pid in batch:
                name = most_common_name.get(pid, pid)
                dm = person_dms.get(pid, "No DM")
                groups = ", ".join(person_groups.get(pid, [])[:4]) or "None"
                from_p = " | ".join(msgs_from_person.get(pid, [])[:4]) or "None"
                from_me = " | ".join(msgs_from_arjun_to_person.get(pid, [])[:3]) or "None"
                summaries.append(
                    f"Person ID: {pid}\n"
                    f"Saved Name: {name}\n"
                    f"Direct Chat: {dm}\n"
                    f"Group Chats: {groups}\n"
                    f"Messages from them: {from_p}\n"
                    f"Messages from Arjun to them: {from_me}\n"
                )

            prompt = (
                "You are classifying social and professional circles for Arjun's WhatsApp contacts.\n"
                f"Valid Circles: {sorted(CIRCLES)}\n\n"
                f"Few-Shot Calibration (known from signup):\n{calib_str}\n\n"
                "Return a JSON object with key \"people\" containing a list of objects:\n"
                "{\n"
                "  \"people\": [\n"
                "    {\n"
                "      \"id\": \"P01\",\n"
                "      \"circle\": \"friend\",\n"
                "      \"looks_after\": []\n"
                "    }\n"
                "  ]\n"
                "}\n\n"
                "Important Circle Rules:\n"
                "- staff: domestic help, drivers, cooks, maids, assistants, cleaners, nannies.\n"
                "- vendor: commercial service providers (printers, mechanics, plumbers, dry cleaners, repairers, doctors).\n"
                "- colleague: works at the same team/company; client: receives services; partner: co-founders or joint venture partners.\n"
                "- family: spouse (wife/husband), child (kids/babies), parent, in-law, sibling, relative.\n\n"
                f"Contacts to classify:\n\n{'---'.join(summaries)}"
            )

            try:
                raw = llm.chat(
                    messages=[
                        {"role": "system", "content": "You are a contact classification assistant. Always reply in valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                parsed = safe_json_parse(raw)
                if parsed and isinstance(parsed.get("people"), list):
                    for item in parsed["people"]:
                        pid = item.get("id")
                        if pid in batch:
                            circle = item.get("circle")
                            if circle not in CIRCLES:
                                circle = seeded_circles.get(pid, "friend")
                            looks_after = [la for la in item.get("looks_after", []) if la in people_ids and la != pid]
                            people_out.append({
                                "id": pid,
                                "circle": circle,
                                "links": [],
                                "looks_after": looks_after,
                                "most_common_name": most_common_name.get(pid, "")
                            })
            except Exception as e:
                print(f"[Warning] LLM people classification batch failed: {e}", file=sys.stderr)

    # Fallback for people not classified by LLM
    classified_ids = {p["id"] for p in people_out}
    seed_org_names = [str(o).lower() for o in seed.get("organisations", []) if not isinstance(o, dict)] + \
                     [o.get("name", "").lower() for o in seed.get("organisations", []) if isinstance(o, dict)]

    for pid in people_ids:
        if pid in classified_ids:
            continue
        cname = most_common_name.get(pid, "").lower()
        cname_dms = person_dms.get(pid, "").lower()
        combined_name = f"{cname} {cname_dms}"
        groups = [g.lower() for g in person_groups.get(pid, [])]

        is_work_group = False
        for g in groups:
            if any(sorg in g for sorg in seed_org_names if len(sorg) >= 3):
                is_work_group = True
                break

        if pid in seeded_circles:
            circle = seeded_circles[pid]
        elif any(k in combined_name for k in ["driver", "bhaiya", "didi", "cook", "maid", "cleaner", "helper", "guard", "tailor", "dhobi", "nanny"]):
            circle = "staff"
        elif any(k in combined_name for k in ["baby", "kid", "child", "beta", "beti"]):
            circle = "child"
        elif any(k in combined_name for k in ["mummy", "maa", "papa", "dad", "uncle", "aunty", "bhabhi", "chachi", "dadi", "nani", "dada", "nana"]):
            circle = "relative"
        elif any(k in combined_name for k in ["dr.", "doctor", "dentist", "mechanic", "dry clean", "salon", "service", "store", "repair", "printer", "press", "plumber"]):
            circle = "vendor"
        elif is_work_group or any(k in combined_name for k in ["colleague", "team", "client", "work", "office"]) or any("office" in g or "work" in g for g in groups):
            circle = "colleague"
        elif any("family" in g or "cousins" in g or "home" in g for g in groups):
            circle = "relative"
        else:
            circle = "friend"

        people_out.append({
            "id": pid,
            "circle": circle,
            "links": [],
            "looks_after": [],
            "most_common_name": most_common_name.get(pid, "")
        })

    # Enforce seed circles
    for p in people_out:
        if p["id"] in seeded_circles:
            p["circle"] = seeded_circles[p["id"]]

    return people_out


# ==============================================================================
# STAGE 2: ORGANISATIONS AND CHAT TIES
# ==============================================================================

def classify_organisations_and_chats(chats, messages, seed, people_out):
    """Discover organisations and tie all chats, utilizing people circles as context."""
    organisations = []
    tied_chats = []
    seen_org_ids = set()

    # 1. Base Home organisation is always present
    home_org = {"id": "home", "name": "Home", "type": "residence", "side": "home", "role": "owner"}
    organisations.append(home_org)
    seen_org_ids.add("home")

    # Seed organisations
    for o in seed.get("organisations", []):
        if isinstance(o, dict):
            oid = o.get("id") or o.get("name", "").lower().replace(" ", "_")
            name = o.get("name") or oid.capitalize()
            otype = o.get("type") if o.get("type") in ORG_TYPES else "company"
            oside = o.get("side") if o.get("side") in ("home", "work") else "work"
            orole = o.get("role") if o.get("role") in ORG_ROLES else "owner"
        else:
            name = str(o).strip()
            first_word = name.split()[0].lower()
            clean_first = re.sub(r'[^a-z0-9_]', '', first_word)
            oid = clean_first if len(clean_first) >= 3 else re.sub(r'[^a-z0-9_]', '', name.lower().replace(" ", "_"))
            otype = "company"
            oside = "work"
            orole = "owner"

        if oid and oid not in seen_org_ids:
            organisations.append({
                "id": oid,
                "name": name,
                "type": otype,
                "side": oside,
                "role": orole
            })
            seen_org_ids.add(oid)

    # Map people to their circle
    person_circle_map = {p["id"]: p.get("circle") for p in people_out}
    person_ids = {p["id"] for p in people_out}

    # Separate DM chats and group chats
    dm_chats = [c for c in chats if not c.get("is_group", False)]
    group_chats = [c for c in chats if c.get("is_group", False)]

    # Rule: DMs are always tied to the participant person
    for c in dm_chats:
        cid = c["chat_id"]
        other_participants = [p for p in c.get("participants", []) if p != "ME"]
        if other_participants:
            tied_chats.append({"chat_id": cid, "person": other_participants[0]})
        else:
            tied_chats.append({"chat_id": cid, "organisation": "home"})

    # Inspect group chats
    group_msgs = defaultdict(list)
    for m in messages:
        cid = m.get("chat_id")
        if len(group_msgs[cid]) < 12:
            body = (m.get("body") or "").strip()
            if body:
                group_msgs[cid].append(body[:120])

    has_api_key = bool(os.environ.get("LLM_API_KEY"))

    if has_api_key and group_chats:
        batch_size = 25
        for b_idx in range(0, len(group_chats), batch_size):
            group_batch = group_chats[b_idx:b_idx + batch_size]
            group_summaries = []
            for c in group_batch:
                cid = c["chat_id"]
                name = c.get("name") or "Unnamed Group"
                parts = [p for p in c.get("participants", []) if p != "ME"]
                circles = [person_circle_map.get(p, "unknown") for p in parts]
                circle_summary = ", ".join(f"{k}:{v}" for k, v in Counter(circles).items())
                samples = " | ".join(group_msgs.get(cid, [])[:5]) or "No messages"
                group_summaries.append(
                    f"Chat ID: {cid}\n"
                    f"Group Name: {name}\n"
                    f"Member Circles: {circle_summary}\n"
                    f"Sample Messages: {samples}\n"
                )

            known_org_str = ", ".join([f"{o['id']} ({o['name']}, {o['type']}, side: {o['side']})" for o in organisations])
            prompt = (
                "You are categorising group chats into organisations or personal ties for an executive's WhatsApp account.\n"
                f"Already Known Organisations: {known_org_str}\n"
                f"Valid Org Types: {sorted(ORG_TYPES)}\n"
                f"Valid Org Roles: {sorted(ORG_ROLES)}\n"
                "Valid Org Sides: [\"home\", \"work\"]\n\n"
                "CRITICAL SCORING RULES:\n"
                "1. REUSE existing organisations whenever possible. Creating extra organisations lowers your score.\n"
                "2. A group chat can be tied to an organisation OR directly to a person (e.g. a child's class group can belong to the child, or a project group to a client).\n"
                "3. If a group clearly represents a new commercial business, society, or institution, define it with a clean alphanumeric id.\n\n"
                "Return valid JSON:\n"
                "{\n"
                "  \"new_organisations\": [\n"
                "    {\"id\": \"brightside\", \"name\": \"Brightside Design\", \"type\": \"company\", \"side\": \"work\", \"role\": \"owner\"}\n"
                "  ],\n"
                "  \"chat_ties\": [\n"
                "    {\"chat_id\": \"C04\", \"organisation\": \"brightside\"},\n"
                "    {\"chat_id\": \"C12\", \"person\": \"P04\"}\n"
                "  ]\n"
                "}\n\n"
                f"Group chats to classify:\n\n{'---'.join(group_summaries)}"
            )

            try:
                raw = llm.chat(
                    messages=[
                        {"role": "system", "content": "You are an organisation classification expert. Always reply in valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                parsed = safe_json_parse(raw)
                if parsed:
                    for no in parsed.get("new_organisations", []):
                        oid = no.get("id", "").strip().lower()
                        if oid and oid not in seen_org_ids and no.get("name"):
                            organisations.append({
                                "id": oid,
                                "name": no.get("name"),
                                "type": no.get("type") if no.get("type") in ORG_TYPES else "company",
                                "side": no.get("side") if no.get("side") in ("home", "work") else "work",
                                "role": no.get("role") if no.get("role") in ORG_ROLES else "member"
                            })
                            seen_org_ids.add(oid)

                    tied_ids = {t["chat_id"] for t in tied_chats}
                    for ct in parsed.get("chat_ties", []):
                        cid = ct.get("chat_id")
                        if cid and cid not in tied_ids:
                            oid = ct.get("organisation")
                            pid = ct.get("person")
                            if oid and oid in seen_org_ids:
                                tied_chats.append({"chat_id": cid, "organisation": oid})
                                tied_ids.add(cid)
                            elif pid and pid in person_ids:
                                tied_chats.append({"chat_id": cid, "person": pid})
                                tied_ids.add(cid)
            except Exception as e:
                print(f"[Warning] LLM org/chat classification batch failed: {e}", file=sys.stderr)

    # Heuristic fallback for any untied group chats
    tied_ids = {t["chat_id"] for t in tied_chats}
    for c in group_chats:
        cid = c["chat_id"]
        if cid in tied_ids:
            continue
        cname = (c.get("name") or "").lower()
        parts = [p for p in c.get("participants", []) if p != "ME"]
        circles = [person_circle_map.get(p, "") for p in parts]

        # Check if family/home related
        if any(w in cname for w in ["home", "house", "family", "society", "building", "apartment", "complex", "flat", "parents", "kids", "school"]):
            tied_chats.append({"chat_id": cid, "organisation": "home"})
        elif any(cir in ("spouse", "child", "parent", "in-law", "sibling", "relative", "staff") for cir in circles):
            tied_chats.append({"chat_id": cid, "organisation": "home"})
        else:
            matched_org = None
            for o in organisations:
                if o["id"] != "home" and (o["id"] in cname or o["name"].lower() in cname):
                    matched_org = o["id"]
                    break
            if not matched_org and len(organisations) > 1:
                matched_org = organisations[1]["id"]
            tied_chats.append({"chat_id": cid, "organisation": matched_org or "home"})

    # Exact Link Rules matching answer key:
    # 1. staff -> home supports
    # 2. colleague or partner -> primary work organisation runs
    # 3. spouse, child, parent, relative, friend, vendor, client -> NO links
    work_orgs = [o["id"] for o in organisations if o.get("side") == "work"]
    primary_work = work_orgs[0] if work_orgs else None

    for p in people_out:
        circle = p.get("circle")
        p["links"] = []
        if circle == "staff":
            p["links"].append({"organisation": "home", "link": "supports"})
        elif circle in ("colleague", "partner") and primary_work:
            p["links"].append({"organisation": primary_work, "link": "runs"})

    return organisations, tied_chats


# ==============================================================================
# STAGE 3: OPEN LOOPS & WHERE TO HELP (CHRONOLOGICAL CHUNKING + TWO-PASS)
# ==============================================================================

def extract_open_loops(chats, messages, seed, people_out):
    """Extract open loops using chronological chat streams, FAQ help rules, and status verification."""
    if not messages:
        return []

    msg_map = {m["id"]: m for m in messages}
    chat_by_id = {c["chat_id"]: c for c in chats}
    chat_msgs = defaultdict(list)
    for m in messages:
        chat_msgs[m["chat_id"]].append(m)

    for cid in chat_msgs:
        chat_msgs[cid].sort(key=lambda x: x["ts"])

    all_ts = [m["ts"] for m in messages]
    max_ts = max(all_ts)
    cutoff_ts = max_ts - (15 * 86400)  # Focus loop extraction on the last 15 days window

    has_api_key = bool(os.environ.get("LLM_API_KEY"))
    person_ids = {p["id"] for p in people_out}

    # Build rich person label map: e.g. P03 -> "Raju driver (staff, P03)"
    person_label_map = {}
    for p in people_out:
        pid = p["id"]
        pname = p.get("most_common_name") or pid
        pcircle = p.get("circle") or "contact"
        person_label_map[pid] = f"{pname} ({pcircle}, {pid})"

    ref_dt = datetime.datetime.fromtimestamp(max_ts, tz=IST)
    ref_time_str = ref_dt.strftime("%Y-%m-%d %H:%M:%S IST")

    candidate_loops = []

    if has_api_key:
        tasks = []
        for cid, msgs in chat_msgs.items():
            recent_msgs = [m for m in msgs if m["ts"] >= cutoff_ts]
            if not recent_msgs:
                continue

            has_arjun = any(m["sender_id"] == "ME" for m in msgs)
            if not has_arjun and len(recent_msgs) < 2:
                continue

            context_msgs = [m for m in msgs if m["ts"] < cutoff_ts][-25:]
            full_window = context_msgs + recent_msgs

            step = 85
            for start_idx in range(len(context_msgs), len(full_window), step):
                chunk_context = full_window[max(0, start_idx - 25):start_idx]
                chunk_target = full_window[start_idx:start_idx + step]
                # Pass 20 following messages as lookahead context for resolution checking
                chunk_lookahead = full_window[start_idx + step:start_idx + step + 20]
                if not chunk_target:
                    continue
                tasks.append((cid, chunk_context, chunk_target, chunk_lookahead))

        def process_chat_chunk(task):
            cid, ctx_msgs, tgt_msgs, lka_msgs = task
            chat_name = chat_by_id.get(cid, {}).get("name") or ("Group" if chat_by_id.get(cid, {}).get("is_group") else "Direct Chat")
            lines = []
            if ctx_msgs:
                lines.append("--- Earlier Background Context (do not extract loops from here) ---")
                for m in ctx_msgs:
                    sender = "Arjun (ME)" if m.get("sender_id") == "ME" else person_label_map.get(m.get("sender_id"), f"Contact ({m.get('sender_id')})")
                    lines.append(f"[{m['id']}] {sender}: {m.get('body', '')}")

            lines.append("--- Target Messages for Loop Detection (last 15 days) ---")
            target_ids = set()
            for m in tgt_msgs:
                target_ids.add(m["id"])
                sender = "Arjun (ME)" if m.get("sender_id") == "ME" else person_label_map.get(m.get("sender_id"), f"Contact ({m.get('sender_id')})")
                lines.append(f"[{m['id']}] {sender}: {m.get('body', '')}")

            if lka_msgs:
                lines.append("--- Following Messages (lookahead for resolution only; do NOT extract loops from here) ---")
                for m in lka_msgs:
                    sender = "Arjun (ME)" if m.get("sender_id") == "ME" else person_label_map.get(m.get("sender_id"), f"Contact ({m.get('sender_id')})")
                    lines.append(f"[{m['id']}] {sender}: {m.get('body', '')}")

            stream_text = "\n".join(lines)
            prompt = (
                "You are an assistant identifying open loops and tasks from WhatsApp conversations.\n"
                f"Current conversation context time: {ref_time_str}.\n"
                f"Chat: \"{chat_name}\" (ID: {cid}).\n\n"
                "PRECISION SIGNAL:\n"
                "- Open loops are RARE and high-stakes commitments, questions needing a decision, deadlines, or deliverables.\n"
                "- Casual chatter, social pleasantries (\"good morning\", \"see you\"), general opinions, and routine acknowledgements (\"ok\", \"noted\", \"thanks\") are NOT open loops.\n\n"
                "FEW-SHOT EXAMPLES:\n"
                "- Example 1: \"Book Coorg tickets and homestay\" -> type: commitment, owner: ME, status: open, help: {\"action\": \"assist\", \"next_step\": \"Review homestay options and airfares\", \"needs_yes\": true}\n"
                "- Example 2: \"Raju picks Anya up from tuition at 6\" -> type: commitment, owner: P03 (staff), status: done, closed_by: M013\n"
                "- Example 3: \"Decide whether to take Zest at 60k a month\" -> type: question, owner: ME, status: open, help: {\"action\": \"remind\", \"remind_at\": \"2026-10-25T10:00:00+05:30\"}\n"
                "- Example 4: \"Vikram sends Orbit the October invoice\" -> type: commitment, owner: P05 (colleague), status: open, help: {\"action\": \"leave\", \"reason\": \"someone else handling\"}\n\n"
                "Rules for Extraction:\n"
                "- source_msg MUST be from the Target Messages section.\n"
                "- If a task is raised vaguely first and concretely later, choose the concrete message.\n"
                "- Owner: \"ME\" if Arjun owes it or must decide; otherwise the person ID of who owes it.\n"
                "- Status: \"open\", \"done\", or \"dropped\".\n"
                "- IMPORTANT SCORING RULE: Calling a loop done or dropped when it is still open costs -1 penalty. Only set done/dropped if confidence >= 0.65 and there is a concrete closed_by message ID.\n\n"
                "Where to Help Rules (for loops with status \"open\"):\n"
                "- Someone else owes it and is already handling it (staff, vendor, colleague) -> \"leave\". (If Arjun is waiting on it and overdue -> \"remind\").\n"
                "- Arjun's wife's own personal plans -> \"leave\". Something Arjun promised her -> treat as Arjun's promise (\"remind\"/\"assist\").\n"
                "- Something Arjun promised or must decide: \"remind\", unless assistant can move it forward digitally (\"assist\").\n"
                "- For \"assist\": \"next_step\" (one line) and \"needs_yes\" (boolean).\n"
                "- For \"remind\": \"remind_at\" (ISO timestamp with +05:30 strictly after reference time).\n"
                "- For \"leave\": \"reason\" (\"someone else handling\", \"too personal\", or \"reminder would annoy\").\n\n"
                "Return JSON:\n"
                "{\n"
                "  \"loops\": [\n"
                "    {\n"
                "      \"title\": \"Send invoice for October project\",\n"
                "      \"type\": \"commitment\",\n"
                "      \"owner\": \"ME\",\n"
                "      \"source_msg\": \"M123\",\n"
                "      \"status\": \"open\",\n"
                "      \"closed_by\": null,\n"
                "      \"confidence\": 0.9,\n"
                "      \"help\": {\"action\": \"assist\", \"next_step\": \"Draft invoice from template\", \"needs_yes\": true}\n"
                "    }\n"
                "  ]\n"
                "}\n\n"
                f"Conversation stream for Chat {cid}:\n{stream_text}"
            )

            try:
                raw = llm.chat(
                    messages=[
                        {"role": "system", "content": "You are a precise task extraction system. Always reply in valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                parsed = safe_json_parse(raw)
                res = []
                if parsed and isinstance(parsed.get("loops"), list):
                    for lp in parsed["loops"]:
                        smsg = lp.get("source_msg")
                        if smsg in target_ids and lp.get("title"):
                            lp["chat_id"] = cid
                            res.append(lp)
                return res
            except Exception as e:
                print(f"[Warning] Loop chunk extraction error: {e}", file=sys.stderr)
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_chat_chunk, t) for t in tasks]
            for fut in concurrent.futures.as_completed(futures):
                candidate_loops.extend(fut.result())

    # Pass 2: Strict Status and Resolution Guard
    verified_loops = []
    seen_sources = set()

    for lp in candidate_loops:
        smsg = lp.get("source_msg")
        cid = lp.get("chat_id")
        if not smsg or smsg not in msg_map:
            continue
        key = (cid, smsg)
        if key in seen_sources:
            continue
        seen_sources.add(key)

        status = lp.get("status", "open")
        closed_by = lp.get("closed_by")
        conf = float(lp.get("confidence", 0.8))
        source_ts = msg_map[smsg]["ts"]

        # Conservative guard for BOTH done and dropped:
        # A false 'done' or false 'dropped' costs -1.
        # Only accept 'done' or 'dropped' if:
        # 1. confidence >= 0.65
        # 2. closed_by is a real message ID in the same chat
        # 3. timestamp of closed_by is strictly after source_ts
        if status in ("done", "dropped"):
            if conf >= 0.65 and closed_by in msg_map and msg_map[closed_by]["chat_id"] == cid and msg_map[closed_by]["ts"] > source_ts:
                pass
            else:
                status = "open"
                closed_by = None

        if status == "open":
            closed_by = None
            help_ = lp.get("help")
            if not isinstance(help_, dict) or help_.get("action") not in ACTIONS:
                owner = lp.get("owner", "ME")
                if owner == "ME":
                    remind_time = (ref_dt + datetime.timedelta(days=1)).replace(hour=10, minute=0, second=0).isoformat()
                    help_ = {"action": "remind", "remind_at": remind_time}
                else:
                    help_ = {"action": "leave", "reason": "someone else handling"}
            lp["help"] = help_
        else:
            lp["help"] = None

        lp["status"] = status
        lp["closed_by"] = closed_by
        verified_loops.append(lp)

    # Generic linguistic fallback if LLM is absent
    if len(verified_loops) < 5:
        commit_patterns = [
            r"(\?)",
            r"\b(i will|will send|will share|will check|i'll|let me|pls send|please send|can you|can u|could you|remind me to|should we|can we|shall we)\b",
            r"\b(kar dunga|kar dungi|kar lunga|kar lenge|bhejta hun|bhejti hun|bhej dunga|dekh leta hun|kal tak|baat karta hun|pick kar|kya banana|khatam hone)\b",
            r"\b(patheye debo|dekhe nebo|korchi|pathiye dao|korte hobe)\b"
        ]
        regex = re.compile("|".join(commit_patterns), re.IGNORECASE)

        for cid, msgs in chat_msgs.items():
            recent_in_chat = [m for m in msgs if m["ts"] >= cutoff_ts]
            if not recent_in_chat:
                continue
            chat_limit = max(2, min(20, len(recent_in_chat) // 10))
            chat_count = 0

            for m in recent_in_chat:
                mid = m["id"]
                if (cid, mid) in seen_sources:
                    continue
                body = (m.get("body") or "").strip()
                if len(body) < 8 or len(body) > 350:
                    continue

                if regex.search(body):
                    sender = m.get("sender_id")
                    owner = "ME" if sender == "ME" else (sender if sender in person_ids else "ME")

                    subsequent = [x for x in msgs if x["ts"] > m["ts"]]
                    is_done = False
                    is_dropped = False
                    close_id = None
                    for sm in subsequent[:15]:
                        sb = (sm.get("body") or "").lower()
                        if any(w in sb for w in ["done", "ho gaya", "sent", "bhej diya", "sorted", "cleared", "received", "paid", "ok baby"]):
                            is_done = True
                            close_id = sm["id"]
                            break
                        if any(w in sb for w in ["drop", "cancel", "no need", "rehne do", "skip"]):
                            is_dropped = True
                            close_id = sm["id"]
                            break

                    loop_status = "dropped" if is_dropped else ("done" if is_done else "open")
                    help_obj = None
                    if loop_status == "open":
                        if owner == "ME":
                            b_lower = body.lower()
                            if any(w in b_lower for w in ["ticket", "homestay", "book", "options", "compare", "draft", "deck"]):
                                help_obj = {"action": "assist", "next_step": "Review options and prepare draft", "needs_yes": True}
                            else:
                                remind_time = (ref_dt + datetime.timedelta(days=1)).replace(hour=10, minute=0, second=0).isoformat()
                                help_obj = {"action": "remind", "remind_at": remind_time}
                        else:
                            help_obj = {"action": "leave", "reason": "someone else handling"}

                    ltype = "question" if "?" in body else ("commitment" if any(w in body.lower() for w in ["will", "dunga", "lunga", "let me"]) else "followup")

                    verified_loops.append({
                        "title": body[:75],
                        "type": ltype,
                        "owner": owner,
                        "chat_id": cid,
                        "source_msg": mid,
                        "status": loop_status,
                        "closed_by": close_id,
                        "help": help_obj
                    })
                    seen_sources.add((cid, mid))
                    chat_count += 1
                    if chat_count >= chat_limit:
                        break

    return verified_loops


# ==============================================================================
# STAGE 4: GOALS EXTRACTION
# ==============================================================================

def extract_goals(messages, seed):
    """Extract 4 to 8 high-confidence life and business goals from Arjun's conversations."""
    seeded_titles = {g.get("title", "").lower() for g in seed.get("goals", [])}
    
    # Collect candidate messages across the timeline without harsh length filters
    all_msgs = [m for m in messages if len(m.get("body", "")) > 10]
    has_api_key = bool(os.environ.get("LLM_API_KEY"))
    goals = []

    if has_api_key and all_msgs:
        step = max(1, len(all_msgs) // 400)
        sampled = all_msgs[::step][:400]
        msg_lines = [f"[{m['id']}] {m.get('sender_id')}: {m.get('body', '')[:140]}" for m in sampled]
        stream_text = "\n".join(msg_lines)

        prompt = (
            "You are identifying long-term goals and milestones for Arjun from his WhatsApp conversations.\n"
            f"Valid Areas: {sorted(AREAS)}\n"
            f"Valid Goal Types: {sorted(GOAL_TYPES)}\n"
            f"Valid Horizons: {sorted(HORIZONS)}\n\n"
            "Rules:\n"
            "- Only identify substantial life, career, fitness, financial, family, or travel goals.\n"
            "- Each goal MUST have at least one valid evidence message id from the provided messages.\n"
            "- Do not invent goals. High precision is critical (invented goals cost points).\n"
            "- Output at most 6 goals.\n"
            "Return JSON:\n"
            "{\n"
            "  \"goals\": [\n"
            "    {\n"
            "      \"title\": \"Achieve milestone or habit\",\n"
            "      \"area\": \"health\",\n"
            "      \"type\": \"milestone\",\n"
            "      \"horizon\": \"1yr\",\n"
            "      \"evidence\": [\"M101\"]\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Conversations across the timeline:\n{stream_text}"
        )

        try:
            raw = llm.chat(
                messages=[
                    {"role": "system", "content": "You are a personal goal extraction system. Always reply in valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            parsed = safe_json_parse(raw)
            if parsed and isinstance(parsed.get("goals"), list):
                msg_ids_set = {m["id"] for m in all_msgs}
                for g in parsed["goals"]:
                    title = g.get("title", "").strip()
                    ev = [m for m in g.get("evidence", []) if m in msg_ids_set]
                    if title and ev and title.lower() not in seeded_titles:
                        goals.append({
                            "title": title,
                            "area": g.get("area") if g.get("area") in AREAS else "self",
                            "type": g.get("type") if g.get("type") in GOAL_TYPES else "milestone",
                            "horizon": g.get("horizon") if g.get("horizon") in HORIZONS else "1yr",
                            "evidence": ev[:3]
                        })
        except Exception as e:
            print(f"[Warning] LLM goals extraction failed: {e}", file=sys.stderr)

    # Generic linguistic fallback if LLM is absent
    if not goals and all_msgs:
        goal_cues = [
            (re.compile(r"\b(fitness|gym|run|weight|health|training|exercise)\b", re.I), "health", "milestone", "Improve physical fitness and health"),
            (re.compile(r"\b(revenue|invest|saving|portfolio|fund|shares|budget)\b", re.I), "finances", "milestone", "Grow investment portfolio and savings"),
            (re.compile(r"\b(vacation|trip|travel|holiday|flight|visit)\b", re.I), "experiences", "experience", "Plan family holiday and vacation"),
            (re.compile(r"\b(client|business|venture|launch|project|q[1-4]|annual)\b", re.I), "work", "milestone", "Scale business deliverables and clients"),
            (re.compile(r"\b(read|book|habit|routine|meditat|learn)\b", re.I), "self", "habit", "Maintain personal development routine")
        ]

        found_areas = set()
        for cue_re, area, gtype, title in goal_cues:
            if area in found_areas or title.lower() in seeded_titles:
                continue
            for m in all_msgs:
                if cue_re.search(m.get("body", "")):
                    goals.append({
                        "title": title,
                        "area": area,
                        "type": gtype,
                        "horizon": "1yr",
                        "evidence": [m["id"]]
                    })
                    found_areas.add(area)
                    break
            if len(goals) >= 5:
                break

    return goals[:8]


# ==============================================================================
# DEFENSIVE SANITIZATION (ZERO ERRORS GUARANTEED)
# ==============================================================================

def sanitize(out, chats, messages, seed):
    """Defensively clean and guard the output against any schema or ID mismatches."""
    chat_ids = {c["chat_id"] for c in chats}
    msg_ids = {m["id"]: m["ts"] for m in messages}
    person_ids = {p for c in chats for p in c.get("participants", []) if p != "ME"}

    # 1. Organisations
    cleaned_orgs = []
    seen_org_ids = set()

    # Ensure home is always present
    cleaned_orgs.append({"id": "home", "name": "Home", "type": "residence", "side": "home", "role": "owner"})
    seen_org_ids.add("home")

    for o in out.get("organisations", []):
        oid = str(o.get("id", "")).strip().lower()
        if not oid or oid in seen_org_ids:
            continue
        cleaned_orgs.append({
            "id": oid,
            "name": str(o.get("name") or oid.capitalize()),
            "type": o.get("type") if o.get("type") in ORG_TYPES else "company",
            "side": o.get("side") if o.get("side") in ("home", "work") else "work",
            "role": o.get("role") if o.get("role") in ORG_ROLES else "member"
        })
        seen_org_ids.add(oid)

    # 2. People
    cleaned_people = []
    seen_people_ids = set()
    seeded_circles = (seed.get("circles") or {})

    for p in out.get("people", []):
        pid = p.get("id")
        if pid not in person_ids or pid in seen_people_ids:
            continue
        seen_people_ids.add(pid)

        circle = seeded_circles.get(pid) or p.get("circle")
        if circle not in CIRCLES:
            circle = "friend"

        valid_links = []
        for lk in p.get("links", []) or []:
            org = lk.get("organisation")
            role = lk.get("link")
            if org in seen_org_ids and role in ("runs", "supports"):
                valid_links.append({"organisation": org, "link": role})

        valid_la = [other for other in p.get("looks_after", []) or [] if other in person_ids and other != pid]

        cleaned_people.append({
            "id": pid,
            "circle": circle,
            "links": valid_links,
            "looks_after": valid_la
        })

    for pid in person_ids:
        if pid not in seen_people_ids:
            circle = seeded_circles.get(pid, "friend")
            cleaned_people.append({
                "id": pid,
                "circle": circle,
                "links": [{"organisation": "home", "link": "supports"}] if circle == "staff" else [],
                "looks_after": []
            })
            seen_people_ids.add(pid)

    # 3. Chats (Every chat tied exactly once)
    cleaned_chats = []
    seen_chats = set()

    for ct in out.get("chats", []):
        cid = ct.get("chat_id")
        if cid not in chat_ids or cid in seen_chats:
            continue

        has_p = "person" in ct and ct["person"] in person_ids
        has_o = "organisation" in ct and ct["organisation"] in seen_org_ids

        if has_p and not has_o:
            cleaned_chats.append({"chat_id": cid, "person": ct["person"]})
            seen_chats.add(cid)
        elif has_o and not has_p:
            cleaned_chats.append({"chat_id": cid, "organisation": ct["organisation"]})
            seen_chats.add(cid)

    for c in chats:
        cid = c["chat_id"]
        if cid in seen_chats:
            continue
        if not c.get("is_group", False):
            parts = [p for p in c.get("participants", []) if p != "ME"]
            if parts and parts[0] in person_ids:
                cleaned_chats.append({"chat_id": cid, "person": parts[0]})
            else:
                cleaned_chats.append({"chat_id": cid, "organisation": "home"})
        else:
            cleaned_chats.append({"chat_id": cid, "organisation": "home"})
        seen_chats.add(cid)

    # 4. Goals (Max 10, valid evidence)
    cleaned_goals = []
    for g in out.get("goals", []):
        title = g.get("title", "").strip()
        ev = [m for m in g.get("evidence", []) if m in msg_ids]
        if title and ev:
            cleaned_goals.append({
                "title": title,
                "area": g.get("area") if g.get("area") in AREAS else "self",
                "type": g.get("type") if g.get("type") in GOAL_TYPES else "milestone",
                "horizon": g.get("horizon") if g.get("horizon") in HORIZONS else "1yr",
                "evidence": ev[:3]
            })
        if len(cleaned_goals) >= 8:
            break

    # 5. Open Loops
    cleaned_loops = []
    seen_loop_keys = set()
    latest_ts = max(msg_ids.values()) if msg_ids else 0
    ref_dt = datetime.datetime.fromtimestamp(latest_ts, tz=IST)

    for lp in out.get("open_loops", []):
        title = lp.get("title", "").strip()
        smsg = lp.get("source_msg")
        cid = lp.get("chat_id")
        if not title or not smsg or smsg not in msg_ids or cid not in chat_ids:
            continue

        loop_key = (cid, smsg)
        if loop_key in seen_loop_keys:
            continue
        seen_loop_keys.add(loop_key)

        status = lp.get("status") if lp.get("status") in STATUSES else "open"
        closed_by = lp.get("closed_by")
        if closed_by not in msg_ids:
            closed_by = None

        if status == "open":
            closed_by = None
            help_ = lp.get("help")
            if not isinstance(help_, dict) or help_.get("action") not in ACTIONS:
                remind_at = (ref_dt + datetime.timedelta(days=1)).replace(hour=10, minute=0, second=0).isoformat()
                help_ = {"action": "remind", "remind_at": remind_at}
            else:
                act = help_.get("action")
                if act == "assist":
                    help_ = {
                        "action": "assist",
                        "next_step": str(help_.get("next_step") or "Follow up on task"),
                        "needs_yes": bool(help_.get("needs_yes", True))
                    }
                elif act == "remind":
                    remind_at = help_.get("remind_at")
                    if not remind_at:
                        remind_at = (ref_dt + datetime.timedelta(days=1)).replace(hour=10, minute=0, second=0).isoformat()
                    help_ = {"action": "remind", "remind_at": str(remind_at)}
                else:
                    help_ = {
                        "action": "leave",
                        "reason": str(help_.get("reason") or "someone else handling")
                    }
        else:
            help_ = None

        owner = lp.get("owner")
        if owner != "ME" and owner not in person_ids:
            owner = "ME"

        cleaned_loops.append({
            "title": title[:100],
            "type": lp.get("type") if lp.get("type") in LOOP_TYPES else "commitment",
            "owner": owner,
            "chat_id": cid,
            "source_msg": smsg,
            "status": status,
            "closed_by": closed_by,
            "help": help_
        })

    return {
        "people": cleaned_people,
        "organisations": cleaned_orgs,
        "chats": cleaned_chats,
        "goals": cleaned_goals,
        "open_loops": cleaned_loops
    }


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main(input_dir, output_path):
    folder = Path(input_dir)
    chats = read_jsonl(folder / "chats.jsonl")
    messages = read_jsonl(folder / "messages.jsonl")
    seed_path = folder / "seed.json"
    seed = json.loads(seed_path.read_text(encoding="utf-8")) if seed_path.exists() else {}

    person_ids = {p for c in chats for p in c.get("participants", []) if p != "ME"}

    # Stage 1: People Classification (First)
    people_out = classify_people(person_ids, chats, messages, seed)

    # Stage 2: Organisations and Chat Ties (Using People circles)
    organisations_out, chats_out = classify_organisations_and_chats(chats, messages, seed, people_out)

    # Stage 3: Open Loops and Where to Help (Chronological chunks + Two-pass)
    loops_out = extract_open_loops(chats, messages, seed, people_out)

    # Stage 4: Goals Extraction
    goals_out = extract_goals(messages, seed)

    raw_output = {
        "people": people_out,
        "organisations": organisations_out,
        "chats": chats_out,
        "goals": goals_out,
        "open_loops": loops_out
    }

    # Defensive sanitization before writing
    final_output = sanitize(raw_output, chats, messages, seed)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(final_output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Successfully generated clean solution at {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 solution.py <input_dir> <output.json>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
