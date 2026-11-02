"""Deterministic guardrails + router (§10). Pure functions except route escalation."""
from __future__ import annotations

import copy
import os
import re

from pt_prompt import URGENCY, vocab

TOXINS = {  # name -> (regex, species it applies to)
    "chocolate": (r"\bchocolate|cocoa|brownie", {"dog", "cat"}),
    "xylitol": (r"xylitol|birch sugar|sugar[- ]free (gum|candy|mints?)", {"dog", "cat"}),
    "grapes/raisins": (r"\bgrapes?\b|raisins?|sultanas?|currants?", {"dog", "cat"}),
    "lilies": (r"\blil(y|ies)\b|day ?lil", {"cat"}),
    "antifreeze": (r"anti-?freeze|ethylene glycol|coolant", {"dog", "cat"}),
    "rodenticide": (r"rat (poison|bait)|mouse (poison|bait)|rodenticide|bromethalin|d-con", {"dog", "cat"}),
    "ibuprofen/NSAID": (r"ibuprofen|advil|motrin|naproxen|aleve", {"dog", "cat"}),
    "acetaminophen": (r"acetaminophen|tylenol|paracetamol", {"dog", "cat"}),
    "onion/garlic": (r"(ate|eaten|got into|whole|bowl|plate|lots of|bunch of)[^.]{0,30}\b(onions?|garlic)\b", {"dog", "cat"}),
    "THC/edibles": (r"\bthc\b|edible|cannabis|marijuana|weed (gummy|brownie)|pot brownie", {"dog", "cat"}),
    "permethrin": (r"permethrin|dog flea (treatment|medicine|drops)", {"cat"}),
}
EMERGENCY_PHRASES = {
    "can't breathe": (r"can'?t breathe|cannot breathe|struggling to breathe|gasping|blue (gums|tongue)", {"dog", "cat"}),
    "collapsed": (r"collapsed|passed out|fainted|unresponsive|won'?t wake", {"dog", "cat"}),
    "seizure": (r"seiz|convuls|fitting", {"dog", "cat"}),
    "straining to pee (cat)": (r"strain\w*[^.]{0,30}(pee|urin|litter)|(can'?t|cannot|not able to) pee|no pee", {"cat"}),
    "bloat + retching (dog)": (r"(bloat\w*|hard belly|swollen belly|distended)[^.]{0,80}(retch|dry heav|trying to vomit|gag)"
                               r"|(retch|dry heav|trying to vomit)[^.]{0,80}(bloat\w*|hard belly|swollen belly|distended)", {"dog"}),
}
NEGATION = re.compile(r"\b(no|not|never|didn'?t|did not|without|wasn'?t)\b[^.]{0,25}$")


def _hits(table: dict, species: str, message: str) -> list[str]:
    msg = message.lower()
    out = []
    for name, (pat, sp) in table.items():
        if species not in sp:
            continue
        for m in re.finditer(pat, msg):
            if not NEGATION.search(msg[max(0, m.start() - 40):m.start()]):
                out.append(name)
                break
    return out


def apply(pred: dict | None, species: str, message: str) -> tuple[dict | None, list[dict]]:
    """Return (pred', audit_log). A None pred (parse error) gets flagged needs_review."""
    audit: list[dict] = []
    if pred is None:
        audit.append({"rule": "parse_error", "action": "needs_review"})
        # Rules must still fire on an unparseable answer: build a minimal emergency answer if any rule matches.
        if not (_hits(TOXINS, species, message) or _hits(EMERGENCY_PHRASES, species, message)):
            return None, audit
        pred = {"urgency": "monitor_at_home", "system": "toxic_ingestion" if _hits(TOXINS, species, message) else "neuro",
                "conditions": [], "red_flags": [], "needs_review": True,
                "plan": {"immediate": [], "watch_for": [], "prevent": [], "recheck": "now"}}
        audit.append({"rule": "rules_on_parse_error", "action": "built emergency answer from rules"})
    p = copy.deepcopy(pred)
    plan = p.setdefault("plan", {})
    imm = plan.setdefault("immediate", [])
    orig_urgency = p.get("urgency")

    for t in _hits(TOXINS, species, message):
        audit.append({"rule": "toxin_override", "match": t})
        flag = f"possible {t} exposure"
        if flag not in p.setdefault("red_flags", []):
            p["red_flags"].append(flag)
        if "call_poison_control" not in imm:
            imm.append("call_poison_control")
        p["urgency"] = "emergency_now"
    for e in _hits(EMERGENCY_PHRASES, species, message):
        audit.append({"rule": "emergency_phrase", "match": e})
        p["urgency"] = "emergency_now"
    if p["urgency"] == "emergency_now" and "go_to_emergency_vet_now" not in imm:
        imm.insert(0, "go_to_emergency_vet_now")
    if p["urgency"] != orig_urgency:
        audit.append({"rule": "urgency_override", "from": orig_urgency, "to": p["urgency"]})

    never = set(vocab()["never"].get(species, []))
    for k in ("immediate", "watch_for", "prevent"):
        removed = [x for x in plan.get(k, []) if x in never]
        if removed:
            plan[k] = [x for x in plan[k] if x not in never]
            audit.append({"rule": "never_list", "removed_harmful": removed, "section": k})

    p["needs_review"] = any(a["rule"] == "urgency_override" for a in audit)
    if p["needs_review"]:
        audit.append({"rule": "confidence", "action": "needs_review", "why": "rule flipped model urgency"})
    return p, audit


def route(pred: dict | None) -> str:
    if pred is None or pred.get("needs_review"):
        return "large"
    return "large" if pred.get("urgency") in ("vet_today", "vet_this_week") else "small"


def large_model() -> str:
    return os.environ.get("FIREWORKS_LARGE_MODEL") or "accounts/fireworks/models/glm-5p3"


def merge_routed(small: dict | None, large: dict | None) -> dict | None:
    """Prefer the large model's answer, but never let routing lower urgency below the small model's."""
    if large is None:
        return small
    if small is None:
        return large
    out = copy.deepcopy(large)
    if URGENCY.index(small["urgency"]) > URGENCY.index(large["urgency"]):
        out["urgency"] = small["urgency"]
    return out
