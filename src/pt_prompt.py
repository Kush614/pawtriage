"""Single source of truth for the PawTriage prompt, output schema, and parser.

Used identically by baseline, harness, training data, fine-tune scoring and the app.
Only the harness passes `rubric_text` (appended to the system prompt).
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOCAB = ROOT / "data" / "vocab"

URGENCY = ["monitor_at_home", "vet_this_week", "vet_today", "emergency_now"]  # ordered low -> high
SYSTEMS = ["gi", "toxic_ingestion", "urinary", "respiratory", "skin", "musculoskeletal",
           "neuro", "eye_ear", "behavior", "wellness"]
RECHECK = ["now", "24h", "72h", "2w", "annual"]
VIDEO_PREFIX = "These are frames from a short video, in order."


@lru_cache(maxsize=None)
def vocab() -> dict:
    checklist = json.loads((VOCAB / "checklist.json").read_text())
    never = json.loads((VOCAB / "never.json").read_text())
    never_items = sorted({x for v in never.values() for x in v})
    img_map_path = VOCAB / "image_label_map.json"
    image_findings = ["no_visible_abnormality"]
    if img_map_path.exists():
        m = json.loads(img_map_path.read_text())
        image_findings = sorted({image_finding_for(c["slug"]) for c in m.get("classes", {}).values()
                                 if c.get("kept")} | {"no_visible_abnormality"})
    return {
        "conditions": json.loads((VOCAB / "conditions.json").read_text()),
        # Never-list items are legal *output* tokens so harmful advice is detectable
        # (and strippable) rather than collapsing into a parse error. Gold never uses them.
        "immediate": checklist["immediate"] + never_items,
        "watch_for": checklist["watch_for"],
        "prevent": checklist["prevent"],
        "never": never,
        "image_findings": image_findings,
    }


def image_finding_for(slug: str) -> str:
    """Dataset class slug -> image_finding value; healthy classes map to no_visible_abnormality."""
    return "no_visible_abnormality" if "healthy" in slug else slug


def schema_text() -> str:
    v = vocab()
    return json.dumps({
        "urgency": f"one of {URGENCY}",
        "system": f"one of {SYSTEMS}",
        "conditions": f"1-3 items, most likely first, from {v['conditions']}",
        "red_flags": "list of short strings",
        "plan": {
            "immediate": f"items from {v['immediate']}",
            "watch_for": f"items from {v['watch_for']}",
            "prevent": f"items from {v['prevent']}",
            "recheck": f"one of {RECHECK}",
        },
        "image_finding": f"ONLY if images are attached: one of {v['image_findings']}",
    }, separators=(",", ":"))


SYSTEM_TEMPLATE = (
    "You are a veterinary triage assistant for dogs and cats. You do not diagnose. "
    "You decide how urgently the pet should see a vet and what the owner should do until then. "
    "Reply with ONLY a JSON object matching this schema: {schema}. "
    "Use only values from the allowed vocabularies."
)


def system_prompt(rubric_text: str | None = None) -> str:
    s = SYSTEM_TEMPLATE.format(schema=schema_text())
    if rubric_text:
        s += "\n\nReference triage rubric (use it):\n" + rubric_text
    return s


def system_prompt_hash(rubric_text: str | None = None) -> str:
    return hashlib.sha256(system_prompt(rubric_text).encode()).hexdigest()[:16]


def _data_url(path: str | Path) -> str:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    b = p.read_bytes()
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(b).decode()}"


def row_images(row: dict) -> list[str]:
    """Ordered image paths for a row: video = contact sheet then frames."""
    if row.get("modality") == "video":
        return [row["sheet_path"], *row.get("frame_paths", [])]
    if row.get("image_path"):
        return [row["image_path"]]
    return []


def user_text(row: dict) -> str:
    prof = json.dumps(row.get("profile", {}), separators=(",", ":"))
    msg = row["message"]
    if row.get("modality") == "video":
        msg = f"{VIDEO_PREFIX} {msg}"
    return f"Pet profile: {prof}\nOwner message: {msg}"


def build_messages(row: dict, rubric_text: str | None = None, prior: list[dict] | None = None) -> list[dict]:
    """Chat messages for one row. `prior` = earlier turns (app follow-ups only)."""
    imgs = row_images(row)
    text = user_text(row)
    if imgs:
        content = [{"type": "image_url", "image_url": {"url": _data_url(p)}} for p in imgs]
        content.append({"type": "text", "text": text})
    else:
        content = text
    return [{"role": "system", "content": system_prompt(rubric_text)}, *(prior or []),
            {"role": "user", "content": content}]


def gold_json(gold: dict, with_image: bool) -> str:
    out = {k: gold[k] for k in ("urgency", "system", "conditions", "red_flags", "plan")}
    if with_image and gold.get("image_finding"):
        out["image_finding"] = gold["image_finding"]
    return json.dumps(out, separators=(",", ":"))


_FENCE = re.compile(r"```(?:json)?")


def _first_json_block(text: str) -> str | None:
    text = _FENCE.sub("", text or "")
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def parse_output(text: str) -> tuple[dict | None, str | None]:
    """Return (pred, error). Any invalid JSON or disallowed value => (None, reason)."""
    block = _first_json_block(text)
    if block is None:
        return None, "no_json"
    try:
        p = json.loads(block)
    except json.JSONDecodeError:
        return None, "invalid_json"
    v = vocab()
    if not isinstance(p, dict):
        return None, "not_object"
    # Structural normalizations (identical for every system); vocabulary stays strict.
    if "plan" not in p and any(k in p for k in ("immediate", "watch_for", "prevent", "recheck")):
        p["plan"] = {k: p.pop(k) for k in ("immediate", "watch_for", "prevent", "recheck") if k in p}
    if p.get("image_finding", 0) is None:
        p.pop("image_finding")
    try:
        if p["urgency"] not in URGENCY:
            return None, f"bad_urgency:{p['urgency']}"
        if p["system"] not in SYSTEMS:
            return None, f"bad_system:{p['system']}"
        conds = p["conditions"]
        if not isinstance(conds, list) or not conds or any(c not in v["conditions"] for c in conds):
            return None, "bad_conditions"
        plan = p["plan"]
        for k in ("immediate", "watch_for", "prevent"):
            if any(x not in v[k] for x in plan.get(k, [])):
                return None, f"bad_plan_{k}"
            plan.setdefault(k, [])
        if plan.get("recheck") not in RECHECK:
            return None, "bad_recheck"
        p.setdefault("red_flags", [])
        if "image_finding" in p and p["image_finding"] not in v["image_findings"]:
            return None, "bad_image_finding"
    except (KeyError, TypeError, AttributeError) as e:
        return None, f"schema:{e}"
    p["conditions"] = conds[:3]
    return p, None


def salvage_output(text: str) -> tuple[dict | None, list[str]]:
    """APP ONLY (never used in scoring): keep a mostly-valid answer by dropping out-of-vocabulary list items.
    Urgency must still be valid. Returns (pred, dropped_items)."""
    block = _first_json_block(text)
    if block is None:
        return None, []
    try:
        p = json.loads(block)
    except json.JSONDecodeError:
        return None, []
    if not isinstance(p, dict) or p.get("urgency") not in URGENCY:
        return None, []
    v = vocab()
    dropped: list[str] = []
    if "plan" not in p and any(k in p for k in ("immediate", "watch_for", "prevent", "recheck")):
        p["plan"] = {k: p.pop(k) for k in ("immediate", "watch_for", "prevent", "recheck") if k in p}
    plan = p.get("plan") if isinstance(p.get("plan"), dict) else {}
    for k in ("immediate", "watch_for", "prevent"):
        items = plan.get(k) if isinstance(plan.get(k), list) else []
        dropped += [f"{k}:{x}" for x in items if x not in v[k]]
        plan[k] = [x for x in items if x in v[k]]
    if plan.get("recheck") not in RECHECK:
        dropped.append(f"recheck:{plan.get('recheck')}")
        plan["recheck"] = "24h"
    p["plan"] = plan
    conds = p.get("conditions") if isinstance(p.get("conditions"), list) else []
    dropped += [f"condition:{c}" for c in conds if c not in v["conditions"]]
    p["conditions"] = [c for c in conds if c in v["conditions"]][:3]
    if p.get("system") not in SYSTEMS:
        dropped.append(f"system:{p.get('system')}")
        p["system"] = "wellness"
    if p.get("image_finding") is not None and p["image_finding"] not in v["image_findings"]:
        dropped.append(f"image_finding:{p['image_finding']}")
        p.pop("image_finding")
    elif "image_finding" in p and p["image_finding"] is None:
        p.pop("image_finding")
    p["red_flags"] = [str(x) for x in p.get("red_flags") or []]
    return p, dropped
