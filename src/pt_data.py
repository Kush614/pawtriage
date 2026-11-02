"""Rubric loading/validation, synthetic generation, dedupe, split + lock."""
from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from pt_prompt import RECHECK, ROOT, SYSTEMS, URGENCY, vocab

DATA = ROOT / "data"
GEN = DATA / "generated"
SPLITS = DATA / "splits"
LIST_COLS = ["conditions", "red_flags", "plan_immediate", "plan_watch_for", "plan_prevent"]


def load_rubric(path: Path = DATA / "rubric.csv") -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for c in LIST_COLS:
                r[c] = [x.strip() for x in (r.get(c) or "").split("|") if x.strip()]
            rows.append(r)
    return rows


def validate_rubric(rows: list[dict]) -> list[str]:
    v = vocab()
    errs = []
    seen = set()
    for r in rows:
        sid = r["scenario_id"]
        if sid in seen:
            errs.append(f"{sid}: duplicate scenario_id")
        seen.add(sid)
        if r["species"] not in ("dog", "cat", "both"):
            errs.append(f"{sid}: species {r['species']}")
        if r["urgency"] not in URGENCY:
            errs.append(f"{sid}: urgency {r['urgency']}")
        if r["system"] not in SYSTEMS:
            errs.append(f"{sid}: system {r['system']}")
        if r["recheck"] not in RECHECK:
            errs.append(f"{sid}: recheck {r['recheck']}")
        if not 1 <= len(r["conditions"]) <= 3:
            errs.append(f"{sid}: needs 1-3 conditions")
        for c in r["conditions"]:
            if c not in v["conditions"]:
                errs.append(f"{sid}: unknown condition {c}")
        for col, key in (("plan_immediate", "immediate"), ("plan_watch_for", "watch_for"), ("plan_prevent", "prevent")):
            for x in r[col]:
                if x not in v[key]:
                    errs.append(f"{sid}: {col} item not in checklist: {x}")
        sp = ["dog", "cat"] if r["species"] == "both" else [r["species"]]
        for s in sp:
            bad = set(r["plan_immediate"] + r["plan_watch_for"] + r["plan_prevent"]) & set(v["never"].get(s, []))
            if bad:
                errs.append(f"{sid}: gold plan contains never-list items for {s}: {bad}")
        if not r.get("source"):
            errs.append(f"{sid}: missing source")
    return errs


def expand_species(rows: list[dict]) -> list[dict]:
    """'both' rows become dog + cat variants: scenario_id gets a species suffix."""
    out = []
    for r in rows:
        if r["species"] == "both":
            for s in ("dog", "cat"):
                out.append({**r, "species": s, "scenario_id": f"{r['scenario_id']}__{s}"})
        else:
            out.append(r)
    return out


def gold_from_rubric(r: dict, image_finding: str | None = None) -> dict:
    g = {"urgency": r["urgency"], "system": r["system"], "conditions": r["conditions"],
         "red_flags": r["red_flags"],
         "plan": {"immediate": r["plan_immediate"], "watch_for": r["plan_watch_for"],
                  "prevent": r["plan_prevent"], "recheck": r["recheck"]}}
    if image_finding:
        g["image_finding"] = image_finding
    return g


def coverage(rows: list[dict]) -> dict:
    ex = expand_species(rows)
    return {"n_rubric_rows": len(rows), "n_species_variants": len(ex),
            "urgency": dict(Counter(r["urgency"] for r in ex)),
            "species": dict(Counter(r["species"] for r in ex)),
            "system": dict(Counter(r["system"] for r in ex)),
            "image_classes": sorted(r["scenario_id"] for r in rows if r["scenario_id"].startswith("img_")),
            "missing_urgency": [u for u in URGENCY if u not in {r["urgency"] for r in ex}]}


# ---------------------------------------------------------------- generation
STYLES = ["panicked", "calm", "terse", "rambling", "worried but trying to stay calm", "matter-of-fact"]

GEN_PROMPT = """You write realistic messages that pet owners type into a vet triage app.
Species: {species}
Situation (vet language, do NOT copy these words; owners describe what they see): {pattern}

Write {n} DIFFERENT owner messages. Vary across them: pet age, breed, weight, tone ({styles}),
typos/no punctuation in some, length 1-5 sentences, time since onset, and include an irrelevant
red-herring detail in about a third of them. Never name a diagnosis the owner wouldn't know.
For each, also give a pet profile consistent with the message.

Return ONLY JSON: {{"items": [{{"message": "...", "profile": {{"name": "...", "age_years": number,
"breed": "...", "weight_kg": number, "meds": [], "conditions": []}}}}]}}"""


def _norm_tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def jaccard(a: str, b: str) -> float:
    x, y = _norm_tokens(a), _norm_tokens(b)
    return len(x & y) / max(1, len(x | y))


def dedupe(items: list[dict], thresh: float = 0.8) -> list[dict]:
    kept: list[dict] = []
    for it in items:
        if all(jaccard(it["message"], k["message"]) <= thresh for k in kept):
            kept.append(it)
    return kept


async def generate_for_scenario(llm, model: str, r: dict, n: int = 40, batch: int = 10,
                                extra: str = "") -> list[dict]:
    """Generator sees only pattern + species + style. Labels are attached from the rubric."""
    items: list[dict] = []
    calls = []
    for b in range(0, n, batch):
        prompt = GEN_PROMPT.format(species=r["species"], pattern=r["pattern"] + extra,
                                   n=min(batch, n - b), styles=", ".join(STYLES))
        calls.append(llm.chat(model, [{"role": "user", "content": prompt}], temperature=1.0,
                              cache_key=f"gen|{r['scenario_id']}|{b}|{extra}", max_tokens=4000,
                              json_mode=True))
    for res in await asyncio.gather(*calls):
        try:
            txt = res["text"]
            obj = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
            items.extend(x for x in obj["items"] if isinstance(x, dict) and x.get("message"))
        except Exception:
            continue
    return dedupe(items)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


# ---------------------------------------------------------------- split + lock
def split_by_scenario(rows: list[dict], test_frac: float = 0.2, seed: int = 0) -> tuple[set, set]:
    """Scenario-level split; stratified so test gets every urgency tier x species seen."""
    # Split key is the base scenario (dog/cat variants of a 'both' row stay together).
    rng = random.Random(seed)
    by_stratum = defaultdict(list)
    meta = {}
    for r in rows:
        meta[base_id(r["scenario_id"])] = (r["gold"]["urgency"], r["species"])
    for sid, strat in meta.items():
        by_stratum[strat].append(sid)
    test_base = set()
    for strat, sids in sorted(by_stratum.items()):
        sids = sorted(sids)
        rng.shuffle(sids)
        k = max(1, round(len(sids) * test_frac)) if len(sids) >= 2 else 0
        test_base.update(sids[:k])
    all_ids = {r["scenario_id"] for r in rows}
    test = {s for s in all_ids if base_id(s) in test_base}
    return all_ids - test, test


def base_id(sid: str) -> str:
    return sid.split("__")[0]


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_lock(paths: list[Path], extra_hashes: dict | None = None) -> None:
    lines = ["# LOCKED test set", "", "Never regenerate or edit these files. sha256:", ""]
    for p in paths:
        lines.append(f"- `{Path(p).relative_to(ROOT)}`: `{file_sha(p)}`")
    if extra_hashes:
        lines += ["", "Test media (resized copies) combined sha256:", "",
                  f"- media: `{hashlib.sha256(''.join(sorted(extra_hashes.values())).encode()).hexdigest()}` "
                  f"({len(extra_hashes)} files)"]
    (SPLITS / "LOCKED.md").write_text("\n".join(lines) + "\n")


def check_lock() -> bool:
    lock = SPLITS / "LOCKED.md"
    if not lock.exists():
        return False
    for m in re.finditer(r"- `([^`]+)`: `([0-9a-f]{64})`", lock.read_text()):
        if file_sha(ROOT / m.group(1)) != m.group(2):
            raise AssertionError(f"{m.group(1)} changed since lock!")
    return True


def assert_no_overlap(train: list[dict], test: list[dict]) -> None:
    a = {base_id(r["scenario_id"]) for r in train if not r["scenario_id"].startswith(("img_", "vid_"))}
    b = {base_id(r["scenario_id"]) for r in test if not r["scenario_id"].startswith(("img_", "vid_"))}
    assert not (a & b), f"text scenario overlap: {a & b}"
    ai = {r.get("image_path") or r.get("clip_path") for r in train if r.get("modality") in ("image", "video", "video_sheet")}
    bi = {r.get("image_path") or r.get("clip_path") for r in test if r.get("modality") in ("image", "video", "video_sheet")}
    assert not ((ai - {None}) & (bi - {None})), "image/clip appears in both train and test"
