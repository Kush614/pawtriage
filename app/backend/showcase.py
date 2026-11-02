"""Demo showcase: recorded fine-tune answers (no GPU needed), care-adherence score, dog-walker multi-Den route."""
from __future__ import annotations

import json
import random
from datetime import timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import select

from . import den, store
from .store import Pet

ROOT = Path(__file__).resolve().parents[2]
router = APIRouter()
SYSTEMS = ["base", "harness_rubric", "sft", "sft_safety"]
_REPLAY: dict | None = None


def _load() -> dict:
    """Join the locked image test rows with every system's *recorded* answer (results/<system>.jsonl)."""
    global _REPLAY
    if _REPLAY is None:
        rows = {}
        for line in (ROOT / "data" / "splits" / "test_images.jsonl").read_text().splitlines():
            r = json.loads(line)
            rows[r["id"]] = {"id": r["id"], "species": r["species"], "message": r["message"], "gold": r["gold"],
                             "image_path": r.get("image_path"), "answers": {}}
        for sysname in SYSTEMS:
            p = ROOT / "results" / f"{sysname}.jsonl"
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                a = json.loads(line)
                if a["id"] in rows:
                    try:
                        parsed = json.loads(a["raw"]) if a.get("raw") else None
                    except Exception:
                        parsed = None
                    rows[a["id"]]["answers"][sysname] = {
                        "pred_urgency": a.get("pred_urgency"), "urgency_correct": a.get("urgency_correct"),
                        "parse_error": a.get("parse_error"), "parse_error_reason": a.get("parse_error_reason"),
                        "image_finding": (parsed or {}).get("image_finding") if isinstance(parsed, dict) else None,
                        "conditions": ((parsed or {}).get("conditions") or [])[:3] if isinstance(parsed, dict) else [],
                        "latency_ms": a.get("latency_ms"), "cost_usd": a.get("cost_usd"), "model": a.get("model"),
                        "raw": (a.get("raw") or "")[:1200]}
        _REPLAY = rows
    return _REPLAY


def _has_image(r: dict) -> bool:
    return bool(r.get("image_path")) and (ROOT / r["image_path"]).exists()


@router.get("/replay")
def replay(pick: str = "wins", n: int = 1, seed: int | None = None):
    """pick=wins: fine-tune right, base wrong · any: random · losses: fine-tune wrong (honesty)."""
    rows = [r for r in _load().values() if "sft" in r["answers"] and "base" in r["answers"] and _has_image(r)]
    if pick == "wins":
        rows = [r for r in rows if r["answers"]["sft"]["urgency_correct"] and not r["answers"]["base"]["urgency_correct"]]
    elif pick == "losses":
        rows = [r for r in rows if not r["answers"]["sft"]["urgency_correct"]]
    if not rows:
        raise HTTPException(404, "no recorded rows (images not present locally?)")
    rng = random.Random(seed)
    out = rng.sample(rows, min(n, len(rows)))
    stats = {s: sum(1 for r in _load().values() if r["answers"].get(s, {}).get("urgency_correct")) for s in SYSTEMS}
    total = sum(1 for r in _load().values() if "sft" in r["answers"])
    return {"recorded": True, "note": "Recorded answers from the locked test set run (the LoRA deployment is shut down).",
            "image_rows": total, "correct_by_system": stats,
            "rows": [{**{k: v for k, v in r.items() if k != "image_path"}, "image_url": f"/replay/img/{r['id']}"} for r in out]}


@router.get("/replay/img/{rid}")
def replay_img(rid: str):
    r = _load().get(rid)
    if not r or not _has_image(r):
        raise HTTPException(404)
    return FileResponse(ROOT / r["image_path"])  # local demo only; the image set is not redistributed


# ------------------------------------------------------------------ adherence (vet / insurer-facing)
def adherence(s, pet_id: int) -> dict:
    qs = s.exec(select(den.Quest).where(den.Quest.pet_id == pet_id, den.Quest.kind.in_(["immediate", "med", "recheck", "contagion"]))).all()
    due = [q for q in qs if q.status == "done" or (q.due_at and den.aware(q.due_at) < den.now())]
    if not due:
        return {"score": None, "done": 0, "due": 0, "on_time": 0}
    done = [q for q in due if q.status == "done"]
    on_time = [q for q in done if not q.due_at or den.aware(q.done_at) <= den.aware(q.due_at) + timedelta(hours=2)]
    return {"score": round(100 * len(on_time) / len(due)), "done": len(done), "due": len(due), "on_time": len(on_time)}


@router.get("/pets/{pet_id}/adherence")
def pet_adherence(pet_id: int):
    with store.session() as s:
        if not s.get(Pet, pet_id):
            raise HTTPException(404)
        return adherence(s, pet_id)


# ------------------------------------------------------------------ dog walker: one helper, many Dens
@router.get("/walker/{name}")
def walker(name: str):
    """Everything a helper with this name has to do today, across every Den they belong to, ordered by due time."""
    with store.session() as s:
        mems = [m for m in s.exec(select(den.Member)).all() if m.name.lower() == name.lower()]
        if not mems:
            raise HTTPException(404, "no member with that name")
        cutoff = den.now() + timedelta(hours=24)
        stops = []
        for m in mems:
            d = s.get(den.Den, m.den_id)
            qs = s.exec(select(den.Quest).where(den.Quest.den_id == d.id, den.Quest.status == "open")).all()
            qs = [q for q in qs if (q.assignee_id in (None, m.id)) and (not q.due_at or den.aware(q.due_at) <= cutoff)
                  and q.kind != "watch"]
            stops.append({"den_id": d.id, "den": d.name, "neighborhood": d.neighborhood, "member_id": m.id,
                          "emergency": den.emergency_state(s, d.id) is not None,
                          "tasks": [{"id": q.id, "pet": s.get(Pet, q.pet_id).name, "title": q.title, "kind": q.kind,
                                     "due_at": den.aware(q.due_at).isoformat() if q.due_at else None,
                                     "mine": q.assignee_id == m.id} for q in sorted(qs, key=lambda q: (q.due_at is None, den.aware(q.due_at) if q.due_at else 0))]})
        return {"walker": name, "stops": stops, "dens": len(stops), "tasks": sum(len(x["tasks"]) for x in stops)}
