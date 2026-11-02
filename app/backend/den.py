"""PawTown "Dens": shared care coordination for multi-pet, multi-caregiver households.

Every game element maps to a health behaviour:
  care-plan steps -> quests (adherence) · rechecks -> timed quests (early catch of worsening)
  meds -> recurring quests with a double-dose guard · contagious case -> linked quests for every pet
  emergency -> game freezes, only the route to care remains · notes -> caregiver handoff in the vet packet
Points reward *care actions*, never "staying home"; nothing is scored during an emergency except getting to the vet.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import Field, SQLModel, select

from . import store
from .store import Case, Pet

ROOT = Path(__file__).resolve().parents[2]
router = APIRouter()

CONTAGIOUS = {  # condition / image finding -> (label, zoonotic: can spread to people)
    "dermatophytosis": ("ringworm", True), "cat_ringworm": ("ringworm", True),
    "sarcoptic_mange": ("mange", True), "cat_scabies_mange": ("mange", True),
    "infectious_tracheobronchitis": ("kennel cough", False), "parvovirus": ("parvovirus", False),
    "intestinal_parasites": ("intestinal parasites", True), "ear_mites": ("ear mites", False),
    "flea_allergy_dermatitis": ("fleas", False), "cat_flea_allergy": ("fleas", False),
    "fungal_infection": ("fungal skin infection", True), "dog_fungal_infection": ("fungal skin infection", True),
}
POINTS = {"immediate": 10, "recheck": 15, "med": 5, "contagion": 10, "preventive": 8, "watch": 3}
DOSE_WINDOW_H = 0.75  # a med instance done again within 75% of its interval = possible double dose
COMMUNITY_MIN = 3     # privacy: never show a neighborhood signal below this many reports


def now() -> datetime:
    return datetime.now(timezone.utc)


def aware(d: datetime | None) -> datetime | None:
    return d if d is None or d.tzinfo else d.replace(tzinfo=timezone.utc)


def nice(x: str) -> str:
    return (x or "").replace("_", " ")


# ------------------------------------------------------------------ models
class Den(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    neighborhood: str = "Mission"


class Member(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    den_id: int = Field(foreign_key="den.id")
    name: str
    role: str = "parent"  # parent | coparent | helper
    emoji: str = "🧑"


class DenPet(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    den_id: int = Field(foreign_key="den.id")
    pet_id: int = Field(foreign_key="pet.id")


class Quest(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    den_id: int
    pet_id: int
    case_id: Optional[int] = None
    kind: str          # immediate | watch | recheck | med | contagion | preventive
    item: str          # vocab item or free label
    title: str
    assignee_id: Optional[int] = None
    due_at: Optional[datetime] = None
    every_hours: Optional[float] = None   # recurring (meds)
    status: str = "open"                  # open | done
    done_by: Optional[int] = None
    done_at: Optional[datetime] = None
    points: int = 0
    created_at: datetime = Field(default_factory=now)


class Event(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    den_id: int
    at: datetime = Field(default_factory=now)
    member_id: Optional[int] = None
    pet_id: Optional[int] = None
    case_id: Optional[int] = None
    kind: str          # quest_done | note | warning | contagion | emergency | at_vet | case | badge
    text: str
    data: str = "{}"


def init() -> None:
    SQLModel.metadata.create_all(store.engine)


# ------------------------------------------------------------------ helpers
def den_of_pet(s, pet_id: int) -> Optional[int]:
    dp = s.exec(select(DenPet).where(DenPet.pet_id == pet_id)).first()
    return dp.den_id if dp else None


def log(s, den_id, kind, text, member_id=None, pet_id=None, case_id=None, data=None):
    s.add(Event(den_id=den_id, kind=kind, text=text, member_id=member_id, pet_id=pet_id, case_id=case_id,
                data=json.dumps(data or {})))


def latest_result(c: Case) -> dict:
    turns = json.loads(c.turns)
    return turns[-1]["result"] if turns else {}


def contagion_of(result: dict) -> tuple[str, bool] | None:
    keys = list(result.get("conditions", [])[:2]) + [result.get("image_finding") or ""]
    for k in keys:
        if k in CONTAGIOUS:
            return CONTAGIOUS[k]
    return None


def on_case(case: Case, pet: Pet) -> None:
    """Called after every triage / follow-up: turn the plan into Den quests (+ contagion fan-out)."""
    with store.session() as s:
        den_id = den_of_pet(s, pet.id)
        if not den_id:
            return
        r = latest_result(case)
        if not r:
            return
        # close stale open plan quests of this case (a follow-up replaces the plan)
        for q in s.exec(select(Quest).where(Quest.case_id == case.id, Quest.status == "open",
                                            Quest.kind.in_(["immediate", "watch", "recheck"]))).all():
            s.delete(q)
        _add = s.add
        open_keys = {(q.pet_id, q.kind, q.item) for q in s.exec(select(Quest).where(Quest.den_id == den_id, Quest.status == "open")).all()}

        def add(obj):  # same open quest already on the board (e.g. a repeat scan) -> don't duplicate it
            if isinstance(obj, Quest):
                k = (obj.pet_id, obj.kind, obj.item)
                if k in open_keys:
                    return
                open_keys.add(k)
            _add(obj)
        s.add = add
        urg = r.get("urgency")
        log(s, den_id, "case", f"{pet.name}: {nice(urg)}" + (f" (possible {nice(r['conditions'][0])})" if r.get("conditions") else ""),
            pet_id=pet.id, case_id=case.id, data={"urgency": urg})
        if urg == "emergency_now":
            log(s, den_id, "emergency", f"EMERGENCY for {pet.name}: go to an emergency vet now", pet_id=pet.id, case_id=case.id)
        for it in r.get("plan", {}).get("immediate", []):
            s.add(Quest(den_id=den_id, pet_id=pet.id, case_id=case.id, kind="immediate", item=it,
                        title=nice(it).capitalize(), due_at=now() + timedelta(hours=2), points=POINTS["immediate"]))
        for it in r.get("plan", {}).get("watch_for", [])[:4]:
            s.add(Quest(den_id=den_id, pet_id=pet.id, case_id=case.id, kind="watch", item=it,
                        title=f"Watch {pet.name} for {nice(it)}", points=POINTS["watch"]))
        if case.recheck_due:
            s.add(Quest(den_id=den_id, pet_id=pet.id, case_id=case.id, kind="recheck", item="recheck",
                        title=f"Recheck {pet.name} and report a change", due_at=case.recheck_due,
                        points=POINTS["recheck"]))
        cg = contagion_of(r)
        if cg:
            label, zoonotic = cg
            others = [s.get(Pet, dp.pet_id) for dp in s.exec(select(DenPet).where(DenPet.den_id == den_id)).all()
                      if dp.pet_id != pet.id]
            log(s, den_id, "contagion", f"Possible {label} in {pet.name}: protecting {len(others)} other pet(s)"
                + (" and the humans" if zoonotic else ""), pet_id=pet.id, case_id=case.id, data={"label": label})
            imm = set(r.get("plan", {}).get("immediate", []))
            if "isolate_from_other_pets" not in imm:
                s.add(Quest(den_id=den_id, pet_id=pet.id, case_id=case.id, kind="contagion", item="isolate_from_other_pets",
                            title=f"Keep {pet.name} apart from the other pets", points=POINTS["contagion"]))
            for o in others:
                s.add(Quest(den_id=den_id, pet_id=o.id, case_id=case.id, kind="contagion", item="check_for_signs",
                            title=f"Check {o.name} for signs of {label}", due_at=now() + timedelta(hours=24),
                            points=POINTS["contagion"]))
                s.add(Quest(den_id=den_id, pet_id=o.id, case_id=case.id, kind="contagion", item="treat_all_pets_in_household",
                            title=f"Ask the vet whether {o.name} needs treating too", points=POINTS["contagion"]))
            if zoonotic and "wash_hands_after_contact" not in imm:
                s.add(Quest(den_id=den_id, pet_id=pet.id, case_id=case.id, kind="contagion", item="wash_hands_after_contact",
                            title=f"Everyone: wash hands after touching {pet.name}; watch your own skin",
                            points=POINTS["contagion"]))
        s.commit()


def emergency_state(s, den_id: int) -> dict | None:
    pets = [dp.pet_id for dp in s.exec(select(DenPet).where(DenPet.den_id == den_id)).all()]
    for c in s.exec(select(Case).where(Case.pet_id.in_(pets), Case.status == "open")).all():
        r = latest_result(c)
        if r.get("urgency") == "emergency_now":
            at_vet = s.exec(select(Event).where(Event.case_id == c.id, Event.kind == "at_vet")).first()
            if not at_vet:
                pet = s.get(Pet, c.pet_id)
                return {"case_id": c.id, "pet_id": pet.id, "pet": pet.name, "red_flags": r.get("red_flags", []),
                        "conditions": r.get("conditions", [])}
    return None


def member_points(s, den_id: int) -> dict:
    pts = Counter()
    for q in s.exec(select(Quest).where(Quest.den_id == den_id, Quest.status == "done")).all():
        if q.done_by:
            pts[q.done_by] += q.points
    for e in s.exec(select(Event).where(Event.den_id == den_id, Event.kind.in_(["note", "at_vet"]))).all():
        if e.member_id:
            pts[e.member_id] += 25 if e.kind == "at_vet" else 3
    return dict(pts)


def badges(s, den_id: int) -> list[dict]:
    out = []
    ev = s.exec(select(Event).where(Event.den_id == den_id)).all()
    qs = s.exec(select(Quest).where(Quest.den_id == den_id)).all()
    if any(e.kind == "at_vet" for e in ev):
        out.append({"id": "listened", "icon": "🏥", "name": "Went when it mattered", "why": "got to the vet when PawTriage said emergency"})
    contributors = {q.done_by for q in qs if q.done_by} | {e.member_id for e in ev if e.kind == "note" and e.member_id}
    if len(contributors) >= 3:
        out.append({"id": "team", "icon": "🤝", "name": "Team Den", "why": "3+ caregivers pitched in"})
    by_case = defaultdict(list)
    for q in qs:
        if q.kind == "immediate" and q.case_id:
            by_case[q.case_id].append(q)
    if any(all(q.status == "done" for q in v) and v for v in by_case.values()):
        out.append({"id": "finisher", "icon": "✅", "name": "Plan finisher", "why": "every 'do now' step of a case done"})
    cq = [q for q in qs if q.kind == "contagion"]
    if cq and all(q.status == "done" for q in cq):
        out.append({"id": "contained", "icon": "🛡️", "name": "Contagion contained", "why": "every contagion step done"})
    if any(q.kind == "recheck" and q.status == "done" and q.due_at and aware(q.done_at) <= aware(q.due_at) + timedelta(hours=6) for q in qs):
        out.append({"id": "ontime", "icon": "⏰", "name": "On-time recheck", "why": "followed up when asked"})
    if any(e.kind == "warning" for e in ev):
        out.append({"id": "caught", "icon": "💊", "name": "Double dose caught", "why": "the Den log stopped a repeat dose"})
    return out


def quest_view(q: Quest, members: dict) -> dict:
    return {"id": q.id, "pet_id": q.pet_id, "case_id": q.case_id, "kind": q.kind, "item": q.item, "title": q.title,
            "assignee": members.get(q.assignee_id), "assignee_id": q.assignee_id, "status": q.status,
            "done_by": members.get(q.done_by), "done_at": q.done_at.isoformat() if q.done_at else None,
            "due_at": aware(q.due_at).isoformat() if q.due_at else None, "points": q.points, "every_hours": q.every_hours}


def den_view(den_id: int, as_member: int | None = None) -> dict:
    with store.session() as s:
        den = s.get(Den, den_id)
        if not den:
            raise HTTPException(404, "den not found")
        mem = s.exec(select(Member).where(Member.den_id == den_id)).all()
        mdict = {m.id: {"id": m.id, "name": m.name, "role": m.role, "emoji": m.emoji} for m in mem}
        viewer = s.get(Member, as_member) if as_member else None
        helper = bool(viewer and viewer.role == "helper")
        pets = []
        for dp in s.exec(select(DenPet).where(DenPet.den_id == den_id)).all():
            p = s.get(Pet, dp.pet_id)
            cases = s.exec(select(Case).where(Case.pet_id == p.id, Case.status == "open").order_by(Case.id.desc())).all()
            open_case = cases[0] if cases else None
            r = latest_result(open_case) if open_case else {}
            from .showcase import adherence as _adh
            pets.append({"adherence": None if helper else _adh(s, p.id), "id": p.id, "name": p.name, "species": p.species, "breed": p.breed, "age_years": p.age_years,
                         "meds": json.loads(p.meds), "conditions": json.loads(p.conditions),
                         "case": None if not open_case else {
                             "id": open_case.id, "urgency": r.get("urgency"), "red_flags": r.get("red_flags", []),
                             "conditions": [] if helper else r.get("conditions", []),
                             "image_finding": None if helper else r.get("image_finding"),
                             "recheck_due": aware(open_case.recheck_due).isoformat() if open_case.recheck_due else None,
                             "cost_usd": (json.loads(open_case.turns)[-1].get("cost_usd") if open_case.turns != "[]" else None),
                             "latency_ms": (json.loads(open_case.turns)[-1].get("latency_ms") if open_case.turns != "[]" else None),
                             "mode": (json.loads(open_case.turns)[-1].get("mode") if open_case.turns != "[]" else None),
                             "media": None if helper else next((t.get("image_url") or t.get("sheet_url") for t in json.loads(open_case.turns) if t.get("image_url") or t.get("sheet_url")), None)}})
        qs = s.exec(select(Quest).where(Quest.den_id == den_id).order_by(Quest.status.desc(), Quest.due_at)).all()
        quests = [quest_view(q, mdict) for q in qs]
        if helper:  # helpers: today's open tasks only
            cutoff = now() + timedelta(hours=24)
            quests = [q for q in quests if q["status"] == "open" and (not q["due_at"] or datetime.fromisoformat(q["due_at"]) <= cutoff)]
        events = s.exec(select(Event).where(Event.den_id == den_id).order_by(Event.id.desc())).all()[:40]
        ev = [{"at": aware(e.at).isoformat(), "kind": e.kind, "text": e.text, "member": mdict.get(e.member_id),
               "pet_id": e.pet_id} for e in events if not (helper and e.kind in ("case", "contagion"))]
        pts = member_points(s, den_id)
        return {"id": den.id, "name": den.name, "neighborhood": den.neighborhood,
                "viewer": mdict.get(as_member), "helper_view": helper,
                "members": [{**mdict[m.id], "points": pts.get(m.id, 0)} for m in mem],
                "pets": pets, "quests": quests, "events": ev,
                "emergency": emergency_state(s, den_id), "badges": [] if helper else badges(s, den_id)}


# ------------------------------------------------------------------ API
class DenIn(BaseModel):
    name: str
    neighborhood: str = "Mission"


class MemberIn(BaseModel):
    name: str
    role: str = "parent"
    emoji: str = "🧑"


class PetLink(BaseModel):
    pet_id: int


class MedIn(BaseModel):
    pet_id: int
    name: str
    every_hours: float = 24
    assignee_id: Optional[int] = None


class AssignIn(BaseModel):
    member_id: Optional[int]


class DoneIn(BaseModel):
    member_id: int
    force: bool = False


class NoteIn(BaseModel):
    member_id: int
    pet_id: int
    text: str


@router.post("/dens")
def create_den(d: DenIn):
    with store.session() as s:
        den = Den(**d.model_dump())
        s.add(den)
        s.commit()
        s.refresh(den)
        return {"id": den.id}


@router.get("/dens")
def list_dens():
    with store.session() as s:
        return [{"id": d.id, "name": d.name, "neighborhood": d.neighborhood} for d in s.exec(select(Den)).all()]


@router.get("/dens/{den_id}")
def get_den(den_id: int, as_member: Optional[int] = None):
    return den_view(den_id, as_member)


@router.post("/dens/{den_id}/members")
def add_member(den_id: int, m: MemberIn):
    if m.role not in ("parent", "coparent", "helper"):
        raise HTTPException(400, "role must be parent|coparent|helper")
    with store.session() as s:
        mm = Member(den_id=den_id, **m.model_dump())
        s.add(mm)
        s.commit()
        s.refresh(mm)
        return {"id": mm.id}


@router.post("/dens/{den_id}/pets")
def link_pet(den_id: int, body: PetLink):
    with store.session() as s:
        if not s.get(Pet, body.pet_id):
            raise HTTPException(404, "pet not found")
        if not s.exec(select(DenPet).where(DenPet.den_id == den_id, DenPet.pet_id == body.pet_id)).first():
            s.add(DenPet(den_id=den_id, pet_id=body.pet_id))
            s.commit()
    return {"ok": True}


@router.post("/dens/{den_id}/meds")
def add_med(den_id: int, m: MedIn):
    with store.session() as s:
        pet = s.get(Pet, m.pet_id)
        q = Quest(den_id=den_id, pet_id=m.pet_id, kind="med", item=m.name, title=f"Give {pet.name} {m.name}",
                  every_hours=m.every_hours, due_at=now(), assignee_id=m.assignee_id, points=POINTS["med"])
        s.add(q)
        s.commit()
        s.refresh(q)
        return {"id": q.id}


@router.post("/quests/{qid}/assign")
def assign(qid: int, body: AssignIn):
    with store.session() as s:
        q = s.get(Quest, qid)
        if not q:
            raise HTTPException(404)
        q.assignee_id = body.member_id
        s.add(q)
        s.commit()
    return {"ok": True}


@router.post("/quests/{qid}/done")
def done(qid: int, body: DoneIn):
    with store.session() as s:
        q = s.get(Quest, qid)
        if not q:
            raise HTTPException(404)
        m = s.get(Member, body.member_id)
        pet = s.get(Pet, q.pet_id)
        if q.kind == "med":
            last = s.exec(select(Quest).where(Quest.den_id == q.den_id, Quest.pet_id == q.pet_id, Quest.kind == "med",
                                              Quest.item == q.item, Quest.status == "done").order_by(Quest.done_at.desc())).first()
            window = timedelta(hours=(q.every_hours or 24) * DOSE_WINDOW_H)
            if last and now() - aware(last.done_at) < window and not body.force:
                by = s.get(Member, last.done_by)
                mins = int((now() - aware(last.done_at)).total_seconds() // 60)
                log(s, q.den_id, "warning", f"Stopped a possible double dose: {by.name if by else 'someone'} already gave "
                    f"{pet.name} {q.item} {mins} min ago", member_id=body.member_id, pet_id=q.pet_id)
                s.commit()
                return {"ok": False, "double_dose": True, "given_by": by.name if by else None, "minutes_ago": mins}
        if q.status == "done":
            return {"ok": True, "already": True}
        if emergency_state(s, q.den_id) and not (q.kind == "immediate" and q.case_id == emergency_state(s, q.den_id)["case_id"]):
            q.points = 0  # emergency: the game is paused, only the emergency's own steps count
        q.status, q.done_by, q.done_at = "done", body.member_id, now()
        s.add(q)
        log(s, q.den_id, "quest_done", f"{m.name} · {q.title}", member_id=body.member_id, pet_id=q.pet_id, case_id=q.case_id)
        if q.kind == "med" and q.every_hours:  # schedule the next dose
            s.add(Quest(den_id=q.den_id, pet_id=q.pet_id, kind="med", item=q.item, title=q.title,
                        every_hours=q.every_hours, assignee_id=q.assignee_id, points=q.points,
                        due_at=now() + timedelta(hours=q.every_hours)))
        s.commit()
        return {"ok": True, "points": q.points}


@router.post("/dens/{den_id}/notes")
def note(den_id: int, n: NoteIn):
    with store.session() as s:
        m = s.get(Member, n.member_id)
        c = s.exec(select(Case).where(Case.pet_id == n.pet_id).order_by(Case.id.desc())).first()
        log(s, den_id, "note", f"{m.emoji} {m.name}: {n.text}", member_id=n.member_id, pet_id=n.pet_id,
            case_id=c.id if c else None)
        s.commit()
    return {"ok": True}


@router.post("/cases/{case_id}/at_vet")
def at_vet(case_id: int, member_id: int):
    with store.session() as s:
        c = s.get(Case, case_id)
        den_id = den_of_pet(s, c.pet_id)
        m = s.get(Member, member_id)
        log(s, den_id, "at_vet", f"{m.name} is taking {s.get(Pet, c.pet_id).name} to the vet", member_id=member_id,
            pet_id=c.pet_id, case_id=case_id)
        s.commit()
    return {"ok": True}


def caregiver_notes(case_id: int) -> list[str]:
    with store.session() as s:
        ev = s.exec(select(Event).where(Event.case_id == case_id, Event.kind.in_(["note", "quest_done", "warning", "at_vet"]))
                    .order_by(Event.id)).all()
        return [f"{aware(e.at).strftime('%m-%d %H:%M')} UTC · {e.text}" for e in ev]


@router.get("/community/weather")
def weather():
    """Neighborhood contagion signal over the last 7 days; suppressed below COMMUNITY_MIN reports (privacy)."""
    with store.session() as s:
        since = now() - timedelta(days=7)
        cnt = Counter()
        for c in s.exec(select(Case)).all():
            if aware(c.created_at) < since:
                continue
            den_id = den_of_pet(s, c.pet_id)
            den = s.get(Den, den_id) if den_id else None
            cg = contagion_of(latest_result(c))
            if den and cg:
                cnt[(den.neighborhood, cg[0])] += 1
        out = []
        for (hood, label), n in cnt.items():
            out.append({"neighborhood": hood, "label": label, "reports": n if n >= COMMUNITY_MIN else None,
                        "shown": n >= COMMUNITY_MIN,
                        "note": f"{n} {label} reports this week" if n >= COMMUNITY_MIN else
                        f"below the {COMMUNITY_MIN}-report privacy threshold"})
        return {"min_reports": COMMUNITY_MIN, "signals": out}


@router.get("/map/places")
def places():
    p = ROOT / "data" / "map" / "sf_places.json"
    return json.loads(p.read_text()) if p.exists() else {"source": None, "vets": [], "dog_parks": []}


@router.get("/pawtown/config")
def pawtown_config():
    import os
    return {"scanner_url": os.environ.get("SCANNER_URL")}  # set without the access key; the owner adds ?key=
