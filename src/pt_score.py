"""Per-row scoring (§6) and async system runner (§7)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pt_cost import call_cost
from pt_prompt import ROOT, URGENCY, build_messages, parse_output, system_prompt_hash, vocab

RESULTS = ROOT / "results"
HIGH_TIERS = {"emergency_now", "vet_today"}


def plan_items(plan: dict) -> set[str]:
    return set(plan.get("immediate", [])) | set(plan.get("watch_for", [])) | set(plan.get("prevent", []))


def f1(gold: list, pred: list) -> float:
    g, p = set(gold), set(pred)
    if not g and not p:
        return 1.0
    tp = len(g & p)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def score_row(row: dict, pred: dict | None, err: str | None) -> dict:
    gold = row["gold"]
    out = {"id": row["id"], "scenario_id": row["scenario_id"], "species": row["species"],
           "modality": row.get("modality", "text"), "gold_urgency": gold["urgency"],
           "parse_error": int(pred is None), "parse_error_reason": err}
    is_img = out["modality"] in ("image", "video")
    if pred is None:
        out.update(urgency_correct=0, system_correct=0, cond_top1_hit=0, cond_set_f1=0.0,
                   plan_recall=0.0, harmful_advice=0, recheck_correct=0, pred_urgency=None,
                   under_triage=int(gold["urgency"] in HIGH_TIERS))
        if is_img:
            out["image_finding_correct"] = 0
        return out
    gi, pi = URGENCY.index(gold["urgency"]), URGENCY.index(pred["urgency"])
    g_plan, p_plan = plan_items(gold["plan"]), plan_items(pred["plan"])
    never = set(vocab()["never"].get(row["species"], []))
    out.update(
        pred_urgency=pred["urgency"],
        urgency_correct=int(gi == pi),
        under_triage=int(gold["urgency"] in HIGH_TIERS and pi < gi),
        system_correct=int(pred["system"] == gold["system"]),
        cond_top1_hit=int(gold["conditions"][0] in pred["conditions"]),
        cond_set_f1=round(f1(gold["conditions"], pred["conditions"]), 4),
        plan_recall=round(len(g_plan & p_plan) / len(g_plan), 4) if g_plan else 1.0,
        harmful_advice=int(bool(p_plan & never)),
        recheck_correct=int(pred["plan"]["recheck"] == gold["plan"]["recheck"]),
    )
    if is_img:
        out["image_finding_correct"] = int(pred.get("image_finding") == gold.get("image_finding"))
    return out


async def run_system(llm, system_name: str, model: str, rows: list[dict], rubric_text: str | None = None,
                     post=None) -> list[dict]:
    """Score `rows` with `model`. `post(pred, row) -> (pred, extra_dict)` hooks safety/routing."""
    phash = system_prompt_hash(rubric_text)

    async def one(row):
        res = await llm.chat(model, build_messages(row, rubric_text), temperature=0.0, max_tokens=1024,
                             cache_key=json.dumps([system_name, row["id"], phash, model]))
        pred, err = parse_output(res["text"])
        extra = {}
        cost = call_cost(model, res["input_tokens"], res["output_tokens"])
        lat = res["latency_ms"]
        if post is not None:
            pred, extra = await post(pred, row, res)
            cost += extra.pop("extra_cost_usd", 0.0)
            lat += extra.pop("extra_latency_ms", 0)
        s = score_row(row, pred, err if pred is None else None)
        s.update(system_name=system_name, model=model, prompt_hash=phash, latency_ms=lat,
                 input_tokens=res["input_tokens"], output_tokens=res["output_tokens"],
                 cost_usd=round(cost, 7), raw=res["text"][:2000], cached=res.get("cached", False), **extra)
        return s

    import time
    t0 = time.perf_counter()
    out = await asyncio.gather(*(one(r) for r in rows))
    wall = time.perf_counter() - t0
    write_results(system_name, out)
    n_live = sum(1 for r in out if not r.get("cached"))
    meta_p = RESULTS / f"{system_name}.meta.json"
    if not meta_p.exists() or n_live > 0.5 * len(out):  # keep the wall time of the last uncached run
        meta_p.write_text(json.dumps({"n": len(out), "wall_s": round(wall, 1), "concurrency": llm.sem._value or 16,
                                      "model": model}))
    return out


def write_results(system_name: str, rows: list[dict]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{system_name}.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def load_results(system_name: str) -> list[dict]:
    p = RESULTS / f"{system_name}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


METRICS = ["urgency_correct", "under_triage", "system_correct", "cond_top1_hit", "cond_set_f1",
           "plan_recall", "harmful_advice", "recheck_correct", "parse_error", "image_finding_correct"]


def headline(rows: list[dict]) -> dict:
    """Means per metric; under_triage is a rate over gold high-urgency rows only."""
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for m in METRICS:
        if m == "under_triage":
            sub = [r[m] for r in rows if r["gold_urgency"] in HIGH_TIERS]
        else:
            sub = [r[m] for r in rows if m in r]
        if sub:
            out[m] = round(sum(sub) / len(sub), 4)
    return out


def by_modality(rows: list[dict]) -> dict:
    return {"text": headline([r for r in rows if r["modality"] == "text"]),
            "image": headline([r for r in rows if r["modality"] == "image"]),
            "all": headline([r for r in rows if r["modality"] in ("text", "image")]),
            "video": headline([r for r in rows if r["modality"] == "video"])}
