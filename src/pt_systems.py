"""The six scored systems (§7). Each returns result rows written to results/<name>.jsonl."""
from __future__ import annotations

import csv
import io
import os

import pt_data as D
import pt_safety
from pt_cost import call_cost
from pt_prompt import build_messages, parse_output
from pt_score import run_system

BASE = os.environ.get("FIREWORKS_BASE_MODEL", "accounts/fireworks/models/glm-5p3-flash")


def test_rows(include_video: bool = True) -> list[dict]:
    if D.check_lock():
        rows = D.read_jsonl(D.SPLITS / "test.jsonl") + D.read_jsonl(D.SPLITS / "test_images.jsonl")
    else:  # pre-lock: score candidates (model calls cache by row id, so re-scoring after review is free)
        print("WARNING: test set not locked yet; scoring candidates")
        rows = D.read_jsonl(D.SPLITS / "test.candidate.jsonl") + D.read_jsonl(D.SPLITS / "test_images.candidate.jsonl")
    if include_video:
        rows += D.read_jsonl(D.SPLITS / "test_video.jsonl")
    return rows


def rubric_text() -> str:
    """Compact rubric table for the harness system prompt (text + image rows)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["scenario", "species", "pattern", "urgency", "system", "conditions", "red_flags", "immediate",
                "watch_for", "prevent", "recheck"])
    for r in D.load_rubric():
        w.writerow([r["scenario_id"], r["species"], r["pattern"], r["urgency"], r["system"], "|".join(r["conditions"]),
                    "|".join(r["red_flags"]), "|".join(r["plan_immediate"]), "|".join(r["plan_watch_for"]),
                    "|".join(r["plan_prevent"]), r["recheck"]])
    return buf.getvalue()


async def base(llm, rows):
    return await run_system(llm, "base", BASE, rows)


async def harness(llm, rows):
    return await run_system(llm, "harness_rubric", BASE, rows, rubric_text=rubric_text())


async def frontier(llm, rows):
    return await run_system(llm, "frontier", os.environ.get("OPENROUTER_FRONTIER_MODEL") or "openai/gpt-6.1-terra-pro", rows)


def sft_model() -> str:
    m = os.environ.get("FIREWORKS_DEPLOYMENT_MODEL")
    if not m:
        raise RuntimeError("FIREWORKS_DEPLOYMENT_MODEL not set (deploy the LoRA first)")
    return m


async def sft(llm, rows, model: str | None = None, name: str = "sft"):
    return await run_system(llm, name, model or sft_model(), rows)


async def _safety_post(pred, row, res):
    p, audit = pt_safety.apply(pred, row["species"], row["message"])
    return p, {"audit": audit, "removed_harmful": int(any(a["rule"] == "never_list" for a in audit))}


async def sft_safety(llm, rows, model: str | None = None):
    """Same answers as `sft` + the deterministic safety layer, computed offline from results/sft.jsonl,
    so sft vs sft_safety differ only by the rules (no second, noisy model call)."""
    import shutil
    from pt_prompt import parse_output
    from pt_score import RESULTS, load_results, score_row, write_results
    by_id = {r["id"]: r for r in rows}
    out = []
    for r in load_results("sft"):
        row = by_id.get(r["id"])
        if row is None:
            continue
        pred, err = parse_output(r["raw"])
        p, audit = pt_safety.apply(pred, row["species"], row["message"])
        s = score_row(row, p, err if p is None else None)
        s.update({k: r[k] for k in ("model", "prompt_hash", "latency_ms", "input_tokens", "output_tokens", "cost_usd", "raw")},
                 system_name="sft_safety", audit=audit, cached=False,
                 removed_harmful=int(any(a["rule"] == "never_list" for a in audit)))
        out.append(s)
    if not out:
        raise RuntimeError("score `sft` first")
    write_results("sft_safety", out)
    if (RESULTS / "sft.meta.json").exists():
        shutil.copy(RESULTS / "sft.meta.json", RESULTS / "sft_safety.meta.json")
    return out


def make_router_post(llm):
    large = pt_safety.large_model()

    async def post(pred, row, res):
        p, audit = pt_safety.apply(pred, row["species"], row["message"])
        extra = {"audit": audit, "routed": "small"}
        if pt_safety.route(p) == "large":
            r2 = await llm.chat(large, build_messages(row), temperature=0.0, max_tokens=1024,
                                cache_key=f"router|{row['id']}|{large}")
            p2, _ = parse_output(r2["text"])
            p2, audit2 = pt_safety.apply(p2, row["species"], row["message"])
            p = pt_safety.merge_routed(p, p2)
            extra.update(routed="large", audit=audit + [{"rule": "router", "to": large}] + audit2,
                         extra_cost_usd=call_cost(large, r2["input_tokens"], r2["output_tokens"]),
                         router_cost_usd=call_cost(large, r2["input_tokens"], r2["output_tokens"]),
                         extra_latency_ms=r2["latency_ms"])
        return p, extra
    return post


async def sft_router(llm, rows, model: str | None = None):
    return await run_system(llm, "sft_router", model or sft_model(), rows, post=make_router_post(llm))
