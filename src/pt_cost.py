"""Token + $ accounting. Prices live in prices.json (fill from pricing pages; date noted there)."""
from __future__ import annotations

import json

import numpy as np

from pt_prompt import ROOT

PRICES_PATH = ROOT / "src" / "prices.json"


def prices() -> dict:
    return json.loads(PRICES_PATH.read_text())


def price_for(model: str) -> dict:
    p = prices()["models"]
    if model in p:
        return p[model]
    # Deployed LoRA of the base model is billed like the base model (serverless LoRA) unless overridden.
    for k, v in p.items():
        if k != "_default" and model.startswith(k):
            return v
    return p["_default"]


def call_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pr = price_for(model)
    return input_tokens / 1e6 * pr["input_per_m"] + output_tokens / 1e6 * pr["output_per_m"]


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    lat = np.array([r["latency_ms"] for r in rows if r.get("latency_ms") is not None], dtype=float)
    cost = np.array([r.get("cost_usd", 0.0) for r in rows], dtype=float)
    return {"usd_per_1k_chats": round(float(cost.mean() * 1000), 4),
            "p50_latency_ms": round(float(np.percentile(lat, 50))) if len(lat) else None,
            "p95_latency_ms": round(float(np.percentile(lat, 95))) if len(lat) else None}


def deployment_usd_per_1k(system_name: str) -> dict | None:
    """Dedicated-GPU serving cost from measured throughput: $/h / (chats/h at the scoring concurrency)."""
    from pathlib import Path
    dep = ROOT / "results" / "deployment.json"
    meta = ROOT / "results" / f"{system_name}.meta.json"
    if not dep.exists() or not meta.exists():
        return None
    d, m = json.loads(dep.read_text()), json.loads(meta.read_text())
    usd_h = d["gpu"]["count"] * d["gpu"]["usd_per_gpu_hour"]
    chats_per_h = m["n"] / m["wall_s"] * 3600
    return {"usd_per_1k_chats_dedicated": round(usd_h / chats_per_h * 1000, 3), "gpu_usd_per_hour": usd_h,
            "measured_chats_per_hour": round(chats_per_h), "concurrency": m["concurrency"],
            "note": "dedicated 4xB300 at measured throughput; higher concurrency lowers this"}
