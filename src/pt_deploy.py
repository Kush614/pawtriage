"""Create / poll / delete the on-demand deployment that serves the fine-tuned LoRA (live merge).

  .venv/bin/python src/pt_deploy.py create <accounts/ACC/models/MODEL_ID>
  .venv/bin/python src/pt_deploy.py status
  .venv/bin/python src/pt_deploy.py delete
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv, set_key

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
API = "https://api.fireworks.ai/v1"
STATE = ROOT / "results" / "deployment.json"
# 4x B300 shape for glm-5p3-flash (smallest available on this account, 2026-11-02).
SHAPE = os.environ.get("PT_DEPLOY_SHAPE", "accounts/fireworks/deploymentShapes/ewt0b3n7")
GPU = {"type": "NVIDIA_B300_288GB", "count": 4, "usd_per_gpu_hour": 15.0}


def h():
    return {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}", "Content-Type": "application/json"}


def account() -> str:
    r = httpx.get(f"{API}/accounts", headers=h(), timeout=30)
    r.raise_for_status()
    return r.json()["accounts"][0]["name"].split("/")[-1]


def create(model: str) -> dict:
    acc = account()
    body = {"baseModel": model, "deploymentShape": SHAPE, "minReplicaCount": 1, "maxReplicaCount": 1,
            "displayName": "pawtriage-sft"}
    r = httpx.post(f"{API}/accounts/{acc}/deployments", headers=h(), json=body, timeout=60)
    if r.status_code >= 400:
        raise SystemExit(f"create failed {r.status_code}: {r.text[:800]}")
    d = r.json()
    st = {"account": acc, "model": model, "deployment": d["name"], "created_at": time.time(), "gpu": GPU,
          "shape": SHAPE}
    STATE.write_text(json.dumps(st, indent=1))
    return st


def status() -> dict:
    st = json.loads(STATE.read_text())
    r = httpx.get(f"{API}/{st['deployment']}", headers=h(), timeout=30)
    r.raise_for_status()
    d = r.json()
    return {"state": d.get("state"), "status": d.get("status"), "replicas": d.get("replicaCount"), **st}


def wait_ready(timeout_s: int = 3600) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        s = status()
        print(time.strftime("%H:%M:%S"), s["state"], flush=True)
        if s["state"] == "READY":
            return s
        if s["state"] in ("FAILED", "DELETED", "DELETING"):
            raise SystemExit(f"deployment {s['state']}: {s.get('status')}")
        time.sleep(30)
    raise SystemExit("timeout waiting for deployment")


def pick_model_string() -> str:
    """Live-merge deployments answer to the model name; fall back to model#deployment."""
    st = json.loads(STATE.read_text())
    for cand in (st["model"], f"{st['model']}#{st['deployment']}"):
        r = httpx.post("https://api.fireworks.ai/inference/v1/chat/completions", headers=h(), timeout=120,
                       json={"model": cand, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 64})
        print(cand, r.status_code, r.text[:200], flush=True)
        if r.status_code == 200:
            set_key(str(ROOT / ".env"), "FIREWORKS_DEPLOYMENT_MODEL", cand, quote_mode="never")
            return cand
    raise SystemExit("no model string worked")


def delete() -> None:
    st = json.loads(STATE.read_text())
    r = httpx.delete(f"{API}/{st['deployment']}", headers=h(), timeout=60)
    print(r.status_code, r.text[:300])
    st["deleted_at"] = time.time()
    st["hours_up"] = round((st["deleted_at"] - st["created_at"]) / 3600, 2)
    STATE.write_text(json.dumps(st, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "create":
        print(create(sys.argv[2]))
        wait_ready()
        print(pick_model_string())
    elif cmd == "status":
        print(json.dumps(status(), indent=1))
    elif cmd == "delete":
        delete()
