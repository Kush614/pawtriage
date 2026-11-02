"""Launch SFT; on capacity_exhausted, delete our orphan sessions, wait, retry. Prints one status line per event."""
import json, os, subprocess, sys, time
from pathlib import Path
import httpx
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pt_sft
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
H = {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}
ACC = "kushise27-y1gkfyqxpn"

def cleanup():
    r = httpx.get(f"https://api.fireworks.ai/v1/accounts/{ACC}/trainingSessions?pageSize=50", headers=H, timeout=30).json()
    for t in r.get("trainingSessions", []):
        httpx.delete(f"https://api.fireworks.ai/v1/{t['name']}", headers=H, timeout=30)

for attempt in range(40):
    meta = pt_sft.launch()
    log = Path(meta["log_dir"]) / "stdout.log"
    print(f"attempt {attempt} run {meta['run_id']} pid {meta['pid']}", flush=True)
    while True:
        time.sleep(10)
        txt = log.read_text(errors="replace") if log.exists() else ""
        if "Step " in txt:
            print(f"TRAINING STARTED run {meta['run_id']}", flush=True)
            sys.exit(0)
        alive = subprocess.run(["kill", "-0", str(meta["pid"])], capture_output=True).returncode == 0
        if not alive:
            reason = "capacity" if "capacity_exhausted" in txt else "other"
            print(f"attempt {attempt} died: {reason} :: {[l for l in txt.splitlines() if 'Error' in l][-1:]}", flush=True)
            if reason != "capacity":
                sys.exit(1)
            cleanup()
            time.sleep(60)
            break
