"""Vision LoRA SFT on Fireworks serverless training (§8).

Builds OpenAI-format chat rows via build_messages() + gold JSON, shuffles text and image
rows together, and runs the Fireworks cookbook `sft_loop` (serverless=True, LoRA r16/a32,
lr 1e-4, 1 epoch, batch 16) in the separate training venv (.venv-train).

Usage:
  .venv/bin/python src/pt_sft.py build           # write data/splits/sft_train.jsonl
  .venv/bin/python src/pt_sft.py launch [--max-text 1000]   # build + train in background
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv  # noqa: E402

import pt_data as D  # noqa: E402
from pt_prompt import ROOT, build_messages, gold_json, system_prompt_hash  # noqa: E402

load_dotenv(ROOT / ".env")
SFT_PATH = D.SPLITS / "sft_train.jsonl"
COOKBOOK = ROOT / "vendor" / "cookbook"
TRAIN_PY = ROOT / ".venv-train" / "bin" / "python"
BASE_MODEL = os.environ.get("FIREWORKS_BASE_MODEL", "accounts/fireworks/models/glm-5p3-flash")
TOKENIZER = "zai-org/GLM-5.3-Flash"


def to_chat(row: dict) -> dict:
    msgs = build_messages(row)
    is_img = row.get("modality") in ("image", "video_sheet", "video")
    msgs.append({"role": "assistant", "content": gold_json(row["gold"], with_image=is_img)})
    return {"messages": msgs}


def build(max_text: int | None = None, image_ratio: float = 0.4, seed: int = 0) -> dict:
    text = D.read_jsonl(D.SPLITS / "train.jsonl")
    imgs = D.read_jsonl(D.SPLITS / "train_images.jsonl") + D.read_jsonl(D.SPLITS / "train_video.jsonl")
    test = (D.read_jsonl(D.SPLITS / "test.jsonl") + D.read_jsonl(D.SPLITS / "test_images.jsonl")
            + D.read_jsonl(D.SPLITS / "test.candidate.jsonl") + D.read_jsonl(D.SPLITS / "test_images.candidate.jsonl"))
    D.assert_no_overlap(text + imgs, test)
    rng = random.Random(seed)
    if max_text and len(text) > max_text:
        text = rng.sample(text, max_text)
    upsample = 1
    if imgs and len(imgs) / (len(imgs) + len(text)) < image_ratio * 0.75:
        upsample = 2  # spec: upsample image rows x2 if scarce
    rows = text + imgs * upsample
    rng.shuffle(rows)
    with open(SFT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(to_chat(r)) + "\n")
    info = {"n_rows": len(rows), "n_text": len(text), "n_image_unique": len(imgs), "image_upsample": upsample,
            "image_frac": round(len(imgs) * upsample / max(1, len(rows)), 3), "prompt_hash": system_prompt_hash(),
            "path": str(SFT_PATH)}
    (ROOT / "results" / "sft_data_info.json").write_text(json.dumps(info, indent=1))
    return info


RUNNER = r'''
import json, os, sys, logging
sys.path.insert(0, os.environ["COOKBOOK"])


def main():
    import training.recipes.sft_loop as sft_loop
    serverless = os.environ.get("SERVERLESS", "1") == "1"
    extra = {}
    if not serverless:
        extra["trainer"] = sft_loop.TrainerConfig(training_shape_id=os.environ["TRAINING_SHAPE"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = sft_loop.Config(
        serverless=serverless,
        log_path=os.environ["LOG_PATH"],
        base_model=os.environ["BASE_MODEL"],
        tokenizer_model=os.environ["TOKENIZER"],
        dataset=os.environ["DATASET"],
        lora_rank=16, lora_alpha=32,
        max_seq_len=int(os.environ.get("MAX_SEQ_LEN", "16384")),
        epochs=1, batch_size=16, learning_rate=1e-4,
        dcp_save_interval=int(os.environ.get("DCP_EVERY", "25")),
        output_model_id=os.environ["OUTPUT_MODEL_ID"],
        **extra,
    )
    out = sft_loop.main(cfg)
    print(json.dumps({"type": "final", "result": out}, default=str), flush=True)


if __name__ == "__main__":  # required: macOS spawns DataLoader workers by re-importing this file
    main()
'''


def launch(max_text: int | None = None, output_model_id: str | None = None, dedicated: bool = False) -> dict:
    info = build(max_text)
    run_id = time.strftime("%m%d-%H%M")
    output_model_id = output_model_id or f"pawtriage-v1-{run_id}"
    log_dir = ROOT / "results" / "sft_logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = log_dir / "runner.py"
    runner.write_text(RUNNER)
    env = {**os.environ, "COOKBOOK": str(COOKBOOK), "LOG_PATH": str(log_dir / "cookbook"), "BASE_MODEL": BASE_MODEL,
           "TOKENIZER": TOKENIZER, "DATASET": str(SFT_PATH), "OUTPUT_MODEL_ID": output_model_id,
           "SERVERLESS": "0" if dedicated else "1",
           "TRAINING_SHAPE": "accounts/fireworks/trainingShapes/glm-5p3-flash-262k-b300-lora"}
    stdout = open(log_dir / "stdout.log", "w")
    proc = subprocess.Popen([str(TRAIN_PY), "-u", str(runner)], env=env, stdout=stdout, stderr=subprocess.STDOUT,
                            cwd=str(COOKBOOK))
    meta = {"pid": proc.pid, "run_id": run_id, "dedicated": dedicated, "output_model_id": output_model_id, "log_dir": str(log_dir), **info}
    (ROOT / "results" / "sft_run.json").write_text(json.dumps(meta, indent=1))
    return meta


def sync_train_log() -> list[dict]:
    """Copy every-10-steps loss lines from the newest run's stdout into results/train_log.jsonl."""
    meta = json.loads((ROOT / "results" / "sft_run.json").read_text())
    lines = []
    for l in open(Path(meta["log_dir"]) / "stdout.log", errors="replace"):
        if l.startswith('{"type": "metrics"'):
            d = json.loads(l)
            if "ce_loss" in d and (d["step"] % 10 == 0 or d["step"] == 1):
                lines.append(d)
    D.write_jsonl(ROOT / "results" / "train_log.jsonl", lines)
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "launch", "log"])
    ap.add_argument("--max-text", type=int, default=None)
    ap.add_argument("--output-model-id", default=None)
    ap.add_argument("--dedicated", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        print(json.dumps(build(a.max_text), indent=1))
    elif a.cmd == "launch":
        print(json.dumps(launch(a.max_text, a.output_model_id, a.dedicated), indent=1))
    else:
        for l in sync_train_log()[-10:]:
            print(l)
