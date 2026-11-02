"""Ingest labelled clips -> frame-sampled rows. Half of clips per scenario to train (video_sheet), half to test."""
import csv, json, random, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pt_data as D, pt_video

rubric = {r["scenario_id"]: r for r in D.load_rubric()}
rows = list(csv.DictReader(open(ROOT / "data/video/labels.csv")))
agreed = [r for r in rows if r["label_a"].strip() and r["label_a"].strip() == r["label_b"].strip()]
print(f"{len(rows)} clips, {len(agreed)} with labeller agreement ({len(rows)-len(agreed)} dropped)")
by = defaultdict(list)
for r in agreed:
    by[r["label_a"].strip()].append(r)
rng = random.Random(0)
train, test = [], []
for sid, clips in sorted(by.items()):
    rng.shuffle(clips)
    for i, c in enumerate(clips):
        clip = ROOT / "data/video/clips" / c["clip_file"]
        meta = pt_video.sample(clip, ROOT / "data/video/frames" / clip.stem)
        rel = lambda p: str(Path(p).relative_to(ROOT))
        base = {"id": f"vid-{clip.stem}", "scenario_id": sid, "species": c["species"], "clip_path": rel(clip),
                "profile": {"name": "Pet"}, "message": c["owner_message"] or "is this normal?",
                "gold": D.gold_from_rubric(rubric[sid])}
        if i % 2 == 0 and len(clips) > 1:
            test.append({**base, "modality": "video", "sheet_path": rel(meta["sheet_path"]),
                         "frame_paths": [rel(p) for p in meta["frame_paths"]]})
        else:  # training: the contact sheet as a single image row
            train.append({**base, "modality": "video_sheet", "image_path": rel(meta["sheet_path"])})
D.write_jsonl(D.SPLITS / "train_video.jsonl", train)
D.write_jsonl(D.SPLITS / "test_video.jsonl", test)
print(json.dumps({"train_sheets": len(train), "test_clips": len(test), "scenarios": sorted(by)}))
