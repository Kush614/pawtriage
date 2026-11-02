"""Notebook 01 logic: generate text + image rows, split, lock."""
from __future__ import annotations

import asyncio
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

import pt_data as D
from pt_prompt import ROOT, image_finding_for

IMG_DIR = ROOT / "data" / "images"
GENERIC_IMAGE_PATTERN = ("owner is attaching a photo of a spot on their pet's skin or coat that they are unsure about; "
                         "the message must NOT describe the lesion's appearance in any detail (no colors, shapes, "
                         "crusts, hair loss, etc.) - just ask whether it is serious / what to do, with some context")


def generator_model() -> str:
    return os.environ.get("OPENROUTER_GENERATOR_MODEL") or os.environ.get("OPENROUTER_FRONTIER_MODEL") \
        or "openai/gpt-6.1-terra-pro"


# ------------------------------------------------------------------ text
async def generate_text_rows(llm, per_scenario: int = 40) -> list[dict]:
    rubric = [r for r in D.expand_species(D.load_rubric()) if not r["scenario_id"].startswith(("img_", "vid_"))]
    model = generator_model()
    results = await asyncio.gather(*(D.generate_for_scenario(llm, model, r, n=per_scenario) for r in rubric))
    rows = []
    for r, items in zip(rubric, results):
        for i, it in enumerate(items):
            rows.append({"id": f"{r['scenario_id']}-{i:03d}", "scenario_id": r["scenario_id"],
                         "species": r["species"], "modality": "text",
                         "profile": it.get("profile", {}), "message": it["message"].strip(),
                         "gold": D.gold_from_rubric(r)})
    D.write_jsonl(D.GEN / "all.jsonl", rows)
    return rows


# ------------------------------------------------------------------ images
def dhash(img: Image.Image, size: int = 8) -> int:
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for y in range(size):
        for x in range(size):
            bits = (bits << 1) | (px[y * (size + 1) + x] > px[y * (size + 1) + x + 1])
    return bits


def dihedral_min_hash(path: Path) -> int:
    """Hash invariant to the rotate/flip augmentation found in the cat set."""
    im = Image.open(path)
    variants = []
    for flip in (False, True):
        base = ImageOps.mirror(im) if flip else im
        for rot in (0, 90, 180, 270):
            variants.append(dhash(base.rotate(rot, expand=True)))
    return min(variants)


def group_near_duplicates(paths: list[Path], max_hamming: int = 10) -> list[int]:
    hashes = [dihedral_min_hash(p) for p in paths]
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if bin(hashes[i] ^ hashes[j]).count("1") <= max_hamming:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(paths))]


async def image_message_pool(llm, species: str, n: int = 60) -> list[dict]:
    r = {"scenario_id": f"imgpool_{species}", "species": species, "pattern": GENERIC_IMAGE_PATTERN}
    return await D.generate_for_scenario(llm, generator_model(), r, n=n)


async def build_image_rows(llm, test_images_per_class: int = 13, msgs_per_test_image: int = 3,
                           seed: int = 0) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    label_map = json.loads((ROOT / "data/vocab/image_label_map.json").read_text())["classes"]
    rubric = {r["scenario_id"]: r for r in D.load_rubric()}
    pools = {s: await image_message_pool(llm, s, n=80) for s in ("dog", "cat")}
    train, test = [], []
    split_log = {}
    for cls in label_map.values():
        if not cls.get("kept"):
            continue
        slug, species = cls["slug"], cls["species"]
        sid = f"img_{slug}"
        rr = rubric[sid]
        paths = sorted((IMG_DIR / slug).glob("*.jpg"))
        groups = group_near_duplicates(paths)
        by_group = defaultdict(list)
        for p, g in zip(paths, groups):
            by_group[g].append(p)
        gids = sorted(by_group)
        rng.shuffle(gids)
        test_paths: list[Path] = []
        for g in gids:  # whole near-duplicate groups go to test until the quota is met
            if len(test_paths) >= test_images_per_class:
                break
            test_paths.extend(by_group[g])
        test_set = set(test_paths)
        split_log[slug] = {"images": len(paths), "dup_groups": len(by_group), "test_images": len(test_set)}
        gold = D.gold_from_rubric(rr, image_finding=image_finding_for(slug))
        pool = pools[species]
        for p in paths:
            is_test = p in test_set
            k = msgs_per_test_image if is_test else 1
            for j, it in enumerate(rng.sample(pool, k)):
                row = {"id": f"img-{slug}-{p.stem}-{j}", "scenario_id": sid, "species": species,
                       "modality": "image", "image_path": str(p.relative_to(ROOT)),
                       "profile": it.get("profile", {}), "message": it["message"].strip(), "gold": gold}
                (test if is_test else train).append(row)
    D.write_jsonl(D.GEN / "images.jsonl", train + test)
    (D.GEN / "image_split_log.json").write_text(json.dumps(split_log, indent=1))
    return train, test


# ------------------------------------------------------------------ split + lock
def split_text(rows: list[dict], seed: int = 0) -> tuple[list[dict], list[dict]]:
    train_ids, test_ids = D.split_by_scenario(rows, 0.2, seed)
    return [r for r in rows if r["scenario_id"] in train_ids], [r for r in rows if r["scenario_id"] in test_ids]


def write_candidates(text_train, text_test, img_train, img_test) -> None:
    """Train files are final now (training can start); test files are candidates until human review + lock."""
    if (D.SPLITS / "LOCKED.md").exists():
        raise RuntimeError("Test set already locked. Refusing to overwrite.")
    D.assert_no_overlap(text_train + img_train, text_test + img_test)
    D.write_jsonl(D.SPLITS / "train.jsonl", text_train)
    D.write_jsonl(D.SPLITS / "train_images.jsonl", img_train)
    D.write_jsonl(D.SPLITS / "test.candidate.jsonl", text_test)
    D.write_jsonl(D.SPLITS / "test_images.candidate.jsonl", img_test)
    import csv
    with open(D.SPLITS / "review_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "species", "gold_urgency", "message", "action"])
        for r in text_test:
            w.writerow([r["id"], r["species"], r["gold"]["urgency"], r["message"], ""])


def apply_review_and_lock() -> dict:
    """Apply review_test.csv (action = drop | fix_label:<urgency>), then write + lock test files."""
    import csv
    if (D.SPLITS / "LOCKED.md").exists():
        raise RuntimeError("Test set already locked. Refusing to overwrite. Delete LOCKED.md only if you mean it.")
    text_train = D.read_jsonl(D.SPLITS / "train.jsonl")
    img_train = D.read_jsonl(D.SPLITS / "train_images.jsonl")
    text_test = D.read_jsonl(D.SPLITS / "test.candidate.jsonl")
    img_test = D.read_jsonl(D.SPLITS / "test_images.candidate.jsonl")
    acts = {r["id"]: (r.get("action") or "").strip() for r in csv.DictReader(open(D.SPLITS / "review_test.csv"))}
    text_test = [r for r in text_test if acts.get(r["id"]) != "drop"]
    for r in text_test:
        a = acts.get(r["id"], "")
        if a.startswith("fix_label:"):
            r["gold"]["urgency"] = a.split(":", 1)[1]
    D.assert_no_overlap(text_train + img_train, text_test + img_test)
    D.write_jsonl(D.SPLITS / "test.jsonl", text_test)
    D.write_jsonl(D.SPLITS / "test_images.jsonl", img_test)
    media = {r["image_path"]: D.file_sha(ROOT / r["image_path"]) for r in img_test}
    D.write_lock([D.SPLITS / "test.jsonl", D.SPLITS / "test_images.jsonl"], media)
    return {"text_train": len(text_train), "text_test": len(text_test), "img_train": len(img_train),
            "img_test": len(img_test),
            "text_test_urgency": dict(Counter(r["gold"]["urgency"] for r in text_test)),
            "text_test_species": dict(Counter(r["species"] for r in text_test)),
            "text_test_scenarios": len({r["scenario_id"] for r in text_test})}
