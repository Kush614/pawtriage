"""Export images from raw datasets into data/images/<slug>/NNNN.jpg and build contact sheets.
Usage: python -I export_and_sheets.py  (reads only image bytes; executes nothing from downloads)"""
import io, random, sys, zipfile, json
from pathlib import Path
from PIL import Image, ImageOps
ROOT = Path("/Users/kush/pawtriage/data")
OUT = ROOT / "images"; VER = ROOT / "raw" / "verify"
VER.mkdir(parents=True, exist_ok=True)
EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
CAP = 250

def sources():
    # dog: sanjayS059/Pet_disease (folders)
    base = ROOT / "raw" / "pet_disease"
    m = {"Bacterial_dermatosis": "dog_bacterial_dermatosis", "Fungal_infections": "dog_fungal_infection",
         "Healthy": "dog_healthy_skin", "Hypersensitivity_allergic_dermatosis": "dog_allergic_dermatosis"}
    for d, slug in m.items():
        files = sorted(p for p in (base / d).iterdir() if p.suffix.lower() in EXT)
        yield slug, [(p.name, lambda p=p: p.read_bytes()) for p in files]
    # cat: shahida12/Russiancat-skin-disease (zip)
    z = zipfile.ZipFile(ROOT / "raw" / "russiancat_skin" / "archive.zip")
    m = {"Flea_Allergy": "cat_flea_allergy", "Health": "cat_healthy_skin", "Ringworm": "cat_ringworm", "Scabies": "cat_scabies_mange"}
    for d, slug in m.items():
        names = sorted(n for n in z.namelist() if n.startswith(f"CAT SKIN DISEASE/{d}/") and n.lower().endswith(EXT))
        yield slug, [(n, lambda n=n: z.read(n)) for n in names]

counts = {}
for slug, items in sources():
    od = OUT / slug; od.mkdir(parents=True, exist_ok=True)
    random.Random(0).shuffle(items)
    n = 0
    for name, rd in items:
        if n >= CAP: break
        try:
            im = Image.open(io.BytesIO(rd())); im = ImageOps.exif_transpose(im).convert("RGB")
        except Exception as e:
            print("skip", name, e); continue
        if max(im.size) > 1024: im.thumbnail((1024, 1024), Image.LANCZOS)
        n += 1; im.save(od / f"{n:04d}.jpg", "JPEG", quality=85)
    counts[slug] = n
    # contact sheet: 6x5 of 30 random
    files = sorted(od.glob("*.jpg")); pick = random.Random(1).sample(files, min(30, len(files)))
    T = 200; sheet = Image.new("RGB", (6 * T, 5 * T), "white")
    for i, f in enumerate(pick):
        t = Image.open(f); t.thumbnail((T, T)); sheet.paste(t, ((i % 6) * T, (i // 6) * T))
    sheet.save(VER / f"{slug}.jpg", quality=80)
    print(slug, n)
json.dump(counts, open(VER / "counts.json", "w"), indent=1)
