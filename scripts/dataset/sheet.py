import sys, json, random
from pathlib import Path
from PIL import Image
out = sys.argv[1]; files = [Path(p) for p in sys.argv[2:]]
files = random.Random(2).sample(files, min(48, len(files))); T = 150
sheet = Image.new("RGB", (8 * T, 6 * T), "white")
for i, f in enumerate(files):
    t = Image.open(f); t.thumbnail((T, T)); sheet.paste(t, ((i % 8) * T, (i // 8) * T))
sheet.save(out, quality=80)
