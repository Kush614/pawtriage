"""Flag the colour-tinted 'fur close-up with coloured side bars' frames in dog classes."""
import sys, json
from pathlib import Path
from PIL import Image, ImageStat
root = Path("/Users/kush/pawtriage/data/images")
res = {}
for d in sorted(root.glob("dog_*")):
    flagged = []
    for f in sorted(d.glob("*.jpg")):
        im = Image.open(f).convert("RGB"); w, h = im.size
        s = im.resize((64, 36))
        L = ImageStat.Stat(s.crop((0, 0, 3, 36))).mean; R = ImageStat.Stat(s.crop((61, 0, 64, 36))).mean
        C = ImageStat.Stat(s.crop((8, 4, 56, 32))).mean
        def dom(m): return max(m) - sorted(m)[1]
        if 1.6 < w / h < 1.9 and (dom(L) > 60 or dom(R) > 60) and (dom(C) > 25 or max(C) < 60):
            flagged.append(f.name)
    res[d.name] = flagged
    print(d.name, len(list(d.glob('*.jpg'))), "flagged", len(flagged))
json.dump(res, open(sys.argv[1], "w"), indent=1)
