"""Video -> N evenly spaced frames + a 2x3 contact sheet (§4.5)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
LONG_SIDE = 1024


def resize_image(src: Path | str, dst: Path | str, long_side: int = LONG_SIDE) -> Path:
    """Shared preprocessing for train + serve: longest side 1024 (no upscale), RGB JPEG q85."""
    from PIL import ImageOps
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    s = long_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "JPEG", quality=85)
    return dst


def duration_s(path: Path | str) -> float:
    r = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "Duration:" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise ValueError(f"could not read duration of {path}")


def sample(video: Path | str, out_dir: Path | str, n: int = 6, max_seconds: float = 30.0) -> dict:
    """Extract n evenly spaced frames (resized) and a 2x3 contact sheet. Returns paths."""
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = duration_s(video)
    if dur > max_seconds + 0.5:
        raise ValueError(f"video is {dur:.1f}s; max {max_seconds}s")
    frames = []
    for i in range(n):
        t = dur * (i + 0.5) / n
        raw = out_dir / f"raw_{i}.jpg"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video),
                        "-frames:v", "1", "-q:v", "2", str(raw)], check=True)
        frames.append(resize_image(raw, out_dir / f"frame_{i}.jpg"))
        raw.unlink(missing_ok=True)
    sheet = contact_sheet(frames, out_dir / "sheet.jpg")
    meta = {"duration_s": round(dur, 2), "frame_paths": [str(f) for f in frames], "sheet_path": str(sheet)}
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return meta


def contact_sheet(frames: list[Path], dst: Path, cols: int = 3, rows: int = 2) -> Path:
    ims = [Image.open(f).convert("RGB") for f in frames[: cols * rows]]
    tw = LONG_SIDE // cols
    th = round(tw * ims[0].height / ims[0].width)
    sheet = Image.new("RGB", (tw * cols, th * rows), "white")
    for i, im in enumerate(ims):
        sheet.paste(im.resize((tw, th), Image.LANCZOS), ((i % cols) * tw, (i // cols) * th))
    return resize_image_obj(sheet, dst)


def resize_image_obj(im: Image.Image, dst: Path) -> Path:
    s = LONG_SIDE / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(dst, "JPEG", quality=85)
    return dst
