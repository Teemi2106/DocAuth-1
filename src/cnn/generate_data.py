"""
Generate synthetic genuine and forged document ELA images for CNN training.

Pipeline for each sample
------------------------
Genuine:
  render doc → save as JPEG (simulate scan) → compute ELA → save PNG

Forged:
  render doc → save as JPEG (first compression cycle)
             → apply manipulation (copy-move / text-alter / splice)
             → save as JPEG again (second cycle — this is what ELA detects)
             → compute ELA → save PNG

Usage:
    python -m src.cnn.generate_data            # 400 per class (default)
    python -m src.cnn.generate_data --n 600
"""
from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.analysis.ela import generate_ela

DATA_DIR    = ROOT / "model_data" / "ela_dataset"
GENUINE_DIR = DATA_DIR / "genuine"
FORGED_DIR  = DATA_DIR / "forged"
W, H = 800, 1100


# ── Document renderer ─────────────────────────────────────────────────────────

def _jpeg_roundtrip(img: Image.Image, quality: int = 85) -> Image.Image:
    """Compress to JPEG and reload (simulates a real scan)."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def _text_rect(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int = 11) -> None:
    shade = random.randint(20, 75)
    draw.rectangle([x, y, x + w, y + h], fill=(shade, shade, shade))


def make_document() -> Image.Image:
    """Render a synthetic certificate / official-letter style document."""
    bg = random.choice([(255, 255, 255), (252, 250, 243), (247, 247, 247)])
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    # outer border
    bw = random.randint(3, 7)
    d.rectangle([bw, bw, W - bw, H - bw], outline=(65, 65, 65), width=bw)

    # header bar
    hh = random.randint(85, 130)
    color = random.choice([
        (15, 55, 115), (115, 20, 35), (15, 75, 45), (60, 30, 100), (90, 60, 10),
    ])
    d.rectangle([25, 25, W - 25, 25 + hh], fill=color)
    for i, lw in enumerate([260, 160, 210]):
        cx = W // 2
        d.rectangle([cx - lw // 2, 44 + i * 25, cx + lw // 2, 55 + i * 25], fill=(255, 255, 255))

    # body text blocks
    y = 210
    for _ in range(random.randint(5, 9)):
        for j in range(random.randint(2, 5)):
            _text_rect(d, 70, y + j * 20, random.randint(220, 640))
        y += random.randint(110, 160)

    # horizontal rule separators
    for _ in range(random.randint(1, 3)):
        yl = random.randint(200, H - 200)
        d.line([55, yl, W - 55, yl], fill=(155, 155, 155), width=1)

    # seal / stamp
    cx = random.choice([130, W - 130])
    cy = H - 185
    r  = random.randint(48, 72)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(65, 65, 65), width=3)
    d.ellipse([cx - r + 9, cy - r + 9, cx + r - 9, cy + r - 9], outline=(65, 65, 65), width=1)

    # signature line
    sx = random.randint(70, 260)
    d.line([sx, H - 115, sx + 190, H - 115], fill=(25, 25, 25), width=2)

    return img


# ── Forgery operators ─────────────────────────────────────────────────────────

def _copy_move(img: Image.Image) -> Image.Image:
    """Copy a rectangular patch and paste it at a different location."""
    w, h = img.size
    pw, ph = random.randint(60, 160), random.randint(50, 120)
    sx, sy = random.randint(0, w - pw), random.randint(0, h - ph)
    patch = img.crop((sx, sy, sx + pw, sy + ph))
    result = img.copy()
    dx, dy = sx, sy
    while abs(dx - sx) < pw // 2 and abs(dy - sy) < ph // 2:
        dx, dy = random.randint(0, w - pw), random.randint(0, h - ph)
    result.paste(patch, (dx, dy))
    return result


def _text_alter(img: Image.Image) -> Image.Image:
    """Overwrite one or more text regions with slightly different marks."""
    result = img.copy()
    d = ImageDraw.Draw(result)
    for _ in range(random.randint(1, 4)):
        x0 = random.randint(70, 400)
        y0 = random.randint(200, H - 300)
        _text_rect(d, x0, y0, random.randint(60, 230), 13)
    return result


def _splice(img: Image.Image, donor: Image.Image) -> Image.Image:
    """Paste a region from a different (donor) document into img."""
    w, h   = img.size
    dw, dh = donor.size
    pw, ph = random.randint(80, 200), random.randint(60, 150)
    sx = random.randint(0, max(1, dw - pw))
    sy = random.randint(0, max(1, dh - ph))
    patch = donor.crop((sx, sy, sx + pw, sy + ph))
    result = img.copy()
    result.paste(patch, (random.randint(0, w - pw), random.randint(0, h - ph)))
    return result


# ── Main generation loop ──────────────────────────────────────────────────────

def generate(n: int = 400) -> None:
    GENUINE_DIR.mkdir(parents=True, exist_ok=True)
    FORGED_DIR.mkdir(parents=True, exist_ok=True)

    # donor pool for splicing
    donors = [_jpeg_roundtrip(make_document()) for _ in range(8)]

    print(f"Generating {n} genuine + {n} forged ELA images → {DATA_DIR}")
    for i in range(n):

        # ── Genuine ───────────────────────────────────────────────────────────
        ela_g = generate_ela(_jpeg_roundtrip(make_document()), quality=90, scale=15)
        ela_g.save(GENUINE_DIR / f"g_{i:04d}.png")

        # ── Forged ────────────────────────────────────────────────────────────
        # First cycle: simulate an existing compressed document
        base = _jpeg_roundtrip(make_document())
        fn = random.choice([
            _copy_move,
            _text_alter,
            lambda x: _splice(x, random.choice(donors)),
        ])
        # Second cycle: save after manipulation — ELA will expose the double-compression
        ela_f = generate_ela(_jpeg_roundtrip(fn(base)), quality=90, scale=15)
        ela_f.save(FORGED_DIR / f"f_{i:04d}.png")

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    print(f"Done. {n} genuine + {n} forged in {DATA_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic ELA training data")
    ap.add_argument("--n", type=int, default=400, help="Images per class (default 400)")
    generate(ap.parse_args().n)
