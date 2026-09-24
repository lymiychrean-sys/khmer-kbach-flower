"""
Turns transparent-background flower PNGs (one per growth stage) into dot
recipes: each dot's position comes from the alpha mask, its color from the
photo.

Usage:
    ./venv/bin/python extract_flower.py
        (reads stage_0.png, stage_1.png, ... from flower-project-assets/)
    ./venv/bin/python extract_flower.py stage0.png stage1.png ... stage4.png
        (or pick the images yourself)

Writes stage_recipes/stage_0.json ... stage_4.json
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

CANVAS = 200
CELL = 3
JITTER = 0.45

OUT_DIR = Path(__file__).parent / "stage_recipes"
ASSETS_DIR = Path(__file__).parent / "flower-project-assets"


def _hash01(ix, iy):
    n = math.sin(ix * 127.1 + iy * 311.7) * 43758.5453
    return n - math.floor(n)


def extract_one(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"Could not read image: {image_path}")
    if img.ndim != 3 or img.shape[2] != 4:
        raise SystemExit(f"{image_path} has no alpha channel -- expected a transparent-background PNG")
    rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    alpha = img[:, :, 3]

    ys, xs = np.where(alpha > 30)
    if len(ys) == 0:
        raise SystemExit(f"No foreground found in {image_path} -- image is fully transparent")
    rgb = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    alpha = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # CLAHE on lightness so petal shading doesn't average out flat
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    h, w = alpha.shape
    scale = (CANVAS * 0.85) / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # centered horizontally, bottom-aligned so every stage's base lines up
    canvas_rgb = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    canvas_alpha = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    off_x, off_y = (CANVAS - new_w) // 2, CANVAS - new_h
    canvas_rgb[off_y:, off_x:off_x + new_w] = rgb
    canvas_alpha[off_y:, off_x:off_x + new_w] = alpha

    dots = []
    for iy in range(-(-CANVAS // CELL)):
        for ix in range(-(-CANVAS // CELL)):
            x0, y0 = ix * CELL, iy * CELL
            patch_a = canvas_alpha[y0:y0 + CELL, x0:x0 + CELL]
            if patch_a.mean() <= 40:
                continue
            r, g, b = canvas_rgb[y0:y0 + CELL, x0:x0 + CELL][patch_a > 40].mean(axis=0).astype(int)
            jx = (_hash01(ix, iy) - 0.5) * CELL * JITTER
            jy = (_hash01(ix + 91, iy + 7) - 0.5) * CELL * JITTER
            dots.append({
                "x": round(x0 + CELL / 2 + jx, 1), "y": round(y0 + CELL / 2 + jy, 1),
                "bgr": [int(b), int(g), int(r)],
            })

    return {"canvas": CANVAS, "dots": dots}


def _stage_images():
    """stage_0.png, stage_1.png, ... from ASSETS_DIR, sorted by number (so stage_10 comes after stage_9)."""
    paths = sorted(ASSETS_DIR.glob("stage_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise SystemExit(f"No stage_*.png images found in {ASSETS_DIR}")
    return paths


def main():
    paths = sys.argv[1:] or _stage_images()

    OUT_DIR.mkdir(exist_ok=True)
    for i, path in enumerate(paths):
        print(f"Processing stage {i}: {path}")
        recipe = extract_one(path)
        out_path = OUT_DIR / f"stage_{i}.json"
        with open(out_path, "w") as f:
            json.dump(recipe, f)
        print(f"  -> {out_path} ({len(recipe['dots'])} dots)")


if __name__ == "__main__":
    main()
