"""
Renders the glow-dot flower from stage_recipes/*.json, cross-fading
between stages by the weights from GrowthEstimator.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

RECIPE_DIR = Path(__file__).parent / "stage_recipes"
NUM_STAGES = 5
STAGE_SCALE = {0: 0.42, 1: 0.62, 2: 0.80, 3: 0.95, 4: 1.0}
DISSOLVE_BAND = 0.22  # wider = softer grow-in between stages

_cache = {}


def _hash01(i, salt):
    n = math.sin(i * 127.1 + salt * 311.7) * 43758.5453
    return n - math.floor(n)


def _boost_saturation(recipe, sat_mult=1.75, val_mult=1.15):
    if not recipe["dots"]:
        return recipe
    bgr = np.array([d["bgr"] for d in recipe["dots"]], dtype="uint8").reshape(-1, 1, 3)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype("float32")
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_mult, 0, 255)
    boosted = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    for d, c in zip(recipe["dots"], boosted):
        d["bgr"] = [int(v) for v in c]
    return recipe


def load_recipes():
    if not _cache:
        for i in range(NUM_STAGES):
            with open(RECIPE_DIR / f"stage_{i}.json") as f:
                recipe = _boost_saturation(json.load(f))
            # fixed per-dot order in which dots appear as the stage's weight rises
            for j, d in enumerate(recipe["dots"]):
                d["priority"] = _hash01(j, i)
            _cache[i] = recipe
    return _cache


def _add_bloom(layer):
    near = cv2.GaussianBlur(layer, (5, 5), 0) * 0.20
    far = cv2.GaussianBlur(layer, (11, 11), 0) * 0.10
    return layer + near + far


def _tonemap(layer, white_point=340.0):
    # soft compression instead of a hard clip, so overlaps keep their hue
    return layer * white_point / (white_point + layer)


def render_glow_layer(weights, canvas=200):
    """weights: {0: w0, ..., 4: w4}. Returns a (canvas, canvas, 3) float32 BGR layer."""
    recipes = load_recipes()
    layer = np.zeros((canvas, canvas, 3), dtype="float32")
    anchor_x, anchor_y = canvas / 2, canvas

    for idx, w in weights.items():
        if w < 0.02:
            continue
        s = STAGE_SCALE[idx]
        # at partial weight only low-priority dots show: a grow-in, not a dim
        thresh = w * (1.0 + DISSOLVE_BAND)
        for d in recipes[idx]["dots"]:
            reveal = (thresh - d["priority"]) / DISSOLVE_BAND + 0.5
            if reveal <= 0:
                continue
            alpha = min(1.0, reveal)
            x = anchor_x + (d["x"] - anchor_x) * s
            y = anchor_y + (d["y"] - anchor_y) * s
            color = tuple(c * alpha for c in d["bgr"])
            cv2.circle(layer, (int(x), int(y)), 1, color, -1, cv2.LINE_AA)

    return layer


def draw_photo_flower(frame, base_xy, weights, canvas=200, scale=2.6):
    """Adds the flower onto frame with its base at base_xy. weights: the
    {"0": w, ...} dict from GrowthEstimator.estimate()."""
    glow = render_glow_layer({int(k): w for k, w in weights.items()}, canvas=canvas)
    glow = _tonemap(_add_bloom(glow))

    new_size = int(canvas * scale)
    glow = cv2.resize(glow, (new_size, new_size), interpolation=cv2.INTER_LINEAR)

    # anchor on the lowest lit pixel so the stem touches the hand at any stage size
    gh, gw = glow.shape[:2]
    ys, xs = np.where(glow.max(axis=2) > 4)
    if len(ys):
        ax, ay = (xs.min() + xs.max()) / 2.0, ys.max()
    else:
        ax, ay = gw / 2.0, gh

    fh, fw = frame.shape[:2]
    x0, y0 = int(base_xy[0] - ax), int(base_xy[1] - ay)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = min(gw, fw - x0), min(gh, fh - y0)
    if sx0 >= sx1 or sy0 >= sy1:
        return

    region = frame[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1].astype("float32")
    added = region + glow[sy0:sy1, sx0:sx1]
    frame[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1] = np.clip(added, 0, 255).astype("uint8")
