"""
Bakes poses.json + stage_recipes/*.json into site/data.json for the website.
Applies the same saturation boost and per-dot reveal order as photo_flower.py,
so the web flower matches the desktop one. Re-run after changing either source.

    ./venv/bin/python build_web_data.py
"""

import json
from pathlib import Path

from growth import EXCLUDE_FEATURES
from photo_flower import NUM_STAGES, STAGE_SCALE, DISSOLVE_BAND, load_recipes

ROOT = Path(__file__).parent

poses = json.loads((ROOT / "poses.json").read_text())
first = next(iter(poses.values()))[0]
names = sorted(n for n in first if n not in EXCLUDE_FEATURES)

data = {
    "names": names,
    "poses": {
        stage: [[round(s[n], 4) for n in names] for s in samples]
        for stage, samples in poses.items()
    },
    "scale": [STAGE_SCALE[i] for i in range(NUM_STAGES)],
    "band": DISSOLVE_BAND,
    # dot = [x, y, r, g, b, priority]
    "stages": [
        [
            [round(d["x"], 1), round(d["y"], 1), d["bgr"][2], d["bgr"][1], d["bgr"][0], round(d["priority"], 3)]
            for d in recipe["dots"]
        ]
        for recipe in (load_recipes()[i] for i in range(NUM_STAGES))
    ],
}

out = ROOT / "site" / "data.json"
out.write_text(json.dumps(data, separators=(",", ":")))
print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
