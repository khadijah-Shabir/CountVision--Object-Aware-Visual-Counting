#!/usr/bin/env python
"""Synthetic circles benchmark (motivation experiment E1): N non-overlapping filled circles on a blank canvas.
N in {5,10,15,20,30,40,50,75,100,150,200}, 20 images each, seed 1234 -> benchmarks/synthetic_circles.jsonl"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]
NS = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]; PER = 20; SIZE = 1024
out = ROOT / "results" / "synthetic_images"; out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1234); rows = []
for n in NS:
    r = int(min(40, max(10, 0.42 * SIZE / np.sqrt(n) / 2)))   # radius shrinks with N so all N fit
    for k in range(PER):
        pts = []
        tries = 0
        while len(pts) < n and tries < 200000:
            tries += 1
            c = rng.integers(r + 2, SIZE - r - 2, 2)
            if all((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 >= (2 * r + 3) ** 2 for p in pts): pts.append(c)
        assert len(pts) == n, (n, len(pts))
        im = Image.new("RGB", (SIZE, SIZE), "white"); d = ImageDraw.Draw(im)
        for (x, y) in pts: d.ellipse([x - r, y - r, x + r, y + r], fill=(30, 100, 200))
        fn = out / f"circles_n{n:03d}_{k:02d}.png"; im.save(fn)
        rows.append(dict(id=f"n{n:03d}_{k:02d}", image_path=str(fn), E="circles", question="How many circles are there?",
                         gt_count=n, source_split="synthetic"))
(ROOT / "benchmarks").mkdir(exist_ok=True)
with open(ROOT / "benchmarks" / "synthetic_circles.jsonl", "w") as f:
    for r_ in rows: f.write(json.dumps(r_) + "\n")
print(len(rows), "images")
