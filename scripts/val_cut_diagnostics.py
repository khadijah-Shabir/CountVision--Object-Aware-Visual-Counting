#!/usr/bin/env python
"""Cut-failure diagnostic on a seeded sample of FSC-147 VAL images (never test).
Caches raw detector/SAM output so post-processing rules can be compared without re-running models."""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lvlm_count import division as D
from lvlm_count.config import load_config, PROJECT, DATASETS
from lvlm_count.grounding import Grounder, detect_areas
from lvlm_count.segmentation import Segmenter, postprocess_masks, filter_group_boxes

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
cfg = load_config(); pp = cfg["postprocess"]
cache = PROJECT / "countvision/results/cache_val_raw2"; cache.mkdir(parents=True, exist_ok=True)
split = json.load(open(DATASETS / "fsc147/Train_Test_Val_FSC_147.json"))["val"]
cls = dict(l.rstrip("\n").split("\t") for l in open(DATASETS / "fsc147/ImageClasses_FSC147.txt"))
ids = list(np.random.default_rng(1234).choice(split, N, replace=False))
g = s = None
raw = {}
for k in ids:
    f = cache / f"{k}.pkl"
    if f.exists():
        raw[k] = pickle.load(open(f, "rb")); continue
    if g is None:
        g = Grounder(str(PROJECT / cfg["grounding"]["model_dir"]), "cuda")
        s = Segmenter(str(PROJECT / cfg["sam"]["checkpoint"]), cfg["sam"]["model_type"], "cuda")
    img = Image.open(DATASETS / "fsc147/images_384_VarV2" / k).convert("RGB"); E = cls[k]
    _, _, areas = detect_areas(g, img, E, 0.1, 0.1)
    out = []
    for x0, y0, x1, y1 in areas:
        crop = np.asarray(img)[y0:y1, x0:x1]
        b, sc = g.detect(Image.fromarray(crop), E.rstrip("s") if not E.endswith("ss") else E, 0.1, 0.1)
        out.append(dict(crop=crop, boxes=b, scores=sc, masks=s.masks_from_boxes(crop, b)))
    raw[k] = dict(E=E, crops=out); pickle.dump(raw[k], open(f, "wb")); print("cached", k, flush=True)

def evaluate(factor, group):
    tot_cuts = fail = n_sub = n_img_all_fail = 0
    for k, r in raw.items():
        img_fail = True
        for c in r["crops"]:
            cm, cs = c["masks"], c["scores"]
            if group and len(c["boxes"]):
                k = filter_group_boxes(c["boxes"]); cm, cs = cm[k], cs[k]
            m, sc = postprocess_masks(cm, cs, nms_iou=pp["mask_nms_iou"], erosion_px=pp["erosion_px"],
                                      refinement_px=pp["refinement_px"], max_area_frac=pp["max_mask_area_frac"],
                                      max_median_factor=factor)
            res = D.object_aware_division(c["crop"], list(m), samples_per_mask=5)
            tot_cuts += len(res.cuts); fail += len(res.failed_cuts); n_sub += len(res.subimages)
            if len(res.cuts) == 0 or len(res.paths): img_fail = False
        n_img_all_fail += img_fail
    return dict(group_box_filter=group, factor=factor, cuts=tot_cuts, failed=fail, fail_rate=round(fail / max(tot_cuts, 1), 3),
                mean_subimages=round(n_sub / len(raw), 2), images_all_cuts_failed=n_img_all_fail)

rep = [evaluate(f, g) for g in (False, True) for f in (None, 25, 10)]
for r in rep: print(r)
json.dump(rep, open(PROJECT / "countvision/results/val_cut_diagnostics.json", "w"), indent=1)
