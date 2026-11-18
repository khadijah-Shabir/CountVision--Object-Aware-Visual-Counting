"""End-to-end LVLM-Count pipeline (+ ablation switches)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PIL import Image

from . import division as D
from .counting import LVLM, count_with_retry
from .grounding import Grounder, detect_areas
from .segmentation import Segmenter, filter_group_boxes, postprocess_masks

log = logging.getLogger("lvlm_count")


def plural_singular(e: str):
    """Paper convention: area detection uses the plural phrase, segmentation the singular.
    FSC-147 category names are already plural; a trivial singulariser is used for segmentation.
    [UNSPECIFIED] handling is logged in DECISIONS.md."""
    words = e.strip()
    def sing(w):
        if w.endswith("ies"): return w[:-3] + "y"
        if w.endswith("ses") or w.endswith("xes") or w.endswith("ches") or w.endswith("shes"): return w[:-2]
        if w.endswith("s") and not w.endswith("ss"): return w[:-1]
        return w
    parts = words.split(" and ")
    sp = []
    for p in parts:
        toks = p.split()
        toks[-1] = sing(toks[-1]) if toks else toks
        sp.append(" ".join(toks))
    return words, " and ".join(sp)


@dataclass
class RunOutput:
    pred: Optional[int]
    sub_counts: List[Optional[int]] = field(default_factory=list)
    n_subimages: int = 0
    n_areas: int = 0
    parse_failures: int = 0
    api_calls: int = 0
    timings: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)


class LVLMCount:
    """variant: full | no_area | area_only | naive | area_naive | sam_all (SAM segment-everything, no detector)."""

    def __init__(self, cfg: dict, lvlm: LVLM, grounder: Grounder = None, segmenter: Segmenter = None):
        self.cfg, self.lvlm, self.g, self.s = cfg, lvlm, grounder, segmenter

    def _t(self, out, key, t0):
        out.timings[key] = out.timings.get(key, 0) + time.time() - t0

    def divide(self, image_path: str, E: str, *, variant="full",
               naive_k=3, keep_debug=False, seed=0) -> RunOutput:
        """Stages 0-3 only (no LVLM). Returns RunOutput with debug['subimages'] always filled."""
        cfg, out = self.cfg, RunOutput(None)
        img = Image.open(image_path).convert("RGB")
        arr = np.asarray(img)
        W, H = img.size
        area_phrase, seg_phrase = plural_singular(E)

        # ---------------- Stage 1: area detection
        t0 = time.time()
        if variant in ("full", "area_only", "area_naive", "full_xy", "maskimg"):
            raw, sc, areas = detect_areas(self.g, img, area_phrase,
                                          cfg["grounding"]["area_box_threshold"],
                                          cfg["grounding"]["area_text_threshold"])
        else:
            raw, sc, areas = np.zeros((0, 4)), np.zeros(0), np.array([[0, 0, W, H]])
        self._t(out, "area_detection", t0)
        out.n_areas = len(areas)
        if keep_debug:
            out.debug.update(raw_boxes=raw, areas=areas)

        subs_all = []
        for ai, (x0, y0, x1, y1) in enumerate(areas):
            crop = arr[y0:y1, x0:x1]
            # ---------------- Stage 2 + 3 per area
            if variant in ("full", "no_area", "sam_all", "full_xy", "maskimg"):
                t0 = time.time()
                if variant == "sam_all":
                    boxes, scores = np.zeros((0, 4)), np.zeros(0)
                else:
                    boxes, scores = self.g.detect(Image.fromarray(crop), seg_phrase,
                                                  cfg["grounding"]["seg_box_threshold"],
                                                  cfg["grounding"]["seg_text_threshold"])
                pp = cfg["postprocess"]
                if pp.get("group_box_filter") and len(boxes):
                    k = filter_group_boxes(boxes, pp.get("group_cover_th", 0.67))
                    boxes, scores = boxes[k], scores[k]
                if variant == "sam_all":
                    masks, scores = self.s.auto_masks(crop)
                else:
                    masks = self.s.masks_from_boxes(crop, boxes)
                masks, scores = postprocess_masks(
                    masks, scores, nms_iou=pp["mask_nms_iou"], erosion_px=pp["erosion_px"],
                    refinement_px=pp["refinement_px"], refinement_op=pp["refinement_op"],
                    max_area_frac=pp["max_mask_area_frac"],
                    max_median_factor=pp.get("max_mask_median_factor"))
                self._t(out, "segmentation", t0)
                t0 = time.time()
                dv = cfg["division"]
                res = D.object_aware_division(crop, list(masks), samples_per_mask=dv["samples_per_mask"],
                                              max_side=dv["max_graph_side"], heuristic=dv["heuristic"],
                                              blank=dv["blank_color"], min_area=dv["min_region_area"],
                                              seed=seed, axes="xy" if variant == "full_xy" else "x")
                self._t(out, "division", t0)
                subs = res.subimages
                if keep_debug:
                    out.debug.setdefault("areas_dbg", []).append(
                        dict(crop=crop, boxes=boxes, masks=masks, res=res, offset=(x0, y0)))
            elif variant in ("naive", "area_naive"):
                subs = D.naive_division(crop, naive_k, blank=cfg["division"]["blank_color"])
            else:  # area_only
                subs = [{"image": crop, "tiny": False, "bbox": (0, 0, x1 - x0, y1 - y0), "area": crop.shape[0] * crop.shape[1]}]
            subs_all.extend(subs)

        if variant == "maskimg":   # baseline: render all post-processed masks as one image (LVLM counts the blobs)
            canvas = np.full_like(arr, 255); rg = np.random.default_rng(0)
            for d_ in out.debug.get("areas_dbg", []):
                ox, oy = d_["offset"]; h_, w_ = d_["crop"].shape[:2]
                for m_ in d_["masks"]:
                    canvas[oy:oy + h_, ox:ox + w_][m_] = rg.integers(30, 230, 3)
            subs_all = [{"image": canvas, "tiny": False, "bbox": (0, 0, W, H), "area": W * H}]
        out.n_subimages = len(subs_all)
        out.debug["subimages"] = subs_all
        out.debug["meta"] = dict(variant=variant, n_areas=out.n_areas, areas=np.asarray(areas).tolist(),
                                 n_raw_boxes=int(len(raw)), timings=out.timings,
                                 cuts=[a_["res"].cuts for a_ in out.debug.get("areas_dbg", [])],
                                 failed_cuts=[a_["res"].failed_cuts for a_ in out.debug.get("areas_dbg", [])],
                                 n_masks=[int(len(a_["masks"])) for a_ in out.debug.get("areas_dbg", [])])
        return out

    def run(self, image_path: str, question: str, E: str, *, variant="full",
            naive_k=3, keep_debug=False, seed=0) -> RunOutput:
        out = self.divide(image_path, E, variant=variant, naive_k=naive_k, keep_debug=True, seed=seed)
        subs_all = out.debug["subimages"]
        # ---------------- Stage 4: counting (parallelisable; sequential here)
        t0 = time.time()
        total, counts = 0, []
        for s in subs_all:
            if s.get("tiny"):
                counts.append(0)
                continue
            r = count_with_retry(self.lvlm, s["image"], question, zero_aware=True,
                                 max_retries=cfg["counting"]["max_retries"])
            out.api_calls += r.attempts
            out.parse_failures += r.failures
            counts.append(r.value)
            total += r.value or 0
        self._t(out, "counting", t0)
        out.sub_counts, out.n_subimages = counts, len(subs_all)
        out.pred = total if all(c is not None for c in counts) else (total if counts else None)
        if not keep_debug:
            out.debug = {}
        return out
