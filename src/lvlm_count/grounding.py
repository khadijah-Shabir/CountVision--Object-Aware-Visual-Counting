"""GroundingDINO wrapper (HuggingFace implementation of the IDEA-Research checkpoints)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class Grounder:
    def __init__(self, model_dir: str, device: str | None = None):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir).to(self.device).eval()

    @torch.no_grad()
    def detect(self, image: Image.Image, text: str, box_threshold: float, text_threshold: float):
        """Return boxes (n,4) xyxy in pixels and scores (n,)."""
        text = text.lower().strip()
        if not text.endswith("."):
            text += "."
        inp = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        try:
            out = self.model(**inp)
            res = self.processor.post_process_grounded_object_detection(
                out, inp.input_ids, box_threshold=box_threshold, text_threshold=text_threshold,
                target_sizes=[image.size[::-1]])[0]
        except RuntimeError:   # e.g. crops too small for the detector (the authors catch the same error)
            return np.zeros((0, 4)), np.zeros(0)
        return res["boxes"].cpu().numpy().reshape(-1, 4), res["scores"].cpu().numpy()


def box_overlap(a, b) -> bool:
    """True if boxes (x0,y0,x1,y1) overlap with positive area."""
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def merge_boxes(boxes: np.ndarray) -> np.ndarray:
    """Connected components of the box-overlap graph -> union bounding box per component."""
    n = len(boxes)
    if n == 0:
        return boxes.reshape(0, 4)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if box_overlap(boxes[i], boxes[j]):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = [[boxes[g, 0].min(), boxes[g, 1].min(), boxes[g, 2].max(), boxes[g, 3].max()]
           for g in groups.values()]
    out = np.array(out, float)
    return out[np.lexsort((out[:, 1], out[:, 0]))]  # left-to-right, deterministic


def detect_areas(grounder: Grounder, image: Image.Image, phrase: str, box_th: float, text_th: float):
    """Stage 1. Returns (raw_boxes, merged non-overlapping integer areas). Falls back to whole image."""
    W, H = image.size
    boxes, scores = grounder.detect(image, phrase, box_th, text_th)
    if len(boxes):
        boxes = np.clip(boxes, 0, [W, H, W, H])
    merged = merge_boxes(boxes) if len(boxes) else np.array([[0, 0, W, H]], float)
    areas = np.stack([np.floor(merged[:, 0]), np.floor(merged[:, 1]),
                      np.ceil(merged[:, 2]), np.ceil(merged[:, 3])], 1).astype(int)
    areas[:, [0, 2]] = areas[:, [0, 2]].clip(0, W)
    areas[:, [1, 3]] = areas[:, [1, 3]].clip(0, H)
    keep = (areas[:, 2] > areas[:, 0]) & (areas[:, 3] > areas[:, 1])
    areas = areas[keep] if keep.any() else np.array([[0, 0, W, H]])
    return boxes, scores, areas
