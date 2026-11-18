"""Stage 2: SAM box-prompted masks + post-processing (NMS -> erosion -> refinement -> drop huge)."""
from __future__ import annotations

import cv2
import numpy as np
import torch


class Segmenter:
    def __init__(self, checkpoint: str, model_type: str = "vit_h", device: str | None = None):
        from segment_anything import SamPredictor, sam_model_registry
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[model_type](checkpoint=checkpoint).to(self.device)
        self.predictor = SamPredictor(sam)

    def auto_masks(self, image_rgb: np.ndarray):
        """SAM 'segment everything' (no detector). Authors' generator settings except crop_n_layers=0
        (speed; DECISIONS.md #34). Returns (masks (n,H,W) bool, predicted_iou (n,))."""
        from segment_anything import SamAutomaticMaskGenerator
        if not hasattr(self, "_amg"):
            self._amg = SamAutomaticMaskGenerator(self.predictor.model, points_per_side=32, pred_iou_thresh=0.86,
                                                  stability_score_thresh=0.92, crop_n_layers=0, min_mask_region_area=100)
        out = self._amg.generate(image_rgb)
        if not out:
            return np.zeros((0,) + image_rgb.shape[:2], bool), np.zeros(0)
        return np.stack([o["segmentation"] for o in out]), np.array([o["predicted_iou"] for o in out])

    @torch.no_grad()
    def masks_from_boxes(self, image_rgb: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """(n,H,W) bool masks, one per box (single best mask per box)."""
        if len(boxes) == 0:
            return np.zeros((0,) + image_rgb.shape[:2], bool)
        self.predictor.set_image(image_rgb)
        b = self.predictor.transform.apply_boxes_torch(
            torch.as_tensor(boxes, dtype=torch.float, device=self.device), image_rgb.shape[:2])
        m, _, _ = self.predictor.predict_torch(point_coords=None, point_labels=None, boxes=b,
                                               multimask_output=False)
        return m[:, 0].cpu().numpy().astype(bool)


def filter_group_boxes(boxes: np.ndarray, cover_th: float = 0.67) -> np.ndarray:
    """Indices of boxes to keep after dropping 'group' boxes (as in the authors' code, `filter_bboxes`).

    Boxes are visited from largest to smallest area; a box is dropped when it covers more than
    `cover_th` of the minimum bounding rectangle (MBR) of all remaining smaller boxes, i.e. it is a
    box around the whole group rather than around one object. Stops at the first box that is not
    a group box and keeps everything from there on. Not described in the paper; see DECISIONS.md.
    """
    n = len(boxes)
    if n <= 1:
        return np.arange(n)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-area, kind="stable")
    for i in range(n - 1):
        rest = boxes[order[i + 1:]]
        mbr = np.array([rest[:, 0].min(), rest[:, 1].min(), rest[:, 2].max(), rest[:, 3].max()])
        cur = boxes[order[i]]
        inter = max(0, min(cur[2], mbr[2]) - max(cur[0], mbr[0])) * max(0, min(cur[3], mbr[3]) - max(cur[1], mbr[1]))
        mbr_area = (mbr[2] - mbr[0]) * (mbr[3] - mbr[1])
        if mbr_area <= 0 or inter / mbr_area <= cover_th:
            return np.sort(order[i:])
    return np.sort(order[-1:])


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def mask_nms(masks: np.ndarray, scores: np.ndarray, iou_th: float = 0.4) -> np.ndarray:
    """Keep indices; if two masks have IoU > iou_th drop the lower-confidence one."""
    order = list(np.argsort(-scores, kind="stable"))
    keep = []
    for i in order:
        if all(mask_iou(masks[i], masks[j]) <= iou_th for j in keep):
            keep.append(int(i))
    return np.array(sorted(keep), int)


def postprocess_masks(masks: np.ndarray, scores: np.ndarray, *, nms_iou=0.4, erosion_px=2,
                      refinement_px=3, refinement_op="close", max_area_frac=0.5, max_median_factor=None,
                      do_nms=True, do_erode=True, do_refine=True):
    """NMS -> erosion -> refinement -> drop empty / excessively large masks. Returns (masks, scores)."""
    if len(masks) == 0:
        return masks, scores
    if do_nms:
        k = mask_nms(masks, scores, nms_iou)
        masks, scores = masks[k], scores[k]
    out = []
    for m in masks:
        u = m.astype(np.uint8)
        if do_erode and erosion_px > 0:
            u = cv2.erode(u, np.ones((erosion_px, erosion_px), np.uint8))
        if do_refine and refinement_px > 0:
            op = cv2.MORPH_CLOSE if refinement_op == "close" else cv2.MORPH_OPEN
            u = cv2.morphologyEx(u, op, np.ones((refinement_px, refinement_px), np.uint8))
        out.append(u.astype(bool))
    out = np.stack(out)
    H, W = out.shape[1:]
    areas = out.sum((1, 2))
    keep = (areas > 0) & (areas <= max_area_frac * H * W)
    if max_median_factor and keep.any():  # authors' rule: drop masks > factor x median mask area
        keep &= areas <= max_median_factor * np.median(areas[areas > 0])
    return out[keep], scores[keep]
