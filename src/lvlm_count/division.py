"""Stage 3: object-aware division.

3a  sample mask pixels -> project to x -> MeanShift -> cut x-coordinates
3b  A* path (8-connected, Manhattan heuristic) from (x,0) to (x,h-1) around obstacle masks
3c  partition the image into regions between paths -> sub-images

Everything here is model-free and deterministic given the masks and the rng seed.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from sklearn.cluster import MeanShift

# ----------------------------------------------------------------------------- 3a


def sample_mask_points(masks: Sequence[np.ndarray], k: int = 5, rng=None) -> np.ndarray:
    """Return up to k (row, col) pixels sampled from each mask. Shape (n, 2)."""
    rng = rng or np.random.default_rng(0)
    pts = []
    for m in masks:
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        idx = rng.choice(len(ys), size=min(k, len(ys)), replace=False)
        pts.append(np.stack([ys[idx], xs[idx]], 1))
    return np.concatenate(pts, 0) if pts else np.zeros((0, 2), int)


def cut_positions(xs: np.ndarray, midpoint: bool = True) -> List[int]:
    """MeanShift (sklearn default bandwidth) on 1-D x values -> cut x-coordinates.

    One cluster (or <2 distinct values) => no cut. The cut between consecutive clusters
    is the midpoint of (max x of the left cluster, min x of the right cluster).
    """
    xs = np.asarray(xs, float).reshape(-1, 1)
    if len(xs) < 2 or np.ptp(xs) == 0:
        return []
    labels = MeanShift().fit(xs).labels_
    clusters = []
    for lab in np.unique(labels):
        v = xs[labels == lab, 0]
        clusters.append((v.min(), v.max()))
    clusters.sort()
    cuts = []
    for (_, left_max), (right_min, _) in zip(clusters[:-1], clusters[1:]):
        c = (left_max + right_min) / 2 if midpoint else left_max
        cuts.append(int(round(c)))
    # de-duplicate, keep order
    return sorted(set(cuts))


# ----------------------------------------------------------------------------- 3b

_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def astar(free: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
          heuristic: str = "manhattan") -> Optional[List[Tuple[int, int]]]:
    """A* on an 8-connected grid. free[r, c] True = walkable. Nodes are (row, col).

    Diagonal moves are not allowed to squeeze between two blocked orthogonal neighbours
    (no corner cutting); this guarantees that straight segments between consecutive
    path nodes stay inside free cells. Step cost is 1 for every move (as in an unweighted
    networkx graph). Manhattan is NOT admissible with unit-cost diagonals; we replicate
    the paper's choice, `heuristic="octile"` gives the admissible alternative.
    Ties are broken deterministically.
    """
    H, W = free.shape
    if not (free[start] and free[goal]):
        return None
    gr, gc = goal

    if heuristic == "manhattan":
        def h(r, c): return abs(r - gr) + abs(c - gc)
    elif heuristic == "octile":
        def h(r, c):
            dr, dc = abs(r - gr), abs(c - gc)
            return max(dr, dc)
    else:
        raise ValueError(heuristic)

    g = {start: 0}
    parent = {}
    heap = [(h(*start), 0, start)]
    closed = set()
    while heap:
        _, gc_, node = heapq.heappop(heap)
        if node in closed:
            continue
        if node == goal:
            path = [node]
            while node in parent:
                node = parent[node]
                path.append(node)
            return path[::-1]
        closed.add(node)
        r, c = node
        for dr, dc in _MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or not free[nr, nc]:
                continue
            if dr != 0 and dc != 0 and not (free[r + dr, c] and free[r, c + dc]):
                continue  # no corner cutting
            ng = gc_ + 1
            if ng < g.get((nr, nc), 1 << 60):
                g[(nr, nc)] = ng
                parent[(nr, nc)] = node
                heapq.heappush(heap, (ng + h(nr, nc), ng, (nr, nc)))
    return None


def _nearest_free_on_row(free_row: np.ndarray, x: int) -> Optional[int]:
    idx = np.nonzero(free_row)[0]
    if len(idx) == 0:
        return None
    return int(idx[np.argmin(np.abs(idx - x))])


def _downscale_obstacles(obstacle: np.ndarray, s: int) -> np.ndarray:
    """Block (s x s) is walkable only if it contains NO obstacle pixel (conservative)."""
    H, W = obstacle.shape
    Hs, Ws = -(-H // s), -(-W // s)
    pad = np.zeros((Hs * s, Ws * s), bool)
    pad[:H, :W] = obstacle
    return ~pad.reshape(Hs, s, Ws, s).any(axis=(1, 3))


def find_cut_path(obstacle: np.ndarray, x: int, max_side: int = 256,
                  heuristic: str = "manhattan") -> Optional[np.ndarray]:
    """Path (n,2) of full-resolution (row, col) pixels from (0,x) to (h-1,x), avoiding obstacles.

    The search runs on a downscaled grid for speed; consecutive grid nodes are joined with
    straight full-resolution segments. Because a downscaled cell is walkable only if every
    pixel in it is free and diagonal corner-cutting is forbidden, the full-resolution path
    never touches an obstacle pixel. If no path exists at a coarse scale, finer scales are
    tried down to full resolution. Returns None when the cut is impossible
    (objects block the whole image).
    """
    H, W = obstacle.shape
    x = int(np.clip(x, 0, W - 1))
    s0 = max(1, -(-max(H, W) // max_side))
    scales = []
    s = s0
    while s >= 1:
        scales.append(s)
        s //= 2
    if scales[-1] != 1:
        scales.append(1)
    for s in scales:
        free_s = _downscale_obstacles(obstacle, s)
        Hs, Ws = free_s.shape
        # endpoints: move to nearest free cell on the same (top / bottom) row of the grid
        top = _nearest_free_on_row(free_s[0], x // s)
        bot = _nearest_free_on_row(free_s[Hs - 1], x // s)
        if top is None or bot is None:
            continue
        path = astar(free_s, (0, top), (Hs - 1, bot), heuristic)
        if path is None:
            continue
        # cell centres in full-res coordinates, clipped to the image
        pts = [(min(r * s + s // 2, H - 1), min(c * s + s // 2, W - 1)) for r, c in path]
        # extend to the true image border rows so the cut spans top to bottom
        pts = [(0, pts[0][1])] + pts + [(H - 1, pts[-1][1])]
        canvas = np.zeros((H, W), np.uint8)
        poly = np.array([[c, r] for r, c in pts], np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], False, 1, thickness=1)
        return np.argwhere(canvas > 0)
    return None


# ----------------------------------------------------------------------------- 3c


def partition_by_paths(shape: Tuple[int, int], paths: Sequence[np.ndarray]) -> np.ndarray:
    """Label every pixel with a region id (0..K-1), regions separated by the paths.

    Free regions are the 4-connected components of the non-path pixels (an 8-connected
    one-pixel line is a barrier for 4-connectivity). Path pixels themselves are assigned
    to the nearest region, so every pixel belongs to exactly one region.
    """
    H, W = shape
    barrier = np.zeros((H, W), bool)
    for p in paths:
        if len(p):
            barrier[p[:, 0], p[:, 1]] = True
    lab, n = ndi.label(~barrier, structure=ndi.generate_binary_structure(2, 1))
    if n == 0:
        return np.zeros((H, W), np.int32)
    if barrier.any():
        idx = ndi.distance_transform_edt(barrier, return_distances=False, return_indices=True)
        lab = lab[idx[0], idx[1]]
    return (lab - 1).astype(np.int32)


def make_subimages(image: np.ndarray, labels: np.ndarray, blank: int = 255,
                   min_area: int = 0):
    """One sub-image per region: region pixels from the original, everything else blank, cropped.

    Returns list of dicts {region, bbox(x0,y0,x1,y1), area, image}. Regions smaller than
    min_area are still returned (flagged 'tiny') so the caller can count them as 0.
    """
    out = []
    for r in np.unique(labels):
        m = labels == r
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = np.full_like(image, blank)
        sub[m] = image[m]
        out.append({"region": int(r), "bbox": (int(x0), int(y0), int(x1), int(y1)),
                    "area": int(m.sum()), "tiny": int(m.sum()) < min_area,
                    "image": sub[y0:y1, x0:x1]})
    # left-to-right order
    out.sort(key=lambda d: d["bbox"][0])
    return out


@dataclass
class DivisionResult:
    cuts: List[int] = field(default_factory=list)          # requested cut x-coordinates
    paths: List[np.ndarray] = field(default_factory=list)  # successful paths (row, col)
    failed_cuts: List[int] = field(default_factory=list)
    labels: Optional[np.ndarray] = None
    subimages: list = field(default_factory=list)
    samples: Optional[np.ndarray] = None
    cuts_y: List[int] = field(default_factory=list)


def object_aware_division(image: np.ndarray, masks: Sequence[np.ndarray], *,
                          samples_per_mask: int = 5, max_side: int = 256,
                          heuristic: str = "manhattan", blank: int = 255,
                          min_area: int = 100, seed: int = 0, axes: str = "x") -> DivisionResult:
    """Full Stage 3 for one crop. `masks`: list of HxW bool arrays (post-processed)."""
    H, W = image.shape[:2]
    res = DivisionResult()
    obstacle = np.zeros((H, W), bool)
    for m in masks:
        obstacle |= m.astype(bool)
    rng = np.random.default_rng(seed)
    res.samples = sample_mask_points(masks, samples_per_mask, rng)
    res.cuts = cut_positions(res.samples[:, 1]) if len(res.samples) and "x" in axes else []
    for x in res.cuts:
        p = find_cut_path(obstacle, x, max_side=max_side, heuristic=heuristic)
        if p is None:
            res.failed_cuts.append(x)
        else:
            res.paths.append(p)
    if "y" in axes and len(res.samples):          # horizontal cuts = vertical cuts on the transposed problem
        ycuts = cut_positions(res.samples[:, 0])
        res.cuts_y = ycuts
        for y in ycuts:
            p = find_cut_path(obstacle.T, y, max_side=max_side, heuristic=heuristic)
            if p is None:
                res.failed_cuts.append(("y", y))
            else:
                res.paths.append(p[:, ::-1].copy())   # (row', col') in transposed coords -> (col', row')
    res.labels = partition_by_paths((H, W), res.paths)
    res.subimages = make_subimages(image, res.labels, blank=blank, min_area=min_area)
    return res


def naive_division(image: np.ndarray, k: int, blank: int = 255) -> list:
    """Baseline: k equal-width vertical strips with straight cuts (ignores objects)."""
    H, W = image.shape[:2]
    edges = np.linspace(0, W, k + 1).round().astype(int)
    out = []
    for i in range(k):
        if edges[i + 1] <= edges[i]:
            continue  # zero-width strip (image narrower than k)
        out.append({"region": i, "bbox": (int(edges[i]), 0, int(edges[i + 1]), H),
                    "area": int((edges[i + 1] - edges[i]) * H), "tiny": False,
                    "image": image[:, edges[i]:edges[i + 1]]})
    return out
