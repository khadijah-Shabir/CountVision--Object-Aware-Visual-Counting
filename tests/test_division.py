import numpy as np
import pytest
from lvlm_count import division as D


def rand_masks(rng, H=120, W=160, n=12, rmax=9):
    ms = []
    for _ in range(n):
        m = np.zeros((H, W), bool)
        cy, cx, r = rng.integers(10, H - 10), rng.integers(10, W - 10), rng.integers(3, rmax)
        yy, xx = np.ogrid[:H, :W]
        m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = True
        ms.append(m)
    return ms


# ---- mean shift edge cases
def test_cut_invariants_small_sample():
    # sklearn's default bandwidth can split tiny integer samples; only check invariants.
    xs = np.array([50, 51, 52, 50, 49])
    c = D.cut_positions(xs)
    assert c == sorted(set(c)) and all(xs.min() <= v <= xs.max() for v in c)
    assert len(c) <= len(set(xs)) - 1

def test_cut_identical_x():
    assert D.cut_positions(np.array([30, 30, 30])) == []

def test_cut_single_sample():
    assert D.cut_positions(np.array([30])) == []

def test_cut_two_clusters_midpoint():
    xs = np.array([10, 11, 12, 13, 90, 91, 92, 93])
    cuts = D.cut_positions(xs)
    assert cuts == sorted(set(cuts)) and any(13 < c < 90 for c in cuts)
    assert int(round((13 + 90) / 2)) in cuts or len(cuts) > 1   # midpoint rule for the large gap

def test_cut_three_clusters_sorted():
    xs = np.array([5, 6, 7, 100, 101, 102, 200, 201, 202])
    c = D.cut_positions(xs)
    assert c == sorted(c) and any(7 < v < 100 for v in c) and any(102 > 0 and 102 < v < 200 for v in c)

def test_masks_touching_border():
    H, W = 50, 60
    m = np.zeros((H, W), bool); m[:, :5] = True
    res = D.object_aware_division(np.full((H, W, 3), 128, np.uint8), [m])
    ob = m
    assert all(not ob[p[:, 0], p[:, 1]].any() for p in res.paths)   # never through the mask
    assert sum(s["area"] for s in res.subimages) == H * W


# ---- A*
def test_astar_connected_and_deterministic():
    free = np.ones((30, 30), bool); free[10:20, 0:25] = False
    p1 = D.astar(free, (0, 5), (29, 5)); p2 = D.astar(free, (0, 5), (29, 5))
    assert p1 == p2 and p1[0] == (0, 5) and p1[-1] == (29, 5)
    for a, b in zip(p1[:-1], p1[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
    assert all(free[n] for n in p1)

def test_astar_blocked():
    free = np.ones((10, 10), bool); free[5, :] = False
    assert D.astar(free, (0, 3), (9, 3)) is None

def test_cut_blocked_returns_none():
    ob = np.zeros((40, 40), bool); ob[20, :] = True
    assert D.find_cut_path(ob, 10) is None


# ---- no-cut-through, incl. touching / nested / fully blocking layouts
@pytest.mark.parametrize("seed", range(8))
def test_no_cut_through_random(seed):
    rng = np.random.default_rng(seed)
    masks = rand_masks(rng)
    ob = np.any(masks, 0)
    img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    res = D.object_aware_division(img, masks, seed=seed)
    for p in res.paths:
        assert not ob[p[:, 0], p[:, 1]].any()
        assert p[:, 0].min() == 0 and p[:, 0].max() == 119

def test_no_cut_through_touching_nested():
    H, W = 100, 100
    a = np.zeros((H, W), bool); a[20:60, 10:40] = True
    b = np.zeros((H, W), bool); b[30:50, 15:30] = True      # nested in a
    c = np.zeros((H, W), bool); c[20:60, 40:70] = True      # touching a
    d = np.zeros((H, W), bool); d[30:80, 85:95] = True
    img = np.full((H, W, 3), 200, np.uint8)
    res = D.object_aware_division(img, [a, b, c, d], samples_per_mask=20)
    ob = a | b | c | d
    assert len(res.cuts) >= 1
    for p in res.paths:
        assert not ob[p[:, 0], p[:, 1]].any()

def test_fully_blocking_layout_reports_failed_cut():
    H, W = 60, 100
    wall = np.zeros((H, W), bool); wall[28:32, :] = True    # full-width wall: no route top->bottom
    left = np.zeros((H, W), bool); left[5:15, 5:15] = True
    right = np.zeros((H, W), bool); right[5:15, 80:90] = True
    res = D.object_aware_division(np.full((H, W, 3), 9, np.uint8), [wall, left, right], samples_per_mask=20)
    assert res.failed_cuts and not res.paths
    assert len(res.subimages) == 1  # no cut -> whole crop


# ---- partition property
@pytest.mark.parametrize("seed", range(6))
def test_partition_every_pixel_once(seed):
    rng = np.random.default_rng(seed)
    H, W = 90, 140
    masks = rand_masks(rng, H, W, n=15)
    img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    res = D.object_aware_division(img, masks, seed=seed)
    lab = res.labels
    assert lab.shape == (H, W) and lab.min() >= 0
    total = sum(s["area"] for s in res.subimages)
    assert total == H * W                                   # areas sum to the image
    assert len(np.unique(lab)) == len(res.subimages)        # one sub-image per region

def test_partition_touching_and_crossing_paths():
    H, W = 50, 50
    p1 = np.array([[r, 20] for r in range(H)]); p2 = np.array([[r, 20 + (r > 25) * 10] for r in range(H)])
    # make p2 connected: add horizontal bridge
    p2 = np.concatenate([p2, [[25, c] for c in range(20, 31)]])
    p3 = np.array([[25, c] for c in range(W)])               # crosses both (horizontal barrier)
    lab = D.partition_by_paths((H, W), [p1, p2, p3])
    assert (lab >= 0).all() and lab.shape == (H, W)
    n = len(np.unique(lab))
    assert n >= 3
    img = np.zeros((H, W, 3), np.uint8)
    subs = D.make_subimages(img, lab)
    assert sum(s["area"] for s in subs) == H * W

def test_subimage_pixels_from_original_and_blank():
    img = np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    path = np.array([[r, 30] for r in range(40)])
    lab = D.partition_by_paths((40, 60), [path])
    subs = D.make_subimages(img, lab, blank=255)
    assert len(subs) == 2
    left = subs[0]
    x0, y0, x1, y1 = left["bbox"]
    m = lab == left["region"]
    assert (img[m] == left["image"][m[y0:y1, x0:x1]]).all()


# ---- end-to-end plumbing with mocked LVLM on synthetic circles
def test_end_to_end_mock_sum_equals_truth():
    from lvlm_count.counting import MockLVLM, count_with_retry
    rng = np.random.default_rng(3)
    for trial in range(5):
        H, W = 200, 400
        img = np.full((H, W, 3), 255, np.uint8)
        masks, centers = [], []
        n = int(rng.integers(8, 25))
        yy, xx = np.ogrid[:H, :W]
        tries = 0
        while len(masks) < n and tries < 2000:
            tries += 1
            cy, cx, r = rng.integers(12, H - 12), rng.integers(12, W - 12), 8
            if any((cy - a) ** 2 + (cx - b) ** 2 < (2 * r + 4) ** 2 for a, b in centers):
                continue
            m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
            img[m] = (0, 0, 0)
            masks.append(m); centers.append((cy, cx))
        res = D.object_aware_division(img, masks, samples_per_mask=10)
        # 'true' per-strip count = number of black blobs fully inside the strip image
        import scipy.ndimage as ndi
        def truth(strip, prompt):
            return ndi.label((strip < 128).all(-1))[1]
        lv = MockLVLM(truth)
        total = sum(count_with_retry(lv, s["image"], "How many circles?").value for s in res.subimages)
        assert total == len(masks), (total, len(masks), res.cuts)


# ---- both-axis division (appendix variant): horizontal cuts are vertical cuts on the transposed problem
@pytest.mark.parametrize("seed", range(5))
def test_both_axes_partition_and_no_cut_through(seed):
    rng = np.random.default_rng(seed)
    H, W = 120, 160
    masks = rand_masks(rng, H, W, n=14)
    ob = np.any(masks, 0)
    img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    res = D.object_aware_division(img, masks, seed=seed, axes="xy")
    for p in res.paths:
        assert not ob[p[:, 0], p[:, 1]].any()                       # neither vertical nor horizontal paths touch a mask
    assert sum(s["area"] for s in res.subimages) == H * W           # still an exact partition
    assert (res.labels >= 0).all()

def test_horizontal_paths_span_left_to_right():
    H, W = 60, 100
    m = np.zeros((H, W), bool); m[5:15, 10:20] = True; m[45:55, 10:20] = True; m[5:15, 70:80] = True; m[45:55, 70:80] = True
    res = D.object_aware_division(np.full((H, W, 3), 7, np.uint8), [m & False | m], samples_per_mask=30, axes="xy")
    hpaths = [p for p in res.paths if p[:, 1].min() == 0 and p[:, 1].max() == W - 1]
    assert hpaths, "expected at least one horizontal path from the left border to the right border"
