"""MAE, RMSE, Acc±k, t-interval over trials, paired bootstrap over images."""
import numpy as np
from scipy import stats


def mae(gt, pred):
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(gt, float))))


def rmse(gt, pred):
    return float(np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(gt, float)) ** 2)))


def acc_within(gt, pred, k=0):
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(gt, float)) <= k) * 100)


def bias(gt, pred):
    return float(np.mean(np.asarray(pred, float) - np.asarray(gt, float)))


def summarize(gt, pred):
    d = {"mae": mae(gt, pred), "rmse": rmse(gt, pred), "bias": bias(gt, pred), "n": int(len(gt))}
    for k in (0, 1, 3, 5, 10):
        d[f"acc_pm{k}"] = acc_within(gt, pred, k)
    return d


def t_ci(values, conf=0.95):
    """Mean and t-interval half-width over trial-level values (e.g. 6 trial MAEs)."""
    v = np.asarray(values, float)
    m = float(v.mean())
    if len(v) < 2:
        return m, float("nan")
    h = float(stats.sem(v) * stats.t.ppf((1 + conf) / 2, len(v) - 1))
    return m, h


def paired_bootstrap(gt, pred_a, pred_b, n_boot=10000, seed=1234):
    """Bootstrap over images of MAE(a) - MAE(b). Positive => b (e.g. LVLM-Count) is better.

    Returns mean diff, 95% CI, and the fraction of resamples in which b wins.
    """
    gt, a, b = (np.asarray(x, float) for x in (gt, pred_a, pred_b))
    ea, eb = np.abs(a - gt), np.abs(b - gt)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gt), size=(n_boot, len(gt)))
    diffs = ea[idx].mean(1) - eb[idx].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(ea.mean() - eb.mean()), "ci": (float(lo), float(hi)),
            "frac_b_wins": float((diffs > 0).mean()), "diffs": diffs}
