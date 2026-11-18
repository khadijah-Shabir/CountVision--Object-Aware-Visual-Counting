#!/usr/bin/env python
"""Evaluation figures E2-E7, E9, E11, E12 (reads only results/). Usage: python scripts/figures/e_figures.py [backbone ...]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import COL, save
from lvlm_count import metrics as M
sys.path.insert(0, str(ROOT / "scripts"))
from release_analysis import bootstrap_samples, read_jsonl

R = ROOT / "results"
BBS = sys.argv[1:] or sorted({p.name for p in R.glob("*/*/*") if p.is_dir() and p.parent.parent.name in ("fsc147", "pascal", "emoji", "penguin")})
DS = [d for d in ("fsc147", "pascal", "emoji", "penguin") if (R / d).exists()]


def load(ds, method, bb):
    d = R / ds / method / bb
    return {int(f.stem.split("_")[1]): pd.read_json(f, lines=True) for f in sorted(d.glob("trial_*.jsonl"))} if d.exists() else {}


def mat(tr):
    ids = sorted(set.intersection(*[set(df.image_id.astype(str)) for df in tr.values()]))
    P = np.stack([df.assign(image_id=df.image_id.astype(str)).set_index("image_id").loc[ids, "pred"].values for df in tr.values()], 1)
    t0 = next(iter(tr.values())).assign(image_id=lambda d: d.image_id.astype(str)).set_index("image_id")
    return ids, t0.loc[ids, "gt"].values, P, t0.loc[ids, "n_subimages"].values


def best_naive(ds, bb):
    best = None
    for k in (2, 3, 4):
        tr = load(ds, f"naive{k}", bb)
        if tr:
            ids, gt, P, _ = mat(tr); m = np.mean([M.mae(gt, P[:, j]) for j in range(P.shape[1])])
            if best is None or m < best[0]: best = (m, f"naive{k}")
    return best[1] if best else None


def pair_ids(ds, bb):
    runs = [load(ds, method, bb) for method in ("base", "full")]
    observed = [set(mat(tr)[0]) for tr in runs if tr]
    return set.intersection(*observed) if observed else set()


# ---------------- E2 headline bars
for bb in BBS:
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2)); ok = False
    for ax, metric in zip(axs, ("MAE", "RMSE")):
        x = 0
        for ds in DS:
            shared = pair_ids(ds, bb)
            for m, c in (("base", COL["base"]), ("full", COL["lvlm_count"])):
                tr = load(ds, m, bb)
                if not tr: continue
                ok = True; ids, gt, P, _ = mat(tr)
                keep = [i for i, key in enumerate(ids) if key in shared]
                gt, P = gt[keep], P[keep]
                vals = [M.mae(gt, P[:, j]) if metric == "MAE" else M.rmse(gt, P[:, j]) for j in range(P.shape[1])]
                mean, h = M.t_ci(vals)
                ax.bar(x, mean, 0.8, color=c, yerr=0 if np.isnan(h) else h, capsize=2, label=("Base LVLM" if m == "base" else "LVLM-Count") if ds == DS[0] else None)
                x += 1
            x += 0.6
        ax.set_xticks([i * 2.6 + 0.5 for i in range(len(DS))]); ax.set_xticklabels(DS); ax.set_ylabel(metric + " (lower is better)")
    if ok:
        axs[0].legend(); fig.suptitle(f"E2 headline: base vs LVLM-Count, backbone {bb}. Error bars: 95% t-interval over 6 trials; common records", y=1.02, fontsize=9)
        save(fig, f"E2_headline_{bb}")
    else: plt.close(fig)

# ---------------- E3 MAE vs GT range (FSC-147) + E7 cumulative accuracy + E4 scatter + E5 signed error + E9 + E11
BINS = [0, 20, 50, 100, 200, 1e9]; LAB = ["[0,20)", "[20,50)", "[50,100)", "[100,200)", "[200,inf)"]
for bb in BBS:
    tb, tf = load("fsc147", "base", bb), load("fsc147", "full", bb)
    if tb and tf:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for tr, c, lab in ((tb, COL["base"], "Base LVLM"), (tf, COL["lvlm_count"], "LVLM-Count")):
            ids, gt, P, _ = mat(tr); b = np.digitize(gt, BINS) - 1; ys, es = [], []
            for k in range(5):
                v = [M.mae(gt[b == k], P[b == k, j]) for j in range(P.shape[1])] if (b == k).any() else [np.nan]
                m_, h_ = M.t_ci(v); ys.append(m_); es.append(0 if np.isnan(h_) else h_)
            ax.errorbar(range(5), ys, yerr=es, marker="o", color=c, label=lab, capsize=2)
        ax.set_xticks(range(5)); ax.set_xticklabels(LAB); ax.set_xlabel("ground-truth count"); ax.set_ylabel("MAE"); ax.legend()
        ax.set_title(f"E3 MAE by GT range, FSC-147 test, {bb}", fontsize=9); save(fig, f"E3_mae_vs_gt_{bb}")

    # E7 cumulative accuracy
    fig, axs = plt.subplots(1, len(DS), figsize=(3.2 * len(DS), 2.9), squeeze=False); ok = False
    for ax, ds in zip(axs[0], DS):
        shared = pair_ids(ds, bb)
        for m, c, lab in (("base", COL["base"], "Base"), ("full", COL["lvlm_count"], "LVLM-Count")):
            tr = load(ds, m, bb)
            if not tr: continue
            ok = True; ids, gt, P, _ = mat(tr); ks = np.arange(0, 21)
            keep = [i for i, key in enumerate(ids) if key in shared]
            gt, P = gt[keep], P[keep]
            ax.plot(ks, [np.mean([M.acc_within(gt, P[:, j], k) for j in range(P.shape[1])]) for k in ks], color=c, label=lab)
        ax.set_title(ds); ax.set_xlabel("tolerance k"); ax.set_ylabel("% within +-k")
    if ok: axs[0][0].legend(); fig.suptitle(f"E7 cumulative accuracy, {bb}", y=1.03, fontsize=9); save(fig, f"E7_cumulative_acc_{bb}")
    else: plt.close(fig)

    # E4 scatter (log-log, trial-averaged prediction)
    fig, axs = plt.subplots(1, len(DS), figsize=(3.3 * len(DS), 3.1), squeeze=False); ok = False
    for ax, ds in zip(axs[0], DS):
        shared = pair_ids(ds, bb)
        for m, c, lab in (("base", COL["base"], "Base"), ("full", COL["lvlm_count"], "LVLM-Count")):
            tr = load(ds, m, bb)
            if not tr: continue
            ok = True; ids, gt, P, _ = mat(tr)
            keep = [i for i, key in enumerate(ids) if key in shared]
            ax.scatter(gt[keep] + 1, P[keep].mean(1) + 1, s=6, alpha=0.5, color=c, label=lab, lw=0)
        lim = [1, 4000]; ax.plot(lim, lim, "k--", lw=0.7); ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(ds); ax.set_xlabel("true count + 1"); ax.set_ylabel("predicted + 1")
    if ok: axs[0][0].legend(markerscale=2); fig.suptitle(f"E4 predicted vs true, {bb} (mean over trials)", y=1.03, fontsize=9); save(fig, f"E4_scatter_{bb}")
    else: plt.close(fig)

    # E5 signed error (FSC: base / best naive / full)
    meths = [("base", "Base", COL["base"]), (best_naive("fsc147", bb), "Naive division", COL["naive"]), ("full", "LVLM-Count", COL["lvlm_count"])]
    data = []
    for m, lab, c in meths:
        tr = load("fsc147", m, bb) if m else {}
        if tr: ids, gt, P, _ = mat(tr); data.append((lab, c, np.clip((P - gt[:, None]).ravel(), -300, 300)))
    if len(data) >= 2:
        fig, ax = plt.subplots(figsize=(4.6, 3.2)); v = ax.violinplot([d[2] for d in data], showmedians=True)
        for body, d in zip(v["bodies"], data): body.set_facecolor(d[1]); body.set_alpha(0.7)
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(1, len(data) + 1)); ax.set_xticklabels([d[0] for d in data]); ax.set_ylabel("pred - GT (clipped at +-300)")
        ax.set_title(f"E5 signed error, FSC-147 test, {bb}", fontsize=9); save(fig, f"E5_signed_error_{bb}")

    # E9 sub-image analysis
    tf = load("fsc147", "full", bb)
    if tf:
        ids, gt, P, ns = mat(tf)
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        axs[0].hist(ns, bins=np.arange(0.5, max(ns) + 1.5), color=COL["lvlm_count"]); axs[0].set_xlabel("number of sub-images"); axs[0].set_ylabel("images")
        g = np.minimum(ns, 4); tb = load("fsc147", "base", bb)
        for k, lab in ((1, "1"), (2, "2"), (3, "3"), (4, "4+")):
            sel = g == k
            if sel.any(): axs[1].bar(k - 0.2, np.mean([M.mae(gt[sel], P[sel, j]) for j in range(P.shape[1])]), 0.35, color=COL["lvlm_count"])
            if sel.any() and tb:
                ib, gtb, Pb, _ = mat(tb); idx = {i: n for n, i in enumerate(ib)}; rows = [idx[i] for i, s in zip(ids, sel) if s and i in idx]
                axs[1].bar(k + 0.2, np.mean([M.mae(gtb[rows], Pb[rows, j]) for j in range(Pb.shape[1])]), 0.35, color=COL["base"])
        axs[1].set_xticks([1, 2, 3, 4]); axs[1].set_xticklabels(["1 (=base)", "2", "3", "4+"]); axs[1].set_ylabel("MAE"); axs[1].set_xlabel("sub-images produced (blue LVLM-Count, grey base)")
        fig.suptitle(f"E9 sub-image analysis, FSC-147 test, {bb}; single-area/no-cut images: {np.mean(ns == 1)*100:.1f}%", y=1.03, fontsize=9); save(fig, f"E9_subimages_{bb}")

    # E11 paired bootstrap histograms
    fig, axs = plt.subplots(1, len(DS), figsize=(3.2 * len(DS), 2.8), squeeze=False); ok = False
    for ax, ds in zip(axs[0], DS):
        tb, tf = load(ds, "base", bb), load(ds, "full", bb)
        if not (tb and tf): ax.axis("off"); continue
        ok = True; ib, gt, Pb, _ = mat(tb); jf, gt2, Pf, _ = mat(tf); common = sorted(set(ib) & set(jf))
        ix = {k: i for i, k in enumerate(ib)}; jx = {k: i for i, k in enumerate(jf)}
        gtc = np.array([gt[ix[k]] for k in common]); a = np.array([Pb[ix[k]].mean() for k in common]); b = np.array([Pf[jx[k]].mean() for k in common])
        manifest = {str(row["id"]): row["image_path"]
                    for row in read_jsonl(ROOT / "benchmarks" / f"{ds}.jsonl")}
        _, groups = np.unique([manifest[k] for k in common], return_inverse=True)
        diffs = bootstrap_samples(abs(a - gtc) - abs(b - gtc), groups)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        ax.hist(diffs, bins=40, color=COL["light"])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{ds}: 95% CI [{lo:.1f}, {hi:.1f}]", fontsize=8)
        ax.set_xlabel("MAE(base) - MAE(LVLM-Count)")
    if ok:
        fig.suptitle(f"E11 paired source-image bootstrap (10,000 resamples), {bb}", y=1.04, fontsize=9)
        save(fig, f"E11_bootstrap_{bb}")
    else: plt.close(fig)

    # E12 per-class change (FSC)
    tb, tf = load("fsc147", "base", bb), load("fsc147", "full", bb)
    if tb and tf:
        man = {json.loads(l)["id"]: json.loads(l)["E"] for l in open(ROOT / "benchmarks/fsc147.jsonl")}
        ib, gt, Pb, _ = mat(tb); jf, _, Pf, _ = mat(tf); common = sorted(set(ib) & set(jf))
        ix = {k: i for i, k in enumerate(ib)}; jx = {k: i for i, k in enumerate(jf)}
        rows = [(man[k], abs(Pf[jx[k]].mean() - gt[ix[k]]) - abs(Pb[ix[k]].mean() - gt[ix[k]])) for k in common]
        df = pd.DataFrame(rows, columns=["cls", "d"]).groupby("cls").d.agg(["mean", "count"]).query("count>=2").sort_values("mean")
        sel = pd.concat([df.head(10), df.tail(10)])
        fig, ax = plt.subplots(figsize=(5, 4.6)); ax.barh(range(len(sel)), sel["mean"], color=[COL["lvlm_count"] if v < 0 else COL["accent"] for v in sel["mean"]])
        ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel.index, fontsize=7); ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("change in absolute error (negative = LVLM-Count better)"); ax.set_title(f"E12 per-class change, 10 best / 10 worst classes (>=2 imgs), {bb}", fontsize=8); save(fig, f"E12_per_class_{bb}")

# ---------------- E6 ablation bars (FSC-147, first backbone with data)
ABL = [("Base", "base"), ("+ naive division", "NAIVE"), ("+ area det. + naive", "AREANAIVE"), ("+ area det. only", "area_only"),
       ("SAM-everything masks", "sam_all"), ("object-aware, no area det.", "no_area"), ("Full LVLM-Count", "full")]
for bb in BBS:
    vals = []
    for lab, m in ABL:
        if m in ("NAIVE", "AREANAIVE"):
            pre = "naive" if m == "NAIVE" else "area_naive"; cand = [(k, load("fsc147", f"{pre}{k}", bb)) for k in (2, 3, 4)]
            cand = [(k, tr) for k, tr in cand if tr]
            if not cand: continue
            res = []
            for k, tr in cand: ids, gt, P, _ = mat(tr); res.append((np.mean([M.mae(gt, P[:, j]) for j in range(P.shape[1])]), M.t_ci([M.mae(gt, P[:, j]) for j in range(P.shape[1])])[1], k))
            mm, hh, kk = min(res); vals.append((f"{lab} (best k={kk})", mm, hh, COL["naive"]))
        else:
            tr = load("fsc147", m, bb)
            if not tr: continue
            ids, gt, P, _ = mat(tr); v = [M.mae(gt, P[:, j]) for j in range(P.shape[1])]; mm, hh = M.t_ci(v)
            vals.append((lab, mm, hh, COL["base"] if m == "base" else COL["lvlm_count"] if m == "full" else COL["light"]))
    if len(vals) >= 3:
        fig, ax = plt.subplots(figsize=(6.5, 3.4)); ax.barh(range(len(vals)), [v[1] for v in vals], xerr=[0 if np.isnan(v[2]) else v[2] for v in vals], color=[v[3] for v in vals], capsize=2)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels([v[0] for v in vals]); ax.invert_yaxis(); ax.set_xlabel("MAE (FSC-147 test, 3 or 6 trials, 95% CI)")
        ax.set_title(f"E6 ablation, {bb}. Naive row uses the best of k in {{2,3,4}} (favourable to naive)", fontsize=8); save(fig, f"E6_ablation_{bb}")
print("E-figures done for", BBS, "->", ROOT / "results/figures")
