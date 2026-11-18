#!/usr/bin/env python
"""Build all result tables from results/<dataset>/<method>/<backbone>/trial_*.jsonl -> results/tables/*.csv|md
Every number in the paper tables comes from here. Mask-count baseline is derived from the cache meta."""
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from lvlm_count import metrics as M

OUT = ROOT / "results" / "tables"; OUT.mkdir(parents=True, exist_ok=True)


def load(ds, method, bb):
    """-> {trial: DataFrame(image_id, gt, pred, n_subimages, parse_failures)}"""
    d = ROOT / "results" / ds / method / bb
    return {int(f.stem.split("_")[1]): pd.read_json(f, lines=True) for f in sorted(d.glob("trial_*.jsonl"))}


def trial_matrix(tr):
    """images x trials matrix of predictions, aligned on image_id (only ids present in every trial)."""
    ids = sorted(set.intersection(*[set(df.image_id) for df in tr.values()]))
    gt = tr[min(tr)].set_index("image_id").loc[ids, "gt"].values
    P = np.stack([df.set_index("image_id").loc[ids, "pred"].values for df in tr.values()], 1)
    return ids, gt, P


def row(ds, method, bb):
    tr = load(ds, method, bb)
    if not tr: return None
    ids, gt, P = trial_matrix(tr)
    maes = [M.mae(gt, P[:, k]) for k in range(P.shape[1])]
    rmses = [M.rmse(gt, P[:, k]) for k in range(P.shape[1])]
    m, h = M.t_ci(maes)
    r = dict(dataset=ds, method=method, backbone=bb, n_images=len(ids), n_trials=P.shape[1],
             MAE=m, MAE_CI95=h, RMSE=float(np.mean(rmses)), bias=float(np.mean(P.mean(1) - gt)))
    for k in (0, 1, 3, 5, 10):
        r[f"acc_pm{k}"] = float(np.mean([M.acc_within(gt, P[:, j], k) for j in range(P.shape[1])]))
    if "pred_raw" in next(iter(tr.values())).columns:      # same metrics without the plausibility bound
        raw = [M.mae(df.set_index("image_id").loc[ids, "gt"].values, df.set_index("image_id").loc[ids, "pred_raw"].values) for df in tr.values()]
        r["MAE_unfiltered"] = float(np.mean(raw)); r["n_implausible_answers"] = int(sum(df.n_implausible.sum() for df in tr.values()))
    r["median_AE"] = float(np.mean([np.median(np.abs(P[:, k] - gt)) for k in range(P.shape[1])]))
    r["subimages_mean"] = float(np.mean([df.n_subimages.mean() for df in tr.values()]))
    r["parse_fail_per_img"] = float(np.mean([df.parse_failures.mean() for df in tr.values()]))
    return r


def mask_count_baseline(ds):
    """Number of surviving post-processed masks summed over detected areas (full variant, deterministic)."""
    man = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "benchmarks" / f"{ds}.jsonl")}
    from precompute import cache_dir
    gt, pr = [], []
    for rid, r in man.items():
        p = cache_dir(ds, rid, "full") / "meta.json"
        if not p.exists(): continue
        gt.append(r["gt_count"]); pr.append(sum(json.load(open(p))["n_masks"]))
    s = M.summarize(gt, pr)
    return dict(dataset=ds, method="mask_count", backbone="-", n_images=s["n"], n_trials=1, MAE=s["mae"], MAE_CI95=np.nan,
                RMSE=s["rmse"], bias=s["bias"], **{f"acc_pm{k}": s[f"acc_pm{k}"] for k in (0, 1, 3, 5, 10)})


def main():
    rows = []
    for dsdir in sorted((ROOT / "results").iterdir()):
        if not (dsdir.is_dir() and (ROOT / "benchmarks" / f"{dsdir.name}.jsonl").exists()): continue
        for mdir in sorted(dsdir.iterdir()):
            for bdir in sorted(mdir.iterdir()) if mdir.is_dir() else []:
                r = row(dsdir.name, mdir.name, bdir.name)
                if r: rows.append(r)
        if (ROOT / "results" / "cache" / dsdir.name).exists():
            rows.append(mask_count_baseline(dsdir.name))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "all_results.csv", index=False)
    # paired bootstrap: full vs base per (dataset, backbone)
    bs = []
    for (ds, bb), g in df[df.method.isin(["base", "full"])].groupby(["dataset", "backbone"]):
        if set(g.method) != {"base", "full"}: continue
        ib, gt, Pb = trial_matrix(load(ds, "base", bb)); ifu, gt2, Pf = trial_matrix(load(ds, "full", bb))
        common = sorted(set(ib) & set(ifu)); ix = {k: i for i, k in enumerate(ib)}; jx = {k: i for i, k in enumerate(ifu)}
        gtc = np.array([gt[ix[k]] for k in common])
        a = np.array([Pb[ix[k]].mean() for k in common]); b = np.array([Pf[jx[k]].mean() for k in common])
        r = M.paired_bootstrap(gtc, a, b, 10000)
        bs.append(dict(dataset=ds, backbone=bb, mae_base=M.mae(gtc, a), mae_full=M.mae(gtc, b), diff=r["diff"],
                       ci_lo=r["ci"][0], ci_hi=r["ci"][1], frac_full_wins=r["frac_b_wins"]))
    pd.DataFrame(bs).to_csv(OUT / "paired_bootstrap.csv", index=False)
    with open(OUT / "main_results.md", "w") as f:
        for ds, g in df.groupby("dataset"):
            f.write(f"\n### {ds}\n\n" + g.drop(columns=["dataset"]).round(2).to_markdown(index=False) + "\n")
        if bs: f.write("\n### paired bootstrap (MAE base - MAE full, trial-averaged predictions; >0 = LVLM-Count better)\n\n"
                       + pd.DataFrame(bs).round(3).to_markdown(index=False) + "\n")
    print((OUT / "main_results.md").read_text())


if __name__ == "__main__":
    main()
