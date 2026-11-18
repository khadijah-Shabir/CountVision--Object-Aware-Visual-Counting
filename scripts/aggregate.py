#!/usr/bin/env python
"""Aggregate worker outputs into per-image predictions: results/<dataset>/<method>/<backbone>/trial_<k>.jsonl
and print metrics. pred = sum of sub-image counts (tiny sub-images count 0; unparsed => 0 and flagged)."""
import argparse, json, sys, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from lvlm_count import metrics as M
from precompute import cache_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outs", nargs="+", required=True); ap.add_argument("--backbone", required=True)
    ap.add_argument("--max-count", type=int, default=10000, help="answers above this are invalid (counted 0, flagged); raw kept as pred_raw")
    a = ap.parse_args()
    val, fail = {}, {}
    for pat in a.outs:
        for fp in glob.glob(pat):
            for l in open(fp):
                r = json.loads(l); val[r["job_id"]] = r["value"]; fail[r["job_id"]] = r["failures"]
    by = defaultdict(lambda: defaultdict(list))   # (ds,method,trial) -> id -> [(sub,val,fail)]
    for jid, v in val.items():
        ds, rid, m, fn, t = jid.split("|"); by[(ds, m, int(t[1:]))][rid].append((fn, v, fail[jid]))
    summary = defaultdict(list)
    for (ds, m, t), per in sorted(by.items()):
        man = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "benchmarks" / f"{ds}.jsonl")}
        out_dir = ROOT / "results" / ds / m / a.backbone; out_dir.mkdir(parents=True, exist_ok=True)
        gt, pr = [], []
        with open(out_dir / f"trial_{t}.jsonl", "w") as f:
            for rid, items in per.items():
                if rid not in man: continue
                r = man[rid]; counts = [v for _, v, _ in items]
                if m != "base":
                    meta = json.load(open(cache_dir(ds, rid, m.split("@")[0] if "@" in m else m) / "meta.json"))
                    n_sub = len(meta["subimages"])
                else: n_sub = 1
                pred_raw = int(sum(c or 0 for c in counts))
                bad = [c for c in counts if c is not None and c > a.max_count]
                pred = int(sum(c or 0 for c in counts if c is None or c <= a.max_count))
                f.write(json.dumps(dict(image_id=rid, question=r["question"], gt=r["gt_count"], pred=pred, n_subimages=n_sub,
                                        sub_counts=counts, pred_raw=pred_raw, n_implausible=len(bad), parse_failures=int(sum(x for _, _, x in items)), latency_s=None)) + "\n")
                gt.append(r["gt_count"]); pr.append(pred)
        s = M.summarize(gt, pr); summary[(ds, m)].append(s["mae"])
        print(f"{ds:8s} {m:12s} trial{t} n={s['n']:5d} MAE={s['mae']:.2f} RMSE={s['rmse']:.2f} bias={s['bias']:+.2f} acc0={s['acc_pm0']:.1f}")
    print("\n== mean MAE over trials ==")
    for (ds, m), v in summary.items():
        mean, h = M.t_ci(v); print(f"{ds:8s} {m:12s} MAE {mean:.2f} ±{h:.2f} (n_trials={len(v)})")


if __name__ == "__main__":
    main()
