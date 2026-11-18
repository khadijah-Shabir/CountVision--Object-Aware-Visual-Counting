#!/usr/bin/env python
"""Build LVLM job files from manifests + cached sub-images.

  python scripts/make_jobs.py --datasets pascal emoji --methods base full --trials 6 --out results/jobs/main.jsonl
methods: base (original image) | full | no_area | area_only | area_naive{2,3,4} | naive{2,3,4}
Job id: dataset|id|method|subfile|t<trial>   (tiny sub-images are not sent; counted 0 at aggregation)
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from lvlm_count.counting import build_prompt
from precompute import cache_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True); ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=6); ap.add_argument("--out", required=True)
    ap.add_argument("--zero-aware", type=int, default=1); ap.add_argument("--cot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    import random
    lines = []
    f = type("L", (), {"write": lambda self, x: lines.append(x)})()
    n = 0
    if True:
        for ds in a.datasets:
            rows = [json.loads(l) for l in open(ROOT / "benchmarks" / f"{ds}.jsonl")]
            if a.limit: rows = rows[:a.limit]
            for r in rows:
                prompt = build_prompt(r["question"], bool(a.zero_aware), bool(a.cot))
                prompt_mask = build_prompt("How many distinct coloured regions are in this image?", bool(a.zero_aware), bool(a.cot))
                for m in a.methods:
                    if m == "base":
                        items = [("base.png", r["image_path"])]
                    else:
                        d = cache_dir(ds, r["id"], m)
                        if not (d / "meta.json").exists(): continue
                        meta = json.load(open(d / "meta.json"))
                        items = [(s["file"], str(d / s["file"])) for s in meta["subimages"] if not s["tiny"]]
                    for fn, path in items:
                        for t in range(a.trials):
                            f.write(json.dumps(dict(job_id=f"{ds}|{r['id']}|{m}{a.tag}|{fn}|t{t}", image_path=path,
                                                    prompt=prompt_mask if m == "maskimg" else prompt, seed=1000 * t + 7)) + "\n"); n += 1
    random.Random(1234).shuffle(lines)     # a crashed shard then loses a random slice, not a whole trial
    open(a.out, "w").write("".join(lines))
    print(n, "jobs ->", a.out)


if __name__ == "__main__":
    main()
