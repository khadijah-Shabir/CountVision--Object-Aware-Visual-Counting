#!/usr/bin/env python
"""Stages 0-3 for every benchmark image/variant, cached to disk (deterministic, model-only-GDINO/SAM).

  python scripts/precompute.py --manifest benchmarks/pascal.jsonl --variants full --shard 0/6 --device cuda
Cache layout: results/cache/<dataset>/<id>/<variant>/{meta.json, sub_000.png, ...}
Variants: full | no_area | area_only | area_naive{2,3,4} | naive{2,3,4}
"""
import argparse, hashlib, json, re, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lvlm_count.config import load_config, config_hash, PROJECT


def safe(s): return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))[:80]


def cache_dir(dataset, rid, variant, root=None):
    return Path(root or ROOT / "results" / "cache") / dataset / safe(rid) / variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--variants", default="full")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config); ch = config_hash(cfg)
    rows = [json.loads(l) for l in open(a.manifest)]
    ds = Path(a.manifest).stem
    if a.limit: rows = rows[:a.limit]
    i, n = map(int, a.shard.split("/")); rows = rows[i::n]

    from lvlm_count.grounding import Grounder
    from lvlm_count.segmentation import Segmenter
    from lvlm_count.pipeline import LVLMCount
    from lvlm_count.counting import MockLVLM
    need_models = any(v in ("full", "no_area", "sam_all", "full_xy", "maskimg", "area_only", "area_naive2", "area_naive3", "area_naive4")
                      for v in a.variants.split(","))
    g = Grounder(str(PROJECT / cfg["grounding"]["model_dir"]), a.device) if need_models else None
    s = Segmenter(str(PROJECT / cfg["sam"]["checkpoint"]), cfg["sam"]["model_type"], a.device) if need_models else None
    pipe = LVLMCount(cfg, MockLVLM(lambda *_: 0), g, s)

    t0, done = time.time(), 0
    for r in rows:
        for v in a.variants.split(","):
            d = cache_dir(ds, r["id"], v, a.cache_root)
            if (d / "meta.json").exists():
                try:
                    if json.load(open(d / "meta.json")).get("config_hash") == ch: continue
                except Exception: pass
            variant, k = v, 3
            m = re.match(r"(area_naive|naive)(\d)$", v)
            if m: variant, k = m.group(1), int(m.group(2))
            try:
                out = pipe.divide(r["image_path"], r["E"], variant=variant, naive_k=k, keep_debug=True, seed=1234)
            except Exception as e:
                print("FAIL", ds, r["id"], v, repr(e)[:200], flush=True); traceback.print_exc(limit=2)
                continue
            try:
                d.mkdir(parents=True, exist_ok=True)
                subs = []
                for sb in out.debug["subimages"]:
                    im = sb["image"]
                    if im.shape[0] == 0 or im.shape[1] == 0:
                        continue
                    fn = f"sub_{len(subs):03d}.png"
                    Image.fromarray(im).save(d / fn)
                    subs.append(dict(file=fn, bbox=list(map(int, sb["bbox"])), area=int(sb["area"]), tiny=bool(sb.get("tiny", False))))
                meta = out.debug["meta"]; meta.update(config_hash=ch, subimages=subs, id=r["id"], dataset=ds)
                json.dump(meta, open(d / "meta.json", "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            except Exception as e:
                print("FAIL-SAVE", ds, r["id"], v, repr(e)[:200], flush=True)
        done += 1
        if done % 20 == 0:
            print(f"[{a.shard}] {done}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
    print(f"[{a.shard}] finished {done} rows in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
