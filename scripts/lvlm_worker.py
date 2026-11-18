#!/usr/bin/env python
"""Run LVLM counting jobs (jsonl: job_id,image_path,prompt,seed) on this process's GPUs. Resumable.

  CUDA_VISIBLE_DEVICES=0 python scripts/lvlm_worker.py --model weights/Qwen2-VL-7B-Instruct \
      --jobs jobs.jsonl --out out_0.jsonl --shard 0/6 --batch 16
"""
import argparse, json, sys, time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lvlm_count.counting import parse_count, ParseError
from lvlm_count.hf_lvlm import HFVLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--max-retries", type=int, default=10)
    ap.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    ap.add_argument("--sampling", default="default", choices=["default", "t1"])
    a = ap.parse_args()
    import torch
    n_gpu = torch.cuda.device_count()
    jobs = [json.loads(l) for l in open(a.jobs)]
    i, n = map(int, a.shard.split("/")); jobs = jobs[i::n]
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["job_id"] for l in open(a.out)}
    jobs = [j for j in jobs if j["job_id"] not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} done)", flush=True)
    if not jobs: return
    # sort by pixel count so batches have similar sizes
    for j in jobs:
        with Image.open(j["image_path"]) as im: j["_px"] = im.size[0] * im.size[1]
    jobs.sort(key=lambda j: j["_px"])
    vlm = HFVLM(a.model, gpus=list(range(n_gpu)), max_pixels=a.max_pixels, sampling=a.sampling)
    print('generation config:', vlm.model.generation_config.to_dict() if a.sampling == 'default' else 'T=1.0 top_p=1 top_k=off', flush=True)
    t0 = time.time(); fout = open(a.out, "a")
    for b in range(0, len(jobs), a.batch):
        batch = jobs[b:b + a.batch]
        res = {j["job_id"]: dict(job_id=j["job_id"], value=None, failures=0, strict_failures=0, raw=None) for j in batch}
        pending = list(batch)
        for attempt in range(a.max_retries):
            if not pending: break
            ims = [Image.open(j["image_path"]).convert("RGB") for j in pending]
            try:
                outs = vlm.ask_batch(ims, [j["prompt"] for j in pending], seed=pending[0]["seed"] + attempt)
            except Exception as e:  # e.g. OOM: halve the batch via single calls
                print("batch error", repr(e)[:120], flush=True); torch.cuda.empty_cache()
                outs = []
                for im, j in zip(ims, pending):
                    try: outs.append(vlm.ask_batch([im], [j["prompt"]], seed=j["seed"] + attempt)[0])
                    except Exception as e2: outs.append("ERROR:" + repr(e2)[:100])
            nxt = []
            for j, o in zip(pending, outs):
                res[j["job_id"]]["raw"] = o
                try:
                    res[j["job_id"]]["value"] = parse_count(o)            # strict spec regex
                except ParseError:
                    res[j["job_id"]]["strict_failures"] += 1               # logged for the failure-rate table
                    try: res[j["job_id"]]["value"] = parse_count(o, lenient=True)
                    except ParseError:
                        res[j["job_id"]]["failures"] += 1; nxt.append(j)
            pending = nxt
        for r in res.values(): fout.write(json.dumps(r) + "\n")
        fout.flush()
        if (b // a.batch) % 10 == 0:
            print(f"{b+len(batch)}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    print("finished", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
