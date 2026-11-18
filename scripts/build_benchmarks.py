#!/usr/bin/env python
"""Build deterministic benchmark manifests -> benchmarks/<name>.jsonl (seed 1234).

fields: image_path, question, E, gt_count, source_split (+ id, extra)
Usage: python scripts/build_benchmarks.py [fsc147 emoji pascal penguin ...]
"""
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT.parent / "datasets"
OUT = ROOT / "benchmarks"
SEED = 1234
VOC = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
       "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
VOC_NAME = {"aeroplane": "aeroplane", "diningtable": "dining table", "motorbike": "motorbike",
            "pottedplant": "potted plant", "tvmonitor": "tv monitor"}


def write(name, rows):
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(rows)} rows -> benchmarks/{name}.jsonl")


def fsc147(split="test"):
    d = DS / "fsc147"
    splits = json.load(open(d / "Train_Test_Val_FSC_147.json"))
    ann = json.load(open(d / "annotation_FSC147_384.json"))
    cls = dict(l.rstrip("\n").split("\t") for l in open(d / "ImageClasses_FSC147.txt"))
    rows = []
    for k in splits[split]:
        E = cls[k]
        rows.append(dict(id=k, image_path=str(d / "images_384_VarV2" / k), E=E,
                         question=f"How many {E} are there?", gt_count=len(ann[k]["points"]),
                         source_split=f"fsc147-{split}"))
    return rows


def fsc147_val_pilot(n=100):
    """100 seeded FSC-147 VAL images for the pilot / debugging (never the test set)."""
    rows = fsc147("val")
    idx = np.random.default_rng(SEED).choice(len(rows), n, replace=False)
    return [rows[i] for i in sorted(idx)]


def emoji():
    d = DS / "emoji_benchmark"
    rows = []
    for r in json.load(open(d / "benchmark_data.json")):
        E = r["object_of_interest"]
        rows.append(dict(id=str(r["id"]), image_path=str(d / "images" / r["image_file"]), E=E,
                         question=f"How many {E} are visible in the image?", gt_count=int(r["answer"]),
                         source_split="emoji-count", extra={"class": r["class"]}))
    return rows


def voc_counts(ann_dir, ids):
    """(image, class) -> (n_all, n_non_difficult)."""
    out = {}
    for i in ids:
        c_all, c_nd = Counter(), Counter()
        for o in ET.parse(ann_dir / f"{i}.xml").findall("object"):
            n = o.find("name").text
            c_all[n] += 1
            if o.find("difficult").text == "0":
                c_nd[n] += 1
        for c in VOC:
            out[(i, c)] = (c_all[c], c_nd[c])
    return out


def pascal():
    """Primary = the authors' released, manually corrected 102 questions (all from VOC07 test).
    Also writes a seeded rebuild (5 questions per distinct GT value) and an agreement report."""
    d = DS / "pascal"
    voc = d / "VOCdevkit" / "VOC2007"
    test_ids = open(voc / "ImageSets/Main/test.txt").read().split()
    auth = pd.read_csv(d / "pascal.csv")
    rows = []
    for r in auth.itertuples():
        E = r.category_name
        rows.append(dict(id=r.image_name, image_path=str(voc / "JPEGImages" / r.image_name), E=E,
                         question=f"How many {E}s are in the image?", gt_count=int(r.answer),
                         source_split="voc07-test (authors' 102 questions)"))
    write("pascal", rows)

    # ---- rebuild from raw annotations (comparison only)
    counts = voc_counts(voc / "Annotations", test_ids)
    by_gt = defaultdict(list)
    for (i, c), (n_all, n_nd) in counts.items():
        by_gt[n_all].append((i, c, n_all))
    rng = np.random.default_rng(SEED)
    rb = []
    for g in sorted(by_gt):
        pool = by_gt[g]
        sel = rng.choice(len(pool), size=min(5, len(pool)), replace=False)
        rb += [pool[j] for j in sorted(sel)]
    rebuilt = [dict(id=f"{i}.jpg", image_path=str(voc / "JPEGImages" / f"{i}.jpg"),
                    E=VOC_NAME.get(c, c), question=f"How many {VOC_NAME.get(c, c)}s are in the image?",
                    gt_count=g, source_split="voc07-test (rebuilt from raw annotations)") for i, c, g in rb]
    write("pascal_rebuilt", rebuilt)
    # agreement of authors' corrected GT with raw annotation counts (difficult objects included / excluded)
    inv = {v: k for k, v in VOC_NAME.items()}
    agree_all = agree_nd = 0
    for r in auth.itertuples():
        i = r.image_name.replace(".jpg", ""); c = inv.get(r.category_name, r.category_name)
        n_all, n_nd = counts[(i, c)]
        agree_all += int(n_all == r.answer); agree_nd += int(n_nd == r.answer)
    pairs_a = {(r["id"], r["E"]) for r in rows}; pairs_b = {(r["id"], r["E"]) for r in rebuilt}
    rep = dict(authors_n=len(rows), rebuilt_n=len(rebuilt), overlap_pairs=len(pairs_a & pairs_b),
               authors_gt_equals_raw_all=agree_all, authors_gt_equals_raw_nondifficult=agree_nd,
               note="difficult-flagged objects: counted (n_all) in the rebuild; see DECISIONS.md")
    json.dump(rep, open(OUT / "pascal_agreement.json", "w"), indent=2)
    print("pascal agreement:", rep)


def penguin(n=100):
    """Separate-site TEST split (21,736 imgs == penguin.csv), GT = max over annotators,
    sampled with probability inversely proportional to the frequency of its GT value."""
    d = DS / "penguin_full"
    df = pd.read_csv(DS / "penguin" / "penguin.csv")
    def find(n_):
        for p in (d / "images" / f"{n_}.JPG", DS / "penguin" / "images" / f"{n_}.JPG"):
            if p.exists(): return p
        for p in d.rglob(f"{n_}.JPG"): return p
        return None
    idx = {}
    for root in (d, DS / "penguin" / "images"):
        for p in root.rglob("*.JPG"): idx[p.stem] = p
    df["path"] = df.image_name.map(idx)
    n_unlabeled = int(df["max"].isna().sum())
    avail = df[df.path.notna() & df["max"].notna()]   # 1,649 test images have no annotations at all
    print(f"penguin: {len(avail)}/{len(df)} test images usable ({n_unlabeled} unlabeled excluded); sites:",
          dict(Counter(avail.image_name.str.split("_").str[0])))
    freq = avail["max"].map(avail["max"].value_counts())
    w = (1 / freq); w /= w.sum()
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(avail), size=n, replace=False, p=w.values)
    sel = avail.iloc[sorted(pick)]
    rows = [dict(id=r.image_name, image_path=str(r.path), E="penguins",
                 question="How many penguins are in the image?", gt_count=int(r.max),
                 source_split="penguin-sepsites-test",
                 extra={"annotator_counts": r.number_vector}) for r in sel.itertuples()]
    write("penguin", rows)
    if len(avail) < len(df):
        print("WARNING: penguin sampled from the locally available subset only")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["fsc147", "emoji", "pascal", "penguin"]
    for t in todo:
        r = globals()[t]()
        if r is not None:
            write(t, r)
