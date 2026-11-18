"""Rebuild publication tables from saved trial records, on CPU."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "tables"
DATASETS = ("fsc147", "pascal", "emoji", "penguin", "synthetic_circles")


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def save(name, rows):
    OUT.mkdir(exist_ok=True)
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_runs(dataset, manifest):
    runs = {}
    for folder in sorted((RESULTS / dataset).glob("*/qwen2vl7b")):
        trials = []
        paths = sorted(folder.glob("trial_*.jsonl"), key=lambda path: int(path.stem.split("_")[1]))
        for trial_id, path in enumerate(paths):
            if path.stem != f"trial_{trial_id}":
                raise ValueError(f"Expected contiguous trial IDs in {folder}")
            rows = read_jsonl(path)
            records = {str(row["image_id"]): row for row in rows}
            if len(records) != len(rows) or any(
                key not in manifest or row["gt"] != manifest[key]["gt_count"]
                for key, row in records.items()
            ):
                raise ValueError(f"Inconsistent trial records in {path}")
            trials.append(records)
        if trials:
            runs[folder.parent.name] = trials
    return runs


def matrix(trials, ids, key):
    return np.array([[trial[i][key] for i in ids] for trial in trials], dtype=float)


def bootstrap_samples(differences, groups, seed=1234):
    """Sample image clusters and weight each sampled question equally."""
    totals = np.bincount(groups, weights=differences)
    sizes = np.bincount(groups)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(20):
        idx = rng.integers(len(sizes), size=(500, len(sizes)))
        samples.extend(totals[idx].sum(axis=1) / sizes[idx].sum(axis=1))
    return np.array(samples)


def summarize_strict(runs_by_dataset):
    jobs = {}
    for pattern in ("main_all_*.jsonl", "batch2_*.jsonl"):
        for path in sorted((RESULTS / "jobs" / "q2vl7b").glob(pattern)):
            for row in read_jsonl(path):
                key = row["job_id"]
                if key in jobs and jobs[key] != row:
                    raise ValueError(f"Conflicting worker records for {key}")
                jobs[key] = row
    groups = defaultdict(list)
    for key, row in jobs.items():
        ds, image, method, _, trial = key.split("|")
        groups[ds, method, int(trial[1:]), image].append(row)
    if not jobs:
        return
    output = []
    for ds, runs in runs_by_dataset.items():
        for method, trials in runs.items():
            matched = []
            expected = 0
            mismatched = 0
            for trial_id, records in enumerate(trials):
                for image, row in records.items():
                    worker = groups.get((ds, method, trial_id, image), [])
                    counts = row["sub_counts"]
                    expected += len(counts)
                    if sorted(map(repr, counts)) != sorted(repr(job["value"]) for job in worker):
                        mismatched += 1
                        continue
                    matched.extend(worker)
            output.append(dict(dataset=ds, method=method, expected_jobs=expected,
                               verified_jobs=len(matched), unmatched_image_trials=mismatched,
                               jobs_with_strict_failure=sum(j["strict_failures"] > 0 for j in matched),
                               strict_failed_attempts=sum(j["strict_failures"] for j in matched),
                               jobs_exhausted=sum(j["value"] is None for j in matched)))
    save("strict_format.csv", output)


def main():
    metrics, pairs, masks = [], [], []
    runs_by_dataset = {}
    for ds in DATASETS:
        manifest = {str(row["id"]): row for row in read_jsonl(ROOT / "benchmarks" / f"{ds}.jsonl")}
        runs = load_runs(ds, manifest)
        runs_by_dataset[ds] = runs
        coverage = {method: set.intersection(*(set(trial) for trial in trials))
                    for method, trials in runs.items()}
        common = set.intersection(*coverage.values())
        for method, trials in runs.items():
            ids = sorted(coverage[method])
            gt = matrix(trials, ids, "gt")
            pred = matrix(trials, ids, "pred")
            raw = matrix(trials, ids, "pred_raw")
            ae = abs(pred - gt)
            pick = [i for i, image in enumerate(ids) if image in common]
            metrics.append(dict(dataset=ds, method=method, records=len(ids), trials=len(trials),
                                mae=ae.mean(), rmse=np.sqrt(((pred - gt)**2).mean(axis=1)).mean(),
                                raw_mae=abs(raw - gt).mean(), median_ae=np.median(ae, axis=1).mean(),
                                exact_pct=100 * (ae == 0).mean(), within5_pct=100 * (ae <= 5).mean(),
                                flagged=sum(row["n_implausible"] for trial in trials for row in trial.values()),
                                common_n=len(common), common_mae=ae[:, pick].mean(),
                                missing_ids=";".join(sorted(set(manifest) - coverage[method]))))
        if {"base", "full"} <= runs.keys():
            ids = sorted(coverage["base"] & coverage["full"])
            gt = matrix(runs["base"], ids, "gt")[0]
            base = abs(matrix(runs["base"], ids, "pred").mean(axis=0) - gt)
            full = abs(matrix(runs["full"], ids, "pred").mean(axis=0) - gt)
            _, groups = np.unique([manifest[i]["image_path"] for i in ids], return_inverse=True)
            samples = bootstrap_samples(base - full, groups)
            lo, hi = np.percentile(samples, [2.5, 97.5])
            wins = np.mean(samples > 0)
            pairs.append(dict(dataset=ds, records=len(ids), source_images=len(set(groups)),
                              base_mae=base.mean(), full_mae=full.mean(), diff=(base-full).mean(),
                              ci_lo=lo, ci_hi=hi, fraction_positive=wins))
    save("evaluation.csv", metrics)
    save("paired_by_image.csv", pairs)
    with (RESULTS / "tables" / "all_results.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["method"] == "mask_count":
                masks.append({key: row[key] for key in ("dataset", "n_images", "MAE", "RMSE")})
    save("mask_baseline.csv", masks)
    summarize_strict(runs_by_dataset)


if __name__ == "__main__":
    main()
