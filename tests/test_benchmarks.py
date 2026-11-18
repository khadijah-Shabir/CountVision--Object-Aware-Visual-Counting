import json
from pathlib import Path
import pytest

B = Path(__file__).resolve().parents[1] / "benchmarks"


def rows(name):
    p = B / f"{name}.jsonl"
    if not p.exists():
        pytest.skip(f"{name}.jsonl not built")
    return [json.loads(l) for l in open(p)]


def test_emoji_sizes():
    r = rows("emoji"); assert len(r) == 415 and len({x["image_path"] for x in r}) == 82
    assert all(30 <= x["gt_count"] <= 50 for x in r)


def test_pascal_sizes_and_zero_kept():
    r = rows("pascal"); assert len(r) == 102 and sum(x["gt_count"] == 0 for x in r) == 5


def test_fsc147_test_split():
    r = rows("fsc147"); assert len(r) == 1190 and len({x["id"] for x in r}) == 1190
    assert all(Path(x["image_path"]).exists() for x in r[:50])


def test_penguin_100_balanced_and_labeled():
    r = rows("penguin"); assert len(r) == 100 and all(0 <= x["gt_count"] <= 213 for x in r)
    assert len({x["gt_count"] for x in r}) > 50          # inverse-frequency sampling spreads the counts


def test_manifest_fields():
    for n in ("emoji", "pascal", "fsc147", "penguin"):
        for x in rows(n)[:5]:
            assert {"image_path", "question", "E", "gt_count", "source_split"} <= set(x)


def test_builders_deterministic(tmp_path):
    import subprocess, sys
    a = (B / "penguin.jsonl").read_bytes() if (B / "penguin.jsonl").exists() else pytest.skip("no penguin")
    root = B.parent / "scripts" / "build_benchmarks.py"
    subprocess.run([sys.executable, str(root), "penguin"], check=True, capture_output=True)
    assert (B / "penguin.jsonl").read_bytes() == a
