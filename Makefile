# LVLM-Count replication.
# Two virtualenvs: ../.venv (detector stack, transformers 4.47.1) and ../.venv_lvlm (open LVLMs, transformers 4.57).
PY_DET?=../.venv/bin/python
PY_LV?=../.venv_lvlm/bin/python
PY_PAPER?=python3
SHARDS?=6
VARIANTS?=full,no_area,area_only,area_naive2,area_naive3,area_naive4,naive2,naive3,naive4,sam_all

test:            ## unit + integration + benchmark tests
	PYTHONPATH=src $(PY_DET) -m pytest tests -q

benchmarks:      ## deterministic manifests (seed 1234)
	$(PY_DET) scripts/build_benchmarks.py fsc147 emoji pascal penguin fsc147_val_pilot
	$(PY_DET) scripts/make_synthetic.py

precompute:      ## make precompute DS=fsc147 : stages 0-3 cached for every image/variant, one process per GPU
	@for i in $$(seq 0 $$(($(SHARDS)-1))); do \
	  CUDA_VISIBLE_DEVICES=$$i $(PY_DET) scripts/precompute.py --manifest benchmarks/$(DS).jsonl --variants $(VARIANTS) --shard $$i/$(SHARDS) & \
	done; wait

jobs:            ## make jobs DSS="fsc147" METHODS="base full" TRIALS=6 OUT=results/jobs/x.jsonl
	$(PY_LV) scripts/make_jobs.py --datasets $(DSS) --methods $(METHODS) --trials $(TRIALS) --out $(OUT)

run:             ## make run MODEL=../weights/Qwen2-VL-7B-Instruct JOBS=... OUT=results/jobs/q2vl7b/x GPUS="0 1 2 3 4 5"
	PY_LV="$(PY_LV)" scripts/run_jobs.sh $(MODEL) $(JOBS) $(OUT) "$(GPUS)"

aggregate:       ## make aggregate OUTS="results/jobs/q2vl7b/x_*.jsonl" BB=qwen2vl7b
	$(PY_LV) scripts/aggregate.py --outs "$(OUTS)" --backbone $(BB)

tables:
	$(PY_LV) scripts/make_tables.py

figures:
	$(PY_PAPER) scripts/figures/e_figures.py

test-cpu:
	PYTHONPATH=src $(PY_PAPER) -m pytest tests/test_division.py tests/test_parser_metrics.py -q

paper-analysis:  ## optional: requires saved trial records, manifests, and worker logs
	$(PY_PAPER) scripts/release_analysis.py

paper:           ## builds from checked-in CSVs, no inference
	$(PY_PAPER) report/build.py

.PHONY: test test-cpu benchmarks precompute jobs run aggregate tables figures paper paper-analysis
