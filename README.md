# CountVision: Object-Aware Image Division for Zero-Shot Visual Counting with Vision-Language Models

CountVision is a training-free reimplementation of LVLM-Count. It detects the requested category, segments candidate objects, partitions the image along paths that avoid detected masks, and sums regional counts from Qwen2-VL-7B. The evaluation compares the full method with whole-image counting and simpler partitioning variants. The report documents the 96-image complete-case Penguin analysis, the baseline-only Synthetic Circles experiment, unresolved Pascal label disagreements, and the effect of extreme numeric answers on FSC-147 metrics.

- [Scientific report](report/countvision.pdf) and [editable manuscript](report/manuscript.md)
- [Evaluation summaries](results/tables/) and [implementation decisions](docs/decisions.md)
- [Citation metadata](CITATION.cff) and [rights notice](LICENSE)

## CPU checks and PDF build

From the repository root with Python 3.11 or a compatible newer version:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make test-cpu PY_PAPER=python
make paper PY_PAPER=python
```

The report builds from the manuscript and summary CSVs. It contains no benchmark photographs and requires no model weights, dataset downloads, or raw prediction logs.

## Reproduction inputs

The source and configuration are included. Place datasets and model checkpoints in the sibling directories expected by the benchmark builder and `configs/lvlm_count.yaml`, or set the model paths in the configuration. Review the [Makefile](Makefile) for the benchmark, preprocessing, job generation, inference, and aggregation targets. The detector and LVLM lockfiles record the environments used in the study. GPU wheels must be installed from a compatible PyTorch wheel index.

With local manifests and saved trial records present, `make paper-analysis` recalculates the publication metrics and `make figures` regenerates evaluation plots on CPU. The strict-format table also requires the original worker output. `results/tables/` includes both the original aggregate snapshot and the publication tables; `paired_by_image.csv` is the report's source-image clustered analysis.

The package excludes datasets, model files, trial and worker logs, and benchmark-derived imagery. See the [rights notice](LICENSE) for licensing status.
