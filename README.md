# Industrial Visual Inspection

Unsupervised defect detection and anomaly localization on industrial images, built on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset.

The models are trained on **defect-free images only** and flag anything that deviates from "normal". Each prediction comes with an anomaly score and a pixel-level heatmap showing where the defect is.

> **Status:** work in progress (week 1 of 6: data pipeline and metrics). No results yet; this README will be updated as experiments finish.

## Goals

- Implement and compare **PaDiM** and **PatchCore** (plus a reference run with Anomalib) on five MVTec AD categories: `bottle`, `cable`, `capsule`, `metal_nut`, `screw`.
- Evaluate with image-level and pixel-level AUROC, AUPRO, and a threshold chosen on a held-out calibration split (never on test data).
- Serve the best model through a FastAPI backend and a small React dashboard, packaged with Docker.

## Project structure

```
src/            data, models, training, evaluation, inference, utils
scripts/        helper scripts (e.g. dataset preparation)
configs/        experiment configs
experiments/    raw experiment outputs
notebooks/      exploration (EDA) and Colab runs
backend/        FastAPI service (planned)
frontend/       React dashboard (planned)
tests/          unit tests
docs/           decision log and reports
data/datasets/  datasets (not tracked by git)
```

## Setup

Requires Python 3.12 (tested on 3.12).

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`.

### GPU note

`requirements.txt` pins `torch` and `torchvision` **without** a CPU/CUDA suffix, so the default PyPI build is installed (CPU-only on Windows). If you have an NVIDIA GPU, install the matching CUDA build of PyTorch by following the selector on [pytorch.org](https://pytorch.org/get-started/locally/) after the step above. See `docs/decision-log.md` for the reasoning.

## Data

The dataset is **not** included in this repository.

1. Download the full MVTec AD archive (`mvtec-ad.zip`) from the [official page](https://www.mvtec.com/company/research/datasets/mvtec-ad) and read its license.
2. Extract the five categories used in this project:

```powershell
python scripts/prepare_mvtec.py --zip path\to\mvtec-ad.zip
```

By default the data is written to `data/datasets/mvtec_ad/`. Options:

| Option | Meaning |
|---|---|
| `--zip` | Path to `mvtec-ad.zip` (required) |
| `--out` | Output folder (default: `data/datasets/mvtec_ad`) |
| `--categories` | One or more category names (default: the five above) |

The script stops with an error if a category has no files in the archive, so a typo cannot silently produce an empty folder.

## Roadmap

| Week | Focus |
|---|---|
| 1 | Repo setup, EDA, dataset/DataLoader, AUROC metrics, kNN baseline |
| 2 | PaDiM from scratch, anomaly maps, comparison with Anomalib |
| 3 | PatchCore (memory bank, coreset, nearest-neighbour scoring) |
| 4 | Main experiments, AUPRO, threshold procedure, ablations |
| 5 | Inference module, FastAPI, React dashboard, Docker |
| 6 | Final tables and figures, README, technical report, demo |

## References

- P. Bergmann et al., *MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- T. Defard et al., *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*, 2020.
- K. Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022.
