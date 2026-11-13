# FineProg

FineProg provides a compact pipeline for single-frame progress encoding, R/M/G semantic classification, and Stage2 fine-tuning, with a registry of six tasks.

Git contains the code, configurations, tests, and selected small input labels and provenance files. Processed H5 datasets, model weights, `datasets/manifest.json`, and run outputs remain local and are excluded from Git. After cloning, provide the local assets at the paths listed in the configuration before running the commands below.

During the local cleanup, the complete legacy project was retained at `../fineprog_backup/`. That separate local archive is not required at runtime. The new project's selected datasets and weights are independent files; raw videos, old embeddings, fitted/predicted outputs, and private reports were not migrated.

## Data and configuration

| Task | Splits (video counts) | Human R/M/G labels |
|---|---|---|
| `pick_mango` | train36 / valid4 / calibration18 / fail18 | v1: train, valid, fail |
| `drawer_place` | train36 / valid4 / calibration6 / subopt1 | No framewise R/M/G; clip review inputs are retained |
| `towel` | train36 / valid4 / calibration13 / subopt1 / fail4 | train, valid, subopt, fail |
| `can` | train36 / valid4 / calibration18 / subopt10 / fail15 | None |
| `square` | train36 / valid4 / calibration18 / subopt10 / fail15 | None |
| `drawer_open` | train36 / valid4 / calibration20 / subopt6 / fail18 | None; this is MetaWorld drawer-open-v2 |

Datasets use `datasets/<task>/<split>.h5`, with uint8 RGB frames at 224×224. Original video IDs are preserved, including `000051`–`000060` for can/square subopt. Each task's `labels/` contains labels and keyframes, while `metadata/` contains source mappings. The local `datasets/manifest.json` records migration provenance and is not used at runtime.

The prepared local assets include a Stage1 Local lower checkpoint for every task. Only Mango has a selected upper checkpoint, Stage2 checkpoint, and complete Stage2 training split. Towel has R/M/G inputs but no configured upper checkpoint or Stage2 recipe. The remaining four tasks need human labels before extending the upper and Stage2 pipeline.

- `configs/tasks.yaml`: dataset, label, and default checkpoint paths; per-stage temperatures and OOD thresholds; Mango's 14/4 Stage2 video split.
- `configs/training.yaml`: lower, upper, and Stage2 training settings. ImageNet initialization requires local `weights/imagenet/resnet50.pth` and `weights/imagenet/efficientnet_b0.pth`.
- `configs/progress.yaml`: PCA, Gaussian, and calibration settings. The CLI applies the task's temperature and OOD threshold for the selected stage.

Relative paths resolve from the project root. `--config-dir` selects another directory containing all three YAML files; paths inside those files still resolve from the project root.

The lower encoder repeats the current frame in two context slots and produces a 128D L2-normalized embedding. Mango's upper model uses EfficientNet-B0, a 32D MLP, and an R/M/G head with `l2_normalize: false`. Stage2 fine-tunes the lower encoder using expert replay and teacher labels; its four held-out validation videos always use human v1 labels.

## Commands

Activate the existing `fineprog` environment and enter the project directory:

```bash
conda activate fineprog
cd /path/to/FineProg
```

The following Mango workflow explicitly connects newly trained checkpoints. Use fresh output directories for training and labeling. Add `--steps 2` for a short training run, or `--device cpu` to select CPU; the default uses CUDA when available, otherwise CPU.

```bash
python -B -m fineprog train-lower --task pick_mango --output runs/mango-lower
python -B -m fineprog train-upper --task pick_mango --output runs/mango-upper

python -B -m fineprog label --task pick_mango --split fail \
  --checkpoint runs/mango-upper/checkpoint.pt --output runs/mango-labels

python -B -m fineprog train-stage2 --task pick_mango \
  --checkpoint runs/mango-lower/checkpoint.pt \
  --labels-csv runs/mango-labels/labels.csv --output runs/mango-stage2

python -B -m fineprog fit-progress --task pick_mango --stage stage2 \
  --checkpoint runs/mango-stage2/checkpoint.pt --output runs/mango-progress

python -B -m fineprog predict --task pick_mango --stage stage2 --split valid \
  --checkpoint runs/mango-stage2/checkpoint.pt \
  --progress-model runs/mango-progress/progress.h5 --output runs/mango-predictions
```

With the selected local checkpoints in place, you can skip the first two commands and omit `--checkpoint` from `label` and `train-stage2`. They then use the configured `upper` and `lower` checkpoints respectively. `label` generates a fresh `labels.csv`; `train-stage2 --labels-csv` is required. Only the 14 training videos use teacher labels; validation continues to read human labels.

Training writes its final checkpoint to `<output>/checkpoint.pt`. For new runs, `label --checkpoint` must select the upper model, and `train-stage2 --checkpoint` must select the Stage1 lower model. The CLI does not discover the latest run automatically.

`fit-progress` extracts train features, fits PCA32 and 20 independent full-covariance Gaussians, calibrates on the calibration split, and writes `progress.h5`. `predict` writes per-video NPZ, CSV, and curve PNG files. Use the same checkpoint and matching `--stage` for both commands: `stage1` for Stage1, `stage2` for Stage2. Omitting `--checkpoint` selects the configured local checkpoint, not the run you just trained.

## Single-frame API

This example uses the artifacts fitted above. Call `reset()` at the start of each new episode:

```python
import h5py
from fineprog.progress import ProgressEstimator

estimator = ProgressEstimator(
    checkpoint="runs/mango-stage2/checkpoint.pt",
    model_path="runs/mango-progress/progress.h5",
    device="cpu",
)
estimator.reset()
with h5py.File("datasets/pick_mango/valid.h5", "r") as data:
    rgb = data["videos/000001/frames"][0]  # RGB uint8 [224, 224, 3]
prediction = estimator.predict_frame(rgb)
print(prediction["gated_progress"], prediction["is_ood"])
```

The Gaussian posterior produces continuous progress. A frame is OOD when every bin's conformal p-value is below the threshold; gated progress then holds the previous in-distribution value. The API also returns `raw_progress`, `max_p`, and `posterior`. Call `predict_frame()` repeatedly to process a live video stream.
