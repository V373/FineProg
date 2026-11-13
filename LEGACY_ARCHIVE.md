# FineProg legacy snapshot — 2026-11-13

`archive/legacy-20261113` preserves the code before the baseline reduction.
It starts from `140a2b3` and includes the pending single-frame, memory,
diagnostics, data-pipeline, semantic-classifier and Stage2 work. The archive
branch and `main` initially point to the same final archival commit.

## Included inputs and code

- Source, configurations, tests, architecture documentation and requirements.
- Phase-label CSVs, processed-video index mappings and the existing source-video
  mapping note. These are inputs and provenance, not model evaluation results.
- The fruit-video converter and drawer-place data-restoration script under
  `datasets/raw/`, which are imported directly by their tests.
- Human crop/review inputs under `datasets/annotations/real/`. Pass that directory
  as `--review-root` to `dataset_preparation/prepare_real_progress_data.py`.
- The Robomimic and MetaWorld single-frame launchers under `scripts/legacy/`.
  Their experiment outputs remain local under `.vibe/`.

`third_party/robomimic` remains a submodule at its recorded commit. Initialize it
with `git submodule update --init third_party/robomimic` when needed; local data
and checkpoints inside that checkout are not included.

## Local resources and known limitations

The snapshot excludes `.vibe/`, private result indexes, evaluation reports and
metrics, model weights, fitted Gaussian/PCA models, predictions, embeddings,
raw media and runtime caches. A fresh checkout therefore needs separately
provided datasets and checkpoints to run real-data training or evaluation.
This is a source-and-input archive, not a complete machine backup.

Existing experimental defaults are preserved:

- `configs_v2/train.yaml` currently names `pick_mango_vid`, which is not a
  registered dataset key. Select an existing registry entry for a real run.
- Some historical launchers use `.vide/` as their default output location.
  Specify an output directory under `.vibe/` when running them.
- `configs_v2/train_stage2.yaml` refers to local `.vibe/` checkpoint and loss
  paths. The referenced loss YAML is identical to the tracked
  `configs_v2/loss/loss_composite_tcc_triplet.yaml`; provide that configuration
  and a local checkpoint when preparing a Stage2 run.
- The semantic and Stage2 default configurations are short smoke recipes,
  not complete training protocols.

Run tests in the existing `fineprog` conda environment. CUDA and real-data
tests may skip when the corresponding local resources are unavailable.

Historical media and generated checkpoint plots are removed from the tracked
snapshot only. Local files and existing Git history are retained; this archive
does not rewrite repository history. Full local backup and baseline reduction
are separate later steps.
