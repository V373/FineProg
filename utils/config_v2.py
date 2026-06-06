"""V2 Configuration Resolver for mytcc.

Loads the configs_v2/ YAML files and resolves logical references
(dataset_ref, checkpoint_ref, embedding_ref) into concrete absolute paths.

This module is read-only: it never writes to any config file, including
the V2 registry YAMLs.  Resolution is read-only.

Quick usage
-----------
    from utils.config_v2 import ConfigV2

    cfg = ConfigV2()                          # auto-finds configs_v2/

    # --- Stage loaders ---
    data_cfg   = cfg.load_data_process()      # data_process.yaml
    train_cfg  = cfg.load_train()             # train.yaml
    extract_cfg = cfg.load_extract()          # extract.yaml

    # --- Eval loaders (one per task) ---
    eval_cfg = cfg.load_eval("kendalls_tau")
    eval_cfg = cfg.load_eval("expert_projection")
    eval_cfg = cfg.load_eval("classification")

    # --- Manual registry lookups ---
    ds  = cfg.resolve_dataset("robomimic_can_ph_180vid_train")
    run = cfg.resolve_run("can_ph_180_ep50k")
    emb = cfg.resolve_embedding("can_mh_okay_ep50k")

    # --- Debug helpers ---
    cfg.print_config(eval_cfg, "Expert Projection")
    missing = [p for _, p, ok in cfg.check_paths(eval_cfg) if not ok]
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Optional

# Project root = directory containing this file's parent (mytcc/)
_PROJ_ROOT  = Path(__file__).resolve().parent.parent
_CONFIGS_V2 = _PROJ_ROOT / "configs_v2"


class ConfigV2:
    """V2 configuration loader and path resolver.

    Parameters
    ----------
    configs_v2_dir:
        Path to the configs_v2/ directory.  Defaults to
        ``<project_root>/configs_v2/``.
    """

    # Maps embedding variant names to filename suffix templates.
    # {stem} is the processed-H5 filename stem, e.g. robomimic_can_ph-180vid_train
    _VARIANT_TMPL: dict[str, str] = {
        "standard":  "{stem}-embd.h5",
        "mean_path": "{stem}-embd-mean_path.h5",
        "labeled":   "{stem}-embd-labeled.h5",
    }

    def __init__(self, configs_v2_dir: Optional[str | Path] = None) -> None:
        root = Path(configs_v2_dir) if configs_v2_dir else _CONFIGS_V2
        self._root      = root
        self._proj_root = _PROJ_ROOT
        self._registry_root = root / "registry"   # registry YAMLs live here

        # Load the three permanent registries.
        self._project  = self._load(root / "project.yaml")
        self._datasets = self._load(self._registry_root / "datasets.yaml").get("datasets", {})
        _runs_raw      = self._load(self._registry_root / "runs.yaml")
        self._runs      = _runs_raw.get("runs", {})
        self._embeddings = _runs_raw.get("embeddings", {})

        # Resolve directory paths once, as absolute strings.
        _dirs_rel = self._project.get("dirs", {})
        self._dirs: dict[str, str] = {
            k: str(self._proj_root / v) for k, v in _dirs_rel.items()
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"[ConfigV2] Config file not found: {path}")
        with open(path, "r") as fh:
            return yaml.safe_load(fh) or {}

    def _abs(self, p: str) -> str:
        """Return p as absolute path (prefix with project root if relative)."""
        pp = Path(p)
        if not pp.is_absolute():
            pp = self._proj_root / pp
        return str(pp)

    # ------------------------------------------------------------------ #
    # Registry resolvers
    # ------------------------------------------------------------------ #

    def resolve_dataset(self, dataset_ref: str) -> dict:
        """Resolve a dataset key to its registry entry + concrete absolute paths.

        Added keys in the returned dict
        --------------------------------
        processed_h5_path   Full absolute path to the processed H5 file.
        h5_stem             Stem of processed_h5 (used in embedding path derivation).
        raw_dir_path        (optional) Absolute path to raw video directory.
        robomimic_hdf5_path (optional) Absolute path to the robomimic HDF5 source.
        phase_labels_csv_path (optional) Absolute path to the phase-labels CSV.
        """
        if dataset_ref not in self._datasets:
            raise KeyError(
                f"[ConfigV2] Unknown dataset_ref: {dataset_ref!r}\n"
                f"  Available keys: {sorted(self._datasets.keys())}"
            )
        entry = dict(self._datasets[dataset_ref])

        # Mandatory
        entry["processed_h5_path"] = str(
            Path(self._dirs["processed"]) / entry["processed_h5"]
        )
        entry["h5_stem"] = Path(entry["processed_h5"]).stem

        # Optional
        if entry.get("raw_dir"):
            entry["raw_dir_path"] = str(Path(self._dirs["raw"]) / entry["raw_dir"])
        if entry.get("robomimic_hdf5"):
            entry["robomimic_hdf5_path"] = self._abs(entry["robomimic_hdf5"])
        if entry.get("phase_labels_csv"):
            entry["phase_labels_csv_path"] = self._abs(entry["phase_labels_csv"])
        # idx_mapping CSV auto-derived from processed H5 stem (generated by mp4vid_to_h5data.py)
        _idx_csv = (
            Path(self._dirs["processed"])
            / "idx_mapping"
            / f"{entry['h5_stem']}_idx_mapping.csv"
        )
        if _idx_csv.exists():
            entry["idx_mapping_csv_path"] = str(_idx_csv)

        return entry

    def resolve_validation_dataset(self, train_dataset_ref: str) -> Optional[str]:
        """Return the curated validation dataset key for a given training dataset, or None.

        Returns the value of the ``validation_dataset_ref`` field from the dataset
        registry when a training dataset has exactly one registered validation partner.
        Returns ``None`` for datasets that have no registered pair (e.g. mixed-quality
        MH datasets) and for keys not present in the registry — callers should fall
        back to the training dataset in that case.

        Example::

            cfg = ConfigV2()
            val_ref = cfg.resolve_validation_dataset("robomimic_can_ph_180vid_train")
            # → "robomimic_can_ph_20vid_valid"

            val_ref = cfg.resolve_validation_dataset("robomimic_can_mh_100vid_okay")
            # → None  (no validation pair registered)
        """
        return self._datasets.get(train_dataset_ref, {}).get("validation_dataset_ref") or None

    def resolve_run(self, run_ref: str) -> dict:
        """Resolve a run alias to its registry entry + concrete checkpoint path.

        Added keys
        ----------
        checkpoint_path   Absolute path to the .pt checkpoint file.
        """
        if run_ref not in self._runs:
            raise KeyError(
                f"[ConfigV2] Unknown run_ref: {run_ref!r}\n"
                f"  Available keys: {sorted(self._runs.keys())}"
            )
        entry = dict(self._runs[run_ref])
        run_name = entry["run_name"]
        epoch    = int(entry["checkpoint_epoch"])

        tmpl = self._project["templates"]["checkpoint"]
        fname = tmpl.format(run_name=run_name, epoch=epoch)
        entry["checkpoint_path"] = str(Path(self._dirs["checkpoints"]) / fname)
        return entry

    def resolve_embedding(self, embedding_ref: str) -> dict:
        """Resolve an embedding alias to its registry entry + concrete H5 path.

        Added keys
        ----------
        embedding_h5_path   Absolute path to the embedding H5 file.
        run_name            Full wandb run name (for downstream output-dir derivation).
        h5_stem             Dataset H5 stem (for downstream output-dir derivation).
        """
        if embedding_ref not in self._embeddings:
            raise KeyError(
                f"[ConfigV2] Unknown embedding_ref: {embedding_ref!r}\n"
                f"  Available keys: {sorted(self._embeddings.keys())}"
            )
        entry = dict(self._embeddings[embedding_ref])

        # Explicit override path (bypasses auto-derivation entirely).
        if entry.get("embedding_h5_path"):
            entry["embedding_h5_path"] = self._abs(entry["embedding_h5_path"])
            return entry

        run_entry     = self.resolve_run(entry["run_ref"])
        dataset_entry = self.resolve_dataset(entry["dataset_ref"])
        variant       = entry.get("variant", "standard")
        tmpl          = self._VARIANT_TMPL.get(variant)
        if tmpl is None:
            raise ValueError(
                f"[ConfigV2] Unknown embedding variant: {variant!r}\n"
                f"  Supported: {sorted(self._VARIANT_TMPL.keys())}"
            )
        stem     = dataset_entry["h5_stem"]
        run_name = run_entry["run_name"]
        fname    = tmpl.format(stem=stem)
        entry["embedding_h5_path"] = str(
            Path(self._dirs["embeddings"]) / run_name / fname
        )
        entry["run_name"] = run_name
        entry["h5_stem"]  = stem
        return entry

    # ------------------------------------------------------------------ #
    # Stage loaders
    # ------------------------------------------------------------------ #

    def load_data_process(self, config_path: Optional[str] = None) -> dict:
        """Load data_process.yaml and resolve process_dataset."""
        cfg = self._load(
            Path(config_path) if config_path else self._root / "data_process.yaml"
        )
        ref = cfg.get("process_dataset")
        if ref:
            cfg["dataset_info"] = self.resolve_dataset(ref)
        return cfg

    def load_train(self, config_path: Optional[str] = None) -> dict:
        """Load train.yaml and resolve train_dataset."""
        cfg = self._load(
            Path(config_path) if config_path else self._root / "train.yaml"
        )
        ref = cfg.get("train_dataset")
        if ref:
            cfg["dataset_info"] = self.resolve_dataset(ref)
            cfg["h5_path"] = cfg["dataset_info"]["processed_h5_path"]
        return cfg

    def load_extract(self, config_path: Optional[str] = None) -> dict:
        """Load extract.yaml, resolve checkpoint and extract dataset.

        Resolution order for checkpoint_path:
          1. extract.yaml :: checkpoint_path  (explicit override)
          2. extract.yaml :: checkpoint_ref   (alias in runs.yaml)

        Resolution order for embedding_save_path:
          1. extract.yaml :: embedding_save_path  (explicit override)
          2. Auto-derived: datasets/embeddings/{run_name}/{dataset_stem}-embd.h5
        """
        cfg = self._load(
            Path(config_path) if config_path else self._root / "extract.yaml"
        )

        # Resolve extract dataset
        ref = cfg.get("extract_dataset")
        if ref:
            cfg["dataset_info"] = self.resolve_dataset(ref)
            cfg["extract_h5_path"] = cfg["dataset_info"]["processed_h5_path"]

        # Resolve checkpoint
        if not cfg.get("checkpoint_path"):
            ckpt_ref = cfg.get("checkpoint_ref")
            if ckpt_ref:
                run_entry = self.resolve_run(ckpt_ref)
                cfg["checkpoint_path"] = run_entry["checkpoint_path"]
                cfg["run_name"]        = run_entry["run_name"]

        # Derive embedding save path if not explicitly set
        if not cfg.get("embedding_save_path"):
            run_name = cfg.get("run_name")
            if run_name and ref:
                stem = cfg["dataset_info"]["h5_stem"]
                cfg["embedding_save_path"] = str(
                    Path(self._dirs["embeddings"]) / run_name / f"{stem}-embd.h5"
                )

        return cfg

    def load_eval(
        self,
        task_name: str,
        config_path: Optional[str] = None,
        overrides: Optional[dict] = None,
    ) -> dict:
        """Load an eval task config and resolve all embedding/dataset refs.

        Parameters
        ----------
        task_name:
            One of: ``"kendalls_tau"``, ``"expert_projection"``, ``"classification"``,
            ``"latent_distance_heatmap"``.
        config_path:
            Optional explicit path to the YAML file.  Defaults to
            ``configs_v2/eval/{task_name}.yaml``.
        overrides:
            Optional dict deep-merged on top of the task YAML *before* resolution.
            Use this to inject runtime values such as ``embedding_h5_path`` or
            ``output_dir`` without editing any YAML or registry file.
            When ``embedding_h5_path`` is supplied here, the resolver will skip
            the ``embedding_ref`` registry lookup so the injected path wins.
        """
        if config_path:
            cfg = self._load(Path(config_path))
        else:
            cfg = self._load(self._root / "eval" / f"{task_name}.yaml")

        # Apply runtime overrides before resolution so resolvers see them.
        if overrides:
            cfg = self._deep_merge(cfg, overrides)

        _resolvers = {
            "kendalls_tau":              self._resolve_kendalls_tau,
            "expert_projection":         self._resolve_expert_projection,
            "classification":            self._resolve_classification,
            "latent_distance_heatmap":   self._resolve_latent_distance_heatmap,
        }
        if task_name not in _resolvers:
            raise ValueError(
                f"[ConfigV2] Unknown eval task: {task_name!r}\n"
                f"  Supported: {sorted(_resolvers.keys())}"
            )
        return _resolvers[task_name](cfg)

    # ------------------------------------------------------------------ #
    # Task-specific resolvers (private)
    # ------------------------------------------------------------------ #

    def _resolve_kendalls_tau(self, cfg: dict) -> dict:
        cfg = dict(cfg)
        emb_ref = cfg.get("embedding_ref")
        # Skip registry lookup when embedding_h5_path is already set
        # (e.g. injected at runtime via load_eval(overrides=...)).
        if emb_ref and not cfg.get("embedding_h5_path"):
            e = self.resolve_embedding(emb_ref)
            cfg["embedding_h5_path"] = e["embedding_h5_path"]
            # Derive output directory for kendall heatmaps
            run_name = e.get("run_name", "")
            stem     = e.get("h5_stem",  "")
            if run_name and stem:
                cfg["output_dir"] = str(
                    Path(self._dirs["outputs"]) / "kendall_heatmap" / run_name / stem
                )
        return cfg

    def _resolve_expert_projection(self, cfg: dict) -> dict:
        cfg = dict(cfg)

        expert_ref = cfg.get("expert_embedding_ref")
        if expert_ref:
            e = self.resolve_embedding(expert_ref)
            cfg["expert_h5_path"] = e["embedding_h5_path"]

        nonexpert_ref = cfg.get("nonexpert_embedding_ref")
        if nonexpert_ref:
            e = self.resolve_embedding(nonexpert_ref)
            cfg["nonexpert_h5_path"] = e["embedding_h5_path"]

        # Resolve nonexpert dataset for robomimic_hdf5_path, mask_key, and video raw dir
        ds_ref = cfg.get("nonexpert_dataset_ref")
        if ds_ref:
            ds = self.resolve_dataset(ds_ref)
            cfg["nonexpert_raw_hdf5_path"] = ds.get("robomimic_hdf5_path", "")
            # mask_key in the eval yaml overrides registry; otherwise use registry value
            if not cfg.get("nonexpert_mask_key"):
                cfg["nonexpert_mask_key"] = ds.get("mask_key", "")
            # video_raw_dir: auto-derive from dataset raw_dir_path if not explicitly set
            if not cfg.get("nonexpert_video_raw_dir"):
                cfg["nonexpert_video_raw_dir"] = ds.get("raw_dir_path", "")
            # idx_mapping CSV: fallback demo_name_map source for non-robomimic datasets
            cfg["nonexpert_idx_mapping_csv"] = ds.get("idx_mapping_csv_path", "")

        return cfg

    def _resolve_classification(self, cfg: dict) -> dict:
        cfg = dict(cfg)

        train_ref = cfg.get("train_embedding_ref")
        if train_ref:
            e = self.resolve_embedding(train_ref)
            cfg["classification_train_h5_path"] = e["embedding_h5_path"]

        val_ref = cfg.get("val_embedding_ref")
        if val_ref:
            e = self.resolve_embedding(val_ref)
            cfg["classification_val_h5_path"] = e["embedding_h5_path"]

        return cfg

    def _resolve_latent_distance_heatmap(self, cfg: dict) -> dict:
        """Resolve ``latent_distance_heatmap`` eval config.

        Resolved keys added
        -------------------
        embedding_h5_path   Absolute path to the embedding H5 file.
        output_dir          Derived as outputs/latent_distance_heatmap/<run_name>/<h5_stem>/
                            unless explicitly overridden in the YAML.
        """
        cfg = dict(cfg)
        emb_ref = cfg.get("embedding_ref")
        # Skip registry lookup when embedding_h5_path is already set
        # (e.g. injected at runtime via load_eval(overrides=...)).
        if emb_ref and not cfg.get("embedding_h5_path"):
            e = self.resolve_embedding(emb_ref)
            cfg["embedding_h5_path"] = e["embedding_h5_path"]
            run_name = e.get("run_name", "")
            stem     = e.get("h5_stem",  "")
            # Only derive output_dir when not explicitly set in the YAML
            if not cfg.get("output_dir") and run_name and stem:
                cfg["output_dir"] = str(
                    Path(self._dirs["outputs"])
                    / "latent_distance_heatmap"
                    / run_name
                    / stem
                )
        return cfg

    # ------------------------------------------------------------------ #
    # Debug utilities
    # ------------------------------------------------------------------ #

    def print_config(self, resolved: dict, title: str = "Resolved Config") -> None:
        """Pretty-print a resolved configuration dict."""
        sep = "=" * 64
        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)
        self._fmt_dict(resolved, indent=0)
        print(f"{sep}\n")

    def _fmt_dict(self, obj: Any, indent: int) -> None:
        pad = "  " * indent
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    print(f"{pad}{k}:")
                    self._fmt_dict(v, indent + 1)
                else:
                    print(f"{pad}{k}: {v}")
        else:
            print(f"{pad}{obj}")

    def check_paths(self, resolved: dict) -> list[tuple[str, str, bool]]:
        """Return a list of (key, path, exists) for every file path in the dict.

        Only entries whose value ends in a known file extension are checked:
        .h5, .hdf5, .pt, .yaml, .csv
        """
        results: list[tuple[str, str, bool]] = []
        self._collect_paths(resolved, results, prefix="")
        return results

    def _collect_paths(self, obj: Any, results: list, prefix: str) -> None:
        _exts = {".h5", ".hdf5", ".pt", ".yaml", ".csv"}
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, str) and Path(v).suffix in _exts:
                    p = Path(v)
                    # Only check absolute paths; relative raw-YAML values are
                    # skipped because the resolver always adds an absolute
                    # *_path companion key (e.g. processed_h5_path).
                    if p.is_absolute():
                        results.append((key, v, p.exists()))
                elif isinstance(v, dict):
                    self._collect_paths(v, results, prefix=key)

    # ------------------------------------------------------------------ #
    # Visualize loaders (load_visualize + helpers)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge *override* on top of *base*.

        For nested dicts the merge recurses; all other value types from
        *override* win unconditionally.  Neither input is mutated.
        """
        result = dict(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = ConfigV2._deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    def load_visualize(
        self,
        name: str,
        config_path: Optional[str] = None,
        overrides: Optional[dict] = None,
    ) -> dict:
        """Load a visualization config by merging base.yaml with per-flow YAML.

        Parameters
        ----------
        name:
            One of: ``"tsne"``, ``"tsne_2groups"``, ``"tsne_phase"``,
            ``"tsne_gap_analysis"``, ``"mean_path"``.
        config_path:
            Optional explicit path to the per-flow YAML.  Defaults to
            ``configs_v2/visualize/{name}.yaml``.
        overrides:
            Optional dict of values to deep-merge on top of the per-flow YAML
            *before* resolution.  Useful for injecting CLI arguments (e.g.
            ``{"embedding_ref": args.embedding_ref}``).

        Returns
        -------
        A resolved config dict with all embedding refs replaced by absolute
        ``embedding_h5_path`` and auto-derived ``output_dir``.
        """
        _supported = {
            "tsne", "tsne_2groups", "tsne_phase", "tsne_gap_analysis", "mean_path"
        }
        if name not in _supported:
            raise ValueError(
                f"[ConfigV2] Unknown visualize name: {name!r}\n"
                f"  Supported: {sorted(_supported)}"
            )

        # Load base defaults
        base_cfg = self._load(self._root / "visualize" / "base.yaml")

        # Load per-flow config
        if config_path:
            flow_cfg = self._load(Path(config_path))
        else:
            flow_cfg = self._load(self._root / "visualize" / f"{name}.yaml")

        # Merge: per-flow overrides base (deep merge for nested sections)
        cfg = self._deep_merge(base_cfg, flow_cfg)

        # Apply CLI / programmatic overrides last
        if overrides:
            cfg = self._deep_merge(cfg, overrides)

        # Dispatch to flow-specific resolver
        _resolvers = {
            "tsne":              self._resolve_visualize_tsne,
            "tsne_2groups":      self._resolve_visualize_tsne_2groups,
            "tsne_phase":        self._resolve_visualize_tsne_phase,
            "tsne_gap_analysis": self._resolve_visualize_tsne_gap_analysis,
            "mean_path":         self._resolve_visualize_mean_path,
        }
        return _resolvers[name](cfg)

    # -- Visualize: per-flow resolvers ---------------------------------- #

    def _resolve_visualize_tsne(self, cfg: dict) -> dict:
        """Resolve single-group t-SNE visualize config."""
        cfg = dict(cfg)
        emb_ref = cfg.get("embedding_ref")
        if emb_ref:
            e = self.resolve_embedding(emb_ref)
            cfg["embedding_h5_path"] = e["embedding_h5_path"]
            run_name = e.get("run_name", "")
            stem     = e.get("h5_stem",  "")
            if not cfg.get("output_dir") and run_name and stem:
                cfg["output_dir"] = str(
                    Path(self._dirs["outputs"]) / "tsne" / run_name / stem
                )
        return cfg

    def _resolve_visualize_tsne_2groups(self, cfg: dict) -> dict:
        """Resolve two-group t-SNE visualize config."""
        cfg = dict(cfg)

        emb_ref = cfg.get("embedding_ref")
        if emb_ref:
            e = self.resolve_embedding(emb_ref)
            cfg["embedding_h5_path"] = e["embedding_h5_path"]
            run_name = e.get("run_name", "")
            stem     = e.get("h5_stem",  "")
            if not cfg.get("output_dir") and run_name and stem:
                stem2 = ""
                ref2  = cfg.get("embedding_ref_group2")
                if ref2:
                    e2   = self.resolve_embedding(ref2)
                    stem2 = e2.get("h5_stem", "")
                suffix = f"{stem}_vs_{stem2}" if stem2 else stem
                cfg["output_dir"] = str(
                    Path(self._dirs["outputs"]) / "tsne" / run_name / suffix
                )

        emb_ref2 = cfg.get("embedding_ref_group2")
        if emb_ref2:
            e2 = self.resolve_embedding(emb_ref2)
            cfg["embedding_h5_path_group2"] = e2["embedding_h5_path"]

        return cfg

    def _resolve_visualize_tsne_phase(self, cfg: dict) -> dict:
        """Resolve phase-labeled t-SNE visualize config."""
        cfg = dict(cfg)

        emb_ref = cfg.get("embedding_ref")
        if emb_ref:
            e = self.resolve_embedding(emb_ref)
            cfg["embedding_h5_path"] = e["embedding_h5_path"]
            run_name = e.get("run_name", "")
            stem     = e.get("h5_stem",  "")
            if not cfg.get("output_dir") and run_name and stem:
                cfg["output_dir"] = str(
                    Path(self._dirs["outputs"]) / "tsne" / run_name / stem / "tsne_phase"
                )

        val_ref = cfg.get("val_labeled_embedding_ref")
        if val_ref:
            e_val = self.resolve_embedding(val_ref)
            cfg["val_labeled_h5_path"] = e_val["embedding_h5_path"]

        return cfg

    def _resolve_visualize_tsne_gap_analysis(self, cfg: dict) -> dict:
        """Resolve gap-analysis t-SNE visualize config."""
        cfg = dict(cfg)

        emb_ref = cfg.get("embedding_ref")
        run_name = ""
        stem1    = ""
        if emb_ref:
            e = self.resolve_embedding(emb_ref)
            cfg["embedding_h5_path"] = e["embedding_h5_path"]
            run_name = e.get("run_name", "")
            stem1    = e.get("h5_stem",  "")
            cfg["h5_stem"] = stem1  # for build_vid_id_to_rawname

        emb_ref2 = cfg.get("embedding_ref_group2")
        stem2    = ""
        if emb_ref2:
            e2   = self.resolve_embedding(emb_ref2)
            cfg["embedding_h5_path_group2"] = e2["embedding_h5_path"]
            stem2 = e2.get("h5_stem", "")
            cfg["h5_stem_group2"] = stem2  # for build_vid_id_to_rawname

        if not cfg.get("output_dir") and run_name:
            suffix = f"{stem1}_vs_{stem2}_gap" if stem2 else f"{stem1}_gap"
            cfg["output_dir"] = str(
                Path(self._dirs["outputs"]) / "tsne" / run_name / suffix
            )

        return cfg

    def _resolve_visualize_mean_path(self, cfg: dict) -> dict:
        """Resolve mean-path visualize config (used by compute_mean_embedding_path.py)."""
        cfg = dict(cfg)

        emb_ref = cfg.get("embedding_ref")
        if emb_ref:
            e = self.resolve_embedding(emb_ref)
            cfg["embedding_h5_path"] = e["embedding_h5_path"]
            run_name = e.get("run_name", "")
            stem     = e.get("h5_stem",  "")
            if not cfg.get("output_dir") and run_name and stem:
                cfg["output_dir"] = str(
                    Path(self._dirs["outputs"]) / "tsne" / run_name / stem / "mean_path"
                )

        return cfg


def load_config_v2(configs_v2_dir: Optional[str | Path] = None) -> ConfigV2:
    """Convenience function — equivalent to ``ConfigV2(configs_v2_dir)``."""
    return ConfigV2(configs_v2_dir)
