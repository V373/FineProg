"""utils/registry_scan.py — Sync configs_v2/registry/ against the filesystem.

Scans three artifact directories and reconciles them with the registry YAMLs:

  datasets/processed/*.h5          ↔  configs_v2/registry/datasets.yaml  (datasets section)
  checkpoint/<run_name>/            ↔  configs_v2/registry/runs.yaml      (runs section)
  datasets/embeddings/**/*.h5       ↔  configs_v2/registry/runs.yaml      (embeddings section)

Actions per entry
-----------------
  [OK]      Present on disk AND in registry  → logged, no change.
  [NEW]     Found on disk, not in registry   → auto-registered via RegistryV2.
  [MISSING] In registry, not found on disk   → entry commented out in YAML.

Usage
-----
  python utils/registry_scan.py                         # scan all three scopes
  python utils/registry_scan.py --dry-run               # preview without writing
  python utils/registry_scan.py --scope datasets        # single scope
  python utils/registry_scan.py --scope runs,embeddings # multiple scopes
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure project root is importable regardless of invocation directory.
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

try:
    # Imported as a module from the project root: python -m utils.registry_scan
    from utils.config_v2 import ConfigV2
    from utils.registry_v2 import RegistryV2
except ImportError:
    # Executed directly: python utils/registry_scan.py  (utils/ is sys.path[0])
    from config_v2 import ConfigV2      # type: ignore[import]
    from registry_v2 import RegistryV2  # type: ignore[import]

# ---------------------------------------------------------------------------
# Text-level: comment out a missing registry entry
# ---------------------------------------------------------------------------

def _comment_out_entry(yaml_path: Path, section: str, alias: str,
                        reason: str = "missing", dry_run: bool = False) -> bool:
    """Prefix every line of *alias*'s block with ``# [reason]``.

    Does nothing if the entry is already commented or not found.
    Returns True if the file was (or would be) modified.
    """
    text = yaml_path.read_text()

    # Idempotency: skip if alias line is already commented.
    if re.search(rf"^\s+#\s*\[{re.escape(reason)}\]\s*{re.escape(alias)}:", text, re.MULTILINE):
        return False

    # Match the 2-space-indented block for this alias, stopping at the next
    # sibling key (also 2-space-indented) or a top-level key or end of file.
    pattern = re.compile(
        rf"(^  {re.escape(alias)}:.*?)(?=\n  \S|\n\S|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        return False

    block = m.group(1)
    commented_lines = []
    for line in block.split("\n"):
        if line.strip():
            # Keep first 2-space base indent, insert comment marker.
            commented_lines.append(f"  # [{reason}] {line[2:]}")
        else:
            commented_lines.append("")

    new_text = text[: m.start()] + "\n".join(commented_lines) + text[m.end() :]

    if not dry_run:
        yaml_path.write_text(new_text)
    return True


# ---------------------------------------------------------------------------
# Run-name parsing helpers
# ---------------------------------------------------------------------------

_KNOWN_BACKBONES   = {"resnet50_conv4c", "resnet50_conv5c", "resnet18", "resnet50", "r3m"}
_KNOWN_TRAIN_BASES = {"only_bn", "frozen", "full", "last_layer"}


def _parse_run_name(run_name: str) -> dict:
    """Decompose ``TCC-{dataset_stem}-{backbone}-{train_base}-{YYYYMMDD}-{HHMMSS}``.

    Returns a dict with keys: dataset_stem, backbone, train_base, timestamp.
    Returns an empty dict if the format is unrecognised.
    """
    if not run_name.startswith("TCC-"):
        return {}
    s = run_name[4:]  # strip "TCC-"

    m_ts = re.search(r"-(\d{8})-(\d{6})$", s)
    if not m_ts:
        return {}
    timestamp = f"{m_ts.group(1)}-{m_ts.group(2)}"
    s = s[: m_ts.start()]

    backbone = train_base = dataset_stem = None
    for bb in _KNOWN_BACKBONES:
        idx = s.find(f"-{bb}-")
        if idx >= 0:
            backbone     = bb
            dataset_stem = s[:idx]
            remainder    = s[idx + len(bb) + 2 :]  # after "-{bb}-"
            for tb in _KNOWN_TRAIN_BASES:
                if remainder == tb or remainder.startswith(tb):
                    train_base = tb
                    break
            train_base = train_base or remainder
            break

    return {
        "dataset_stem": dataset_stem or s,
        "backbone":     backbone     or "",
        "train_base":   train_base   or "",
        "timestamp":    timestamp,
    }


def _max_epoch_in_folder(folder: Path) -> Optional[int]:
    """Return the maximum epoch number from ``encoder_epoch*.pt`` files."""
    epochs = [
        int(m.group(1))
        for pt in folder.glob("encoder_epoch*.pt")
        if (m := re.match(r"encoder_epoch(\d+)\.pt$", pt.name))
    ]
    return max(epochs) if epochs else None


# ---------------------------------------------------------------------------
# Registry look-up helpers
# ---------------------------------------------------------------------------

def _find_dataset_ref_by_stem(stem: str, datasets: dict) -> Optional[str]:
    """Return the alias whose ``processed_h5`` stem equals *stem*."""
    for alias, entry in datasets.items():
        if Path(entry.get("processed_h5", "")).stem == stem:
            return alias
    return None


def _find_run_ref_by_run_name(run_name: str, runs: dict) -> Optional[str]:
    """Return the alias whose ``run_name`` value equals *run_name*."""
    for alias, entry in runs.items():
        if entry.get("run_name") == run_name:
            return alias
    return None


def _infer_embedding_variant(h5_name: str) -> str:
    stem = Path(h5_name).stem
    if "-mean_path" in stem:
        return "mean_path"
    if "-labeled" in stem:
        return "labeled"
    return "standard"


def _infer_raw_dir(stem: str) -> str:
    """Guess the raw video subdirectory from an H5 stem.

    Examples::
        robomimic_can_ph-180vid_train  →  robomimic_can_ph
        pouring-2vid                   →  pouring
        pouring_all_training-70vid     →  pouring_all_training
    """
    m = re.match(r"^(.+?)[-_]\d+vid", stem)
    return m.group(1) if m else stem


def _unique_alias(base: str, existing: set[str]) -> str:
    """Append a numeric suffix until the alias is unique."""
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


# ---------------------------------------------------------------------------
# Scope: datasets/processed/
# ---------------------------------------------------------------------------

def scan_datasets(reg: RegistryV2, cfg_v2: ConfigV2, dry_run: bool) -> dict:
    """Scan ``datasets/processed/`` and reconcile with ``datasets.yaml``."""
    processed_dir = Path(cfg_v2._dirs["processed"])
    datasets_yaml = reg._datasets_yaml

    registered     = cfg_v2._datasets                                   # alias → entry
    registered_h5s = {entry["processed_h5"] for entry in registered.values()}
    existing_aliases = set(registered.keys())
    counts = {"ok": 0, "new": 0, "missing": 0}

    print(f"\n{'='*64}")
    print(f"  Scope: datasets   ({processed_dir})")
    print(f"{'='*64}")

    # --- Check registered entries for missing files ---
    for alias, entry in sorted(registered.items()):
        h5_path = Path(cfg_v2._dirs["processed"]) / entry["processed_h5"]
        if not h5_path.exists():
            tag = "  [DRY]" if dry_run else ""
            print(f"  [MISSING] {alias:<42}  {entry['processed_h5']}{tag}")
            _comment_out_entry(datasets_yaml, "datasets", alias, dry_run=dry_run)
            counts["missing"] += 1
        else:
            print(f"  [OK]      {alias:<42}  {entry['processed_h5']}")
            counts["ok"] += 1

    # --- Scan disk for unregistered files ---
    if not processed_dir.exists():
        print(f"  (processed dir not found: {processed_dir})")
        return counts

    for h5_file in sorted(processed_dir.glob("*.h5")):
        if h5_file.name in registered_h5s:
            continue
        stem  = h5_file.stem
        alias = _unique_alias(
            stem.replace("-", "_").replace(" ", "_").lower(),
            existing_aliases,
        )
        raw_dir = _infer_raw_dir(stem)
        tag = "  [DRY]" if dry_run else ""
        print(f"  [NEW]     {alias:<42}  {h5_file.name}  raw_dir={raw_dir}{tag}")
        if not dry_run:
            reg.register_dataset(
                alias        = alias,
                processed_h5 = h5_file.name,
                display_name = stem,
                raw_dir      = raw_dir,
            )
        existing_aliases.add(alias)
        counts["new"] += 1

    return counts


# ---------------------------------------------------------------------------
# Scope: checkpoint/
# ---------------------------------------------------------------------------

def scan_runs(reg: RegistryV2, cfg_v2: ConfigV2, dry_run: bool) -> dict:
    """Scan ``checkpoint/`` run folders and reconcile with runs section."""
    ckpt_dir   = Path(cfg_v2._dirs["checkpoints"])
    runs_yaml  = reg._runs_yaml
    registered = cfg_v2._runs                                           # alias → entry
    registered_run_names = {entry["run_name"]: alias for alias, entry in registered.items()}
    existing_aliases     = set(registered.keys())
    counts = {"ok": 0, "new": 0, "missing": 0}

    print(f"\n{'='*64}")
    print(f"  Scope: runs   ({ckpt_dir})")
    print(f"{'='*64}")

    # --- Check registered runs for missing checkpoint files ---
    for alias, entry in sorted(registered.items()):
        run_name  = entry["run_name"]
        epoch     = int(entry["checkpoint_epoch"])
        ckpt_path = ckpt_dir / run_name / f"encoder_epoch{epoch:06d}.pt"
        if not ckpt_path.exists():
            tag = "  [DRY]" if dry_run else ""
            print(f"  [MISSING] {alias:<42}  epoch {epoch:>6}  {run_name[:50]}{tag}")
            _comment_out_entry(runs_yaml, "runs", alias, dry_run=dry_run)
            counts["missing"] += 1
        else:
            print(f"  [OK]      {alias:<42}  epoch {epoch:>6}  {run_name[:50]}")
            counts["ok"] += 1

    # --- Scan disk for unregistered run folders ---
    if not ckpt_dir.exists():
        print(f"  (checkpoint dir not found: {ckpt_dir})")
        return counts

    for folder in sorted(ckpt_dir.iterdir()):
        if not folder.is_dir():
            continue
        run_name = folder.name
        if run_name in registered_run_names:
            continue
        max_epoch = _max_epoch_in_folder(folder)
        if max_epoch is None:
            continue  # empty folder / no .pt files

        parsed       = _parse_run_name(run_name)
        dataset_stem = parsed.get("dataset_stem", "")
        backbone     = parsed.get("backbone",     "resnet50_conv4c")
        train_base   = parsed.get("train_base",   "only_bn")

        # Try to resolve dataset_ref from dataset_stem
        train_dataset = (
            _find_dataset_ref_by_stem(dataset_stem, cfg_v2._datasets) or dataset_stem
        )
        alias = _unique_alias(
            RegistryV2.suggest_run_alias(train_dataset, max_epoch),
            existing_aliases,
        )

        tag = "  [DRY]" if dry_run else ""
        print(
            f"  [NEW]     {alias:<42}  epoch {max_epoch:>6}"
            f"  {run_name[:50]}  dataset_ref={train_dataset}{tag}"
        )
        if not dry_run:
            reg.register_run(
                alias            = alias,
                run_name         = run_name,
                train_dataset    = train_dataset,
                backbone         = backbone,
                train_base       = train_base,
                checkpoint_epoch = max_epoch,
                description      = f"{dataset_stem}, epoch {max_epoch} [auto-discovered]",
            )
        existing_aliases.add(alias)
        counts["new"] += 1

    return counts


# ---------------------------------------------------------------------------
# Scope: datasets/embeddings/
# ---------------------------------------------------------------------------

def scan_embeddings(reg: RegistryV2, cfg_v2: ConfigV2, dry_run: bool) -> dict:
    """Scan ``datasets/embeddings/`` recursively and reconcile with embeddings section."""
    emb_dir      = Path(cfg_v2._dirs["embeddings"])
    runs_yaml    = reg._runs_yaml
    registered   = cfg_v2._embeddings                                   # alias → entry
    existing_aliases = set(registered.keys())
    counts = {"ok": 0, "new": 0, "missing": 0}

    print(f"\n{'='*64}")
    print(f"  Scope: embeddings   ({emb_dir})")
    print(f"{'='*64}")

    # Resolve every registered embedding → abs path, collect known paths.
    known_paths: dict[str, str] = {}  # abs_path_str → alias
    for alias in list(registered.keys()):
        try:
            resolved = cfg_v2.resolve_embedding(alias)
            known_paths[resolved["embedding_h5_path"]] = alias
        except Exception:
            pass

    # --- Check registered embeddings for missing files ---
    for alias, entry in sorted(registered.items()):
        try:
            resolved  = cfg_v2.resolve_embedding(alias)
            emb_path  = Path(resolved["embedding_h5_path"])
            if not emb_path.exists():
                tag = "  [DRY]" if dry_run else ""
                print(f"  [MISSING] {alias:<42}  {emb_path.name}{tag}")
                _comment_out_entry(runs_yaml, "embeddings", alias, dry_run=dry_run)
                counts["missing"] += 1
            else:
                print(f"  [OK]      {alias:<42}  {emb_path.name}")
                counts["ok"] += 1
        except Exception as exc:
            print(f"  [ERROR]   {alias:<42}  resolve failed: {exc}")

    # --- Scan disk for unregistered embedding files ---
    if not emb_dir.exists():
        print(f"  (embeddings dir not found: {emb_dir})")
        return counts

    for h5_file in sorted(emb_dir.rglob("*.h5")):
        if str(h5_file) in known_paths:
            continue

        variant = _infer_embedding_variant(h5_file.name)
        parent  = h5_file.parent

        # Infer run_ref from parent folder name (if not the top-level emb_dir)
        run_ref = ""
        if parent != emb_dir:
            run_ref = _find_run_ref_by_run_name(parent.name, cfg_v2._runs) or ""

        # Strip embedding suffix to recover the dataset H5 stem
        stem_no_sfx = h5_file.stem
        for sfx in ("-embd-mean_path", "-embd-labeled", "-embd"):
            if stem_no_sfx.endswith(sfx):
                stem_no_sfx = stem_no_sfx[: -len(sfx)]
                break
        dataset_ref = _find_dataset_ref_by_stem(stem_no_sfx, cfg_v2._datasets) or ""

        alias = _unique_alias(
            RegistryV2.suggest_embedding_alias(
                run_ref or "unknown", dataset_ref or stem_no_sfx, variant
            ),
            existing_aliases,
        )
        rel_path = h5_file.relative_to(emb_dir)
        tag = "  [DRY]" if dry_run else ""
        print(
            f"  [NEW]     {alias:<42}  {rel_path}"
            f"  run_ref={run_ref or '?'}  dataset_ref={dataset_ref or '?'}"
            f"  variant={variant}{tag}"
        )
        if not dry_run:
            reg.register_embedding(
                alias       = alias,
                run_ref     = run_ref,
                dataset_ref = dataset_ref,
                variant     = variant,
                description = f"{dataset_ref or stem_no_sfx} ({variant}) [auto-discovered]",
            )
        existing_aliases.add(alias)
        counts["new"] += 1

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan artifact directories and sync configs_v2/registry/ YAMLs.\n"
            "NEW artifacts are auto-registered; MISSING entries are commented out."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview all actions without modifying any file.",
    )
    parser.add_argument(
        "--scope", type=str, default="all",
        metavar="SCOPE[,SCOPE]",
        help=(
            "Which artifact scopes to scan.  Comma-separated subset of: "
            "datasets, runs, embeddings.  Default: all."
        ),
    )
    args = parser.parse_args()

    valid_scopes = {"datasets", "runs", "embeddings"}
    if args.scope == "all":
        scopes = valid_scopes
    else:
        scopes = {s.strip() for s in args.scope.split(",")}
        unknown = scopes - valid_scopes
        if unknown:
            parser.error(f"Unknown scope(s): {', '.join(sorted(unknown))}. "
                         f"Valid: {', '.join(sorted(valid_scopes))}")

    cfg_v2 = ConfigV2()
    reg    = RegistryV2()

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nRegistry scan  {date_str}")
    if args.dry_run:
        print("  mode: DRY-RUN — no files will be modified")
    else:
        print(f"  runs.yaml    : {reg._runs_yaml}")
        print(f"  datasets.yaml: {reg._datasets_yaml}")

    total: dict[str, int] = {"ok": 0, "new": 0, "missing": 0}

    if "datasets" in scopes:
        for k, v in scan_datasets(reg, cfg_v2, args.dry_run).items():
            total[k] += v

    if "runs" in scopes:
        for k, v in scan_runs(reg, cfg_v2, args.dry_run).items():
            total[k] += v

    if "embeddings" in scopes:
        for k, v in scan_embeddings(reg, cfg_v2, args.dry_run).items():
            total[k] += v

    print(f"\n{'='*64}")
    print(
        f"  Summary  "
        f"{total['ok']} OK  |  "
        f"{total['new']} NEW registered  |  "
        f"{total['missing']} MISSING commented out"
    )
    if args.dry_run:
        print("  (dry-run — nothing written)")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
