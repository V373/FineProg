#!/usr/bin/env python3
"""Build real-task FineProg H5 files from reviewed, zero-based inclusive clips.

No frame filtering, temporal resampling, automatic success labeling or training.
Review JSON files are <review-root>/<task>_review.json. See the session report
for the robot-assisted visual review that supplies their episode boundaries.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random

import cv2
import h5py
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS = {"drawer_place": (256, 168, 224), "towel": (80, 0, 480)}
COUNTS = {"drawer_place": 50, "towel": 55}
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sort_key(episode: dict):
    name = episode["source_episode"]
    return int(name.rsplit("_", 1)[-1]) if name.startswith("episode_init_") else name


def raw_episode(task: str, name: str) -> Path:
    return PROJECT_ROOT / "datasets/raw" / task / "collect_data" / name


def load_reviews(root: Path) -> dict:
    tasks = {}
    for task, roi in TASKS.items():
        review_path = root / f"{task}_review.json"
        review = json.loads(review_path.read_text())
        episodes = sorted(review["episodes"], key=sort_key)
        names = [e["source_episode"] for e in episodes]
        actual_names = {p.name for p in raw_episode(task, "").iterdir() if p.is_dir()}
        if len(names) != COUNTS[task] or len(set(names)) != len(names) or set(names) != actual_names:
            raise ValueError(f"{task}: review must cover every source episode exactly once")
        for e in episodes:
            if e["status"] not in ("keep", "exclude", "review"):
                raise ValueError(f"Invalid status: {e}")
            e["source_path"] = str(raw_episode(task, e["source_episode"]).resolve())
            e["split"] = None
            e["video_id"] = None
            if e["status"] == "keep":
                s, c, end, n = (e[k] for k in ("start_frame", "completion_frame", "end_frame", "source_frames"))
                if not all(isinstance(v, int) and not isinstance(v, bool) for v in (s, c, end, n)):
                    raise ValueError(f"Integer frame indices required: {e}")
                if not (0 <= s <= c <= end < n and end <= c + 3):
                    raise ValueError(f"Invalid inclusive clip or semantic tail: {e}")
                e["retained_frames"] = end - s + 1
            else:
                e["retained_frames"] = 0
                if not e.get("reason"):
                    raise ValueError(f"Exclusion/review reason required: {e}")
        accepted = [e for e in episodes if e["status"] == "keep"]
        if len(accepted) < 40:
            raise ValueError(f"{task}: only {len(accepted)} reviewed successes; need 40 for train36/valid4")
        random.Random(SEED).shuffle(accepted)
        selections = {"train": accepted[:36], "valid": accepted[36:40], "calibration": accepted[40:]}
        for split, items in selections.items():
            for i, e in enumerate(items, 1):
                e.update(split=split, video_id=f"{i:06d}")
        tasks[task] = {"review_path": str(review_path.resolve()), "review_sha256": sha256(review_path),
                       "camera": "topFullImg", "roi_xy_size": list(roi), "episodes": episodes,
                       "selections": selections}
    return tasks


def read_rgb(path: Path, roi: tuple | list) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (480, 640, 3):
        raise ValueError(f"Unreadable or unexpected source image: {path}")
    x, y, size = roi
    rgb = cv2.cvtColor(bgr[y:y + size, x:x + size], cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA) if size != 224 else rgb


def validate_source_indices(task: str, episodes: list[dict]) -> None:
    for e in episodes:
        root = Path(e["source_path"])
        expected = set(range(e["source_frames"]))
        for folder in ("topFullImg", "topImg", "leftImg", "rightImg", "observation"):
            ext = ".pkl" if folder == "observation" else ".jpg"
            paths = list((root / folder).glob(f"*{ext}"))
            if len(paths) != len(expected) or {int(p.stem) for p in paths} != expected:
                raise ValueError(f"{task}/{root.name}/{folder}: source indices/count mismatch")


def write_h5(path: Path, task: str, split: str, episodes: list[dict], roi: list, manifest: Path) -> dict:
    frame_digest = hashlib.sha256()
    with h5py.File(path, "x") as h5:
        h5.attrs.update(task=task, split=split, frame_index_base=0, end_inclusive=True,
                        duplicate_filter=False, temporal_stride=1, nominal_fps=20.0,
                        fps_is_nominal=True, camera="topFullImg", roi_xy_size=roi,
                        resize_interpolation="area", source_manifest=str(manifest.resolve()))
        videos = h5.create_group("videos")
        for e in episodes:
            n = e["retained_frames"]
            root = Path(e["source_path"])
            group = videos.create_group(e["video_id"])
            group.attrs.update(action_id=0, action_name=e["source_episode"], source_episode=e["source_episode"],
                               task=task, path=str(root / "topFullImg"), source_episode_path=str(root),
                               source_start_idx=e["start_frame"], source_end_idx=e["end_frame"],
                               completion_frame=e["completion_frame"], source_num_frames=e["source_frames"],
                               num_frames=n, fps=20.0, fps_is_nominal=True, camera="topFullImg",
                               roi_xy_size=roi, frame_index_base=0, end_inclusive=True)
            indices = np.arange(e["start_frame"], e["end_frame"] + 1, dtype=np.int64)
            group.create_dataset("source_frame_indices", data=indices)
            frames = group.create_dataset("frames", (n, 224, 224, 3), dtype=np.uint8,
                                          chunks=(min(8, n), 224, 224, 3))
            source_digest = hashlib.sha256()
            for begin in range(0, n, 8):
                batch = []
                for idx in indices[begin:begin + 8]:
                    source = root / "topFullImg" / f"{idx}.jpg"
                    payload = source.read_bytes()
                    source_digest.update(int(idx).to_bytes(8, "little"))
                    source_digest.update(len(payload).to_bytes(8, "little"))
                    source_digest.update(payload)
                    batch.append(read_rgb(source, roi))
                array = np.stack(batch)
                frames[begin:begin + len(array)] = array
                frame_digest.update(array.tobytes())
            group.attrs["source_selected_jpeg_sha256"] = source_digest.hexdigest()
            e["source_selected_jpeg_sha256"] = source_digest.hexdigest()
            print(f"{task}/{split}/{e['video_id']} {e['source_episode']}: "
                  f"[{e['start_frame']},{e['end_frame']}] -> {n}", flush=True)
    return {"path": str(path.resolve()), "alias": f"{task}_{len(episodes)}vid_{split}",
            "count": len(episodes), "frames": sum(e["retained_frames"] for e in episodes),
            "max_episode_frames": max(e["retained_frames"] for e in episodes),
            "sha256": sha256(path), "rgb_sha256": frame_digest.hexdigest()}


def register_datasets(tasks: dict) -> None:
    registry_path = PROJECT_ROOT / "configs_v2/registry/datasets.yaml"
    existing = yaml.safe_load(registry_path.read_text())["datasets"]
    new = {}
    for task, details in tasks.items():
        for split, spec in details["splits"].items():
            entry = {"display_name": f"{task} reviewed real demonstrations ({split})",
                     "processed_h5": str(Path(spec["path"]).relative_to(PROJECT_ROOT / "datasets/processed")),
                     "raw_dir": f"{task}/collect_data"}
            if split == "train":
                entry["validation_dataset_ref"] = details["splits"]["valid"]["alias"]
            alias = spec["alias"]
            if alias in existing and existing[alias] != entry:
                raise ValueError(f"Registry entry already exists with different content: {alias}")
            if alias not in existing:
                new[alias] = entry
    if new:
        # Append only our new keys: preserve pre-existing edits and comments.
        text = yaml.safe_dump(new, sort_keys=False, allow_unicode=True)
        with registry_path.open("a") as handle:
            handle.write("\n  # Real-task reviewed clips (zero-based inclusive, no frame filter)\n")
            handle.write("\n".join("  " + line if line else line for line in text.splitlines()) + "\n")


def build(args) -> None:
    tasks = load_reviews(args.review_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task, detail in tasks.items():
        validate_source_indices(task, detail["episodes"])
        for split, items in detail["selections"].items():
            path = args.output_dir / f"{task}-{len(items)}vid_{split}.h5"
            if items and path.exists():
                raise FileExistsError(f"Refusing to overwrite processed data: {path}")
    if args.manifest.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {args.manifest}")
    for task, detail in tasks.items():
        detail["splits"] = {}
        for split, items in detail.pop("selections").items():
            if items:
                path = args.output_dir / f"{task}-{len(items)}vid_{split}.h5"
                detail["splits"][split] = write_h5(path, task, split, items, detail["roi_xy_size"], args.manifest)
    manifest = {"schema_version": 1, "created_at": datetime.now().astimezone().isoformat(),
                "split_seed": SEED, "frame_index_base": 0, "end_inclusive": True,
                "duplicate_filter": False, "nominal_fps": 20, "fps_is_nominal": True,
                "tasks": tasks}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    fields = ["task", "source_episode", "source_frames", "status", "split", "video_id",
              "start_frame", "completion_frame", "end_frame", "retained_frames", "reason", "start_reason", "end_reason"]
    with args.manifest.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for task, detail in tasks.items():
            writer.writerows(dict(e, task=task) for e in detail["episodes"])
    if args.register:
        register_datasets(tasks)
    print(f"Wrote {args.manifest}", flush=True)


def validate(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    summary = {"manifest": str(manifest_path.resolve()), "all_passed": True, "tasks": {}}
    for task, details in manifest["tasks"].items():
        source_sets = []
        task_result = {}
        for split, spec in details["splits"].items():
            path = Path(spec["path"])
            if sha256(path) != spec["sha256"]:
                raise ValueError(f"H5 checksum mismatch: {path}")
            rgb_digest = hashlib.sha256()
            count, total, identities = 0, 0, set()
            with h5py.File(path, "r") as h5:
                for vid, group in h5["videos"].items():
                    name = group.attrs["source_episode"]
                    identities.add(name)
                    s, end = int(group.attrs["source_start_idx"]), int(group.attrs["source_end_idx"])
                    indices = group["source_frame_indices"][:]
                    frames = group["frames"]
                    if frames.dtype != np.uint8 or frames.shape != (end - s + 1, 224, 224, 3):
                        raise ValueError(f"Frame schema/count mismatch: {path}/{vid}")
                    if not np.array_equal(indices, np.arange(s, end + 1)):
                        raise ValueError(f"Source mapping mismatch: {path}/{vid}")
                    # All processed bytes are read. Boundary and internal source images
                    # are independently checked for source selection and color/crop errors.
                    for begin in range(0, len(indices), 64):
                        rgb_digest.update(frames[begin:begin + 64].tobytes())
                    for offset in sorted({0, len(indices) // 4, len(indices) // 2, 3 * len(indices) // 4, len(indices) - 1}):
                        expected = read_rgb(raw_episode(task, name) / "topFullImg" / f"{indices[offset]}.jpg", details["roi_xy_size"])
                        if not np.array_equal(expected, frames[offset]):
                            raise ValueError(f"RGB/source correspondence mismatch: {path}/{vid}/{offset}")
                    count += 1
                    total += len(indices)
            if count != spec["count"] or len(identities) != count or total != spec["frames"] or rgb_digest.hexdigest() != spec["rgb_sha256"]:
                raise ValueError(f"Dataset aggregate mismatch: {path}")
            if any(identities & previous for previous in source_sets):
                raise ValueError(f"Split leakage: {task}/{split}")
            source_sets.append(identities)
            task_result[split] = {"count": count, "frames": total, "sha256_verified": True,
                                  "all_frames_read": True, "source_checks_per_episode": 5}
        summary["tasks"][task] = task_result
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "datasets/processed")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = validate(args.manifest)
        output = args.manifest.parent / "data_validation.json"
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    else:
        if args.review_root is None:
            parser.error("--review-root is required when building")
        build(args)


if __name__ == "__main__":
    main()
