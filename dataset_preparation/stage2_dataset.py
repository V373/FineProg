"""Current video frames plus R/M/G sidecars for Stage2 replay and set queries."""

import csv

import h5py
import numpy as np
import torch

from .h5vid_dataset import H5VideoDataset


LABEL_IDS = {"R": 0, "M": 1, "G": 2}


def read_labels(labels_csv, lengths):
    """Read original frame coordinates; missing/unknown semantics remain -1."""
    labels = {vid: np.full(n, -1, dtype=np.int64) for vid, n in lengths.items()}
    seen = {vid: np.zeros(n, dtype=bool) for vid, n in lengths.items()}
    with open(labels_csv, newline="") as file:
        for row in csv.DictReader(file):
            vid, step = row["id"], int(row["frame_idx"])
            if vid not in labels:
                raise ValueError(f"Unknown video_id {vid!r} in {labels_csv}")
            if not 0 <= step < lengths[vid]:
                raise ValueError(f"Frame out of range: {vid}/{step}")
            if seen[vid][step]:
                raise ValueError(f"Duplicate label: {vid}/{step}")
            seen[vid][step] = True
            name, number = row["label"].strip(), row["label_id"].strip()
            if not name or not number:
                continue
            try:
                number = int(number)
            except ValueError:
                number = -1
            if name in LABEL_IDS or number in LABEL_IDS.values():
                if LABEL_IDS.get(name) != number:
                    raise ValueError(f"Inconsistent label: {vid}/{step}: {name}/{number}")
                labels[vid][step] = number
    return labels


class Stage2VideoDataset(H5VideoDataset):
    """Reuse Stage1 time sampling, adding expert references after replay targets.

    Only selected episodes' pixels are cached. Metadata and labels are checked
    against the entire H5 so one sidecar can serve an explicit episode subset.
    """

    def __init__(self, h5_path, labels_csv, *, encoder_config, role,
                 video_ids=None, refs_per_class=4, query_steps=20,
                 sampling_strategy="offset_uniform", random_offset=1):
        if role not in ("expert", "query"):
            raise ValueError("role must be 'expert' or 'query'")
        if sampling_strategy not in ("offset_uniform", "stride"):
            raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")
        self.h5_path, self.role = str(h5_path), role
        self.clip_len = int(encoder_config["clip_len"] if role == "expert" else query_steps)
        self.context_size = int(encoder_config["context_size"])
        self.context_stride = int(encoder_config["context_stride"])
        self.full_episode = bool(encoder_config.get("memory_config", {}).get("enabled", False))
        self.sampling_strategy, self.random_offset = sampling_strategy, random_offset
        self.refs_per_class = int(refs_per_class)
        if self.clip_len < 1 or self.refs_per_class < 1:
            raise ValueError("Sample counts must be positive")
        with h5py.File(h5_path, "r") as file:
            videos = file["videos"]
            self.lengths = {vid: len(videos[vid]["frames"]) for vid in videos}
            self.video_ids = sorted(self.lengths) if video_ids is None else list(video_ids)
            if (not self.video_ids or len(set(self.video_ids)) != len(self.video_ids)
                    or set(self.video_ids) - self.lengths.keys()):
                raise ValueError("video_ids must be a nonempty, unique subset of H5 episodes")
            if any(self.lengths[vid] < 1 for vid in self.video_ids):
                raise ValueError("Cannot sample an empty episode")
            self.labels = read_labels(labels_csv, self.lengths)
            self._frames_cache = {
                vid: {"frames": videos[vid]["frames"][:]} for vid in self.video_ids
            }

    def __getitem__(self, index):
        vid = self.video_ids[index]
        frames = self._frames_cache[vid]["frames"]
        labels = self.labels[vid]
        sampler = self._sample_stride if self.sampling_strategy == "stride" else self._sample_offset_uniform
        steps = sampler(len(frames)).astype(np.int64)
        replay_count = self.clip_len if self.role == "expert" else 0
        if self.role == "expert":
            reference_steps, reference_labels = [], []
            valid = np.flatnonzero(labels >= 0)
            for category in range(3):
                candidates = np.flatnonzero(labels == category)
                if len(candidates):
                    selected = np.random.choice(candidates, self.refs_per_class,
                                                replace=len(candidates) < self.refs_per_class)
                    reference_labels.extend([category] * self.refs_per_class)
                else:
                    # Preserve fixed batch shape, but never invent a missing class.
                    selected = np.random.choice(valid if len(valid) else [0], self.refs_per_class)
                    reference_labels.extend([-1] * self.refs_per_class)
                reference_steps.extend(selected)
            sampled_labels = np.concatenate((labels[steps], reference_labels))
            steps = np.concatenate((steps, reference_steps))
        else:
            goals = np.flatnonzero(labels == LABEL_IDS["G"])
            if len(goals) and not np.any(labels[steps] == LABEL_IDS["G"]):
                steps[-1] = np.random.choice(goals)
                steps.sort()
            sampled_labels = labels[steps]

        item = {"target_steps": torch.from_numpy(steps),
                "labels": torch.from_numpy(sampled_labels.astype(np.int64)),
                "replay_count": replay_count, "seq_len": len(frames), "video_id": vid}
        if self.full_episode:
            item["frames_full"] = torch.from_numpy(frames).permute(0, 3, 1, 2)
        else:
            contexts = np.stack([self._construct_causal_context(step) for step in steps])
            item["frames"] = torch.from_numpy(frames[contexts]).permute(0, 1, 4, 2, 3)
        return item


def collate_stage2(batch):
    """Stack sampled supervision, retaining complete episodes at their real lengths."""
    replay_count = batch[0]["replay_count"]
    if any(item["replay_count"] != replay_count for item in batch):
        raise ValueError("Expert and query samples require separate batches")
    result = {"target_steps": torch.stack([item["target_steps"] for item in batch]),
              "labels": torch.stack([item["labels"] for item in batch]),
              "seq_len": torch.tensor([item["seq_len"] for item in batch], dtype=torch.long),
              "video_id": [item["video_id"] for item in batch],
              "replay_count": replay_count}
    if "frames_full" in batch[0]:
        result["frames_full"] = [item["frames_full"] for item in batch]
    else:
        result["frames"] = torch.stack([item["frames"] for item in batch])
    return result
