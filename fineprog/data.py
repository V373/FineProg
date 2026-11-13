"""RGB H5 trajectories and frame-aligned R/M/G CSV sidecars."""

import csv

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_IDS = {"R": 0, "M": 1, "G": 2}


def read_videos(path, video_ids=None):
    with h5py.File(path, "r") as file:
        videos = file["videos"]
        ids = sorted(videos) if video_ids is None else list(video_ids)
        return {vid: videos[vid]["frames"][:] for vid in ids}


def read_labels(path, lengths):
    labels = {vid: np.full(n, -1, dtype=np.int64) for vid, n in lengths.items()}
    seen = set()
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            vid, frame = row["id"], int(row["frame_idx"])
            if vid not in labels or not 0 <= frame < len(labels[vid]):
                raise ValueError(f"Label coordinate is outside the H5: {vid}/{frame}")
            if (vid, frame) in seen:
                raise ValueError(f"Duplicate label coordinate: {vid}/{frame}")
            seen.add((vid, frame))
            name, number = row["label"].strip(), row["label_id"].strip()
            if not name or not number:
                continue
            if name not in LABEL_IDS or number != str(LABEL_IDS[name]):
                raise ValueError(f"Inconsistent R/M/G label: {vid}/{frame}")
            labels[vid][frame] = LABEL_IDS[name]
    return labels


def sample_steps(length, count, random_offset=1):
    """The original offset-uniform sampling, including short-video end padding."""
    first = random_offset if length > random_offset else 0
    available = np.arange(first, length)
    if len(available) < count:
        steps = np.concatenate([available, np.full(count - len(available), length - 1)])
    else:
        steps = np.random.choice(available, count, replace=False)
    return np.sort(steps).astype(np.int64)


def frame_tensor(frames):
    return torch.from_numpy(frames).permute(0, 3, 1, 2)


class VideoDataset(Dataset):
    def __init__(self, path, clip_len=20, video_ids=None, labels_path=None,
                 role="replay", refs_per_class=4, random_offset=1):
        all_videos = read_videos(path)
        self.labels = (read_labels(labels_path, {v: len(x) for v, x in all_videos.items()})
                       if labels_path is not None else None)
        self.video_ids = sorted(all_videos) if video_ids is None else list(video_ids)
        self.videos = {vid: all_videos[vid] for vid in self.video_ids}
        self.clip_len, self.role = clip_len, role
        self.refs_per_class, self.random_offset = refs_per_class, random_offset

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        vid = self.video_ids[index]
        frames = self.videos[vid]
        steps = sample_steps(len(frames), self.clip_len, self.random_offset)
        replay_count = self.clip_len
        if self.role == "expert":
            reference_steps, reference_labels = [], []
            labels = self.labels[vid]
            for category in range(3):
                choices = np.flatnonzero(labels == category)
                if len(choices):
                    selected = np.random.choice(choices, self.refs_per_class,
                                                replace=len(choices) < self.refs_per_class)
                    reference_labels.extend([category] * self.refs_per_class)
                else:
                    valid = np.flatnonzero(labels >= 0)
                    selected = np.random.choice(valid if len(valid) else [0], self.refs_per_class)
                    reference_labels.extend([-1] * self.refs_per_class)
                reference_steps.extend(selected)
            sampled_labels = np.concatenate([labels[steps], reference_labels])
            steps = np.concatenate([steps, reference_steps])
        elif self.role == "query":
            goals = np.flatnonzero(self.labels[vid] == 2)
            if len(goals) and not (self.labels[vid][steps] == 2).any():
                steps[-1] = np.random.choice(goals)
                steps.sort()
            sampled_labels = self.labels[vid][steps]
        item = {"frames": frame_tensor(frames[steps]), "target_steps": torch.from_numpy(steps),
                "seq_len": len(frames), "video_id": vid, "replay_count": replay_count}
        if self.role != "replay":
            item["labels"] = torch.from_numpy(sampled_labels)
        return item


class SemanticFrameDataset(Dataset):
    def __init__(self, path, labels_path):
        videos = read_videos(path)
        labels = read_labels(labels_path, {v: len(x) for v, x in videos.items()})
        self.frames = np.concatenate([x[labels[vid] >= 0] for vid, x in videos.items()])
        self.labels = np.concatenate([x[x >= 0] for x in labels.values()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.from_numpy(self.frames[index]).permute(2, 0, 1), int(self.labels[index])
