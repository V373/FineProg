"""Labelled single frames (image, R/M/G) for the upper semantic classifier."""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .stage2_dataset import read_labels


class SemanticFrameDataset(Dataset):
    """Every labelled frame of one H5, cached in memory as uint8.

    video_ids / frame_idx only trace a sample back to its source frame; they are
    never model inputs. Frames without an R/M/G label are left out.
    """

    def __init__(self, h5_path, labels_csv):
        with h5py.File(h5_path, "r") as file:
            videos = file["videos"]
            ids = sorted(videos)
            labels = read_labels(labels_csv, {vid: len(videos[vid]["frames"]) for vid in ids})
            keep = [np.flatnonzero(labels[vid] >= 0) for vid in ids]
            self.frames = np.concatenate([videos[vid]["frames"][:][steps] for vid, steps in zip(ids, keep)])
        self.labels = np.concatenate([labels[vid][steps] for vid, steps in zip(ids, keep)])
        self.video_ids = np.repeat(ids, [len(steps) for steps in keep])
        self.frame_idx = np.concatenate(keep)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.from_numpy(self.frames[index]).permute(2, 0, 1), int(self.labels[index])

    def balanced_sampler(self, num_samples):
        """Draw R/M/G with equal probability so the long R phase does not dominate."""
        weights = 1.0 / np.bincount(self.labels, minlength=3)[self.labels]
        return WeightedRandomSampler(torch.from_numpy(weights), num_samples, replacement=True)
