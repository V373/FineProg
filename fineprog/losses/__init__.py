"""The baseline's deterministic TCC, temporal triplet and R/G set losses."""

import torch
from torch.nn import functional as F


def tcc_loss(embeddings, steps, lengths, temperature=0.001, variance_lambda=0.001):
    """Variance-aware cycle-back regression over all ordered video pairs."""
    batch, time, dim = embeddings.shape
    if batch < 2:
        raise ValueError("TCC needs at least two expert videos per batch")
    z = embeddings
    dot = torch.einsum("iad,jbd->ijab", z, z)
    norms = torch.sum(torch.square(z), dim=-1)
    logits = -(norms[:, None, :, None] + norms[None, :, None, :] - 2.0 * dot).clamp_min(0)
    weights = (logits / float(dim) / temperature).softmax(-1)
    neighbors = torch.einsum("ijab,jbd->ijad", weights, z)
    reverse = torch.einsum("ijad,ibd->ijab", neighbors, z)
    neighbor_norm = torch.sum(torch.square(neighbors), dim=-1)
    logits = -(neighbor_norm[..., None] + norms[:, None, None, :] - 2.0 * reverse).clamp_min(0)
    pairs = ~torch.eye(batch, device=z.device, dtype=torch.bool)
    logits = logits / float(dim) / temperature
    logits = logits[pairs].reshape(-1, time)
    labels = torch.eye(time, dtype=z.dtype, device=z.device).expand(batch * (batch - 1), -1, -1).reshape(-1, time)
    source = pairs.nonzero(as_tuple=True)[0]
    sampled = steps[source].unsqueeze(1).expand(-1, time, -1).reshape(-1, time)
    durations = lengths[source].unsqueeze(1).expand(-1, time).reshape(-1)
    times = sampled.to(logits.dtype) / durations.to(logits.dtype)[:, None].expand(-1, time)
    beta = F.softmax(logits, dim=-1)
    actual = torch.sum(times * labels, dim=1)
    predicted = torch.sum(times * beta, dim=1)
    predicted_tiled = predicted.unsqueeze(1).expand(-1, time)
    variance = torch.sum(torch.square(times - predicted_tiled) * beta, dim=1).clamp_min(1e-8)
    log_variance = torch.log(variance)
    squared_error = torch.square(actual - predicted)
    loss = torch.exp(-log_variance) * squared_error + variance_lambda * log_variance
    return torch.mean(loss)


def temporal_triplet_loss(embeddings, steps, margin=0.2):
    """Mean fixed-margin hinge over every valid within-video temporal triplet."""
    _, time, _ = embeddings.shape
    indices = torch.arange(time, device=embeddings.device)
    anchor, first, second = torch.meshgrid(indices, indices, indices, indexing="ij")
    candidates = (anchor != first) & (anchor != second) & (first < second)
    anchor, first, second = (x[candidates] for x in (anchor, first, second))
    gap_first = (steps[:, anchor] - steps[:, first]).abs()
    gap_second = (steps[:, anchor] - steps[:, second]).abs()
    valid = gap_first != gap_second
    distances = []
    for x in embeddings:
        norm = (x ** 2).sum(dim=1, keepdim=True)
        dot = x @ x.T
        distances.append((norm + norm.T - 2.0 * dot).clamp_min(0))
    distance = torch.stack(distances)
    first_distance, second_distance = distance[:, anchor, first], distance[:, anchor, second]
    closer = gap_first < gap_second
    positive = torch.where(closer, first_distance, second_distance)
    negative = torch.where(closer, second_distance, first_distance)
    hinge = (positive - negative + margin).clamp_min(0)
    sum_dtype = torch.float32 if hinge.dtype in (torch.float16, torch.bfloat16) else hinge.dtype
    return (hinge.masked_fill(~valid, 0).sum(dtype=sum_dtype) / valid.sum().clamp_min(1)).to(hinge.dtype)


def replay_loss(embeddings, steps, lengths, temperature=0.001,
                variance_lambda=0.001, triplet_margin=0.2, triplet_weight=0.5):
    """Expert objective used unchanged in lower Stage1 and Stage2 replay."""
    tcc = tcc_loss(embeddings, steps, lengths, temperature, variance_lambda)
    triplet = temporal_triplet_loss(embeddings, steps, triplet_margin)
    return tcc + triplet_weight * triplet


def semantic_set_loss(query, references, query_labels, reference_labels, tau=0.128):
    """R/G query NLL of same-class reference mass; M remains a negative class."""
    with torch.autocast(device_type=query.device.type, enabled=False):
        query, references = query.float(), references.float()
        zero = query.sum() * 0 + references.sum() * 0
        known = (reference_labels >= 0) & (reference_labels <= 2)
        references, reference_labels = references[known], reference_labels[known]
        same = query_labels[:, None] == reference_labels[None, :]
        valid = ((query_labels == 0) | (query_labels == 2)) & same.any(1) & (~same).any(1)
        if not valid.any():
            return zero
        logits = -(query[valid, None] - references[None]).square().sum(-1) / tau
        log_all = torch.logsumexp(logits, dim=1)
        positive = torch.logsumexp(logits.masked_fill(~same[valid], -torch.inf), dim=1)
        return (log_all - positive).mean()
