"""R/G query supervision against current expert R/M/G embedding sets."""

import math

import torch


def semantic_set_loss(query, references, query_labels, reference_labels, tau=0.128):
    """Return set NLL and scalar metrics; labels are R=0, M=1, G=2.

    Embeddings are [Q,D] / [N,D]. Unknown reference labels are excluded;
    a query needs both a same-class reference and a different-class reference.
    M/unknown queries receive no direct supervision. Both embedding sides keep
    their gradients, including a connected zero when no query can be used.
    """
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    query_labels = query_labels.to(device=query.device).reshape(-1)
    reference_labels = reference_labels.to(device=query.device).reshape(-1)
    with torch.autocast(device_type=query.device.type, enabled=False):
        query, references = query.float(), references.float()
        zero = query.sum() * 0 + references.sum() * 0
        metrics = {name: zero.detach() for name in (
            "set_loss", "valid_queries", "r_queries", "g_queries",
            "positive_mass", "mass_R", "mass_M", "mass_G",
        )}
        known = (reference_labels >= 0) & (reference_labels <= 2)
        refs, labels = references[known], reference_labels[known]
        same = query_labels[:, None] == labels[None, :]
        valid = ((query_labels == 0) | (query_labels == 2)) & same.any(1) & (~same).any(1)
        if not valid.any():
            return {"loss": zero, "metrics": metrics}

        selected_labels = query_labels[valid]
        logits = -(query[valid, None, :] - refs[None, :, :]).square().sum(-1) / tau
        log_all = torch.logsumexp(logits, dim=1)
        log_positive = torch.logsumexp(logits.masked_fill(~same[valid], -torch.inf), dim=1)
        per_query = log_all - log_positive
        loss = per_query.mean()
        probabilities = (logits - log_all[:, None]).exp().detach()
        metrics.update(
            set_loss=loss.detach(), valid_queries=valid.sum(),
            r_queries=(selected_labels == 0).sum(),
            g_queries=(selected_labels == 2).sum(),
            positive_mass=(-per_query.detach()).exp().mean(),
        )
        for label, name in enumerate(("R", "M", "G")):
            metrics[f"mass_{name}"] = probabilities[:, labels == label].sum(1).mean()
        return {"loss": loss, "metrics": metrics}
