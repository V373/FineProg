"""
Temporal InfoNCE Contrastive Loss for Encoder Embeddings.

Mathematical definition
-----------------------
For anchor frame i inside video b, let s[b,i] = target_steps[b,i].

  temporal_gap(i,j) = |s[b,i] - s[b,j]|

  step_range(b)  = max(s[b]) - min(s[b])          (per-video)
  abs_pos(b)     = pos_threshold * step_range(b)
  abs_neg(b)     = neg_threshold * step_range(b)

  P(i) = { j != i  |  0 < gap(i,j) <= abs_pos(b) }
  N(i) = { k       |  gap(i,k) >= abs_neg(b) }

Both thresholds are fractions of the per-video step range so that the
positive/negative bands scale consistently across videos of different
lengths. Typical values: pos_threshold=0.05, neg_threshold=0.3.
Frames in the middle band (abs_pos < gap < abs_neg) are ignored.

  logit[i,j] = -||z_i - z_j||^2 / temperature          (squared_l2=True)
             = -||z_i - z_j||   / temperature           (squared_l2=False)

  L_i = -log( sum_{j in P(i)} exp(logit[i,j])
             / ( sum_{j in P(i)} exp(logit[i,j])
               + sum_{k in N(i)} exp(logit[i,k]) ) )

  loss = mean over anchors that have at least one P and at least one N.

Difference from TCCLoss
-----------------------
TCCLoss enforces cross-video temporal cycle-consistency alignment.
TemporalInfoNCELoss uses intra-video temporal proximity as the sole
positive/negative signal and does NOT require pairs of videos.

Anchor sampling modes
---------------------
deterministic : every timestep in each video is a candidate anchor (A = T).
stochastic    : a random subset of A < T anchors is sampled per forward pass,
                reducing compute from O(B*T*T*D) to O(B*A*T*D).

Config keys consumed
--------------------
  pos_threshold         float    (required) fraction of per-video step range → abs positive distance
  neg_threshold         float    (required) fraction of per-video step range → abs negative distance
  temperature           float    default 0.2
  squared_l2            bool     default True
  eps                   float    default 1e-8
  use_target_steps      bool     default True
  normalize_embeddings  bool     default False
  anchor_sampling       str      default "deterministic"  ("deterministic" | "stochastic")
  num_anchors           int|null default null  -- explicit anchor budget; preferred when set
  anchor_fraction       float    default 0.5   -- fraction of T used when num_anchors absent
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
import yaml

# Allow running this file directly for quick tests
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from algos.loss.encoder_loss import BaseEncoderLoss


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
_DEFAULT_CFG = {
    "temperature":           0.2,
    "squared_l2":            True,
    "eps":                   1e-8,
    "use_target_steps":      True,
    "normalize_embeddings":  False,
    "anchor_sampling":       "deterministic",
    "num_anchors":           None,
    "anchor_fraction":       0.5,
}


# ---------------------------------------------------------------------------
# TemporalInfoNCELoss
# ---------------------------------------------------------------------------

class TemporalInfoNCELoss(BaseEncoderLoss):
    """
    Single-video temporal-threshold InfoNCE loss.

    See module docstring for full mathematical definition.

    Args:
        loss_cfg:    Dict with config keys (see module docstring).
        config_path: Optional YAML path.  Loaded before loss_cfg.
    """

    def __init__(
        self,
        loss_cfg: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        cfg = dict(_DEFAULT_CFG)

        # Layer 1: load from YAML
        yaml_cfg = self._load_yaml(config_path)
        cfg.update(yaml_cfg)

        # Layer 2: explicit overrides
        if loss_cfg:
            cfg.update(loss_cfg)

        # Mandatory fields
        if "pos_threshold" not in cfg:
            raise ValueError("[TemporalInfoNCELoss] 'pos_threshold' is required in config")
        if "neg_threshold" not in cfg:
            raise ValueError("[TemporalInfoNCELoss] 'neg_threshold' is required in config")

        self.pos_threshold:    float = float(cfg["pos_threshold"])
        self.neg_threshold:    float = float(cfg["neg_threshold"])
        self.temperature:      float = float(cfg["temperature"])
        self.squared_l2:       bool  = bool(cfg["squared_l2"])
        self.eps:              float = float(cfg["eps"])
        self.use_target_steps: bool  = bool(cfg["use_target_steps"])
        self.normalize_embs:   bool  = bool(cfg["normalize_embeddings"])
        self.anchor_sampling:  str   = str(cfg["anchor_sampling"]).lower().strip()
        self.num_anchors               = cfg["num_anchors"]          # int | None
        self.anchor_fraction:  float = float(cfg["anchor_fraction"])

        if self.anchor_sampling not in ("deterministic", "stochastic"):
            raise ValueError(
                f"[TemporalInfoNCELoss] anchor_sampling must be 'deterministic' or 'stochastic', "
                f"got '{self.anchor_sampling}'"
            )
        if self.pos_threshold >= self.neg_threshold:
            raise ValueError(
                f"[TemporalInfoNCELoss] pos_threshold ({self.pos_threshold}) must be "
                f"< neg_threshold ({self.neg_threshold})"
            )

        print(f"[TemporalInfoNCELoss] Initialized:")
        print(f"  pos_threshold={self.pos_threshold} (fraction)  neg_threshold={self.neg_threshold} (fraction)")
        print(f"  temperature={self.temperature}  squared_l2={self.squared_l2}")
        print(f"  anchor_sampling={self.anchor_sampling}  "
              f"num_anchors={self.num_anchors}  anchor_fraction={self.anchor_fraction}")
        print(f"  normalize_embeddings={self.normalize_embs}  use_target_steps={self.use_target_steps}")

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(config_path: Optional[str]) -> Dict[str, Any]:
        if config_path is None:
            # Default V2 path fallback
            v2 = _project_root / "configs_v2" / "loss" / "loss_temporal_infonce.yaml"
            if v2.exists():
                config_path = str(v2)
        if config_path is None:
            return {}
        p = Path(config_path)
        if not p.exists():
            print(f"[TemporalInfoNCELoss] Warning: config not found: {config_path}")
            return {}
        with open(p, "r") as fh:
            result = yaml.safe_load(fh) or {}
        print(f"[TemporalInfoNCELoss] Loaded config from {config_path}")
        return result

    # ------------------------------------------------------------------
    # Anchor index selection
    # ------------------------------------------------------------------

    def _select_anchor_indices(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Return a 1-D LongTensor of anchor indices (length A).

        deterministic : arange(T)
        stochastic    : sample A = min(T, resolved_num_anchors) without replacement
        """
        if self.anchor_sampling == "deterministic":
            return torch.arange(T, device=device)

        # Stochastic: resolve A
        if self.num_anchors is not None:
            A = int(self.num_anchors)
        else:
            A = max(1, int(round(T * self.anchor_fraction)))
        A = max(1, min(A, T))

        # Per-batch the same anchor indices are used (simpler, still random per step)
        perm = torch.randperm(T, device=device)
        return perm[:A]

    # ------------------------------------------------------------------
    # Pairwise squared-L2 distance  [A, T]  (norm-expansion, no [A,T,D] alloc)
    # ------------------------------------------------------------------

    @staticmethod
    def _squared_l2_dist(anchor_embs: torch.Tensor, all_embs: torch.Tensor) -> torch.Tensor:
        """
        Compute squared L2 distances between anchor_embs [A, D] and all_embs [T, D].
        Uses ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b  to avoid [A, T, D] allocation.

        Returns: dist [A, T]
        """
        norm_a = (anchor_embs ** 2).sum(dim=1, keepdim=True)     # [A, 1]
        norm_t = (all_embs    ** 2).sum(dim=1, keepdim=True).t()  # [1, T]
        dot    = anchor_embs @ all_embs.t()                       # [A, T]
        dist   = norm_a + norm_t - 2.0 * dot
        return dist.clamp(min=0.0)                                 # [A, T]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embeddings: torch.Tensor,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute Temporal InfoNCE loss.

        Args:
            embeddings: [B, T, D] encoder output
            batch:      dict with at least 'target_steps' [B, T] and 'seq_len' [B]

        Returns:
            {"loss": scalar tensor, "metrics": dict}
        """
        B, T, D = embeddings.shape
        device  = embeddings.device

        # Optional embedding normalisation
        if self.normalize_embs:
            embeddings = F.normalize(embeddings, dim=-1)

        # ── Temporal indices ────────────────────────────────────────────
        if self.use_target_steps and "target_steps" in batch:
            steps = batch["target_steps"].to(device=device, dtype=torch.float32)  # [B, T]
        else:
            steps = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)

        # ── Anchor index selection (same for all videos in the batch) ──
        anchor_idx = self._select_anchor_indices(T, device)  # [A]
        A = anchor_idx.shape[0]

        # ── Per-video loss accumulation ─────────────────────────────────
        total_loss        = embeddings.new_zeros(())
        n_valid_anchors   = 0
        sum_pos_dist      = 0.0
        sum_neg_dist      = 0.0
        n_pos_pairs_total = 0
        n_neg_pairs_total = 0

        for b in range(B):
            z       = embeddings[b]                # [T, D]
            s       = steps[b]                     # [T]  float

            z_anchor = z[anchor_idx]               # [A, D]
            s_anchor = s[anchor_idx]               # [A]  float

            # Temporal gap matrix: [A, T]  (|s_anchor_i - s_j|)
            gap = (s_anchor.unsqueeze(1) - s.unsqueeze(0)).abs()  # [A, T]

            # Per-video absolute thresholds from fractional config
            step_range = s.max() - s.min()                             # scalar
            abs_pos    = self.pos_threshold * step_range               # scalar
            abs_neg    = self.neg_threshold * step_range               # scalar

            # Masks
            # anchor cannot match itself: s_anchor[i] vs s[i] when anchor_idx[i]==i → gap==0
            pos_mask = (gap > 0) & (gap <= abs_pos)                    # [A, T]
            neg_mask = (gap >= abs_neg)                                # [A, T]

            # Pairwise distances and logits
            dist2 = self._squared_l2_dist(z_anchor, z)            # [A, T]
            if self.squared_l2:
                dist_for_logit = dist2
            else:
                dist_for_logit = (dist2 + self.eps).sqrt()

            logits = -dist_for_logit / self.temperature            # [A, T]

            # Valid anchor: has at least one pos and at least one neg
            has_pos = pos_mask.any(dim=1)                          # [A]
            has_neg = neg_mask.any(dim=1)                          # [A]
            valid   = has_pos & has_neg                            # [A]

            if not valid.any():
                continue

            # Mask invalid positions with -inf before logsumexp
            NEG_INF = torch.finfo(logits.dtype).min

            # Numerator: sum_{j in P(i)} exp(logit[i,j])
            logits_pos = logits.masked_fill(~pos_mask, NEG_INF)    # [A, T]
            log_num    = torch.logsumexp(logits_pos, dim=1)        # [A]

            # Denominator: sum_{j in P(i) or N(i)} exp(logit[i,j])
            in_denom      = pos_mask | neg_mask                    # [A, T]
            logits_denom  = logits.masked_fill(~in_denom, NEG_INF) # [A, T]
            log_denom     = torch.logsumexp(logits_denom, dim=1)   # [A]

            # InfoNCE loss per anchor
            loss_per_anchor = log_denom - log_num                  # [A]

            # Only average over valid anchors
            loss_valid = loss_per_anchor[valid]
            total_loss = total_loss + loss_valid.sum()
            n_valid_anchors += int(valid.sum().item())

            # Diagnostics (detached)
            with torch.no_grad():
                if pos_mask[valid].any():
                    sum_pos_dist += float(dist2[valid][pos_mask[valid]].mean().item())
                    n_pos_pairs_total += 1
                if neg_mask[valid].any():
                    sum_neg_dist += float(dist2[valid][neg_mask[valid]].mean().item())
                    n_neg_pairs_total += 1

        # ── Final loss ─────────────────────────────────────────────────
        if n_valid_anchors == 0:
            safe_loss = embeddings.sum() * 0.0
            return {
                "loss": safe_loss,
                "metrics": {
                    "loss_total":             float("nan"),
                    "loss_temporal_infonce":  float("nan"),
                    "num_valid_anchors":      0,
                    "num_sampled_anchors":    A,
                    "mean_pos_dist2":         float("nan"),
                    "mean_neg_dist2":         float("nan"),
                },
            }

        contrastive_loss = total_loss / n_valid_anchors

        mean_pos = sum_pos_dist / max(n_pos_pairs_total, 1)
        mean_neg = sum_neg_dist / max(n_neg_pairs_total, 1)

        return {
            "loss": contrastive_loss,
            "metrics": {
                "loss_total":            contrastive_loss.detach().item(),
                "loss_temporal_infonce": contrastive_loss.detach().item(),
                "num_valid_anchors":     n_valid_anchors,
                "num_sampled_anchors":   A,
                "mean_pos_dist2":        mean_pos,
                "mean_neg_dist2":        mean_neg,
            },
        }
