"""
Temporal Triplet Loss for Encoder Embeddings.

Mathematical definition
-----------------------
For a video clip b, let z = embeddings[b] with shape [T, D] and
s = target_steps[b] with shape [T].

For each valid triplet (a, p, n) drawn from the same clip, where:
  |s[a] - s[p]| < |s[a] - s[n]|        (positive is temporally closer to anchor)

Compute squared L2 distances:
  d_ap = ||z[a] - z[p]||_2^2
  d_an = ||z[a] - z[n]||_2^2

Apply hinge loss:
  L = max(0, d_ap - d_an + margin)

Batch loss = mean over all sampled valid triplets across all clips:
  L_batch = mean(L_triplet over valid triplets)

Triplet construction
--------------------
For each anchor a, consider all unordered pairs (j, k) with j < k, j != a, k != a.
Assign positive/negative by temporal distance. Skip pairs where |s[a]-s[j]| == |s[a]-s[k]|.

This vectorized enumeration avoids open-ended resampling loops.
With clip_len=20, each clip has at most 20 * C(19, 2) = 20 * 171 = 3420 candidate
triplet slots, which is cheap enough to enumerate fully.

Sampling budget
---------------
Control compute via num_triplets_fraction: after enumerating N_valid triplets for a
clip, sample K = max(1, round(N_valid * num_triplets_fraction)) without replacement.
Default fraction=1.0 keeps all valid triplets.

Empty-batch behavior
--------------------
If no valid triplets exist for the entire batch, returns embeddings.sum() * 0.0 as a
differentiable zero loss, plus metrics with zero counts and NaN for average statistics.
This matches the stability pattern used by TemporalInfoNCELoss.

Config keys consumed
--------------------
  margin                  float    (required) hinge loss margin; e.g. 1.0
  num_triplets_fraction   float    default 1.0  fraction of valid triplets to sample per clip; (0, 1]
  max_resample_attempts   int      default 16   reserved for forward compatibility; not used by vectorized implementation
  reduction               str      default "mean"   currently only "mean" is supported
  normalize_embeddings    bool     default False
  squared_l2              bool     default True     only True is supported in this version
  capped                  bool     default False    if True, clips d_an at negative_distance_cap before computing hinge
  negative_distance_cap   float    default None     required when capped=True; d_an_eff = min(d_an, c_neg)

Dynamic margin plugin (optional, off by default)
-------------------------------------------------
When dynamic_margin_enabled=True the scalar margin is replaced by a per-triplet
margin computed from the normalized temporal-gap surplus:

  time_gap      = Delta_an - Delta_ap     (always > 0 for valid triplets)
  norm_gap      = time_gap / seq_len_b    (in (0, 1] when steps < seq_len_b)
  m_dyn         = clamp(alpha * norm_gap, margin_min, margin_max)
  L_triplet     = max(0, d_ap - d_an_eff + m_dyn)

This requires seq_len to be present and valid in every batch; no fallback is
provided so the loss raises an explicit error when seq_len is missing or
inconsistent with target_steps.  normalize_embeddings must be False when
the plugin is enabled (raw embedding distances only).

Additional config keys consumed when dynamic_margin_enabled=True
  dynamic_margin_enabled  bool     default False
  dynamic_margin_alpha    float    required; scaling coefficient > 0
  dynamic_margin_min      float    required; lower clamp on m_dyn
  dynamic_margin_max      float    required; upper clamp on m_dyn (>= min)
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml

# Allow running this file directly for quick tests
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from algos.loss.encoder_loss import BaseEncoderLoss


def _to_metric_tensor(value: float, *, device: torch.device) -> torch.Tensor:
    """Create a detached float32 tensor for epoch-level metric aggregation."""
    return torch.tensor(value, device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
_DEFAULT_CFG: Dict[str, Any] = {
    "num_triplets_fraction": 1.0,
    "max_resample_attempts": 16,   # reserved; not used in vectorized implementation
    "reduction":             "mean",
    "normalize_embeddings":  False,
    "squared_l2":            True,
    "capped":                False,
    "negative_distance_cap": None,  # required when capped=True; ignored when capped=False
    # dynamic margin plugin (optional; off by default)
    "dynamic_margin_enabled":  False,
    "dynamic_margin_alpha":    None,   # required when enabled
    "dynamic_margin_min":      None,   # required when enabled
    "dynamic_margin_max":      None,   # required when enabled
}


# ---------------------------------------------------------------------------
# TemporalTripletLoss
# ---------------------------------------------------------------------------

class TemporalTripletLoss(BaseEncoderLoss):
    """
    Intra-video temporal triplet hinge loss.

    For each video clip, constructs triplets (anchor, positive, negative) using
    temporal order: the positive frame is temporally closer to the anchor than
    the negative frame.

    See module docstring for full mathematical definition and construction strategy.

    Args:
        loss_cfg:    Dict with config keys (see module docstring). 'margin' is required.
        config_path: Optional YAML path. Loaded before loss_cfg overrides.
    """

    def __init__(
        self,
        loss_cfg: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        cfg = dict(_DEFAULT_CFG)

        # Layer 1: load from YAML (config_path, or fallback to V2 default)
        yaml_cfg = self._load_yaml(config_path)
        cfg.update(yaml_cfg)

        # Layer 2: explicit overrides from caller
        if loss_cfg:
            cfg.update(loss_cfg)

        # Mandatory field
        if "margin" not in cfg:
            raise ValueError(
                "[TemporalTripletLoss] 'margin' is required in config. "
                "Add 'margin: 1.0' to your loss YAML or pass it via loss_cfg."
            )

        self.margin: float                 = float(cfg["margin"])
        self.num_triplets_fraction: float  = float(cfg["num_triplets_fraction"])
        self.reduction: str                = str(cfg["reduction"]).lower().strip()
        self.normalize_embs: bool          = bool(cfg["normalize_embeddings"])
        self.squared_l2: bool              = bool(cfg["squared_l2"])
        self.max_resample_attempts: int    = int(cfg["max_resample_attempts"])
        self.capped: bool                  = bool(cfg["capped"])
        _raw_cap                           = cfg["negative_distance_cap"]
        if self.capped:
            if _raw_cap is None:
                raise ValueError(
                    "[TemporalTripletLoss] 'negative_distance_cap' is required when "
                    "capped=True. Add it to your loss YAML or pass via loss_cfg."
                )
            self.negative_distance_cap: float = float(_raw_cap)
            if self.negative_distance_cap <= 0:
                raise ValueError(
                    f"[TemporalTripletLoss] negative_distance_cap must be > 0, "
                    f"got {self.negative_distance_cap}."
                )
        else:
            self.negative_distance_cap = float(_raw_cap) if _raw_cap is not None else float("inf")

        # Validation
        if self.reduction != "mean":
            raise ValueError(
                f"[TemporalTripletLoss] Only reduction='mean' is supported, "
                f"got '{self.reduction}'."
            )
        if not self.squared_l2:
            raise ValueError(
                "[TemporalTripletLoss] Only squared_l2=True is supported in this version. "
                "Set squared_l2: true in your config."
            )
        if not (0.0 < self.num_triplets_fraction <= 1.0):
            raise ValueError(
                f"[TemporalTripletLoss] num_triplets_fraction must be in (0, 1], "
                f"got {self.num_triplets_fraction}."
            )

        # ── Dynamic margin plugin ──────────────────────────────────────────────
        self.dynamic_margin_enabled: bool = bool(cfg.get("dynamic_margin_enabled", False))
        if self.dynamic_margin_enabled:
            for _key in ("dynamic_margin_alpha", "dynamic_margin_min", "dynamic_margin_max"):
                if cfg.get(_key) is None:
                    raise ValueError(
                        f"[TemporalTripletLoss] '{_key}' is required when "
                        f"dynamic_margin_enabled=True."
                    )
            self.dynamic_margin_alpha: float = float(cfg["dynamic_margin_alpha"])
            self.dynamic_margin_min:   float = float(cfg["dynamic_margin_min"])
            self.dynamic_margin_max:   float = float(cfg["dynamic_margin_max"])
            if self.dynamic_margin_alpha <= 0:
                raise ValueError(
                    f"[TemporalTripletLoss] dynamic_margin_alpha must be > 0, "
                    f"got {self.dynamic_margin_alpha}."
                )
            if self.dynamic_margin_min > self.dynamic_margin_max:
                raise ValueError(
                    f"[TemporalTripletLoss] dynamic_margin_min ({self.dynamic_margin_min}) "
                    f"must be <= dynamic_margin_max ({self.dynamic_margin_max})."
                )
            if self.normalize_embs:
                raise ValueError(
                    "[TemporalTripletLoss] dynamic_margin_enabled=True requires "
                    "normalize_embeddings=False (raw embedding distances only)."
                )
        else:
            self.dynamic_margin_alpha = float("nan")
            self.dynamic_margin_min   = float("nan")
            self.dynamic_margin_max   = float("nan")

        print(f"[TemporalTripletLoss] Initialized:")
        print(f"  margin={self.margin}  num_triplets_fraction={self.num_triplets_fraction}")
        print(f"  reduction={self.reduction}  squared_l2={self.squared_l2}")
        print(f"  normalize_embeddings={self.normalize_embs}")
        print(f"  capped={self.capped}  negative_distance_cap={self.negative_distance_cap}")
        if self.dynamic_margin_enabled:
            print(f"  dynamic_margin_enabled=True  alpha={self.dynamic_margin_alpha}"
                  f"  min={self.dynamic_margin_min}  max={self.dynamic_margin_max}")
        else:
            print(f"  dynamic_margin_enabled=False")

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(config_path: Optional[str]) -> Dict[str, Any]:
        if config_path is None:
            # Default V2 path fallback
            v2 = _project_root / "configs_v2" / "loss" / "loss_temporal_triplet.yaml"
            if v2.exists():
                config_path = str(v2)
        if config_path is None:
            return {}
        p = Path(config_path)
        if not p.exists():
            print(f"[TemporalTripletLoss] Warning: config not found: {config_path}")
            return {}
        with open(p, "r") as fh:
            result = yaml.safe_load(fh) or {}
        print(f"[TemporalTripletLoss] Loaded config from {config_path}")
        return result

    # ------------------------------------------------------------------
    # Triplet enumeration (vectorized, per-clip)
    # ------------------------------------------------------------------

    @staticmethod
    def _enumerate_valid_triplets(
        s: torch.Tensor,  # [T] integer time steps
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Enumerate all valid (anchor, positive, negative) index triples from
        a single video clip using vectorized enumeration.

        Strategy
        --------
        For each anchor a, consider all unordered pairs (j, k) with j < k,
        j != a, k != a.
          - if |s[a]-s[j]| < |s[a]-s[k]|: valid triplet (a, p=j, n=k)
          - if |s[a]-s[k]| < |s[a]-s[j]|: valid triplet (a, p=k, n=j)
          - if equal: skip

        With T=20: meshgrid is [20,20,20] = 8000 entries; filter to
        ~3420 candidates before removing equal-gap cases.

        Returns
        -------
        (anchor_idx, pos_idx, neg_idx): each [N_valid] LongTensor
        or None if T < 3 or no valid triplets exist.
        """
        T = s.shape[0]
        if T < 3:
            return None

        device = s.device
        idx = torch.arange(T, device=device)

        # Build all (a, j, k) with a != j, a != k, j < k
        # j < k ensures each unordered pair counted once per anchor
        A, J, K = torch.meshgrid(idx, idx, idx, indexing="ij")  # [T, T, T]
        mask = (A != J) & (A != K) & (J < K)

        a_flat = A[mask]   # [N_candidates]
        j_flat = J[mask]
        k_flat = K[mask]

        # Temporal gaps; use float to avoid int32 overflow on large step values
        s_f    = s.float()
        gap_aj = (s_f[a_flat] - s_f[j_flat]).abs()
        gap_ak = (s_f[a_flat] - s_f[k_flat]).abs()

        # Drop pairs where the two candidates are equidistant from the anchor
        valid = gap_aj != gap_ak
        if not valid.any():
            return None

        a_v   = a_flat[valid]
        j_v   = j_flat[valid]
        k_v   = k_flat[valid]
        gaj_v = gap_aj[valid]
        gak_v = gap_ak[valid]

        # Assign positive = temporally closer candidate
        j_closer = gaj_v < gak_v
        pos_idx  = torch.where(j_closer, j_v, k_v)
        neg_idx  = torch.where(j_closer, k_v, j_v)

        return a_v, pos_idx, neg_idx

    # ------------------------------------------------------------------
    # Pairwise distance (per-clip full matrix)
    # ------------------------------------------------------------------

    @staticmethod
    def _squared_l2_dist_full(z: torch.Tensor) -> torch.Tensor:
        """
        Compute the full [T, T] pairwise squared-L2 distance matrix for one clip.

        Uses the norm-expansion identity
            ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        to reduce the cost to one [T,T] GEMM plus two broadcasts,
        avoiding the [T, T, D] intermediate tensor that direct subtraction needs.

        Args:
            z: [T, D] float embeddings for a single clip.

        Returns:
            pdist2: [T, T]  where  pdist2[i, j] = ||z[i] - z[j]||_2^2.
        """
        norm = (z ** 2).sum(dim=1, keepdim=True)          # [T, 1]
        dot  = z @ z.t()                                   # [T, T]  (single GEMM)
        return (norm + norm.t() - 2.0 * dot).clamp(min=0.0)  # [T, T]

    # ------------------------------------------------------------------
    # Dynamic margin plugin
    # ------------------------------------------------------------------

    def _compute_dynamic_margin(
        self,
        s: torch.Tensor,      # [T] integer time steps for this clip
        a_idx: torch.Tensor,  # [K] anchor frame indices (sampled)
        p_idx: torch.Tensor,  # [K] positive frame indices (sampled)
        n_idx: torch.Tensor,  # [K] negative frame indices (sampled)
        seq_len_b: float,     # scalar: sequence length for this clip (> 0)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-triplet dynamic margin from normalized temporal-gap surplus.

        For each sampled triplet (a, p, n):
          Delta_ap = |tau_a - tau_p|
          Delta_an = |tau_a - tau_n|
          time_gap = Delta_an - Delta_ap    (> 0 by valid-triplet construction)
          norm_gap = time_gap / seq_len_b   (in (0, 1] when steps < seq_len_b)
          m_dyn    = clamp(alpha * norm_gap, margin_min, margin_max)

        Args:
            s:         [T] integer time steps for this clip.
            a_idx, p_idx, n_idx: [K] sampled triplet frame indices.
            seq_len_b: video sequence length; must be > 0.

        Returns:
            m_dyn:     [K] per-triplet margin tensor (used in hinge).
            time_gap:  [K] raw (unnormalized) temporal-gap surplus (for logging).
        """
        s_f      = s.float()
        gap_ap   = (s_f[a_idx] - s_f[p_idx]).abs()    # [K]  Delta_ap
        gap_an   = (s_f[a_idx] - s_f[n_idx]).abs()    # [K]  Delta_an
        time_gap = gap_an - gap_ap                     # [K]  > 0 by construction
        norm_gap = time_gap / seq_len_b
        m_dyn    = norm_gap.mul(self.dynamic_margin_alpha).clamp(
            min=self.dynamic_margin_min,
            max=self.dynamic_margin_max,
        )
        return m_dyn, time_gap

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embeddings: torch.Tensor,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute Temporal Triplet loss.

        Args:
            embeddings: [B, T, D] encoder output
            batch:      dict with at least 'target_steps' [B, T] and 'seq_len' [B]

        Returns:
            {"loss": scalar tensor, "metrics": dict}
        """
        # ── Shape and key validation ─────────────────────────────────────
        if embeddings.dim() != 3:
            raise ValueError(
                f"[TemporalTripletLoss] embeddings must be rank 3 [B, T, D], "
                f"got shape {tuple(embeddings.shape)}"
            )
        B, T, D = embeddings.shape

        if T < 3:
            raise ValueError(
                f"[TemporalTripletLoss] clip_len (T={T}) must be >= 3 to form triplets. "
                "Check dataset sampling configuration (clip_len in train.yaml)."
            )

        if "target_steps" not in batch:
            raise ValueError("[TemporalTripletLoss] batch must contain 'target_steps'")
        if "seq_len" not in batch:
            raise ValueError("[TemporalTripletLoss] batch must contain 'seq_len'")

        target_steps = batch["target_steps"]
        seq_len      = batch["seq_len"]

        if tuple(target_steps.shape) != (B, T):
            raise ValueError(
                f"[TemporalTripletLoss] target_steps shape must be [{B}, {T}], "
                f"got {tuple(target_steps.shape)}"
            )
        if seq_len.shape[0] != B:
            raise ValueError(
                f"[TemporalTripletLoss] seq_len must have batch dimension {B}, "
                f"got shape {tuple(seq_len.shape)}"
            )

        device      = embeddings.device
        steps       = target_steps.to(device=device, dtype=torch.long)
        seq_len_dev = seq_len.to(device=device)

        # ── Optional embedding normalisation ────────────────────────────
        if self.normalize_embs:
            embeddings = F.normalize(embeddings, dim=-1)

        # ── Dynamic margin plugin: validate seq_len and target_steps ─────
        if self.dynamic_margin_enabled:
            for _b in range(B):
                _sl = int(seq_len_dev[_b].item())
                if _sl <= 0:
                    raise ValueError(
                        f"[TemporalTripletLoss] dynamic_margin_enabled=True requires "
                        f"seq_len[{_b}] > 0, got {_sl}."
                    )
                _max_step = int(steps[_b].max().item())
                if _max_step >= _sl:
                    raise ValueError(
                        f"[TemporalTripletLoss] dynamic_margin_enabled=True requires "
                        f"target_steps < seq_len for valid [0,1] normalization; "
                        f"clip {_b}: max(target_steps)={_max_step} >= seq_len={_sl}."
                    )

        # ── Per-clip triplet construction and loss accumulation ──────────
        all_hinge:       list = []
        all_d_ap:        list = []
        all_d_an:        list = []
        all_d_an_eff:    list = []
        all_margin_term: list = []  # populated only when dynamic_margin_enabled
        all_time_gap:    list = []  # populated only when dynamic_margin_enabled

        n_clips_with_valid   = 0
        total_valid_triplets = 0

        for b in range(B):
            s = steps[b]        # [T]
            z = embeddings[b]   # [T, D]

            result = self._enumerate_valid_triplets(s)
            if result is None:
                continue

            a_idx, p_idx, n_idx = result  # each [N_valid]
            N_valid = a_idx.shape[0]
            total_valid_triplets += N_valid

            # Subsample according to num_triplets_fraction
            K = max(1, round(N_valid * self.num_triplets_fraction))
            K = min(K, N_valid)

            if K < N_valid:
                perm  = torch.randperm(N_valid, device=device)
                sel   = perm[:K]
                a_idx = a_idx[sel]
                p_idx = p_idx[sel]
                n_idx = n_idx[sel]

            # Pairwise squared-L2 matrix for this clip [T, T].
            # Computed once via norm expansion + single GEMM; all K sampled
            # triplets then read their distances by cheap index lookup.
            pdist2 = self._squared_l2_dist_full(z)    # [T, T]

            # Squared L2 distances via index lookup (no gather+subtract needed)
            d_ap = pdist2[a_idx, p_idx]               # [K]
            d_an = pdist2[a_idx, n_idx]               # [K]

            # Apply capping to negative distance when capped=True
            d_an_eff = d_an.clamp(max=self.negative_distance_cap) if self.capped else d_an

            # ── Select margin source (fixed scalar or dynamic-margin plugin) ──
            if self.dynamic_margin_enabled:
                m_term, time_gap_k = self._compute_dynamic_margin(
                    s, a_idx, p_idx, n_idx,
                    seq_len_b=float(seq_len_dev[b].item()),
                )
                all_margin_term.append(m_term.detach())
                all_time_gap.append(time_gap_k.detach())
            else:
                m_term = self.margin  # scalar, broadcast by clamp

            hinge = torch.clamp(d_ap - d_an_eff + m_term, min=0.0)  # [K]

            all_hinge.append(hinge)
            all_d_ap.append(d_ap.detach())
            all_d_an.append(d_an.detach())
            all_d_an_eff.append(d_an_eff.detach())
            n_clips_with_valid += 1

        # ── Embedding L2 norm (mean over all B×T frames, non-squared) ───
        mean_emb_l2norm = float(
            embeddings.detach().norm(dim=-1).mean().item()
        )

        # ── Safe zero-loss fallback ─────────────────────────────────────
        if not all_hinge:
            safe_loss = embeddings.sum() * 0.0
            nan_metric = _to_metric_tensor(float("nan"), device=device)
            return {
                "loss": safe_loss,
                "metrics": {
                    "loss_total":                    nan_metric,
                    "loss_temporal_triplet":         nan_metric,
                    "num_valid_triplets":            _to_metric_tensor(0.0, device=device),
                    "num_sampled_triplets":          _to_metric_tensor(0.0, device=device),
                    "num_clips_with_valid_triplets": _to_metric_tensor(0.0, device=device),
                    "mean_d_ap":                     nan_metric,
                    "mean_d_an":                     nan_metric,
                    "mean_d_an_capped":              nan_metric,
                    "triplet_accuracy_margin":       nan_metric,
                    "active_triplet_fraction":       nan_metric,
                    "mean_emb_l2norm":               _to_metric_tensor(mean_emb_l2norm, device=device),
                    "mean_dynamic_margin":           nan_metric,
                    "mean_time_gap":                 nan_metric,
                },
            }

        # ── Aggregate loss ───────────────────────────────────────────────
        hinge_all    = torch.cat(all_hinge)      # [N_sampled_total]
        d_ap_all     = torch.cat(all_d_ap)
        d_an_all     = torch.cat(all_d_an)
        d_an_eff_all = torch.cat(all_d_an_eff)

        batch_loss  = hinge_all.mean()
        N_sampled   = hinge_all.shape[0]

        mean_d_ap_val        = float(d_ap_all.mean().item())
        mean_d_an_val        = float(d_an_all.mean().item())
        # mean_d_an_capped: effective negative distance used in training.
        # Equals mean_d_an when capped=False; clamped at c_neg when capped=True.
        mean_d_an_capped_val = float(d_an_eff_all.mean().item())
        active_frac          = float((hinge_all.detach() > 0).float().mean().item())

        if self.dynamic_margin_enabled:
            margin_all          = torch.cat(all_margin_term)   # [N_sampled]
            time_gap_cat        = torch.cat(all_time_gap)       # [N_sampled]
            mean_dyn_margin_val = float(margin_all.mean().item())
            mean_time_gap_val   = float(time_gap_cat.mean().item())
            # triplet_accuracy_margin: d_ap + m_dyn <= d_an (raw, uncapped)
            acc_margin          = float(((d_ap_all + margin_all) <= d_an_all).float().mean().item())
        else:
            mean_dyn_margin_val = float("nan")
            mean_time_gap_val   = float("nan")
            # triplet_accuracy_margin: fraction satisfying d_ap + margin <= d_an.
            # Always uses raw (uncapped) d_an to reflect true geometric accuracy.
            acc_margin          = float(((d_ap_all + self.margin) <= d_an_all).float().mean().item())

        return {
            "loss": batch_loss,
            "metrics": {
                "loss_total":                    batch_loss.detach(),
                "loss_temporal_triplet":         batch_loss.detach(),
                "num_valid_triplets":            _to_metric_tensor(float(total_valid_triplets), device=device),
                "num_sampled_triplets":          _to_metric_tensor(float(N_sampled), device=device),
                "num_clips_with_valid_triplets": _to_metric_tensor(float(n_clips_with_valid), device=device),
                "mean_d_ap":                     _to_metric_tensor(mean_d_ap_val, device=device),
                "mean_d_an":                     _to_metric_tensor(mean_d_an_val, device=device),
                "mean_d_an_capped":              _to_metric_tensor(mean_d_an_capped_val, device=device),
                "triplet_accuracy_margin":       _to_metric_tensor(acc_margin, device=device),
                "active_triplet_fraction":       _to_metric_tensor(active_frac, device=device),
                "mean_emb_l2norm":               _to_metric_tensor(mean_emb_l2norm, device=device),
                "mean_dynamic_margin":           _to_metric_tensor(mean_dyn_margin_val, device=device),
                "mean_time_gap":                 _to_metric_tensor(mean_time_gap_val, device=device),
            },
        }
