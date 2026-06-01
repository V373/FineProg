"""
CompositeEncoderLoss — Weighted sum of multiple BaseEncoderLoss components.

Architecture
------------
CompositeEncoderLoss owns N child BaseEncoderLoss instances (registered as
nn.ModuleDict so .to(device) and train()/eval() propagate automatically).
Its forward() passes the same embeddings and batch to every child, then
returns a weighted scalar:

    loss = Σ_i  weight_i * child_i.loss

This is a drop-in replacement for any single loss: train.py sees one
BaseEncoderLoss-compatible module with the standard {loss, metrics} contract.

Configuration schema (composite YAML)
--------------------------------------
components:
  - alias:       tcc           # unique key; used as metric prefix
    name:        tcc           # loss_name dispatched by build_loss()
    weight:      1.0           # scalar multiplier on the child loss
    config_file: loss_tcc.yaml # child YAML, relative to composite YAML dir

  - alias:       temporal_infonce
    name:        temporal_infonce
    weight:      0.5
    config_file: loss_temporal_infonce.yaml

Metric namespacing policy
--------------------------
Summary metrics (top-level, no prefix):
  loss_total                           — the weighted-sum scalar (for wandb main curve)
  loss_composite                       — same as loss_total (explicit alias)
  component_weight/<alias>             — static weight per child (constant)
  component_raw_loss/<alias>           — unweighted child loss value
  component_weighted_loss/<alias>      — weight * raw loss contribution

Child-detail metrics (prefixed by alias):
  <alias>/<child_metric_key>           — all keys from child.metrics dict
  Note: child "loss_total" gets prefixed too → "tcc/loss_total" etc.

Constraints
-----------
- Weights are static scalars (no schedule, no gradient balancing).
- Nested composite losses are NOT supported (guards in __init__).
- All child losses share the same embeddings and batch — the training loop
  computes encoder outputs exactly once, as before.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from algos.loss.encoder_loss import BaseEncoderLoss, build_loss


# ---------------------------------------------------------------------------
# CompositeEncoderLoss
# ---------------------------------------------------------------------------

class CompositeEncoderLoss(BaseEncoderLoss):
    """
    Weighted-sum composite of multiple encoder losses.

    Args:
        loss_cfg (dict):    Parsed composite YAML content supplied by build_loss().
                            Must contain a "components" list (see module docstring).
        config_path (str):  Path to the composite YAML file; used to resolve
                            component config_file paths relative to its directory.
    """

    def __init__(
        self,
        loss_cfg: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        if loss_cfg is None:
            loss_cfg = {}

        # Resolve the directory that holds the composite YAML so we can resolve
        # relative config_file paths for child components.
        if config_path is not None:
            parent_dir = Path(config_path).resolve().parent
        else:
            # Default fallback: configs_v2/ in project root
            parent_dir = _project_root / "configs_v2"

        components_cfg: List[Dict] = loss_cfg.get("components", [])
        if not components_cfg:
            raise ValueError(
                "[CompositeEncoderLoss] 'components' list is empty or missing in config. "
                "At least two child components are required."
            )

        # Build name → weight and name → child module mappings
        aliases: List[str] = []
        weights: List[float] = []
        modules: Dict[str, BaseEncoderLoss] = {}

        for idx, spec in enumerate(components_cfg):
            alias       = str(spec.get("alias", f"loss_{idx}"))
            child_name  = str(spec.get("name", ""))
            weight      = float(spec.get("weight", 1.0))
            cfg_file    = spec.get("config_file", None)

            if not child_name:
                raise ValueError(
                    f"[CompositeEncoderLoss] Component #{idx} is missing 'name' key."
                )
            if child_name == "composite":
                raise ValueError(
                    "[CompositeEncoderLoss] Nested composite losses are not supported. "
                    f"Component '{alias}' uses name='composite'."
                )
            if alias in modules:
                raise ValueError(
                    f"[CompositeEncoderLoss] Duplicate alias '{alias}' in components list."
                )

            # Resolve child config path
            if cfg_file is not None:
                child_cfg_path = str(parent_dir / cfg_file)
            else:
                child_cfg_path = None

            print(
                f"[CompositeEncoderLoss] Building child '{alias}' "
                f"(name={child_name}, weight={weight}, config={child_cfg_path})"
            )
            child_module = build_loss(child_name, config_path=child_cfg_path)

            aliases.append(alias)
            weights.append(weight)
            modules[alias] = child_module

        # Register as ModuleDict so device + train/eval state propagates
        self.children_losses = torch.nn.ModuleDict(modules)
        self._aliases: List[str] = aliases
        self._weights: Dict[str, float] = dict(zip(aliases, weights))

        total_w = sum(weights)
        print(
            f"[CompositeEncoderLoss] Initialized with {len(aliases)} components: "
            + ", ".join(f"{a}×{w}" for a, w in zip(aliases, weights))
            + f"  (Σweights={total_w:.4f})"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embeddings: torch.Tensor,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute weighted-sum composite loss.

        Args:
            embeddings: [B, T, D]
            batch:      dict — same contract as individual loss forward()

        Returns:
            {"loss": scalar, "metrics": dict}
        """
        total_loss: Optional[torch.Tensor] = None
        combined_metrics: Dict[str, Any] = {}

        for alias in self._aliases:
            weight = self._weights[alias]
            child  = self.children_losses[alias]

            out        = child(embeddings, batch)
            child_loss = out["loss"]               # scalar tensor (gradient OK)
            child_mets = out.get("metrics", {})    # detached dict

            weighted = weight * child_loss

            # Accumulate weighted sum
            if total_loss is None:
                total_loss = weighted
            else:
                total_loss = total_loss + weighted

            # Summary metrics for this component
            combined_metrics[f"component_weight/{alias}"]        = weight
            combined_metrics[f"component_raw_loss/{alias}"]      = child_loss.detach().item()
            combined_metrics[f"component_weighted_loss/{alias}"] = weighted.detach().item()

            # Prefix every child metric key with the alias
            for k, v in child_mets.items():
                combined_metrics[f"{alias}/{k}"] = v

        if total_loss is None:
            # No components — should never happen after __init__ guard
            total_loss = embeddings.sum() * 0.0

        # Top-level summary
        combined_metrics["loss_composite"] = total_loss.detach().item()
        combined_metrics["loss_total"]     = total_loss.detach().item()

        return {"loss": total_loss, "metrics": combined_metrics}
