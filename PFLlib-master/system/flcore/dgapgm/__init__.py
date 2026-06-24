"""Reusable components for the DGAPGM algorithm."""

from .losses import MechanismPrototypeContrastiveLoss
from .mechanism import build_dga_distance_matrix, build_margin_matrix, build_mixup_adjacency
from .mixup import PrototypeGuidedGasMixup, gas_to_dga_rgb
from .prototype import aggregate_prototypes, compute_local_prototypes

__all__ = [
    "MechanismPrototypeContrastiveLoss", "PrototypeGuidedGasMixup",
    "aggregate_prototypes", "build_dga_distance_matrix", "build_margin_matrix",
    "build_mixup_adjacency", "compute_local_prototypes", "gas_to_dga_rgb",
]
