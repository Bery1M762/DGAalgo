"""Mechanism-aware prototype contrastive loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MechanismPrototypeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, global_prototypes, margin_matrix, proto_mask=None):
        if global_prototypes is None or global_prototypes.numel() == 0:
            return features.sum() * 0.0
        if proto_mask is None:
            proto_mask = torch.ones(global_prototypes.shape[0], dtype=torch.bool,
                                    device=global_prototypes.device)
        proto_mask = proto_mask.to(features.device).bool()
        valid_samples = proto_mask[labels]
        if not valid_samples.any():
            return features.sum() * 0.0
        features, labels = features[valid_samples], labels[valid_samples]
        prototypes = F.normalize(global_prototypes.to(features.device), dim=1, eps=1e-12)
        similarities = F.normalize(features, dim=1, eps=1e-12) @ prototypes.t()
        available = proto_mask.unsqueeze(0).expand(features.shape[0], -1)
        margins = margin_matrix.to(features.device)[labels]
        logits = (similarities + margins) / self.temperature
        # Positive logits must not include a margin; unavailable prototypes are excluded.
        row_indices = torch.arange(features.shape[0], device=features.device)
        logits[row_indices, labels] = similarities[row_indices, labels] / self.temperature
        logits = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
        return F.cross_entropy(logits, labels)
