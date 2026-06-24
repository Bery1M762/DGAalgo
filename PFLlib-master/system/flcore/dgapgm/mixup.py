"""Gas-space mixup utilities for DGAPGM."""
import torch
import torch.nn.functional as F


def gas_to_dga_rgb(gas, eps=1e-6):
    """Map five DGA gases to the specified three-channel 5x5 representation."""
    if gas.ndim == 1:
        gas = gas.unsqueeze(0)
    if gas.ndim != 2 or gas.shape[1] != 5:
        raise ValueError("gas must have shape [B, 5] or [5]")
    left, right = gas.unsqueeze(2), gas.unsqueeze(1)
    red = left / (left + right + eps)
    green = torch.log1p((left - right).abs())
    green = green / (green.amax(dim=(1, 2), keepdim=True) + eps)
    blue = (left * right) / (gas.amax(dim=1, keepdim=True).unsqueeze(2).pow(2) + eps)
    return torch.stack((red, green, blue), dim=1)


class PrototypeGuidedGasMixup:
    def __init__(self, num_classes, minority_gamma=1.0, eps=1e-6):
        self.num_classes = num_classes
        self.minority_gamma = minority_gamma
        self.eps = eps

    def generate(self, gases, labels, global_prototypes, adjacency, alpha, proto_mask=None):
        if gases is None or gases.shape[0] < 2:
            return None
        device = gases.device
        adjacency = adjacency.to(device).bool()
        class_counts = torch.bincount(labels, minlength=self.num_classes).float()
        present = class_counts > 0
        weights = (class_counts + self.eps).pow(-self.minority_gamma) * present
        if weights.sum() == 0:
            return None
        source_classes = torch.multinomial(weights / weights.sum(), labels.shape[0], replacement=True)
        if proto_mask is None:
            proto_mask = torch.zeros(self.num_classes, dtype=torch.bool, device=device)
        else:
            proto_mask = proto_mask.to(device).bool()
        mixed_gases, targets, proto_targets, proto_valid = [], [], [], []
        beta = torch.distributions.Beta(float(alpha), float(alpha))
        for source_class in source_classes.tolist():
            source_indices = torch.where(labels == source_class)[0]
            candidate_classes = torch.where(adjacency[source_class] & present)[0]
            if source_indices.numel() == 0 or candidate_classes.numel() == 0:
                continue
            # Same class is preferred when it supplies a distinct sample.
            if source_indices.numel() >= 2:
                target_class = source_class
            else:
                nonself = candidate_classes[candidate_classes != source_class]
                target_class = int((nonself if nonself.numel() else candidate_classes)[
                    torch.randint(len(nonself) if nonself.numel() else len(candidate_classes), (1,), device=device)
                ].item())
            target_indices = torch.where(labels == target_class)[0]
            first = source_indices[torch.randint(source_indices.numel(), (1,), device=device)]
            if target_class == source_class and source_indices.numel() >= 2:
                # Same-class mixup must use two different samples.
                available_seconds = source_indices[source_indices != first]
                second = available_seconds[torch.randint(available_seconds.numel(), (1,), device=device)]
            else:
                second = target_indices[torch.randint(target_indices.numel(), (1,), device=device)]
            if first.item() == second.item():
                continue
            lam = beta.sample().to(device)
            mixed_gases.append(lam * gases[first] + (1.0 - lam) * gases[second])
            targets.append(lam * F.one_hot(labels[first], self.num_classes).float()
                           + (1.0 - lam) * F.one_hot(labels[second], self.num_classes).float())
            pair_has_prototypes = (global_prototypes is not None
                                   and bool(proto_mask[labels[first]].item())
                                   and bool(proto_mask[labels[second]].item()))
            if pair_has_prototypes:
                proto_targets.append(lam * global_prototypes[labels[first]]
                                     + (1.0 - lam) * global_prototypes[labels[second]])
            else:
                # Preserve batch alignment while making the target unusable for MSE.
                proto_targets.append(torch.zeros_like(global_prototypes[0]).unsqueeze(0)
                                     if global_prototypes is not None else torch.empty(0, device=device))
            proto_valid.append(pair_has_prototypes)
        if not mixed_gases:
            return None
        mixed_gases = torch.cat(mixed_gases, dim=0)
        proto_targets = torch.cat(proto_targets, dim=0) if global_prototypes is not None else None
        return (gas_to_dga_rgb(mixed_gases, self.eps), torch.cat(targets, dim=0),
                proto_targets, torch.tensor(proto_valid, dtype=torch.bool, device=device))
