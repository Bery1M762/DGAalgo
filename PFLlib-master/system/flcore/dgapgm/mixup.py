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

    def generate(self, gases, labels, global_prototypes, adjacency, alpha):
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
        mixed_gases, targets, proto_targets = [], [], []
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
            second = target_indices[torch.randint(target_indices.numel(), (1,), device=device)]
            if first.item() == second.item() and source_indices.numel() < 2:
                continue
            lam = beta.sample().to(device)
            mixed_gases.append(lam * gases[first] + (1.0 - lam) * gases[second])
            targets.append(lam * F.one_hot(labels[first], self.num_classes).float()
                           + (1.0 - lam) * F.one_hot(labels[second], self.num_classes).float())
            if global_prototypes is not None:
                proto_targets.append(lam * global_prototypes[labels[first]]
                                     + (1.0 - lam) * global_prototypes[labels[second]])
        if not mixed_gases:
            return None
        mixed_gases = torch.cat(mixed_gases, dim=0)
        return (gas_to_dga_rgb(mixed_gases, self.eps), torch.cat(targets, dim=0),
                torch.cat(proto_targets, dim=0) if proto_targets else None)
