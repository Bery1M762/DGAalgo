"""Local prototype statistics and reliability-weighted global aggregation."""
import torch


def compute_local_prototypes(features, labels, num_classes):
    if features.ndim != 2 or labels.ndim != 1:
        raise ValueError("features must be [N, d] and labels must be [N]")
    dimension = features.shape[1]
    prototypes = features.new_zeros((num_classes, dimension))
    counts = torch.zeros(num_classes, dtype=torch.long, device=features.device)
    compactness = features.new_zeros(num_classes)
    mask = torch.zeros(num_classes, dtype=torch.bool, device=features.device)
    for label in range(num_classes):
        class_features = features[labels == label]
        if class_features.numel() == 0:
            continue
        prototype = class_features.mean(dim=0)
        prototypes[label] = prototype
        counts[label] = class_features.shape[0]
        compactness[label] = (class_features - prototype).pow(2).sum(dim=1).mean()
        mask[label] = True
    return {"prototypes": prototypes, "counts": counts,
            "compactness": compactness, "mask": mask}


def aggregate_prototypes(client_payloads, old_global_prototypes, beta, rho, old_proto_mask=None):
    """Aggregate only reported classes; absent classes retain their previous state."""
    global_prototypes = old_global_prototypes.detach().clone()
    num_classes = global_prototypes.shape[0]
    proto_mask = (old_proto_mask.detach().clone() if old_proto_mask is not None
                  else torch.zeros(num_classes, dtype=torch.bool, device=global_prototypes.device))
    for label in range(num_classes):
        weighted_sum = torch.zeros_like(global_prototypes[label])
        total_weight = 0.0
        for payload in client_payloads:
            if not bool(payload["mask"][label].item()):
                continue
            count = float(payload["counts"][label].item())
            compactness = float(payload["compactness"][label].item())
            reliability = count / (count + float(beta)) / (1.0 + compactness)
            weighted_sum += payload["prototypes"][label].to(global_prototypes.device) * reliability
            total_weight += reliability
        if total_weight > 0:
            aggregate = weighted_sum / total_weight
            global_prototypes[label] = ((1.0 - float(rho)) * global_prototypes[label]
                                        + float(rho) * aggregate)
            proto_mask[label] = True
    return global_prototypes, proto_mask
