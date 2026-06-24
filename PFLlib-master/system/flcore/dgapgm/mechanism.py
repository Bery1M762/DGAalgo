"""DGA fault-mechanism priors used by DGAPGM.

Class order is fixed: medium overheating=0, high overheating=1,
low-energy discharge=2, high-energy discharge=3, normal=4,
low-temperature overheating=5, partial discharge=6.
"""
import torch


def build_dga_distance_matrix(num_classes=7):
    if num_classes != 7:
        raise ValueError("DGAPGM uses the fixed seven-class DGA label order")
    distance = torch.full((7, 7), 3.0)
    distance.fill_diagonal_(0.0)
    # Normal versus any fault.
    distance[4, :] = 4.0
    distance[:, 4] = 4.0
    distance[4, 4] = 0.0
    # Ordered fault families: low -> medium -> high.
    for left, right, value in ((5, 0, 1.0), (0, 1, 1.0), (5, 1, 2.0),
                               (6, 2, 1.0), (2, 3, 1.0), (6, 3, 2.0)):
        distance[left, right] = value
        distance[right, left] = value
    return distance


def build_margin_matrix(distance_matrix, eta):
    return distance_matrix.float() * float(eta)


def build_mixup_adjacency(num_classes=7):
    if num_classes != 7:
        raise ValueError("DGAPGM uses the fixed seven-class DGA label order")
    adjacency = torch.eye(7, dtype=torch.bool)
    for left, right in ((5, 0), (0, 1), (6, 2), (2, 3)):
        adjacency[left, right] = True
        adjacency[right, left] = True
    return adjacency
