"""Generate PFLlib client files for DGA-RGB data.

Each saved sample retains ``g`` (H2, CH4, C2H6, C2H4, C2H2), because DGAPGM
performs mixup in gas space rather than in the derived RGB representation.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


# Fixed DGAPGM label order. Do not reorder without also changing mechanism.py.
LABEL_MAP = {
    "中温过热": 0, "高温过热": 1, "低能放电": 2, "高能放电": 3,
    "正常": 4, "低温过热": 5, "局部放电": 6,
}
GAS_COLUMNS = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]


def gas_to_dga_rgb(gas, eps=1e-6):
    left, right = gas[:, :, None], gas[:, None, :]
    red = left / (left + right + eps)
    green = np.log1p(np.abs(left - right))
    green /= green.max(axis=(1, 2), keepdims=True) + eps
    blue = left * right / (gas.max(axis=1)[:, None, None] ** 2 + eps)
    return np.stack((red, green, blue), axis=1).astype(np.float32)


def partition_indices(labels, num_clients, partition, alpha, rng):
    if partition == "iid":
        indices = rng.permutation(len(labels))
        return [part.astype(np.int64) for part in np.array_split(indices, num_clients)]
    if partition != "dir":
        raise ValueError("partition must be 'iid' or 'dir'")
    client_indices = [[] for _ in range(num_clients)]
    for label in range(7):
        indices = rng.permutation(np.where(labels == label)[0])
        if not len(indices):
            continue
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        cut_points = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
        for client, piece in enumerate(np.split(indices, cut_points)):
            client_indices[client].extend(piece.tolist())
    return [np.asarray(rng.permutation(part), dtype=np.int64) for part in client_indices]


def save_client_file(path, rgb, labels, gases, indices):
    payload = {"x": rgb[indices], "y": labels[indices], "g": gases[indices]}
    with open(path, "wb") as file:
        np.savez_compressed(file, data=payload)


def generate(args):
    frame = pd.read_excel(args.xlsx)
    required = GAS_COLUMNS + ["故障类型"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("xlsx is missing columns: " + ", ".join(missing))
    labels_as_text = frame["故障类型"].astype(str).str.strip()
    unknown = sorted(set(labels_as_text) - set(LABEL_MAP))
    if unknown:
        raise ValueError("unknown DGA labels: " + ", ".join(unknown))
    gases = frame[GAS_COLUMNS].apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
    if not np.isfinite(gases).all() or (gases < 0).any():
        raise ValueError("gas values must be finite and non-negative")
    labels = labels_as_text.map(LABEL_MAP).to_numpy(np.int64)
    rgb = gas_to_dga_rgb(gases)
    rng = np.random.default_rng(args.seed)
    client_parts = partition_indices(labels, args.num_clients, args.partition, args.alpha, rng)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset_name)
    train_dir, test_dir = os.path.join(output_dir, "train"), os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    statistics = []
    for client_id, indices in enumerate(client_parts):
        if len(indices) < 2:
            raise ValueError("client {} has fewer than two samples; reduce --num_clients or change partition".format(client_id))
        split_at = min(max(1, int(round(len(indices) * args.train_ratio))), len(indices) - 1)
        train_indices, test_indices = indices[:split_at], indices[split_at:]
        save_client_file(os.path.join(train_dir, "{}.npz".format(client_id)), rgb, labels, gases, train_indices)
        save_client_file(os.path.join(test_dir, "{}.npz".format(client_id)), rgb, labels, gases, test_indices)
        statistics.append({int(label): int((labels[indices] == label).sum()) for label in np.unique(labels[indices])})
    config = {"num_clients": args.num_clients, "num_classes": 7,
              "non_iid": args.partition == "dir", "balance": args.partition == "iid",
              "partition": args.partition, "alpha": args.alpha,
              "Size of samples for labels in clients": statistics}
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    print("Saved DGA-RGB dataset to {}".format(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True)
    parser.add_argument("--dataset_name", default="DGA_RGB")
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--partition", choices=("iid", "dir"), default="iid")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    if not 0 < arguments.train_ratio < 1 or arguments.alpha <= 0:
        parser.error("--train_ratio must be in (0, 1) and --alpha must be positive")
    generate(arguments)
