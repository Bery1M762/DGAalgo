import argparse
import glob
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np


def _load_npz_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(file_path, "rb") as f:
        data = np.load(f, allow_pickle=True)["data"].tolist()

    x = np.asarray(data["x"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    return x, y


def load_reference_dataset(dataset_dir: str, use_train: bool = True, use_test: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    split_files = []

    if use_train:
        split_files.extend(sorted(glob.glob(os.path.join(dataset_dir, "train", "*.npz"))))
    if use_test:
        split_files.extend(sorted(glob.glob(os.path.join(dataset_dir, "test", "*.npz"))))

    if not split_files:
        raise FileNotFoundError(f"No npz files found in {dataset_dir}/train or {dataset_dir}/test")

    all_x = []
    all_y = []
    for file_path in split_files:
        x, y = _load_npz_data(file_path)
        all_x.append(x)
        all_y.append(y)

    x_all = np.concatenate(all_x, axis=0).astype(np.float32)
    y_all = np.concatenate(all_y, axis=0).astype(np.int64)
    return x_all, y_all


def _normalize_proportions(labels: np.ndarray, label_proportions: Optional[Dict[int, float]]) -> Dict[int, float]:
    unique_labels = [int(v) for v in labels.tolist()]

    if label_proportions is None:
        return {label: 1.0 / len(unique_labels) for label in unique_labels}

    values = {label: float(max(0.0, label_proportions.get(label, 0.0))) for label in unique_labels}
    total = sum(values.values())
    if total <= 0:
        raise ValueError("label_proportions are all zeros or unmatched with dataset labels")

    return {label: values[label] / total for label in unique_labels}


def _allocate_counts(total_samples: int, proportions: Dict[int, float]) -> Dict[int, int]:
    labels = list(proportions.keys())
    raw = {label: total_samples * proportions[label] for label in labels}
    counts = {label: int(np.floor(raw[label])) for label in labels}

    remainder = total_samples - sum(counts.values())
    if remainder > 0:
        order = sorted(labels, key=lambda k: raw[k] - counts[k], reverse=True)
        for label in order[:remainder]:
            counts[label] += 1

    return counts


def _parse_label_proportions(raw: str) -> Optional[Dict[int, float]]:
    if raw is None or raw.strip() == "":
        return None

    items = raw.split(",")
    proportions: Dict[int, float] = {}
    for item in items:
        if ":" not in item:
            raise ValueError("Invalid --label_proportions format. Example: 0:0.1,1:0.2")
        k, v = item.split(":", 1)
        proportions[int(k.strip())] = float(v.strip())

    return proportions


def generate_distribution_similar_data(
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    total_samples: int,
    label_proportions: Optional[Dict[int, float]] = None,
    noise_sigma: float = 0.10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data with distribution similar to reference dataset.

    Returns:
    - x_flat: shape [N, 784]
    - x_img:  shape [N, 1, 28, 28]
    - y:      shape [N]
    """
    if total_samples <= 0:
        raise ValueError("total_samples must be > 0")
    if noise_sigma < 0:
        raise ValueError("noise_sigma must be >= 0")

    if reference_x.ndim != 4 or reference_x.shape[1:] != (1, 28, 28):
        raise ValueError(f"Expected reference_x shape [N,1,28,28], got {reference_x.shape}")

    rng = np.random.default_rng(random_state)
    labels = np.unique(reference_y)
    proportions = _normalize_proportions(labels, label_proportions)
    counts = _allocate_counts(total_samples, proportions)

    x_flat_parts = []
    y_parts = []

    flat_ref = reference_x.reshape(reference_x.shape[0], -1)

    for label in labels.tolist():
        label = int(label)
        n = counts.get(label, 0)
        if n <= 0:
            continue

        mask = reference_y == label
        class_flat = flat_ref[mask]
        if class_flat.shape[0] == 0:
            continue

        sampled_idx = rng.choice(class_flat.shape[0], size=n, replace=True)
        sampled = class_flat[sampled_idx].copy()

        # Multiplicative noise keeps non-negativity and better preserves local contrast patterns.
        pos_mask = sampled > 0
        if np.any(pos_mask):
            noise = rng.lognormal(mean=0.0, sigma=noise_sigma, size=sampled.shape)
            sampled[pos_mask] = sampled[pos_mask] * noise[pos_mask]

        q_low = np.percentile(class_flat, 1, axis=0)
        q_high = np.percentile(class_flat, 99, axis=0)
        sampled = np.clip(sampled, q_low, q_high)
        sampled = np.clip(sampled, 0.0, 1.0)

        x_flat_parts.append(sampled.astype(np.float32))
        y_parts.append(np.full((n,), label, dtype=np.int64))

    if not x_flat_parts:
        raise RuntimeError("No synthetic samples generated. Check label proportions and reference data.")

    x_flat = np.concatenate(x_flat_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    perm = rng.permutation(x_flat.shape[0])
    x_flat = x_flat[perm]
    y = y[perm]

    x_img = x_flat.reshape(-1, 1, 28, 28).astype(np.float32)
    return x_flat, x_img, y


def evaluate_distribution_similarity(
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    synthetic_x: np.ndarray,
    synthetic_y: np.ndarray,
) -> Dict[str, object]:
    labels = np.unique(reference_y)

    ref_counts = {int(k): int((reference_y == k).sum()) for k in labels.tolist()}
    syn_counts = {int(k): int((synthetic_y == k).sum()) for k in labels.tolist()}

    ref_total = max(int(reference_y.shape[0]), 1)
    syn_total = max(int(synthetic_y.shape[0]), 1)
    ref_probs = {k: ref_counts[k] / ref_total for k in ref_counts}
    syn_probs = {k: syn_counts[k] / syn_total for k in syn_counts}

    tv_distance = 0.5 * sum(abs(ref_probs[k] - syn_probs[k]) for k in ref_probs)

    summary = {
        "reference_total": ref_total,
        "synthetic_total": syn_total,
        "label_counts_reference": ref_counts,
        "label_counts_synthetic": syn_counts,
        "label_probs_reference": ref_probs,
        "label_probs_synthetic": syn_probs,
        "total_variation_distance": float(tv_distance),
        "pixel_mean_reference": float(reference_x.mean()),
        "pixel_mean_synthetic": float(synthetic_x.mean()),
        "pixel_std_reference": float(reference_x.std()),
        "pixel_std_synthetic": float(synthetic_x.std()),
    }
    return summary


def save_synthetic_pool(
    output_path: str,
    x_flat: np.ndarray,
    x_img: np.ndarray,
    y: np.ndarray,
    similarity_report: Dict[str, object],
) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        output_path,
        x_flat=x_flat.astype(np.float32),
        x_img=x_img.astype(np.float32),
        y=y.astype(np.int64),
    )

    report_path = os.path.splitext(output_path)[0] + "_report.json"
    with open(report_path, "w") as f:
        json.dump(similarity_report, f, indent=2)

    print(f"Synthetic pool saved to: {output_path}")
    print(f"Similarity report saved to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DGA-like synthetic samples (flat + 28x28 image)")
    parser.add_argument("--dataset_dir", type=str, default="../dataset/DGA", help="Path to dataset folder with train/test")
    parser.add_argument("--use_train", action="store_true", help="Use train split as reference")
    parser.add_argument("--use_test", action="store_true", help="Use test split as reference")
    parser.add_argument("--total_samples", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--label_proportions", type=str, default="", help="Format: 0:0.1,1:0.2,...")
    parser.add_argument("--noise_sigma", type=float, default=0.10, help="Lognormal noise sigma")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="../results/DGA_synthetic_pool.npz", help="Output npz path")
    args = parser.parse_args()

    use_train = args.use_train or (not args.use_train and not args.use_test)
    use_test = args.use_test

    ref_x, ref_y = load_reference_dataset(args.dataset_dir, use_train=use_train, use_test=use_test)
    label_proportions = _parse_label_proportions(args.label_proportions)

    x_flat, x_img, y_syn = generate_distribution_similar_data(
        reference_x=ref_x,
        reference_y=ref_y,
        total_samples=args.total_samples,
        label_proportions=label_proportions,
        noise_sigma=args.noise_sigma,
        random_state=args.seed,
    )

    report = evaluate_distribution_similarity(
        reference_x=ref_x,
        reference_y=ref_y,
        synthetic_x=x_img,
        synthetic_y=y_syn,
    )

    save_synthetic_pool(args.output, x_flat, x_img, y_syn, report)
    print("Label counts (synthetic):", report["label_counts_synthetic"])
    print("Total variation distance (labels):", round(report["total_variation_distance"], 6))


if __name__ == "__main__":
    main()
