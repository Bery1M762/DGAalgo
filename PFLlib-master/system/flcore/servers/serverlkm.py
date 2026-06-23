import json
import os
import time
import glob

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from flcore.clients.clientlkm import clientLKM
from flcore.servers.serverbase import Server


class FedLKM(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientLKM)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.label_cluster_result = {}

    def _load_reference_train_pool(self):
        train_dir = os.path.join("..", "dataset", self.dataset, "train")
        npz_files = sorted(glob.glob(os.path.join(train_dir, "*.npz")))
        if len(npz_files) == 0:
            raise FileNotFoundError(f"No train npz files found in {train_dir}")

        all_x = []
        all_y = []
        for file_path in npz_files:
            with open(file_path, "rb") as f:
                data = np.load(f, allow_pickle=True)["data"].tolist()
            all_x.append(np.asarray(data["x"], dtype=np.float32))
            all_y.append(np.asarray(data["y"], dtype=np.int64))

        ref_x = np.concatenate(all_x, axis=0).astype(np.float32)
        ref_y = np.concatenate(all_y, axis=0).astype(np.int64)

        if ref_x.ndim != 4:
            raise ValueError(f"Expected reference data to be 4D [N,C,H,W], got shape {ref_x.shape}")

        self.data_shape = ref_x.shape[1:]  # e.g. (1, 28, 28) or (5, 3, 3)
        return ref_x, ref_y

    def _generate_synthetic_pool_by_class(self, reference_x, reference_y, class_total_need):
        noise_sigma = float(getattr(self.args, "lkm_noise_sigma", 0.10))
        if noise_sigma < 0:
            raise ValueError("lkm_noise_sigma must be >= 0")

        seed = int(getattr(self.args, "lkm_seed", 42))
        rng = np.random.default_rng(seed)
        flat_ref = reference_x.reshape(reference_x.shape[0], -1)

        pool_by_class = {}
        for class_id in range(self.num_classes):
            need = int(class_total_need[class_id])
            if need <= 0:
                pool_by_class[class_id] = np.zeros((0, 1, 28, 28), dtype=np.float32)
                continue

            class_mask = reference_y == class_id
            class_flat = flat_ref[class_mask]
            if class_flat.shape[0] == 0:
                raise ValueError(f"Cannot synthesize class {class_id}: no reference samples found")

            sampled_idx = rng.choice(class_flat.shape[0], size=need, replace=True)
            sampled = class_flat[sampled_idx].copy()

            pos_mask = sampled > 0
            if np.any(pos_mask):
                noise = rng.lognormal(mean=0.0, sigma=noise_sigma, size=sampled.shape)
                sampled[pos_mask] = sampled[pos_mask] * noise[pos_mask]

            q_low = np.percentile(class_flat, 1, axis=0)
            q_high = np.percentile(class_flat, 99, axis=0)
            sampled = np.clip(sampled, q_low, q_high)
            sampled = np.clip(sampled, 0.0, 1.0)

            pool_by_class[class_id] = sampled.reshape(-1, *self.data_shape).astype(np.float32)

        return pool_by_class

    def _dispatch_synthetic_data_to_clients(self, generation_quota, pool_by_class):
        n_clients, n_classes = generation_quota.shape
        cursor = {c: 0 for c in range(n_classes)}
        assigned = np.zeros((n_clients, n_classes), dtype=np.int64)
        assigned_totals = np.zeros(n_clients, dtype=np.int64)

        for client_id, client in enumerate(self.clients):
            x_parts = []
            y_parts = []

            for class_id in range(n_classes):
                need = int(generation_quota[client_id, class_id])
                if need <= 0:
                    continue

                class_pool = pool_by_class[class_id]
                start = cursor[class_id]
                end = start + need
                if end > class_pool.shape[0]:
                    raise ValueError(
                        f"Synthetic pool exhausted for class {class_id}: need {need}, available {class_pool.shape[0] - start}"
                    )

                x_slice = class_pool[start:end]
                y_slice = np.full((need,), class_id, dtype=np.int64)
                x_parts.append(x_slice)
                y_parts.append(y_slice)

                cursor[class_id] = end
                assigned[client_id, class_id] = need

            if len(x_parts) > 0:
                x_client = np.concatenate(x_parts, axis=0)
                y_client = np.concatenate(y_parts, axis=0)
                perm = np.random.permutation(x_client.shape[0])
                x_client = x_client[perm]
                y_client = y_client[perm]
            else:
                x_client = np.zeros((0, 1, 28, 28), dtype=np.float32)
                y_client = np.zeros((0,), dtype=np.int64)

            client.set_synthetic_train_data(x_client, y_client)
            assigned_totals[client_id] = int(y_client.shape[0])

        for class_id in range(n_classes):
            if cursor[class_id] != pool_by_class[class_id].shape[0]:
                raise ValueError(
                    f"Synthetic class {class_id} dispatch mismatch: dispatched={cursor[class_id]}, generated={pool_by_class[class_id].shape[0]}"
                )

        return assigned, assigned_totals

    def _build_classwise_allocation_plan(self, count_matrix, cluster_labels):
        n_clients = count_matrix.shape[0]
        n_classes = count_matrix.shape[1]

        generation_quota = np.zeros((n_clients, n_classes), dtype=np.int64)
        target_matrix = np.zeros((n_clients, n_classes), dtype=np.int64)
        cluster_waterlines = {}

        unique_clusters = sorted(np.unique(cluster_labels).tolist())
        for cluster_id in unique_clusters:
            member_ids = np.where(cluster_labels == cluster_id)[0]
            member_counts = count_matrix[member_ids]

            # Class-wise waterline: sum over all real samples in the same cluster for each class.
            waterline = np.sum(member_counts, axis=0).astype(np.int64)
            cluster_waterlines[int(cluster_id)] = waterline

            # Per-class quota for client i: G_{i,c} = W_c - S_{i,c}
            quota = waterline[None, :] - member_counts
            generation_quota[member_ids] = quota
            target_matrix[member_ids] = member_counts + quota

        return generation_quota, target_matrix, cluster_waterlines

    def _collect_label_statistics(self):
        count_matrix = []
        prob_matrix = []

        for client in self.clients:
            counts = client.get_label_count_vector()
            probs = counts.astype(np.float64)
            total = probs.sum()
            if total <= 0:
                probs = np.ones(self.num_classes, dtype=np.float64) / max(self.num_classes, 1)
            else:
                probs = probs / total

            count_matrix.append(counts)
            prob_matrix.append(probs)

        return np.asarray(count_matrix), np.asarray(prob_matrix)

    def _select_cluster_count(self, prob_matrix):
        n_clients = prob_matrix.shape[0]
        if n_clients <= 1:
            return 1

        forced_k = getattr(self.args, "label_cluster_k", 0)
        if forced_k and forced_k > 1:
            return min(forced_k, n_clients)

        if n_clients < 3:
            return n_clients

        max_k = getattr(self.args, "label_cluster_max_k", 8)
        max_k = min(max_k, n_clients - 1)

        best_k = 2
        best_score = -1.0

        for k in range(2, max_k + 1):
            try:
                model = KMeans(n_clusters=k, random_state=0, n_init=20)
                labels = model.fit_predict(prob_matrix)
                if len(np.unique(labels)) < 2:
                    continue
                score = silhouette_score(prob_matrix, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue

        if best_score < 0:
            return 2

        return best_k

    def _save_label_cluster_result(self):
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        file_name = f"{self.dataset}_{self.algorithm}_{self.goal}_{self.times}_label_clusters.json"
        file_path = os.path.join(result_path, file_name)

        with open(file_path, "w") as f:
            json.dump(self.label_cluster_result, f, indent=2)

        print(f"Label-cluster summary saved to: {file_path}")

    def _save_cluster_figure(self, prob_matrix, cluster_labels, cluster_centers):
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"Skip cluster figure export: matplotlib unavailable ({e})")
            return

        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        file_name = f"{self.dataset}_{self.algorithm}_{self.goal}_{self.times}_label_clusters.png"
        file_path = os.path.join(result_path, file_name)

        n_clients = prob_matrix.shape[0]
        cluster_ids = np.asarray(cluster_labels)
        sort_idx = np.lexsort((np.arange(n_clients), cluster_ids))
        sorted_probs = prob_matrix[sort_idx]

        # Use PCA projection only for visualization while clustering itself stays in original space.
        if n_clients < 2:
            points_2d = np.zeros((n_clients, 2), dtype=np.float64)
            centers_2d = np.zeros((cluster_centers.shape[0], 2), dtype=np.float64)
        else:
            n_comp = min(2, prob_matrix.shape[0], prob_matrix.shape[1])
            pca = PCA(n_components=n_comp, random_state=0)
            points_proj = pca.fit_transform(prob_matrix)
            centers_proj = pca.transform(cluster_centers)

            if n_comp == 1:
                points_2d = np.concatenate([points_proj, np.zeros((points_proj.shape[0], 1))], axis=1)
                centers_2d = np.concatenate([centers_proj, np.zeros((centers_proj.shape[0], 1))], axis=1)
            else:
                points_2d = points_proj
                centers_2d = centers_proj

        unique_clusters = np.unique(cluster_ids)
        cmap = plt.cm.get_cmap("tab10", max(len(unique_clusters), 1))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

        heat = axes[0].imshow(sorted_probs, aspect="auto", cmap="viridis")
        axes[0].set_title("Client Label Probability Heatmap")
        axes[0].set_xlabel("Label")
        axes[0].set_ylabel("Client (sorted by cluster)")
        axes[0].set_yticks(np.arange(n_clients))
        axes[0].set_yticklabels([str(int(i)) for i in sort_idx])
        plt.colorbar(heat, ax=axes[0], fraction=0.046, pad=0.04)

        for c in unique_clusters:
            mask = cluster_ids == c
            axes[1].scatter(
                points_2d[mask, 0],
                points_2d[mask, 1],
                s=40,
                alpha=0.85,
                color=cmap(int(c) % max(len(unique_clusters), 1)),
                label=f"Cluster {int(c)}",
            )

        axes[1].scatter(
            centers_2d[:, 0],
            centers_2d[:, 1],
            marker="X",
            s=140,
            c="black",
            label="Centroids",
        )
        axes[1].set_title("Client Clusters (PCA Projection)")
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        axes[1].legend(loc="best", fontsize=8)
        axes[1].grid(alpha=0.2)

        fig.suptitle(f"{self.dataset} Label-Distribution Clustering", fontsize=12)
        fig.tight_layout()
        fig.savefig(file_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Label-cluster figure saved to: {file_path}")

    def cluster_clients_by_label_distribution(self):
        count_matrix, prob_matrix = self._collect_label_statistics()
        n_clients = len(self.clients)
        cluster_k = self._select_cluster_count(prob_matrix)

        if cluster_k == 1:
            cluster_labels = np.zeros(n_clients, dtype=np.int64)
            cluster_centers = np.mean(prob_matrix, axis=0, keepdims=True)
        else:
            kmeans = KMeans(n_clusters=cluster_k, random_state=0, n_init=20)
            cluster_labels = kmeans.fit_predict(prob_matrix)
            cluster_centers = kmeans.cluster_centers_

        clusters = {}
        for cid, label in enumerate(cluster_labels):
            cluster_id = int(label)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(cid)

        print("\n-------------Round number: 0-------------")
        print("Uploading label count vectors and clustering clients ...")
        for cid in range(n_clients):
            count_vec = count_matrix[cid].tolist()
            prob_vec = [round(float(x), 6) for x in prob_matrix[cid].tolist()]
            print(f"Client {cid} label counts: {count_vec}")
            print(f"Client {cid} label probs:  {prob_vec}")

        print(f"\nSelected cluster count: {cluster_k}")
        for cluster_id in sorted(clusters.keys()):
            print(f"Cluster {cluster_id}: clients {clusters[cluster_id]}")

        generation_quota, target_matrix, cluster_waterlines = self._build_classwise_allocation_plan(
            count_matrix=count_matrix,
            cluster_labels=cluster_labels,
        )

        class_total_need = np.sum(generation_quota, axis=0).astype(np.int64)
        reference_x, reference_y = self._load_reference_train_pool()
        pool_by_class = self._generate_synthetic_pool_by_class(
            reference_x=reference_x,
            reference_y=reference_y,
            class_total_need=class_total_need,
        )
        assigned_matrix, assigned_totals = self._dispatch_synthetic_data_to_clients(
            generation_quota=generation_quota,
            pool_by_class=pool_by_class,
        )

        print("\nClass-wise synthetic allocation plan (per client):")
        for cid in range(n_clients):
            print(f"Client {cid} generated quota G_i: {generation_quota[cid].astype(int).tolist()}")
            print(f"Client {cid} received synthetic: {int(assigned_totals[cid])}")

        self.label_cluster_result = {
            "dataset": self.dataset,
            "algorithm": self.algorithm,
            "num_clients": n_clients,
            "num_classes": self.num_classes,
            "cluster_k": int(cluster_k),
            "client_label_counts": [row.astype(int).tolist() for row in count_matrix],
            "client_label_probs": [[float(x) for x in row.tolist()] for row in prob_matrix],
            "cluster_labels": [int(x) for x in cluster_labels.tolist()],
            "clusters": {str(k): v for k, v in clusters.items()},
            "cluster_centers": [[float(x) for x in center.tolist()] for center in cluster_centers],
            "allocation_rule": "G_{i,c} = W_c - S_{i,c}, W_c is class-wise cluster waterline",
            "cluster_class_waterlines": {
                str(k): cluster_waterlines[k].astype(int).tolist() for k in cluster_waterlines
            },
            "client_generated_quota": [row.astype(int).tolist() for row in generation_quota],
            "client_target_after_allocation": [row.astype(int).tolist() for row in target_matrix],
            "synthetic_generation_method": "class-conditional resampling + multiplicative noise + percentile clipping",
            "synthetic_total_need_per_class": class_total_need.astype(int).tolist(),
            "synthetic_assigned_per_client_class": [row.astype(int).tolist() for row in assigned_matrix],
            "synthetic_assigned_per_client_total": assigned_totals.astype(int).tolist(),
            "client_real_train_samples": [int(client.real_train_samples) for client in self.clients],
            "client_total_train_samples_after_injection": [int(client.train_samples) for client in self.clients],
        }
        self._save_label_cluster_result()
        self._save_cluster_figure(prob_matrix, cluster_labels, cluster_centers)

    def train(self):
        self.cluster_clients_by_label_distribution()

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientLKM)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
