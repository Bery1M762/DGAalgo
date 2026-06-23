import numpy as np
import torch
from torch.utils.data import DataLoader

from flcore.clients.clientavg import clientAVG
from utils.data_utils import read_client_data


class clientLKM(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self._label_count_vector = None
        self.real_train_samples = train_samples
        self.synthetic_train_data = []

    def get_label_count_vector(self):
        if self._label_count_vector is None:
            train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
            counts = np.zeros(self.num_classes, dtype=np.int64)
            for _, y in train_data:
                label = int(y.item()) if hasattr(y, "item") else int(y)
                if 0 <= label < self.num_classes:
                    counts[label] += 1
            self._label_count_vector = counts

        return self._label_count_vector.copy()

    def get_label_probability_vector(self):
        counts = self.get_label_count_vector().astype(np.float64)
        total = counts.sum()
        if total <= 0:
            return np.ones(self.num_classes, dtype=np.float64) / max(self.num_classes, 1)
        return counts / total

    def set_synthetic_train_data(self, x_img, y):
        self.synthetic_train_data = []

        if x_img is None or y is None:
            self.train_samples = self.real_train_samples
            return

        x_img = np.asarray(x_img, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if x_img.ndim != 4:
            raise ValueError(f"Synthetic x must be 4D [N,C,H,W], got {x_img.shape}")
        if y.ndim != 1 or x_img.shape[0] != y.shape[0]:
            raise ValueError(f"Synthetic y shape mismatch, x={x_img.shape}, y={y.shape}")

        for i in range(y.shape[0]):
            x_tensor = torch.tensor(x_img[i], dtype=torch.float32)
            y_tensor = torch.tensor(y[i], dtype=torch.int64)
            self.synthetic_train_data.append((x_tensor, y_tensor))

        self.train_samples = self.real_train_samples + len(self.synthetic_train_data)

    def get_synthetic_label_count_vector(self):
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for _, y in self.synthetic_train_data:
            label = int(y.item()) if hasattr(y, "item") else int(y)
            if 0 <= label < self.num_classes:
                counts[label] += 1
        return counts

    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        if len(self.synthetic_train_data) > 0:
            train_data = train_data + self.synthetic_train_data

        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
