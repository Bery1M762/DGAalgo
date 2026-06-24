import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader

from flcore.clients.clientbase import Client
from flcore.dgapgm.losses import MechanismPrototypeContrastiveLoss
from flcore.dgapgm.mixup import PrototypeGuidedGasMixup
from flcore.dgapgm.prototype import compute_local_prototypes
from utils.data_utils import read_client_data


class clientDGAPGM(Client):
    """Local-only DGAPGM learner. Model parameters never leave this client."""
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.lambda_mpc = args.lambda_mpc
        self.lambda_pgm = args.lambda_pgm
        self.pgm_mu = args.pgm_mu
        self.mixup_alpha = args.mixup_alpha
        self.use_mpc = args.use_mpc
        self.use_pgm = args.use_pgm
        self.global_prototypes = None
        self.proto_mask = torch.zeros(self.num_classes, dtype=torch.bool)
        self.margin_matrix = None
        self.mixup_adjacency = None
        self.mpc_loss = MechanismPrototypeContrastiveLoss(args.contrast_tau)
        self.mixup = PrototypeGuidedGasMixup(self.num_classes, args.pgm_minority_gamma)
        self.prototype_payload = None

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        return batch[0], batch[1], None

    def load_train_data(self, batch_size=None):
        """Keep final small DGA batches; prototype statistics must see every real sample."""
        batch_size = batch_size or self.batch_size
        data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        return DataLoader(data, batch_size=batch_size, drop_last=False, shuffle=True)

    def _forward(self, x):
        output = self.model(x)
        if isinstance(output, tuple):
            return output
        if hasattr(self.model, "extract_features"):
            return output, self.model.extract_features(x)
        raise TypeError("DGAPGM requires a model returning (logits, features) or extract_features(x)")

    def set_dgapgm_state(self, global_prototypes, proto_mask, margin_matrix, mixup_adjacency):
        self.global_prototypes = global_prototypes.detach().clone().to(self.device)
        self.proto_mask = proto_mask.detach().clone().to(self.device)
        self.margin_matrix = margin_matrix.detach().clone().to(self.device)
        self.mixup_adjacency = mixup_adjacency.detach().clone().to(self.device)

    def train(self):
        start_time = time.time()
        self.model.train()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = max(1, np.random.randint(1, max_local_epochs // 2 + 1))
        for _ in range(max_local_epochs):
            for batch in self.load_train_data():
                x, y, gas = self._unpack_batch(batch)
                x, y = x.to(self.device), y.to(self.device)
                gas = gas.to(self.device) if gas is not None else None
                logits, features = self._forward(x)
                loss = self.loss(logits, y)
                if self.use_mpc and self.global_prototypes is not None and self.proto_mask.any():
                    loss = loss + self.lambda_mpc * self.mpc_loss(
                        features, y, self.global_prototypes, self.margin_matrix, self.proto_mask)
                if self.use_pgm and gas is not None and self.global_prototypes is not None:
                    generated = self.mixup.generate(
                        gas, y, self.global_prototypes, self.mixup_adjacency, self.mixup_alpha)
                    if generated is not None:
                        mixed_x, soft_targets, proto_targets = generated
                        mixed_logits, mixed_features = self._forward(mixed_x.to(self.device))
                        pgm_loss = -(soft_targets.to(self.device) * F.log_softmax(mixed_logits, dim=1)).sum(1).mean()
                        if proto_targets is not None:
                            pgm_loss = pgm_loss + self.pgm_mu * F.mse_loss(
                                mixed_features, proto_targets.to(self.device))
                        loss = loss + self.lambda_pgm * pgm_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.prototype_payload = self.collect_prototype_payload()
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def collect_prototype_payload(self):
        features, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in self.load_train_data(batch_size=self.batch_size):
                x, y, _ = self._unpack_batch(batch)
                _, embedding = self._forward(x.to(self.device))
                features.append(embedding.detach())
                labels.append(y.to(self.device))
        if not features:
            return compute_local_prototypes(torch.empty((0, self.model.feature_dim), device=self.device),
                                            torch.empty(0, dtype=torch.long, device=self.device), self.num_classes)
        return compute_local_prototypes(torch.cat(features), torch.cat(labels), self.num_classes)

    def test_metrics(self):
        self.model.eval()
        correct = total = 0
        probabilities, targets = [], []
        with torch.no_grad():
            for batch in self.load_test_data():
                x, y, _ = self._unpack_batch(batch)
                logits, _ = self._forward(x.to(self.device))
                y = y.to(self.device)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.numel()
                probabilities.append(logits.cpu())
                targets.append(y.cpu())
        auc = 0.0
        if total and probabilities:
            try:
                auc = metrics.roc_auc_score(torch.cat(targets).numpy(), torch.cat(probabilities).numpy(),
                                            multi_class="ovr", average="micro")
            except ValueError:
                pass
        return correct, total, auc

    def train_metrics(self):
        self.model.eval()
        total = 0
        losses = 0.0
        with torch.no_grad():
            for batch in self.load_train_data():
                x, y, _ = self._unpack_batch(batch)
                logits, _ = self._forward(x.to(self.device))
                y = y.to(self.device)
                losses += self.loss(logits, y).item() * y.numel()
                total += y.numel()
        # Server.train_metrics() follows PFLlib's two-value client contract.
        return losses, total
