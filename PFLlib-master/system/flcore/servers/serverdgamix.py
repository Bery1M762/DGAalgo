import time

from flcore.clients.clientdgamix import clientDGAPGM
from flcore.dgapgm.mechanism import (build_dga_distance_matrix, build_margin_matrix,
                                     build_mixup_adjacency)
from flcore.dgapgm.prototype import aggregate_prototypes
from flcore.servers.serverbase import Server


class DGAPGM(Server):
    """Server that communicates prototypes only; it never broadcasts model weights."""
    def __init__(self, args, times):
        super().__init__(args, times)
        if args.num_classes != 7:
            raise ValueError("DGAPGM requires the seven DGA classes in the documented label order")
        self.set_slow_clients()
        self.set_clients(clientDGAPGM)
        self.global_prototypes = self.global_model.classifier.weight.detach().new_zeros(
            (args.num_classes, args.feature_dim))
        self.proto_mask = self.global_prototypes.new_zeros(args.num_classes, dtype=bool)
        self.distance_matrix = build_dga_distance_matrix(args.num_classes).to(self.device)
        self.margin_matrix = build_margin_matrix(self.distance_matrix, args.margin_eta).to(self.device)
        self.mixup_adjacency = build_mixup_adjacency(args.num_classes).to(self.device)
        self.Budget = []
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating DGAPGM server and clients.")

    def send_dgapgm_state(self):
        for client in self.selected_clients:
            start_time = time.time()
            client.set_dgapgm_state(self.global_prototypes, self.proto_mask,
                                    self.margin_matrix, self.mixup_adjacency)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_prototypes(self):
        payloads = []
        for client in self.selected_clients:
            if client.prototype_payload is not None:
                payloads.append({key: value.detach().cpu() for key, value in client.prototype_payload.items()})
        return payloads

    def train(self):
        for round_index in range(self.global_rounds + 1):
            start_time = time.time()
            self.selected_clients = self.select_clients()
            if round_index % self.eval_gap == 0:
                print(f"\n-------------Round number: {round_index}-------------")
                print("\nEvaluate local DGAPGM models")
                self.evaluate()
            self.send_dgapgm_state()
            for client in self.selected_clients:
                client.train()
            payloads = self.receive_prototypes()
            self.global_prototypes, self.proto_mask = aggregate_prototypes(
                payloads, self.global_prototypes, self.args.proto_beta, self.args.proto_ema, self.proto_mask)
            self.Budget.append(time.time() - start_time)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
            if self.auto_break and self.check_done([self.rs_test_acc], top_cnt=self.top_cnt):
                break
        print("\nBest accuracy.")
        print(max(self.rs_test_acc) if self.rs_test_acc else 0.0)
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()
