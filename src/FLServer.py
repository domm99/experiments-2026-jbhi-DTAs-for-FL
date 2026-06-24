import copy
import torch
from collections.abc import Iterable
from utils import GlucoseClassifierLSTM
from LearningConfig import LearningConfig

class FLServer:

    def __init__(self, config: LearningConfig):
        self._config = config
        self.clients_models = []
        self._model = GlucoseClassifierLSTM(
            hidden_size=self._config.hidden_size,
            num_layers=self._config.layers,
            dropout=self._config.dropout,
        ).to(self._config.device)

    def aggregate(self):
        """
        Aggregates N weighted model updates following the FedAvg algorithm.
        :return: Nothing
        """
        client_updates = self.clients_models
        if not client_updates:
            print('Skipping global aggregation: no client updates received')
            return

        total_weight = sum(weight for _, weight in client_updates)
        if total_weight <= 0:
            print('Skipping global aggregation: client updates have zero total weight')
            return

        first_model = client_updates[0][0]
        w_avg = copy.deepcopy(first_model)

        for key in w_avg.keys():
            w_avg[key] = torch.mul(w_avg[key], 0.0)
        for key in w_avg.keys():
            for model, weight in client_updates:
                w_avg[key] += model[key] * weight
            w_avg[key] = torch.div(w_avg[key], total_weight)
        self._model.load_state_dict(w_avg)

    def receive_client_update(self, client_updates: Iterable[tuple[dict[str, torch.Tensor], int]]):
        self.clients_models = [
            (model, weight)
            for model, weight in client_updates
            if weight > 0
        ]

    @property
    def model(self):
        return self._model.state_dict()
