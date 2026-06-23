from LearningConfig import LearningConfig

class FLServer:

    def __init__(self, config: LearningConfig):
        self._config = config
        self.clients_models = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = GlucoseClassifierLSTM(
            hidden_size=self._config.hidden_size,
            num_layers=self._config.layers,
            dropout=self._config.dropout,
        ).to(self._config.device)

    def aggregate(self):
        """
        Aggregates N models following the FedAvg algorithm.
        :return: Nothing
        """
        models = self.clients_models
        w_avg = copy.deepcopy(models[0])

        for key in w_avg.keys():
            w_avg[key] = torch.mul(w_avg[key], 0.0)
        for key in w_avg.keys():
            for i in range(0, len(models)):
                w_avg[key] += models[i][key]
            w_avg[key] = torch.div(w_avg[key], len(models))
        self._model.load_state_dict(w_avg)

    def receive_client_update(self, clients_models):
        self.clients_models = clients_models

    @property
    def model(self):
        return self._model.state_dict()