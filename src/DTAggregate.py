import torch
import pandas as pd
from dataclasses import dataclass
from DT import DT
from LearningConfig import LearningConfig
from utils import (
    CLASS_NAMES,
    GlucoseClassifierLSTM,
    classification_metrics_from_confusion_matrix,
    compute_class_weights,
    compute_train_stats,
    cross_entropy_batch,
    normalize_series,
    create_train_val_loaders,
    evaluate,
    update_confusion_matrix,
)


_NAN_VAL_METRICS = {
    'loss': float('nan'),
    'accuracy': float('nan'),
    'precision': float('nan'),
    'recall': float('nan'),
    'f1_score': float('nan'),
    **{f'precision_{name}': float('nan') for name in CLASS_NAMES},
    **{f'recall_{name}': float('nan') for name in CLASS_NAMES},
}


@dataclass
class DTATrainingResult:
    model: dict[str, torch.Tensor]
    sample_count: int


def _copy_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in state.items()}


class DTAggregate:

    def __init__(self, mid, config: LearningConfig, experiment: str, initial_model, seed: int):
        self._mid = mid
        self._device = config.device
        self._config = config
        self._dts_data = {}
        self._seed = seed
        self._active_dts = {}
        # Every DT that has ever been registered: the local data set of the DTA
        # (Definition 1) also contains the data of patients that are no longer
        # monitored.
        self._registered_dts = {}
        self._last_mean = 0.0
        self._last_std = 1.0
        self._has_statistics = False
        self._experiment = experiment
        # _global_model is the last model received from the FL server (or the
        # local model when learning only locally); _deployed_model is the model
        # deployed to the DTs of this DTA, which differs from the global one
        # when personalization is enabled.
        self._global_model = _copy_state(initial_model)
        self._deployed_model = self._global_model
        # True once a trained model is available (received from the FL server or
        # trained locally), i.e. something that can be deployed to the DTs.
        self._has_model = False

    def _training_source_dts(self) -> dict[str, DT]:
        if self._config.retain_inactive_patient_data:
            return self._registered_dts
        return self._active_dts

    def update_data_from_dts(self, current_time: pd.Timestamp) -> None:
        self._dts_data = {}
        for dt_id, dt in self._training_source_dts().items():
            try:
                patient_series = dt.get_data(current_time)
                if patient_series is None:
                    print(f'Skipping DT {dt_id} during training: not enough history yet')
                    continue
                self._dts_data[dt_id] = patient_series
            except Exception as exc:
                print(f'Skipping DT {dt_id} during training: {exc}')

    def register_active_dt(self, local_dt: DT, patient_id: str) -> None:
        self._active_dts[patient_id] = local_dt
        self._registered_dts[patient_id] = local_dt

    def unregister_active_dt(self, patient_id: str) -> None:
        self._active_dts.pop(patient_id, None)

    @property
    def active_dts(self) -> list[DT]:
        return list(self._active_dts.values())

    @property
    def trainable_dt_count(self) -> int:
        return sum(1 for series in self._dts_data.values() if series is not None)

    @property
    def model(self) -> dict[str, torch.Tensor]:
        """Model deployed to the DTs of this DTA."""
        return self._deployed_model

    @property
    def dta_id(self) -> str:
        return self._mid

    @property
    def global_model(self) -> dict[str, torch.Tensor]:
        return self._global_model

    @global_model.setter
    def global_model(self, value: dict[str, torch.Tensor]) -> None:
        """Receives a new shared model; it is also the deployed one until personalized."""
        self._global_model = _copy_state(value)
        self._deployed_model = self._global_model
        self._has_model = True

    @property
    def deployed_model(self) -> dict[str, torch.Tensor]:
        return self._deployed_model

    @deployed_model.setter
    def deployed_model(self, value: dict[str, torch.Tensor]) -> None:
        self._deployed_model = _copy_state(value)
        self._has_model = True

    @property
    def has_model(self) -> bool:
        return self._has_model

    @property
    def statistics(self) -> tuple[float, float]:
        return self._last_mean, self._last_std

    @property
    def has_statistics(self) -> bool:
        return self._has_statistics

    def _statistics_source_dts(self) -> dict[str, DT]:
        return self._training_source_dts()

    def refresh_statistics(self, current_time: pd.Timestamp) -> bool:
        """
        Makes sure that local normalization statistics are available. A DTA that
        did not take part in a training (e.g. no data at that time) computes them
        from the data currently available from its DTs, so that it can still
        deploy the shared model. Returns False if no DT has enough data yet.
        """
        if self._has_statistics:
            return True
        available_series = []
        for dt in self._statistics_source_dts().values():
            try:
                series = dt.get_data(current_time)
            except Exception as exc:
                print(f'Skipping DT {dt} for normalization statistics: {exc}')
                continue
            if series is not None:
                available_series.append(series)
        if not available_series:
            return False
        self._last_mean, self._last_std = compute_train_stats(available_series)
        self._has_statistics = True
        return True

    def deploy_to(self, dt: DT, current_time: pd.Timestamp) -> bool:
        """Deploys the current model to a DT, if a model and statistics are available."""
        if not self._has_model:
            return False
        if not self.refresh_statistics(current_time):
            return False
        dt.model = (self._deployed_model, self._last_mean, self._last_std, current_time)
        return True

    def notify_new_model(self, current_time: pd.Timestamp) -> None:
        deployed = sum(self.deploy_to(dt, current_time) for dt in self._active_dts.values())
        if deployed < len(self._active_dts):
            print(
                f'DTA {self._mid}: model deployed to {deployed}/{len(self._active_dts)} DTs, '
                'the others will receive it as soon as local data is available'
            )

    def train(
        self,
        current_time: pd.Timestamp,
        log_label: str,
        start_from: str,
        epochs: int,
        mu: float,
    ) -> DTATrainingResult | None:
        """
        Trains a copy of the global ('global') or of the deployed ('deployed')
        model on the local data set for the given number of epochs, with a
        FedProx proximal term of coefficient mu towards the starting model.
        The trained weights are returned; the caller decides whether they
        become the global or the deployed model of the DTA.
        """
        if start_from == 'global':
            start_state = self._global_model
        elif start_from == 'deployed':
            start_state = self._deployed_model
        else:
            raise ValueError(f'Unsupported start_from: {start_from}')

        patients_series_raw = [series for series in self._dts_data.values() if series is not None]
        if not patients_series_raw:
            print('Skipping training: no DT has enough history for training windows yet')
            return None

        model = GlucoseClassifierLSTM(
            hidden_size=self._config.hidden_size,
            num_layers=self._config.layers,
            dropout=self._config.dropout,
        ).to(self._config.device)
        model.load_state_dict(start_state)

        anchor_weights = [
            param.detach().clone()
            for param in model.parameters()
        ]

        optimizer = torch.optim.Adam(model.parameters(), lr=self._config.learning_rate)
        history: list[dict[str, float]] = []
        mean, std = compute_train_stats(patients_series_raw)
        normalized_series = normalize_series(patients_series_raw, mean, std)
        train_loader, val_loader = create_train_val_loaders(
            patient_series = normalized_series,
            sequence_length = self._config.sequence_length,
            stride = self._config.stride,
            batch_size = self._config.batch_size,
        )
        if len(train_loader.dataset) == 0:
            print(f'Skipping training for DTA {self._mid}: no observable training windows yet')
            return None
        class_weights, class_counts = compute_class_weights(
            patient_series=normalized_series,
            sequence_length=self._config.sequence_length,
            stride=self._config.stride,
            split="train",
        )
        sample_count = int(class_counts.sum().item())
        loss_weights = class_weights.to(self._device)
        print(
            "Training class balance | "
            + " | ".join(
                f"{name}={int(count)}"
                for name, count in zip(CLASS_NAMES, class_counts.tolist(), strict=False)
            )
            + " | "
            + " | ".join(
                f"{name}_weight={weight:.4f}"
                for name, weight in zip(CLASS_NAMES, class_weights.tolist(), strict=False)
            )
        )

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_loss_normalization = 0.0
            train_confusion_matrix = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
            for x, y in train_loader:
                x = x.to(self._device)
                y = y.to(self._device)

                optimizer.zero_grad()
                logits = model(x)
                loss, loss_sum, loss_normalization = cross_entropy_batch(logits, y, loss_weights)
                if mu > 0:
                    loss = loss + self.proximal_term(model, anchor_weights, mu)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss_sum
                train_loss_normalization += loss_normalization
                predictions = logits.argmax(dim=1)
                update_confusion_matrix(train_confusion_matrix, y, predictions)

            if len(val_loader.dataset) > 0:
                val_metrics = evaluate(model, val_loader, self._device, class_weights=class_weights)
            else:
                val_metrics = _NAN_VAL_METRICS
            train_metrics = classification_metrics_from_confusion_matrix(train_confusion_matrix)

            epoch_log = {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_loss_normalization, 1.0),
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1_score": train_metrics["f1_score"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1_score": val_metrics["f1_score"],
            }
            for class_name in CLASS_NAMES:
                epoch_log[f"train_precision_{class_name}"] = train_metrics[f"precision_{class_name}"]
                epoch_log[f"train_recall_{class_name}"] = train_metrics[f"recall_{class_name}"]
                epoch_log[f"val_precision_{class_name}"] = val_metrics[f"precision_{class_name}"]
                epoch_log[f"val_recall_{class_name}"] = val_metrics[f"recall_{class_name}"]
            history.append(epoch_log)
            print(
                f"Epoch {epoch:02d} | "
                f"train_acc={epoch_log['train_accuracy']:.4f} | "
                f"val_acc={epoch_log['val_accuracy']:.4f} | "
                f"val_f1={epoch_log['val_f1_score']:.4f} | "
                f"val_loss={epoch_log['val_loss']:.4f}"
            )

        metrics_df = pd.DataFrame(history)
        metrics_df.to_csv(f'{self._config.data_export_path}/{self._experiment}/training_{current_time}-hospital_{self._mid}-{log_label}-seed_{self._seed}.csv', index=False)
        self._last_mean = mean
        self._last_std = std
        self._has_statistics = True
        return DTATrainingResult(model=_copy_state(model.state_dict()), sample_count=sample_count)

    @staticmethod
    def proximal_term(trained_model, anchor_weights, mu: float):
        prox_term = 0.0
        for p_i, param in enumerate(trained_model.parameters()):
            prox_term += (mu / 2) * torch.norm((param - anchor_weights[p_i])) ** 2
        return prox_term
