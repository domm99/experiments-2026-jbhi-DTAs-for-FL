import torch
import pandas as pd
from collections import OrderedDict
from LearningConfig import LearningConfig
from utils import (
    load_patient_series,
    load_test_patient_series,
    PatientSeries,
    GlucoseClassifierLSTM,
    evaluate,
    create_test_loaders,
    normalize_series,
)

class DT:

    def __init__(self, mid: str, data_path: str, experiment: str, config: LearningConfig, seed: int):
        self._mid = mid
        self._time = None
        self._seed = seed
        self._model = None
        self._last_std = 0.0
        self._last_mean = 0.0
        self._config = config
        self._is_active = False
        self._dt_aggregate = None
        self._experiment = experiment
        # Time at which the current model was deployed and time up to which the
        # predictions of the deployed models have already been evaluated.
        self._model_time = None
        self._evaluated_until = None
        self._data = self.__upload_data(data_path, mid)

    def activate(self, current_time: pd.Timestamp):
        self._is_active = True
        self._time = current_time

    def deactivate(self):
        self._is_active = False

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, data: tuple[OrderedDict[str, torch.Tensor], float, float, pd.Timestamp]):
        model, mean, std, deployed_at = data
        fresh_model = GlucoseClassifierLSTM(
            hidden_size=self._config.hidden_size,
            num_layers=self._config.layers,
            dropout=self._config.dropout,
        ).to(self._config.device)
        fresh_model.load_state_dict(model)
        self._model = fresh_model
        self._last_mean = mean
        self._last_std = std
        self._model_time = deployed_at

    @property
    def model_time(self) -> pd.Timestamp | None:
        return self._model_time

    @property
    def dt_aggregate(self):
        return self._dt_aggregate

    @dt_aggregate.setter
    def dt_aggregate(self, dt):
        self._dt_aggregate = dt

    def __filter_data_by_time(self, current_time: pd.Timestamp, last_train_time: pd.Timestamp = None) -> pd.DataFrame:
        if last_train_time is None:
            return self._data[self._data['timestamp'] <= current_time]
        return self._data[
            (self._data['timestamp'] >= last_train_time) &
            (self._data['timestamp'] <= current_time)
        ]

    def __get_patient_series(self, current_time: pd.Timestamp, last_train_time: pd.Timestamp = None) -> PatientSeries:
        filtered_df = self.__filter_data_by_time(current_time, last_train_time)
        series = load_patient_series(
            patient_id=self._mid,
            patient_dataframe=filtered_df,
            sequence_length=self._config.sequence_length,
            train_ratio=self._config.train_ratio,
            label_column=self._config.label_column,
            label_horizon_steps=self._config.label_horizon_steps,
        )
        return series

    def get_data(self, current_time: pd.Timestamp) -> PatientSeries:
        my_series = self.__get_patient_series(current_time)
        return my_series

    def inference(
        self,
        current_time: pd.Timestamp,
        last_training_time: pd.Timestamp,
        window_start_time: pd.Timestamp | None = None,
    ) -> dict:
        if self._model is None:
            metrics = {
                'status': 'skipped_no_model',
                'num_points': 0,
                'prediction_count': 0,
                'prediction_correct_count': 0,
                'prediction_error_count': 0,
                'loss': float('nan'),
                'accuracy': float('nan'),
                'precision': float('nan'),
                'recall': float('nan'),
                'f1_score': float('nan'),
                'prediction_y_true': [],
                'prediction_y_pred': [],
            }
            metrics['dt_id'] = self._mid
            return metrics
        loader, num_points = self.__test_loader_from_data(
            current_time,
            last_training_time,
            window_start_time=window_start_time,
        )
        self._evaluated_until = current_time
        if loader is None:
            metrics = {
                'status': 'skipped_short_test_window',
                'num_points': num_points,
                'prediction_count': 0,
                'prediction_correct_count': 0,
                'prediction_error_count': 0,
                'loss': float('nan'),
                'accuracy': float('nan'),
                'precision': float('nan'),
                'recall': float('nan'),
                'f1_score': float('nan'),
                'prediction_y_true': [],
                'prediction_y_pred': [],
            }
            metrics['dt_id'] = self._mid
            return metrics
        metrics = evaluate(self._model, loader, self._config.device, return_predictions=True)
        metrics['status'] = 'evaluated'
        metrics['num_points'] = num_points
        metrics['dt_id'] = self._mid
        return metrics

    def __upload_data(self, data_path: str, mid: str) -> pd.DataFrame:
        data = pd.read_csv(f'{data_path}/{mid}.csv')
        data['timestamp'] = pd.to_datetime(
            data['Measurement_date'] + ' ' + data['Measurement_time']
        )
        return data

    def __test_loader_from_data(
        self,
        current_time: pd.Timestamp,
        last_training_time: pd.Timestamp,
        window_start_time: pd.Timestamp | None = None,
    ):
        if window_start_time is None:
            # Cumulative evaluation of everything observed since the last training.
            filtered_df = self.__filter_data_by_time(current_time, last_training_time)
            target_start_time = self._model_time
        else:
            # Incremental evaluation: only predictions not evaluated yet, and only
            # those made by the currently deployed model. Tracked per DT so that a
            # DT which received its model later (or whose DTA did not retrain)
            # neither skips nor re-evaluates any period.
            filtered_df = self.__filter_data_by_time(current_time)
            # window_start_time is only a fallback: in global retraining it equals
            # the deployment time anyway.
            candidates = [time for time in (self._evaluated_until, self._model_time) if time is not None]
            target_start_time = max(candidates) if candidates else window_start_time
        window_df = filtered_df[filtered_df['timestamp'] <= current_time]
        if target_start_time is not None:
            window_df = window_df[window_df['timestamp'] > target_start_time]
        num_points = len(window_df)

        series = load_test_patient_series(
            patient_id=self._mid,
            patient_dataframe=filtered_df,
            sequence_length=self._config.sequence_length,
            label_column=self._config.label_column,
            label_horizon_steps=self._config.label_horizon_steps,
        )
        if series is None:
            return None, num_points
        normalized_series = normalize_series(series, self._last_mean, self._last_std)
        loader = create_test_loaders(
            normalized_series,
            self._config.sequence_length,
            self._config.stride,
            self._config.batch_size,
            target_start_time=target_start_time,
            target_start_inclusive=False,
            target_end_time=current_time,
        )
        if len(loader.dataset) == 0:
            return None, num_points
        return loader, num_points
