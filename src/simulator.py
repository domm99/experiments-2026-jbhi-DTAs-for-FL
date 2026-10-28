import heapq
from abc import ABC
from pathlib import Path
import pandas as pd
from DT import DT
from dataclasses import dataclass, field
from DTAggregate import DTAggregate
from LearningConfig import LearningConfig, effective_local_epochs
from FLServer import FLServer


@dataclass(order=True)
class Event:
    time: pd.Timestamp
    priority: int
    event_type: str = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)

class EventQueue:

    def __init__(self):
        self._heap = []

    def push(self, event: Event):
        heapq.heappush(self._heap, event)

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def empty(self) -> bool:
        return len(self._heap) == 0

class SimulationState:
    def __init__(self):
        self.active_patients = set()
        self.local_dts = {}
        self.last_training_time = None
        self.last_inference_results = []

class Monitor(ABC):

    def __init__(self, simulator):
        self._simulator = simulator
        self._simulator.add_monitor(self)

    def on_start(self) -> None:
        """Called when the simulation starts"""

    def on_finish(self) -> None:
        """Called when the simulation ends"""

    def on_event(self, event: Event) -> None:
        """Called after an event has been processed"""

class Simulator:

    def __init__(self,
                 data_folder: str,
                 experiment: str,
                 starting_time: pd.Timestamp,
                 ending_time: pd.Timestamp,
                 config: LearningConfig,
                 seed: int,
                 mapping_dtas_dts: dict[str, list[str]],
                 only_local_learning: bool):
        self._queue = EventQueue()
        self.data_folder = data_folder
        self.seed = seed
        self._config = config
        self.time = starting_time
        self._ending_time = ending_time
        self._state = SimulationState()
        self._handlers = {
            'PATIENT_BECOMES_ACTIVE': self.__handle_patient_becomes_active,
            'PATIENT_BECOMES_INACTIVE': self.__handle_patient_becomes_inactive,
            'TRAIN': self.__handle_train,
            'INFERENCE': self.__handle_inference,
        }
        self._fl_server = FLServer(config)
        initial_model = self._fl_server.model
        self._dtas = {
            h_id : DTAggregate(h_id, config, experiment, initial_model, seed)
            for h_id in mapping_dtas_dts.keys()
        }
        self._mapping_dtas_dts = mapping_dtas_dts
        self._monitors = []
        self._experiment = experiment
        self._only_local_learning = only_local_learning
        self._output_dir = Path(config.data_export_path) / experiment
        self._handled_inference_keys: set[tuple[pd.Timestamp, pd.Timestamp]] = set()
        self._retraining_event_count = 0

    def schedule_event(self, event: Event) -> bool:
        if event.time > self._ending_time:
            return False
        self._queue.push(event)
        return True

    def add_monitor(self, monitor) -> None:
        self._monitors.append(monitor)

    @property
    def state(self) -> SimulationState:
        return self._state

    @property
    def config(self) -> LearningConfig:
        return self._config

    @property
    def ending_time(self) -> pd.Timestamp:
        return self._ending_time

    @property
    def experiment(self) -> str:
        return self._experiment

    @property
    def dta_ids(self) -> list[str]:
        return list(self._dtas.keys())

    def dta_id_for_dt(self, dt_id: str) -> str:
        return self.__lookup_dta_from_patient_id(dt_id)

    def dta_population_size(self, dta_id: str) -> int:
        """Number of patients assigned to the institution of a DTA."""
        return len(self._mapping_dtas_dts.get(dta_id, []))

    @property
    def only_local_learning(self) -> bool:
        return self._only_local_learning

    def dta_ids_without_model(self) -> list[str]:
        return [dta_id for dta_id, dta in self._dtas.items() if not dta.has_model]

    def dta_has_training_data(self, dta_id: str, current_time: pd.Timestamp) -> bool:
        return self._dtas[dta_id].has_training_data(current_time)

    def __dispatch(self, event: Event):
        self._handlers[event.event_type](event)

    def start(self):

        for monitor in self._monitors:
            monitor.on_start()

        while not self._queue.empty():
            event = self._queue.pop()
            self.time = event.time
            self.__dispatch(event)

            for monitor in self._monitors:
                monitor.on_event(event)

        for monitor in self._monitors:
            monitor.on_finish()

    def __handle_patient_becomes_active(self, event: Event):
        patient_id = event.payload['patient_id']
        current_time = event.time

        print(f'========= Patient becoming active at:{current_time} =========')

        if patient_id in self._state.active_patients:
            return

        if patient_id not in self._state.local_dts:
            self._state.local_dts[patient_id] = DT(patient_id, self.data_folder, self._experiment, self._config, self.seed)

        local_dt = self._state.local_dts[patient_id]
        local_dt.activate(current_time)
        self._state.active_patients.add(patient_id)
        dta_id = self.__lookup_dta_from_patient_id(event.payload['patient_id'])
        dta = self._dtas[dta_id]
        dta.register_active_dt(local_dt, patient_id)
        dta.deploy_to(local_dt, current_time)

    def __lookup_dta_from_patient_id(self, patient_id):
        for dta_id, dts in self._mapping_dtas_dts.items():
            if patient_id in dts:
                return dta_id
        raise Exception('No corresponding DTA found')

    def __handle_patient_becomes_inactive(self, event: Event):
        patient_id = event.payload['patient_id']

        print(f'========= Patient becoming inactive at:{event.time} =========')

        dt = self._state.local_dts[patient_id]
        if dt is not None:
            dt.deactivate()
            dta_id = self.__lookup_dta_from_patient_id(event.payload['patient_id'])
            dta = self._dtas[dta_id]
            dta.unregister_active_dt(patient_id)
            self._state.active_patients.remove(patient_id)

    def __handle_train(self, event: Event):
        current_time = event.time
        print(f'========= Training at:{current_time} =========')

        # DTAs to retrain: all of them, unless the event targets specific DTAs
        # (local-only learning with per-DTA retraining).
        target_dta_ids = event.payload.get('dta_ids')
        if target_dta_ids is not None and not self._only_local_learning:
            raise ValueError('Retraining of single DTAs is only supported with only_local_learning')
        target_dtas = [
            dta for dta in self._dtas.values()
            if target_dta_ids is None or dta.dta_id in target_dta_ids
        ]

        for dta in target_dtas:
            dta.update_data_from_dts(current_time)

        # Per-DTA record of the training work done in this event (for the cost
        # analysis): epochs per phase and number of training windows.
        work = {
            dta.dta_id: {'federated_rounds': 0, 'federated_epochs': 0, 'personalization_epochs': 0, 'local_epochs': 0, 'sample_count': 0}
            for dta in self._dtas.values()
        }

        def record(dta_id: str, phase: str, epochs: int, sample_count: int) -> None:
            work[dta_id][f'{phase}_epochs'] += epochs
            work[dta_id]['sample_count'] = max(work[dta_id]['sample_count'], sample_count)
            if phase == 'federated':
                work[dta_id]['federated_rounds'] += 1

        updated_dtas = []
        if self._only_local_learning:
            local_epochs = effective_local_epochs(self._config)
            for dta in target_dtas:
                if dta.trainable_dt_count == 0:
                    print(f'========= Training skipped for DTA {dta.dta_id}: no trainable active DTs =========')
                    continue
                # Same per-event training budget as the federated strategy and
                # no proximal term (there is no shared model to stay close to).
                training_result = dta.train(
                    current_time,
                    'GR_0',
                    start_from='deployed',
                    epochs=local_epochs,
                    mu=self._config.local_mu,
                )
                if training_result is not None:
                    dta.global_model = training_result.model
                    updated_dtas.append(dta)
                    record(dta.dta_id, 'local', local_epochs, training_result.sample_count)
        else:
            updated_dtas = list(self._dtas.values())
            completed_training = False
            for gr in range(self._config.fl_global_rounds):
                print(f'Global round:{gr}')
                ## 1. Local training, always starting from the current global model
                client_updates = []
                for dta in self._dtas.values():
                    if dta.trainable_dt_count == 0:
                        print(f'========= Training skipped for DTA {dta.dta_id}: no trainable active DTs =========')
                        continue
                    training_result = dta.train(
                        current_time,
                        f'GR_{gr}',
                        start_from='global',
                        epochs=self._config.fl_local_epochs,
                        mu=self._config.mu,
                    )
                    if training_result is not None:
                        client_updates.append((training_result.model, training_result.sample_count))
                        record(dta.dta_id, 'federated', self._config.fl_local_epochs, training_result.sample_count)

                ## 2. Get local models from DTAs
                if not client_updates:
                    print('========= Global aggregation skipped: no DTA produced an update =========')
                    break

                ## 3. Global aggregation
                self._fl_server.receive_client_update(client_updates)
                self._fl_server.aggregate()
                completed_training = True

                ## 4. Notify new model to DTAs
                for dta in self._dtas.values():
                    dta.global_model = self._fl_server.model

            if not completed_training:
                print(f'========= Training at {current_time} produced no global model update =========')
                self.__export_retraining_log(event, work, completed=False)
                return

            ## 5. Local personalization: fine-tuning of the global model on the
            ## local data set, deployed only to the DTs of the same DTA. The
            ## global model is kept for the next federated round.
            if self._config.personalization:
                personalization_mu = (
                    self._config.mu
                    if self._config.personalization_mu is None
                    else self._config.personalization_mu
                )
                for dta in self._dtas.values():
                    if dta.trainable_dt_count == 0:
                        print(f'========= Personalization skipped for DTA {dta.dta_id}: no trainable active DTs =========')
                        continue
                    training_result = dta.train(
                        current_time,
                        'PERS',
                        start_from='global',
                        epochs=self._config.personalization_epochs,
                        mu=personalization_mu,
                    )
                    if training_result is not None:
                        dta.deployed_model = training_result.model
                        record(dta.dta_id, 'personalization', self._config.personalization_epochs, training_result.sample_count)

        ## 6. Notification of new model to HDTs (only DTAs whose model changed,
        ## so that the evaluation windows of the other DTs are not reset)
        for dta in updated_dtas:
            dta.notify_new_model(current_time)

        self._state.last_training_time = current_time
        self._state.last_inference_results = []
        self.__export_retraining_log(event, work, completed=True)

    def __export_retraining_log(self, event: Event, work: dict[str, dict[str, int]], completed: bool) -> None:
        """One row per DTA and retraining event, with the training work it did."""
        self._retraining_event_count += 1
        target_ids = event.payload.get('dta_ids') or list(self._dtas)
        rows = []
        for dta_id, dta_work in work.items():
            total_epochs = dta_work['federated_epochs'] + dta_work['personalization_epochs'] + dta_work['local_epochs']
            rows.append({
                'retraining_index': self._retraining_event_count,
                'time': event.time,
                'reason': event.payload.get('reasons', {}).get(dta_id, event.payload.get('reason')),
                'policy': event.payload.get('policy'),
                'mode': 'local' if self._only_local_learning else 'federated',
                'completed': completed,
                'dta_id': dta_id,
                'trained': total_epochs > 0,
                'dts_with_data': self._dtas[dta_id].trainable_dt_count if dta_id in target_ids else None,
                'sample_count': dta_work['sample_count'],
                'federated_rounds': dta_work['federated_rounds'],
                'federated_epochs': dta_work['federated_epochs'],
                'personalization_epochs': dta_work['personalization_epochs'],
                'local_epochs': dta_work['local_epochs'],
                'total_epochs': total_epochs,
                # Number of training windows processed (sample_count x epochs)
                'sample_epochs': dta_work['sample_count'] * total_epochs,
            })
        output_path = self._output_dir / f'retraining_log-seed_{self.seed}.csv'
        pd.DataFrame(rows).to_csv(output_path, mode='a', header=not output_path.exists(), index=False)

    def __handle_inference(self, event: Event):
        current_time = event.time
        print(f'========= Inference at:{current_time} =========')
        last_training_time = event.payload['last_training_time']
        if self._state.last_training_time is None:
            print('========= Inference skipped: no completed training yet =========')
            return
        if last_training_time != self._state.last_training_time:
            print('========= Inference skipped: stale inference event =========')
            return
        inference_key = (current_time, last_training_time)
        if inference_key in self._handled_inference_keys:
            print('========= Inference skipped: duplicate inference event =========')
            return
        self._handled_inference_keys.add(inference_key)
        inference_results = []
        window_start_time = event.payload.get('window_start_time')
        # Lazy deployment: DTs that could not receive the model yet (their DTA
        # had no local data when the model was distributed) get it now; they are
        # evaluated from the next inference on.
        for dta in self._dtas.values():
            for dt in dta.active_dts:
                if dt.model is None:
                    dta.deploy_to(dt, current_time)
        active_dts_from_dtas = [dta.active_dts for dta in self._dtas.values()]
        dts = [dt for active_dts in active_dts_from_dtas for dt in active_dts]
        for local_dt in dts:
            metrics = local_dt.inference(
                current_time,
                last_training_time,
                window_start_time=window_start_time,
            )
            if metrics is not None:
                inference_results.append(metrics)
        self._state.last_inference_results = inference_results
        self.__export_test_metrics(inference_results, current_time)

    def __export_test_metrics(self, inference_results: list[dict], current_time: pd.Timestamp) -> None:
        if not inference_results:
            return
        output_path = self._output_dir / f'test_{current_time}-seed_{self.seed}.csv'
        if output_path.exists():
            raise RuntimeError(
                f'{output_path} already exists: results of different runs would be mixed. '
                'Use an empty output folder (or overwrite=true).'
            )
        rows = [
            {
                key: value for key, value in metrics.items()
                if key not in {'prediction_y_true', 'prediction_y_pred'}
            }
            for metrics in inference_results
        ]
        pd.DataFrame(rows).to_csv(output_path, index=False)
