import heapq
from abc import ABC
import pandas as pd
from DT import DT
from dataclasses import dataclass, field
from DTAggregate import DTAggregate
from LearningConfig import LearningConfig
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
        if dta.has_statistics:
            mean, std = dta.statistics
            local_dt.model = (dta.model, mean, std)

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

        for dta in self._dtas.values():
            dta.update_data_from_dts(current_time)

        if self._only_local_learning:
            for dta in self._dtas.values():
                if dta.trainable_dt_count == 0:
                    print(f'========= Training skipped for DTA {dta.dta_id}: no trainable active DTs =========')
                    continue
                dta.train(current_time, 0)
        else:
            completed_training = False
            for gr in range(self._config.fl_global_rounds):
                print(f'Global round:{gr}')
                ## 1. Local training
                client_updates = []
                for dta in self._dtas.values():
                    if dta.trainable_dt_count == 0:
                        print(f'========= Training skipped for DTA {dta.dta_id}: no trainable active DTs =========')
                        continue
                    training_result = dta.train(current_time, gr)
                    if training_result is not None:
                        client_updates.append((training_result.model, training_result.sample_count))

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
                    dta.model = self._fl_server.model

            if not completed_training:
                print(f'========= Training at {current_time} produced no global model update =========')
                return

        ## 5. Notification of new model to HDTs
        for dta in self._dtas.values():
            dta.notify_new_model()

        self._state.last_training_time = current_time
        self._state.last_inference_results = []

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
        inference_results = []
        window_start_time = event.payload.get('window_start_time')
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
