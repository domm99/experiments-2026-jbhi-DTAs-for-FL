import heapq
from abc import ABC
import pandas as pd
from src.distributed.DT import DT
from dataclasses import dataclass, field
from src.distributed.DTAggregate import DTAggregate
from src.distributed.LearningConfig import LearningConfig

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

    def __init__(self, data_folder: str, experiment: str, starting_time: pd.Timestamp, ending_time: pd.Timestamp, config: LearningConfig, seed: int):
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
        self._dt_aggregate = DTAggregate(config, experiment, seed)
        self._monitors = []
        self._experiment = experiment

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
    def dt_aggregate(self) -> DTAggregate:
        return self._dt_aggregate

    @property
    def config(self) -> LearningConfig:
        return self._config

    @property
    def ending_time(self) -> pd.Timestamp:
        return self._ending_time

    @property
    def experiment(self) -> str:
        return self._experiment

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
        self._dt_aggregate.register_active_dt(local_dt, patient_id)
        mean, std = self._dt_aggregate.statistics
        local_dt.model = (self._dt_aggregate.model, mean, std)

    def __handle_patient_becomes_inactive(self, event: Event):
        patient_id = event.payload['patient_id']

        print(f'========= Patient becoming inactive at:{event.time} =========')

        dt = self._state.local_dts[patient_id]
        if dt is not None:
            dt.deactivate()
            self._dt_aggregate.unregister_active_dt(patient_id)
            self._state.active_patients.remove(patient_id)

    def __handle_train(self, event: Event):
        current_time = event.time
        print(f'========= Training at:{current_time} =========')
        self._dt_aggregate.update_data_from_dts(current_time)
        if self._dt_aggregate.trainable_dt_count == 0:
            print('========= Training skipped: no trainable active DTs =========')
            return
        self._dt_aggregate.train(current_time)
        self._dt_aggregate.notify_new_model()
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
        for local_dt in self._dt_aggregate.active_dts:
            metrics = local_dt.inference(
                current_time,
                last_training_time,
                window_start_time=window_start_time,
            )
            if metrics is not None:
                inference_results.append(metrics)
        self._state.last_inference_results = inference_results
