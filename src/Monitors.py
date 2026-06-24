import pandas as pd
from simulator import Event, Monitor, Simulator

class PeriodicInferenceMonitor(Monitor):

    def __init__(
        self,
        simulator: Simulator,
        inference_interval_days: int,
        inference_priority: int = 2,
    ) -> None:
        super().__init__(simulator)
        self._inference_interval_days = inference_interval_days
        self._inference_priority = inference_priority
        self._source = 'periodic_evaluation'

    def on_event(self, event: Event) -> None:
        if event.event_type == 'TRAIN':
            if self._simulator.state.last_training_time == event.time:
                self._schedule_next_inference(event.time, event.time, event.time)
            return

        if event.event_type != 'INFERENCE':
            return

        if event.payload.get('source') != self._source:
            return

        last_training_time = event.payload['last_training_time']
        if self._simulator.state.last_training_time != last_training_time:
            return

        self._schedule_next_inference(event.time, last_training_time, event.time)

    def _schedule_next_inference(
        self,
        current_time: pd.Timestamp,
        last_training_time: pd.Timestamp,
        window_start_time: pd.Timestamp,
    ) -> bool:
        return self._simulator.schedule_event(
            Event(
                time=current_time + pd.DateOffset(days=self._inference_interval_days),
                priority=self._inference_priority,
                event_type='INFERENCE',
                payload={
                    'last_training_time': last_training_time,
                    'window_start_time': window_start_time,
                    'source': self._source,
                },
            )
        )
