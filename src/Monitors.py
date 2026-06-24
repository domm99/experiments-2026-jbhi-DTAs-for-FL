import pandas as pd
from pathlib import Path
from simulator import Event, Monitor, Simulator
from RetrainingPolicy import ADWINErrorDecentralizedRetrainingPolicy, ADWINErrorUpdate, ADWINErrorDecentralizedDecision

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


class PerformanceDriftMonitor(Monitor):

    def __init__(
        self,
        simulator: Simulator,
        bootstrap_months: int,
        inference_interval_days: int,
        retraining_delay_days: int,
        metric_name: str,
        degradation_threshold: float,
        degraded_dt_fraction_threshold: float,
        metric_floor: float | None = None,
        min_comparable_dts: int = 1,
        threshold_mode: str = 'relative',
        higher_is_worse: bool = True,
        train_priority: int = 1,
        inference_priority: int = 2,
    ) -> None:
        super().__init__(simulator)
        self._bootstrap_months = bootstrap_months
        self._inference_interval_days = inference_interval_days
        self._retraining_delay_days = retraining_delay_days
        self._metric_name = metric_name
        self._degradation_threshold = degradation_threshold
        self._degraded_dt_fraction_threshold = degraded_dt_fraction_threshold
        self._metric_floor = metric_floor
        self._min_comparable_dts = min_comparable_dts
        self._threshold_mode = threshold_mode
        self._higher_is_worse = higher_is_worse
        self._train_priority = train_priority
        self._inference_priority = inference_priority
        self._baseline_metrics: dict[str, float] = {}
        self._baseline_timestamps: dict[str, pd.Timestamp] = {}
        self._training_pending = False

    def on_start(self) -> None:
        self._schedule_train(
            self._simulator.time + pd.DateOffset(months=self._bootstrap_months),
            reason='bootstrap_window',
        )

    def on_event(self, event: Event) -> None:
        if event.event_type == 'TRAIN':
            if self._simulator.state.last_training_time != event.time:
                self._training_pending = False
                return
            self._baseline_metrics = {}
            self._baseline_timestamps = {}
            self._training_pending = False
            self._schedule_inference(
                event.time + pd.DateOffset(days=self._inference_interval_days),
                event.time,
            )
            return

        if event.event_type != 'INFERENCE':
            return

        last_training_time = event.payload['last_training_time']
        detection_time = event.time
        if self._simulator.state.last_training_time != last_training_time:
            return

        evaluated_results = []
        for result in self._simulator.state.last_inference_results:
            if result.get('status') != 'evaluated':
                continue
            metric_value = result.get(self._metric_name)
            if metric_value is None or pd.isna(metric_value):
                continue
            evaluated_results.append(result)

        if self._metric_floor is not None:
            comparable_results = evaluated_results
        else:
            comparable_results = []
            for result in evaluated_results:
                dt_id = result['dt_id']
                metric_value = float(result[self._metric_name])
                if dt_id not in self._baseline_metrics:
                    self._baseline_metrics[dt_id] = metric_value
                    self._baseline_timestamps[dt_id] = detection_time
                    continue
                comparable_results.append(result)

        degraded_results = [
            result for result in comparable_results
            if self._is_degraded(result['dt_id'], float(result[self._metric_name]))
        ]
        degraded_count = len(degraded_results)
        comparable_count = len(comparable_results)

        if comparable_count >= self._min_comparable_dts:
            degraded_fraction = degraded_count / comparable_count
            if degraded_fraction >= self._degraded_dt_fraction_threshold and not self._training_pending:
                reason = 'low_accuracy' if self._metric_floor is not None else 'performance_drift'
                scheduled_training_time = detection_time + pd.DateOffset(days=self._retraining_delay_days)
                train_event_scheduled = self._schedule_train(
                    scheduled_training_time,
                    reason=reason,
                )
                self._training_pending = train_event_scheduled
                self._export_drift_event(
                    last_training_time=last_training_time,
                    detection_time=detection_time,
                    scheduled_training_time=scheduled_training_time,
                    train_event_scheduled=train_event_scheduled,
                    comparable_results=comparable_results,
                    degraded_results=degraded_results,
                    degraded_fraction=degraded_fraction,
                    reason=reason,
                )

        self._schedule_inference(
            detection_time + pd.DateOffset(days=self._inference_interval_days),
            last_training_time,
        )

    def _schedule_train(self, time: pd.Timestamp, reason: str) -> bool:
        return self._simulator.schedule_event(
            Event(
                time=time,
                priority=self._train_priority,
                event_type='TRAIN',
                payload={'reason': reason},
            )
        )

    def _schedule_inference(self, time: pd.Timestamp, last_training_time: pd.Timestamp) -> bool:
        return self._simulator.schedule_event(
            Event(
                time=time,
                priority=self._inference_priority,
                event_type='INFERENCE',
                payload={'last_training_time': last_training_time},
            )
        )

    def _reference_metric(self, dt_id: str) -> float:
        if self._metric_floor is not None:
            return float(self._metric_floor)
        return self._baseline_metrics[dt_id]

    def _absolute_degradation(self, dt_id: str, current_metric: float) -> float:
        reference_metric = self._reference_metric(dt_id)
        if self._higher_is_worse:
            return current_metric - reference_metric
        return reference_metric - current_metric

    def _relative_degradation(self, dt_id: str, current_metric: float) -> float:
        reference_metric = self._reference_metric(dt_id)
        degradation = self._absolute_degradation(dt_id, current_metric)
        reference_scale = abs(reference_metric)
        if reference_scale == 0:
            return 1.0 if degradation > 0 else 0.0
        return degradation / reference_scale

    def _mean_or_nan(self, values: list[float]) -> float:
        if not values:
            return float('nan')
        return float(sum(values) / len(values))

    def _days_between(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> float:
        delta = end_time - start_time
        return float(delta.total_seconds() / 86400.0)

    def _summarize_results(
        self,
        results: list[dict],
        reference_time: pd.Timestamp,
    ) -> dict[str, float]:
        current_metrics: list[float] = []
        reference_metrics: list[float] = []
        absolute_degradations: list[float] = []
        relative_degradations: list[float] = []
        reference_ages_days: list[float] = []

        for result in results:
            dt_id = result['dt_id']
            current_metric = float(result[self._metric_name])
            current_metrics.append(current_metric)
            reference_metrics.append(self._reference_metric(dt_id))
            absolute_degradations.append(self._absolute_degradation(dt_id, current_metric))
            relative_degradations.append(self._relative_degradation(dt_id, current_metric))

            baseline_time = self._baseline_timestamps.get(dt_id)
            if baseline_time is not None:
                reference_ages_days.append(self._days_between(baseline_time, reference_time))

        return {
            'mean_current_metric': self._mean_or_nan(current_metrics),
            'mean_reference_metric': self._mean_or_nan(reference_metrics),
            'mean_absolute_degradation': self._mean_or_nan(absolute_degradations),
            'mean_relative_degradation': self._mean_or_nan(relative_degradations),
            'mean_reference_age_days': self._mean_or_nan(reference_ages_days),
            'max_reference_age_days': max(reference_ages_days) if reference_ages_days else float('nan'),
        }

    def _export_drift_event(
        self,
        last_training_time: pd.Timestamp,
        detection_time: pd.Timestamp,
        scheduled_training_time: pd.Timestamp,
        train_event_scheduled: bool,
        comparable_results: list[dict],
        degraded_results: list[dict],
        degraded_fraction: float,
        reason: str,
    ) -> None:
        comparable_summary = self._summarize_results(comparable_results, detection_time)
        degraded_summary = self._summarize_results(degraded_results, detection_time)
        metrics = {
            'reason': reason,
            'metric_name': self._metric_name,
            'threshold_mode': self._threshold_mode,
            'metric_floor': self._metric_floor,
            'degradation_threshold': self._degradation_threshold,
            'degraded_dt_fraction_threshold': self._degraded_dt_fraction_threshold,
            'higher_is_worse': self._higher_is_worse,
            'last_training_time': last_training_time,
            'detection_time': detection_time,
            'scheduled_training_time': scheduled_training_time,
            'train_event_scheduled': train_event_scheduled,
            'detection_latency_days': self._days_between(last_training_time, detection_time),
            'schedule_latency_days': self._days_between(detection_time, scheduled_training_time),
            'end_to_end_latency_days': self._days_between(last_training_time, scheduled_training_time),
            'comparable_dt_count': len(comparable_results),
            'degraded_dt_count': len(degraded_results),
            'degraded_fraction': degraded_fraction,
            'degraded_dt_ids': '|'.join(sorted(result['dt_id'] for result in degraded_results)),
            'simulation_end_time': self._simulator.ending_time,
            **comparable_summary,
            'mean_degraded_current_metric': degraded_summary['mean_current_metric'],
            'mean_degraded_reference_metric': degraded_summary['mean_reference_metric'],
            'mean_degraded_absolute_degradation': degraded_summary['mean_absolute_degradation'],
            'mean_degraded_relative_degradation': degraded_summary['mean_relative_degradation'],
            'mean_degraded_reference_age_days': degraded_summary['mean_reference_age_days'],
            'max_degraded_reference_age_days': degraded_summary['max_reference_age_days'],
        }

        output_dir = Path(self._simulator.config.data_export_path) / self._simulator.experiment
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'drift_events-seed_{self._simulator.seed}.csv'

        if output_path.exists():
            metrics_df = pd.read_csv(output_path)
            metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)
        else:
            metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(output_path, index=False)

    def _is_degraded(self, dt_id: str, current_metric: float) -> bool:
        if self._metric_floor is not None:
            if self._higher_is_worse:
                return current_metric > self._metric_floor
            return current_metric < self._metric_floor

        baseline_metric = self._baseline_metrics[dt_id]
        if self._higher_is_worse:
            delta = current_metric - baseline_metric
        else:
            delta = baseline_metric - current_metric

        if self._threshold_mode == 'absolute':
            return delta >= self._degradation_threshold

        baseline_scale = abs(baseline_metric)
        if baseline_scale == 0:
            return delta > 0
        return (delta / baseline_scale) >= self._degradation_threshold


class ADWINErrorDecentralizedRetrainingMonitor(Monitor):
    POLICY_NAME = 'ADWINErrorHierarchicalRetrainingPolicy'

    def __init__(
        self,
        simulator: Simulator,
        bootstrap_months: int,
        inference_interval_days: int,
        retraining_delay_days: int,
        delta: float = 0.01,
        drifted_dt_fraction_threshold: float = 0.3,
        min_comparable_dts: int = 1,
        drifted_dta_fraction_threshold: float = 0.3,
        min_comparable_dtas: int = 1,
        reset_after_retrain: bool = True,
        train_priority: int = 1,
        inference_priority: int = 2,
    ) -> None:
        super().__init__(simulator)
        self._bootstrap_months = bootstrap_months
        self._inference_interval_days = inference_interval_days
        self._retraining_delay_days = retraining_delay_days
        self._train_priority = train_priority
        self._inference_priority = inference_priority
        self._source = self.POLICY_NAME
        self._training_pending = False
        self._prediction_timestep = 0
        self._delta = delta
        self._drifted_dt_fraction_threshold = drifted_dt_fraction_threshold
        self._min_comparable_dts = min_comparable_dts
        self._drifted_dta_fraction_threshold = drifted_dta_fraction_threshold
        self._min_comparable_dtas = min_comparable_dtas
        self._reset_after_retrain = reset_after_retrain
        self._dta_policies: dict[str, ADWINErrorDecentralizedRetrainingPolicy] = {}
        self._first_dt_drift_times: dict[str, pd.Timestamp] = {}
        self._pending_dta_signal_times: dict[str, pd.Timestamp] = {}
        self._pending_dta_first_dt_drift_times: dict[str, pd.Timestamp] = {}
        self._total_dt_drifts = 0
        self._total_dta_signals = 0
        self._total_retraining_triggers = 0

    def on_start(self) -> None:
        self._schedule_train(
            self._simulator.time + pd.DateOffset(months=self._bootstrap_months),
            reason='adwin_error_hierarchical_bootstrap',
        )

    def on_event(self, event: Event) -> None:
        if event.event_type == 'TRAIN':
            if self._simulator.state.last_training_time != event.time:
                self._training_pending = False
                return
            self._export_completed_retraining_delays(event.time)
            adwin_reset = self._reset_policies_after_retrain()
            self._first_dt_drift_times = {}
            self._pending_dta_signal_times = {}
            self._pending_dta_first_dt_drift_times = {}
            self._training_pending = False
            print(
                f'{self.POLICY_NAME} | retraining_timestep={event.time} | '
                f'delta={self._delta} | adwin_reset={adwin_reset} | '
                f'total_dt_drifts={self._total_dt_drifts} | '
                f'total_dta_signals={self._total_dta_signals} | '
                f'total_retraining_triggers={self._total_retraining_triggers}'
            )
            self._schedule_inference(
                event.time + pd.DateOffset(days=self._inference_interval_days),
                event.time,
                event.time,
            )
            return

        if event.event_type != 'INFERENCE':
            return

        if event.payload.get('source') != self._source:
            return

        last_training_time = event.payload['last_training_time']
        detection_time = event.time
        if self._simulator.state.last_training_time != last_training_time:
            return

        evaluated_dt_counts_by_dta: dict[str, int] = {}
        drifted_dt_ids_by_dta: dict[str, set[str]] = {}
        evaluated_dt_count = 0
        prediction_count = 0
        error_count = 0
        drift_updates: list[tuple[str, str, ADWINErrorUpdate]] = []
        widths: list[float] = []
        estimations: list[float] = []

        for result in self._simulator.state.last_inference_results:
            if result.get('status') != 'evaluated':
                continue
            dt_id = result['dt_id']
            dta_id = self._simulator.dta_id_for_dt(dt_id)
            y_true_values = result.get('prediction_y_true') or []
            y_pred_values = result.get('prediction_y_pred') or []
            if not y_true_values or not y_pred_values:
                continue

            evaluated_dt_count += 1
            evaluated_dt_counts_by_dta[dta_id] = evaluated_dt_counts_by_dta.get(dta_id, 0) + 1
            dta_policy = self._policy_for_dta(dta_id)
            for y_true, y_pred in zip(y_true_values, y_pred_values, strict=False):
                self._prediction_timestep += 1
                update = dta_policy.update(
                    dt_id=dt_id,
                    y_true=y_true,
                    y_pred=y_pred,
                    timestep=self._prediction_timestep,
                )
                prediction_count += 1
                error_count += update.error
                self._collect_number(widths, update.width)
                self._collect_number(estimations, update.estimation)
                if not update.drift_detected:
                    continue

                self._total_dt_drifts += 1
                self._first_dt_drift_times.setdefault(dta_id, detection_time)
                drifted_dt_ids_by_dta.setdefault(dta_id, set()).add(dt_id)
                drift_updates.append((dta_id, dt_id, update))
                self._export_dt_drift_event(
                    detection_time=detection_time,
                    last_training_time=last_training_time,
                    dta_id=dta_id,
                    dt_id=dt_id,
                    update=update,
                )

        new_signaling_dta_ids: list[str] = []
        for dta_id, dta_evaluated_dt_count in sorted(evaluated_dt_counts_by_dta.items()):
            dta_policy = self._policy_for_dta(dta_id)
            decision = dta_policy.decide(
                drifted_dt_ids=drifted_dt_ids_by_dta.get(dta_id, set()),
                evaluated_dt_count=dta_evaluated_dt_count,
            )
            if not decision.retrain or dta_id in self._pending_dta_signal_times:
                continue
            self._pending_dta_signal_times[dta_id] = detection_time
            self._pending_dta_first_dt_drift_times[dta_id] = self._first_dt_drift_times.get(dta_id, detection_time)
            self._total_dta_signals += 1
            new_signaling_dta_ids.append(dta_id)
            self._export_dta_signal_event(
                detection_time=detection_time,
                last_training_time=last_training_time,
                dta_id=dta_id,
                decision=decision,
            )

        dta_count = max(len(self._simulator.dta_ids), 1)
        signaled_dta_ids = sorted(self._pending_dta_signal_times)
        signaled_dta_fraction = len(signaled_dta_ids) / dta_count
        global_retrain = (
            dta_count >= self._min_comparable_dtas
            and signaled_dta_fraction >= self._drifted_dta_fraction_threshold
        )
        retraining_timestep = None
        train_event_scheduled = False
        if global_retrain and not self._training_pending:
            scheduled_training_time = detection_time + pd.DateOffset(days=self._retraining_delay_days)
            train_event_scheduled = self._schedule_train(
                scheduled_training_time,
                reason='adwin_error_hierarchical_drift',
            )
            self._training_pending = train_event_scheduled
            if train_event_scheduled:
                self._total_retraining_triggers += 1
                retraining_timestep = scheduled_training_time
            self._export_retraining_event(
                detection_time=detection_time,
                last_training_time=last_training_time,
                scheduled_training_time=scheduled_training_time,
                dta_count=dta_count,
                signaled_dta_ids=signaled_dta_ids,
                signaled_dta_fraction=signaled_dta_fraction,
                train_event_scheduled=train_event_scheduled,
            )

        self._print_log(
            detection_time=detection_time,
            prediction_count=prediction_count,
            error_count=error_count,
            evaluated_dt_count=evaluated_dt_count,
            dta_count=dta_count,
            signaled_dta_ids=signaled_dta_ids,
            signaled_dta_fraction=signaled_dta_fraction,
            mean_width=self._mean_or_none(widths),
            max_width=max(widths) if widths else None,
            mean_estimation=self._mean_or_none(estimations),
            retraining_timestep=retraining_timestep,
        )
        self._export_log(
            detection_time=detection_time,
            last_training_time=last_training_time,
            prediction_count=prediction_count,
            error_count=error_count,
            drift_count_in_batch=len(drift_updates),
            evaluated_dt_count=evaluated_dt_count,
            evaluated_dta_count=len(evaluated_dt_counts_by_dta),
            dta_count=dta_count,
            new_signaling_dta_ids=new_signaling_dta_ids,
            signaled_dta_ids=signaled_dta_ids,
            signaled_dta_fraction=signaled_dta_fraction,
            mean_width=self._mean_or_none(widths),
            max_width=max(widths) if widths else None,
            mean_estimation=self._mean_or_none(estimations),
            retraining_timestep=retraining_timestep,
            train_event_scheduled=train_event_scheduled,
        )
        self._schedule_inference(
            detection_time + pd.DateOffset(days=self._inference_interval_days),
            last_training_time,
            detection_time,
        )

    def _schedule_train(self, time: pd.Timestamp, reason: str) -> bool:
        return self._simulator.schedule_event(
            Event(
                time=time,
                priority=self._train_priority,
                event_type='TRAIN',
                payload={'reason': reason, 'policy': self.POLICY_NAME},
            )
        )

    def _schedule_inference(
        self,
        time: pd.Timestamp,
        last_training_time: pd.Timestamp,
        window_start_time: pd.Timestamp,
    ) -> bool:
        return self._simulator.schedule_event(
            Event(
                time=time,
                priority=self._inference_priority,
                event_type='INFERENCE',
                payload={
                    'last_training_time': last_training_time,
                    'window_start_time': window_start_time,
                    'source': self._source,
                    'policy': self.POLICY_NAME,
                },
            )
        )

    def _policy_for_dta(self, dta_id: str) -> ADWINErrorDecentralizedRetrainingPolicy:
        if dta_id not in self._dta_policies:
            self._dta_policies[dta_id] = ADWINErrorDecentralizedRetrainingPolicy(
                delta=self._delta,
                drifted_dt_fraction_threshold=self._drifted_dt_fraction_threshold,
                min_comparable_dts=self._min_comparable_dts,
                reset_after_retrain=self._reset_after_retrain,
            )
        return self._dta_policies[dta_id]

    def _reset_policies_after_retrain(self) -> bool:
        reset_any = False
        for policy in self._dta_policies.values():
            reset_any = policy.on_retrain() or reset_any
        return reset_any

    def _print_log(
        self,
        detection_time: pd.Timestamp,
        prediction_count: int,
        error_count: int,
        evaluated_dt_count: int,
        dta_count: int,
        signaled_dta_ids: list[str],
        signaled_dta_fraction: float,
        mean_width,
        max_width,
        mean_estimation,
        retraining_timestep: pd.Timestamp | None,
    ) -> None:
        print(
            f'{self.POLICY_NAME} | timestep={detection_time} | '
            f'delta={self._delta} | '
            f'prediction_count={prediction_count} | '
            f'prediction_error_count={error_count} | '
            f'evaluated_dt_count={evaluated_dt_count} | '
            f'signaled_dta_count={len(signaled_dta_ids)} | '
            f'dta_count={dta_count} | '
            f'signaled_dta_fraction={signaled_dta_fraction:.6f} | '
            f'drifted_dt_fraction_threshold={self._drifted_dt_fraction_threshold} | '
            f'drifted_dta_fraction_threshold={self._drifted_dta_fraction_threshold} | '
            f'total_dt_drifts={self._total_dt_drifts} | '
            f'total_dta_signals={self._total_dta_signals} | '
            f'total_retraining_triggers={self._total_retraining_triggers} | '
            f'adwin_mean_width={mean_width} | '
            f'adwin_max_width={max_width} | '
            f'adwin_mean_estimation={mean_estimation} | '
            f'retraining_timestep={retraining_timestep}'
        )

    def _export_log(
        self,
        detection_time: pd.Timestamp,
        last_training_time: pd.Timestamp,
        prediction_count: int,
        error_count: int,
        drift_count_in_batch: int,
        evaluated_dt_count: int,
        evaluated_dta_count: int,
        dta_count: int,
        new_signaling_dta_ids: list[str],
        signaled_dta_ids: list[str],
        signaled_dta_fraction: float,
        mean_width,
        max_width,
        mean_estimation,
        retraining_timestep: pd.Timestamp | None,
        train_event_scheduled: bool,
    ) -> None:
        metrics = {
            'policy': self.POLICY_NAME,
            'timestep': detection_time,
            'last_training_time': last_training_time,
            'prediction_count': prediction_count,
            'prediction_error_count': error_count,
            'adwin_drift_count_in_batch': drift_count_in_batch,
            'evaluated_dt_count': evaluated_dt_count,
            'evaluated_dta_count': evaluated_dta_count,
            'dta_count': dta_count,
            'new_signaling_dta_count': len(new_signaling_dta_ids),
            'new_signaling_dta_ids': '|'.join(new_signaling_dta_ids),
            'signaled_dta_count': len(signaled_dta_ids),
            'signaled_dta_ids': '|'.join(signaled_dta_ids),
            'signaled_dta_fraction': signaled_dta_fraction,
            'drifted_dt_fraction_threshold': self._drifted_dt_fraction_threshold,
            'drifted_dta_fraction_threshold': self._drifted_dta_fraction_threshold,
            'min_comparable_dts': self._min_comparable_dts,
            'min_comparable_dtas': self._min_comparable_dtas,
            'adwin_mean_width': mean_width,
            'adwin_max_width': max_width,
            'adwin_mean_estimation': mean_estimation,
            'delta': self._delta,
            'reset_after_retrain': self._reset_after_retrain,
            'total_dt_drifts': self._total_dt_drifts,
            'total_dta_signals': self._total_dta_signals,
            'total_retraining_triggers': self._total_retraining_triggers,
            'retraining_timestep': retraining_timestep,
            'train_event_scheduled': train_event_scheduled,
            'simulation_end_time': self._simulator.ending_time,
        }
        self._append_csv(metrics, f'{self.POLICY_NAME}_log-seed_{self._simulator.seed}.csv')

    def _export_dt_drift_event(
        self,
        detection_time: pd.Timestamp,
        last_training_time: pd.Timestamp,
        dta_id: str,
        dt_id: str,
        update: ADWINErrorUpdate,
    ) -> None:
        metrics = {
            'policy': self.POLICY_NAME,
            'timestep': detection_time,
            'last_training_time': last_training_time,
            'dta_id': dta_id,
            'dt_id': dt_id,
            'stream_timestep': update.timestep,
            'error': update.error,
            'delta': update.delta,
            'adwin_width': update.width,
            'adwin_estimation': update.estimation,
            'dt_total_drifts': update.total_drifts,
            'total_dt_drifts': self._total_dt_drifts,
            'simulation_end_time': self._simulator.ending_time,
        }
        self._append_csv(metrics, f'{self.POLICY_NAME}_dt_drift_events-seed_{self._simulator.seed}.csv')

    def _export_dta_signal_event(
        self,
        detection_time: pd.Timestamp,
        last_training_time: pd.Timestamp,
        dta_id: str,
        decision: ADWINErrorDecentralizedDecision,
    ) -> None:
        first_dt_drift_time = self._pending_dta_first_dt_drift_times[dta_id]
        metrics = {
            'policy': self.POLICY_NAME,
            'timestep': detection_time,
            'last_training_time': last_training_time,
            'dta_id': dta_id,
            'first_local_dt_drift_time': first_dt_drift_time,
            'dta_signal_time': detection_time,
            'local_signal_delay_days': self._days_between(first_dt_drift_time, detection_time),
            'evaluated_dt_count': decision.evaluated_dt_count,
            'drifted_dt_count': len(decision.drifted_dt_ids),
            'drifted_dt_fraction': decision.drifted_dt_fraction,
            'drifted_dt_fraction_threshold': decision.drifted_dt_fraction_threshold,
            'min_comparable_dts': decision.min_comparable_dts,
            'drifted_dt_ids': '|'.join(decision.drifted_dt_ids),
            'total_dt_drifts': self._total_dt_drifts,
            'total_dta_signals': self._total_dta_signals,
            'simulation_end_time': self._simulator.ending_time,
        }
        self._append_csv(metrics, f'{self.POLICY_NAME}_dta_signal_events-seed_{self._simulator.seed}.csv')

    def _export_retraining_event(
        self,
        detection_time: pd.Timestamp,
        last_training_time: pd.Timestamp,
        scheduled_training_time: pd.Timestamp,
        dta_count: int,
        signaled_dta_ids: list[str],
        signaled_dta_fraction: float,
        train_event_scheduled: bool,
    ) -> None:
        metrics = {
            'policy': self.POLICY_NAME,
            'timestep': detection_time,
            'last_training_time': last_training_time,
            'scheduled_training_time': scheduled_training_time,
            'train_event_scheduled': train_event_scheduled,
            'dta_count': dta_count,
            'signaled_dta_count': len(signaled_dta_ids),
            'signaled_dta_fraction': signaled_dta_fraction,
            'drifted_dta_fraction_threshold': self._drifted_dta_fraction_threshold,
            'min_comparable_dtas': self._min_comparable_dtas,
            'signaled_dta_ids': '|'.join(signaled_dta_ids),
            'total_dt_drifts': self._total_dt_drifts,
            'total_dta_signals': self._total_dta_signals,
            'total_retraining_triggers': self._total_retraining_triggers,
            'delta': self._delta,
            'simulation_end_time': self._simulator.ending_time,
        }
        self._append_csv(metrics, f'{self.POLICY_NAME}_retraining_events-seed_{self._simulator.seed}.csv')

    def _export_completed_retraining_delays(self, retraining_time: pd.Timestamp) -> None:
        if not self._pending_dta_signal_times:
            return

        for dta_id in sorted(self._pending_dta_signal_times):
            first_dt_drift_time = self._pending_dta_first_dt_drift_times[dta_id]
            dta_signal_time = self._pending_dta_signal_times[dta_id]
            metrics = {
                'policy': self.POLICY_NAME,
                'dta_id': dta_id,
                'first_local_dt_drift_time': first_dt_drift_time,
                'dta_signal_time': dta_signal_time,
                'retraining_time': retraining_time,
                'first_local_dt_to_retrain_delay_days': self._days_between(first_dt_drift_time, retraining_time),
                'dta_signal_to_retrain_delay_days': self._days_between(dta_signal_time, retraining_time),
                'total_dt_drifts': self._total_dt_drifts,
                'total_dta_signals': self._total_dta_signals,
                'total_retraining_triggers': self._total_retraining_triggers,
                'simulation_end_time': self._simulator.ending_time,
            }
            self._append_csv(metrics, f'{self.POLICY_NAME}_dta_retraining_delays-seed_{self._simulator.seed}.csv')

    def _append_csv(self, metrics: dict, filename: str) -> None:
        output_dir = Path(self._simulator.config.data_export_path) / self._simulator.experiment
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(output_path, mode='a', header=not output_path.exists(), index=False)

    def _collect_number(self, values: list[float], value) -> None:
        if value is None or pd.isna(value):
            return
        values.append(float(value))

    def _mean_or_none(self, values: list[float]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))

    def _days_between(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> float:
        delta = end_time - start_time
        return float(delta.total_seconds() / 86400.0)
