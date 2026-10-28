from __future__ import annotations
from dataclasses import dataclass
from DTAggregate import DTAggregate
from typing import TYPE_CHECKING, Any
from simulator import Event, Simulator

@dataclass
class ADWINErrorUpdate:
    retrain: bool
    drift_detected: bool
    error: int
    delta: float
    total_drifts: int
    width: Any = None
    estimation: Any = None
    timestep: Any = None
    # Change detected by ADWIN regardless of its direction, and its direction
    # ('increase' / 'decrease' of the error rate, None if no change).
    raw_drift_detected: bool = False
    direction: str | None = None

@dataclass
class ADWINErrorDecentralizedDecision:
    retrain: bool
    drifted_dt_ids: list[str]
    evaluated_dt_count: int
    drifted_dt_fraction: float
    drifted_dt_fraction_threshold: float
    min_comparable_dts: int
    total_dt_drifts: int
    # Enough DTs have been evaluated for the quorum to be meaningful
    eligible: bool = False

class RetrainingPolicy:

    def on_event(self, event: Event, simulator: Simulator, dt_aggregate: DTAggregate):
        pass


class ADWINErrorRetrainingPolicy(RetrainingPolicy):
    """Detects performance drift from the binary 0/1 model error stream."""

    def __init__(
        self,
        delta: float = 0.01,
        reset_after_retrain: bool = True,
        one_sided: bool = True,
    ) -> None:
        self.delta = delta
        self.reset_after_retrain = reset_after_retrain
        # ADWIN is two-sided: it also fires when the error rate decreases. With
        # one_sided only increases of the error (i.e. degradation) count as drift.
        self.one_sided = one_sided
        self.total_drifts = 0
        self.ignored_improvements = 0
        self.detector = self._new_detector()

    def _new_detector(self):
        from river import drift

        return drift.ADWIN(delta=self.delta)

    def update(self, y_true, y_pred, timestep=None) -> ADWINErrorUpdate:
        error = int(y_true != y_pred)
        estimation_before = self.estimation
        # ADWIN detects statistically significant changes in model errors,
        # not drift directly on glucose features.
        update_result = self.detector.update(error)
        raw_drift_detected = self._read_drift_flag(update_result)
        if raw_drift_detected is None:
            raw_drift_detected = self._read_drift_flag(self.detector)
        raw_drift_detected = bool(raw_drift_detected)

        direction = None
        if raw_drift_detected:
            # After a detection ADWIN drops the oldest sub-window, so its estimate
            # moves towards the recent error rate.
            estimation_after = self.estimation
            direction = (
                'increase'
                if estimation_before is None or estimation_after is None or estimation_after > estimation_before
                else 'decrease'
            )
        drift_detected = raw_drift_detected and (not self.one_sided or direction == 'increase')

        if drift_detected:
            self.total_drifts += 1
        elif raw_drift_detected:
            self.ignored_improvements += 1

        return ADWINErrorUpdate(
            retrain=drift_detected,
            drift_detected=drift_detected,
            error=error,
            delta=self.delta,
            total_drifts=self.total_drifts,
            width=self._detector_metric(('width', 'window_size', 'n_samples')),
            estimation=self._detector_metric(('estimation', 'mean', 'mean_', 'total_mean')),
            timestep=timestep,
            raw_drift_detected=raw_drift_detected,
            direction=direction,
        )

    def reset(self) -> None:
        self.detector = self._new_detector()

    def on_retrain(self) -> bool:
        if not self.reset_after_retrain:
            return False
        self.reset()
        return True

    @property
    def width(self):
        return self._detector_metric(('width', 'window_size', 'n_samples'))

    @property
    def estimation(self):
        return self._detector_metric(('estimation', 'mean', 'mean_', 'total_mean'))

    def _read_drift_flag(self, obj) -> bool | None:
        if obj is None:
            return None
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, dict):
            for key in ('drift_detected', 'change_detected', 'in_drift'):
                if key in obj:
                    return bool(obj[key])
            return None
        if isinstance(obj, (tuple, list)):
            for item in obj:
                if isinstance(item, bool):
                    return item
            return None

        for attr_name in ('drift_detected', 'change_detected', 'in_drift'):
            if not hasattr(obj, attr_name):
                continue
            value = getattr(obj, attr_name)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            return bool(value)
        return None

    def _detector_metric(self, names: tuple[str, ...]):
        for name in names:
            if not hasattr(self.detector, name):
                continue
            value = getattr(self.detector, name)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            return value
        return None

class ADWINErrorDecentralizedRetrainingPolicy(RetrainingPolicy):
    """
    Keeps one ADWIN error detector per DT and thresholds DT drift votes
    (intra-institution quorum p, Eq. 2).

    With sticky_dt_flags a DT that detected degradation keeps its flag until the
    next retraining, and the quorum is computed over the DTs evaluated since
    then; otherwise only the DTs whose detector fired in the current inference
    batch are counted. The minimum number of evaluated DTs is capped by the
    population of the DTA, so that small institutions can also signal.
    """

    def __init__(
        self,
        delta: float = 0.01,
        drifted_dt_fraction_threshold: float = 0.3,
        min_comparable_dts: int = 1,
        reset_after_retrain: bool = True,
        one_sided: bool = True,
        sticky_dt_flags: bool = True,
    ) -> None:
        self.delta = delta
        self.drifted_dt_fraction_threshold = drifted_dt_fraction_threshold
        self.min_comparable_dts = min_comparable_dts
        self.reset_after_retrain = reset_after_retrain
        self.one_sided = one_sided
        self.sticky_dt_flags = sticky_dt_flags
        self.total_dt_drifts = 0
        self._dt_policies: dict[str, ADWINErrorRetrainingPolicy] = {}
        self._drifted_since_retrain: set[str] = set()
        self._evaluated_since_retrain: set[str] = set()

    def update(self, dt_id: str, y_true, y_pred, timestep=None) -> ADWINErrorUpdate:
        if dt_id not in self._dt_policies:
            self._dt_policies[dt_id] = self._new_dt_policy()
        policy = self._dt_policies[dt_id]
        update = policy.update(y_true=y_true, y_pred=y_pred, timestep=timestep)
        if update.drift_detected:
            self.total_dt_drifts += 1
        return update

    def decide(
        self,
        drifted_dt_ids: set[str],
        evaluated_dt_ids: set[str],
        population_size: int | None = None,
    ) -> ADWINErrorDecentralizedDecision:
        if self.sticky_dt_flags:
            self._drifted_since_retrain |= set(drifted_dt_ids)
            self._evaluated_since_retrain |= set(evaluated_dt_ids)
            drifted, evaluated = self._drifted_since_retrain, self._evaluated_since_retrain
        else:
            drifted, evaluated = set(drifted_dt_ids), set(evaluated_dt_ids)

        min_required = self.min_comparable_dts
        if population_size is not None and population_size > 0:
            min_required = min(min_required, population_size)

        sorted_drifted_dt_ids = sorted(drifted)
        evaluated_dt_count = len(evaluated)
        drifted_dt_fraction = 0.0
        if evaluated_dt_count > 0:
            drifted_dt_fraction = len(sorted_drifted_dt_ids) / evaluated_dt_count

        eligible = evaluated_dt_count >= min_required
        retrain = eligible and drifted_dt_fraction >= self.drifted_dt_fraction_threshold
        return ADWINErrorDecentralizedDecision(
            retrain=retrain,
            drifted_dt_ids=sorted_drifted_dt_ids,
            evaluated_dt_count=evaluated_dt_count,
            drifted_dt_fraction=drifted_dt_fraction,
            drifted_dt_fraction_threshold=self.drifted_dt_fraction_threshold,
            min_comparable_dts=min_required,
            total_dt_drifts=self.total_dt_drifts,
            eligible=eligible,
        )

    def on_retrain(self) -> bool:
        # Flags refer to the model that has just been replaced.
        self._drifted_since_retrain = set()
        self._evaluated_since_retrain = set()
        if not self.reset_after_retrain:
            return False
        for policy in self._dt_policies.values():
            policy.reset()
        return True

    def _new_dt_policy(self) -> ADWINErrorRetrainingPolicy:
        return ADWINErrorRetrainingPolicy(
            delta=self.delta,
            reset_after_retrain=False,
            one_sided=self.one_sided,
        )

