import glob
import random
import pandas as pd
from pathlib import Path
from utils import seed_everything
from simulator import Simulator, Event
from LearningConfig import LearningConfig
from Monitors import PeriodicInferenceMonitor, PerformanceDriftMonitor, ADWINErrorDecentralizedRetrainingMonitor

def balanced_split(xs, n):
    if n <= 0:
        raise ValueError("n must be greater than zero")
    if n > len(xs):
        raise ValueError("n must not be greater than xs length")

    k, r = divmod(len(xs), n)

    return [
        xs[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)]
        for i in range(n)
    ]

def count_patient_labels(
    patient_id: str,
    data_dir: Path,
    label_col: str = "target_any_within_t",
) -> dict[str, int]:
    csv_path = data_dir / f"{patient_id}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found for patient {patient_id}: {csv_path}")

    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(
            f"Column '{label_col}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    labels = df[label_col].astype(str).str.strip().str.lower()
    counts = labels.value_counts().to_dict()

    return {
        "normal": counts.get("normal", 0),
        "hyper": counts.get("hyper", 0),
        "hypo": counts.get("hypo", 0),
        "total": len(labels),
    }

def get_patient_score(
    counts: dict[str, int],
    target_label: str,
) -> float:
    total = counts["total"]

    if total <= 0:
        return 0.0

    if target_label == "normal":
        return counts["normal"] / total

    if target_label == "hyper":
        return counts["hyper"] / total

    if target_label == "hypo":
        return counts["hypo"] / total

    if target_label == "event":
        return (counts["hyper"] + counts["hypo"]) / total

    raise ValueError(
        "target_label must be one of: "
        "'normal', 'hyper', 'hypo', 'event'"
    )

def split_patients_label_skew(
    all_patients,
    config,
    data_dir: str | Path,
    label_col: str = "target_any_within_t",
    target_label: str = "event",
    ascending: bool = True,
    shuffle_ties: bool = True,
) -> dict[str, list[str]]:
    """
    Patient-level label-skew split.

    all_patients only needs to contain:
        {"patient_id": "LIB193263"}

    The function reads each patient CSV, computes the target label ratio,
    sorts patients by that ratio, and assigns consecutive blocks to hospitals.

    target_label:
        "event"  -> hyper + hypo ratio
        "hyper"  -> hyper ratio
        "hypo"   -> hypo ratio
        "normal" -> normal ratio
    """

    data_dir = Path(data_dir)

    patient_scores = []

    for patient in all_patients:
        patient_id = patient["patient_id"]

        counts = count_patient_labels(
            patient_id=patient_id,
            data_dir=data_dir,
            label_col=label_col,
        )

        score = get_patient_score(
            counts=counts,
            target_label=target_label,
        )

        patient_scores.append({
            "patient_id": patient_id,
            "score": score,
            "counts": counts,
        })

    if shuffle_ties:
        random.shuffle(patient_scores)

    patient_scores = sorted(
        patient_scores,
        key=lambda x: x["score"],
        reverse=not ascending,
    )

    patient_ids = [p["patient_id"] for p in patient_scores]

    split = balanced_split(patient_ids, config.number_of_hospitals)

    return {
        f"Hospital-{h}": patients
        for h, patients in enumerate(split)
    }

def split_patients(all_patients, config) -> dict[str, list[str]]:
    ids = [p['patient_id'] for p in all_patients]
    random.shuffle(ids)
    split = balanced_split(ids, config.number_of_hospitals)
    return {
        f'Hospital-{h}': patients
        for h, patients in enumerate(split)
    }

def schedule_trainings(experiment: str, simulator: Simulator, min_time: pd.Timestamp, max_time: pd.Timestamp) -> None:
    if experiment == 'RetrainAfterTime':
        PeriodicInferenceMonitor(
            simulator=simulator,
            inference_interval_days=simulator.config.drift_inference_interval_days,
        )
        current_time = min_time + pd.DateOffset(months=simulator.config.drift_bootstrap_months)
        months_step = 3
        while current_time < max_time:
            train_event = Event(
                time=current_time,
                priority=1,
                event_type='TRAIN',
                payload={},
            )
            simulator.schedule_event(train_event)
            current_time = current_time + pd.DateOffset(months=months_step)
    elif experiment == 'RetrainAfterPerformanceDrift':
        PerformanceDriftMonitor(
            simulator=simulator,
            bootstrap_months=config.drift_bootstrap_months,
            inference_interval_days=config.drift_inference_interval_days,
            retraining_delay_days=config.drift_retraining_delay_days,
            metric_name=config.drift_metric_name,
            degradation_threshold=config.drift_degradation_threshold,
            degraded_dt_fraction_threshold=config.degraded_dt_fraction_threshold,
            metric_floor=config.drift_metric_floor,
            min_comparable_dts=config.drift_min_comparable_dts,
            threshold_mode=config.drift_threshold_mode,
            higher_is_worse=config.drift_higher_is_worse,
        )
    elif experiment in {
        'ADWINErrorDecentralizedRetrainingPolicy',
        'ADWINErrorHierarchicalRetrainingPolicy',
    }:
        ADWINErrorDecentralizedRetrainingMonitor(
            simulator=simulator,
            bootstrap_months=config.drift_bootstrap_months,
            inference_interval_days=config.adwin_inference_interval_days,
            retraining_delay_days=config.drift_retraining_delay_days,
            delta=config.adwin_delta,
            drifted_dt_fraction_threshold=config.adwin_drifted_dt_fraction_threshold,
            min_comparable_dts=config.adwin_min_evaluated_dts,
            drifted_dta_fraction_threshold=config.adwin_drifted_dta_fraction_threshold,
            min_comparable_dtas=config.adwin_min_evaluated_dtas,
            reset_after_retrain=config.adwin_reset_after_retrain,
        )
    else:
        raise ValueError(f'Unsupported experiment: {experiment}')

def load_patients(data_folder: str) -> tuple[list[dict], pd.Timestamp, pd.Timestamp]:
    patients = []
    global_min, global_max = None, None

    for patient in sorted(glob.glob(f'{data_folder}/*.csv')):
        df = pd.read_csv(patient)
        df['timestamp'] = pd.to_datetime(
            df['Measurement_date'] + ' ' + df['Measurement_time']
        )
        patient_id = df['Patient_ID'].iloc[0]
        ts = df['timestamp']
        min_time, max_time = ts.min(), ts.max()
        patients.append({'data': df, 'patient_id': patient_id, 'min_time': min_time, 'max_time': max_time})
        if global_min is None and global_max is None:
            global_min = min_time
            global_max = max_time
        else:
            if min_time < global_min:
                global_min = min_time
            if max_time > global_max:
                global_max = max_time
    return patients, global_min, global_max


def export_hospital_patient_mapping(
    mapping_dtas_dts: dict[str, list[str]],
    experiment: str,
    config,
    seed: int,
) -> None:
    rows = [
        {'hospital_id': hospital_id, 'patient_id': patient_id}
        for hospital_id, patient_ids in mapping_dtas_dts.items()
        for patient_id in patient_ids
    ]
    Path(f'{config.data_export_path}/{experiment}').mkdir(parents=True, exist_ok=True)
    mapping_df = pd.DataFrame(rows).sort_values(['hospital_id', 'patient_id'])
    mapping_df.to_csv(
        f'{config.data_export_path}/{experiment}/hospital_patient_mapping_seed_{seed}.csv',
        index=False,
    )


def run_simulation(seed: int, experiment: str, config, data_folder: str, iid: bool) -> None:
    seed_everything(seed)
    all_patients, min_time, max_time = load_patients(data_folder)

    print(f'Found {len(all_patients)} patients')
    print(f'Min: {min_time}, Max: {max_time}')

    if iid:
        mapping_dtas_dts = split_patients(all_patients, config)
    else:
        mapping_dtas_dts = split_patients_label_skew(all_patients, config, data_folder)

    export_hospital_patient_mapping(mapping_dtas_dts, experiment, config, seed)

    print('================================ Mapping hospitals-patients ================================')
    for h_id, patients in mapping_dtas_dts.items():
        print(f'Hospital ID: {h_id} has {len(patients)} patients')

    simulator = Simulator(data_folder, experiment, min_time, max_time, config, seed, mapping_dtas_dts)

    # Schedule patients activation and deactivation
    for patient in all_patients:
        event_active = Event(
            time=patient['min_time'],
            priority=0,
            event_type='PATIENT_BECOMES_ACTIVE',
            payload=patient,
        )

        event_inactive = Event(
            time=patient['max_time'],
            priority=0,
            event_type='PATIENT_BECOMES_INACTIVE',
            payload=patient,
        )
        simulator.schedule_event(event_active)
        simulator.schedule_event(event_inactive)

    # Schedule trainings and inferences
    schedule_trainings(experiment, simulator, min_time, max_time)

    simulator.start()


if __name__ == '__main__':

    config = LearningConfig()
    data_folder = 'T1DiabetesGranada/split-labeled'
    seeds = [0]
    experiments = ['RetrainAfterTime'] # ADWINErrorHierarchicalRetrainingPolicy
    iid = False

    for experiment in experiments:
        print(f'Running experiment {experiment}')
        exp_folder = f'{experiment}'
        Path(f'{config.data_export_path}/{exp_folder}').mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            run_simulation(seed, exp_folder, config, data_folder, iid)
