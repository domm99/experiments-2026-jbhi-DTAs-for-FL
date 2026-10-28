import argparse
import glob
import random
import shutil
import pandas as pd
from pathlib import Path
from utils import seed_everything
from simulator import Simulator, Event
from LearningConfig import LearningConfig, experiment_folder_name, load_config, save_config
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

def cap_patients_for_hospitals(
    mapping_dtas_dts: dict[str, list[str]],
    hospital_patient_caps: dict[str, int],
) -> dict[str, list[str]]:
    if not hospital_patient_caps:
        return mapping_dtas_dts

    capped_hospitals = set(hospital_patient_caps)
    receiver_hospitals = [
        hospital_id for hospital_id in mapping_dtas_dts
        if hospital_id not in capped_hospitals
    ]
    if not receiver_hospitals:
        raise ValueError(
            'At least one uncapped hospital is required to receive extra patients'
        )

    updated_mapping = {
        hospital_id: list(patient_ids)
        for hospital_id, patient_ids in mapping_dtas_dts.items()
    }
    extra_patients: list[str] = []

    for hospital_id, patient_cap in hospital_patient_caps.items():
        if hospital_id not in updated_mapping:
            raise ValueError(f'Unknown hospital id: {hospital_id}')
        if patient_cap < 0:
            raise ValueError(f'Patient cap must be non-negative for {hospital_id}')

        patients = updated_mapping[hospital_id]
        updated_mapping[hospital_id] = patients[:patient_cap]
        extra_patients.extend(patients[patient_cap:])

    for index, patient_id in enumerate(extra_patients):
        receiver_id = receiver_hospitals[index % len(receiver_hospitals)]
        updated_mapping[receiver_id].append(patient_id)

    return updated_mapping

def schedule_trainings(simulator: Simulator, min_time: pd.Timestamp, max_time: pd.Timestamp) -> None:
    config = simulator.config
    if config.experiment == 'RetrainAfterTime':
        PeriodicInferenceMonitor(
            simulator=simulator,
            inference_interval_days=config.retrain_after_time_inference_interval_days,
        )
        current_time = min_time + pd.DateOffset(months=config.drift_bootstrap_months)
        while current_time < max_time:
            train_event = Event(
                time=current_time,
                priority=1,
                event_type='TRAIN',
                payload={'reason': 'time_based'},
            )
            simulator.schedule_event(train_event)
            current_time = current_time + pd.DateOffset(months=config.time_based_interval_months)
    elif config.experiment == 'RetrainAfterPerformanceDrift':
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
    elif config.experiment in {
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
        raise ValueError(f'Unsupported experiment: {config.experiment}')

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


def run_simulation(seed: int, exp_folder: str, config: LearningConfig) -> None:
    seed_everything(seed)
    data_folder = config.data_folder
    all_patients, min_time, max_time = load_patients(data_folder)

    print(f'Found {len(all_patients)} patients')
    print(f'Min: {min_time}, Max: {max_time}')

    if config.iid:
        mapping_dtas_dts = split_patients(all_patients, config)
    else:
        mapping_dtas_dts = split_patients_label_skew(all_patients, config, data_folder, label_col=config.label_column)

    mapping_dtas_dts = cap_patients_for_hospitals(
        mapping_dtas_dts,
        config.hospital_patient_caps,
    )

    export_hospital_patient_mapping(mapping_dtas_dts, exp_folder, config, seed)

    print('================================ Mapping hospitals-patients ================================')
    for h_id, patients in mapping_dtas_dts.items():
        print(f'Hospital ID: {h_id} has {len(patients)} patients')

    simulator = Simulator(
        data_folder,
        exp_folder,
        min_time,
        max_time,
        config,
        seed,
        mapping_dtas_dts,
        config.only_local_learning,
    )

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
    schedule_trainings(simulator, min_time, max_time)

    simulator.start()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """
    Makes sure that the results of a run are never mixed with those of a
    previous one: the output folder must be empty unless overwrite is set,
    in which case its content is removed.
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f'Output folder {output_dir} is not empty. Remove it, choose another '
                'experiment_name or run with --set overwrite=true.'
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the DT/DTA federated learning simulation.')
    parser.add_argument(
        '--config',
        default=None,
        help='TOML file with LearningConfig values (see configs/).',
    )
    parser.add_argument(
        '--set',
        dest='overrides',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='Override a configuration value using TOML syntax, e.g. --set iid=true. Can be repeated.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    exp_folder = experiment_folder_name(config)
    output_dir = Path(config.data_export_path) / exp_folder

    print(f'Running experiment {config.experiment} -> {output_dir}')
    prepare_output_dir(output_dir, config.overwrite)
    save_config(config, output_dir)

    for seed in config.seeds:
        run_simulation(seed, exp_folder, config)


if __name__ == '__main__':
    main()
