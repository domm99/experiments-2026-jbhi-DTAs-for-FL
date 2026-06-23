import glob
import random
import pandas as pd
from pathlib import Path
from utils import seed_everything
from simulator import Simulator, Event
from LearningConfig import LearningConfig
from Monitors import PeriodicInferenceMonitor

def balanced_split(xs, n):
    if n <= 0:
        raise ValueError("n must be greater than xs length")

    k, r = divmod(len(xs), n)

    return [
        xs[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)]
        for i in range(n)
    ]

def split_patients(all_patients, config) -> map[str, list[str]]:
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
        current_time = min_time
        i = 0
        months_step = 3
        while current_time < max_time:
            train_event = Event(
                time=min_time + pd.DateOffset(months=months_step*i),
                priority=1,
                event_type='TRAIN',
                payload={},
            )
            simulator.schedule_event(train_event)
            current_time = current_time + pd.DateOffset(months=months_step)
            i += 1


def load_patients(data_folder: str) -> tuple[list[dict], pd.Timestamp, pd.Timestamp]:
    patients = []
    global_min, global_max = None, None

    for patient in glob.glob(f'{data_folder}/*.csv'):
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


def run_simulation(seed: int, experiment: str, config) -> None:
    seed_everything(seed)
    all_patients, min_time, max_time = load_patients(data_folder)

    print(f'Found {len(all_patients)} patients')
    print(f'Min: {min_time}, Max: {max_time}')

    mapping_dtas_dts = split_patients(all_patients, config)

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
    experiments = ['RetrainAfterTime']

    for experiment in experiments:
        print(f'Running experiment {experiment}')
        exp_folder = f'{experiment}'
        Path(f'{config.data_export_path}/{exp_folder}').mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            run_simulation(seed, exp_folder, config)
