"""
Builds the per-patient files used by the simulator from the original
T1DiabetesGranada glucose measurements:

    original/Glucose_measurements.csv
        -> split/<Patient_ID>.csv           (rows of each patient, original order)
        -> split-labeled/<Patient_ID>.csv   (with the early-warning labels)

Labels (Eq. 4 of the paper), with a horizon of HORIZON_STEPS rows (4 x 15 min):
    hypo    if any of the next HORIZON_STEPS values is  < 70 mg/dL
    hyper   otherwise, if any of them is                > 180 mg/dL
    normal  otherwise
The last HORIZON_STEPS rows of each patient have no label (NaN). The horizon
is counted in rows, as in the original labeling, regardless of sensor gaps.

Usage:
    python src/preprocess.py --dataset-dir T1DiabetesGranada
    python src/preprocess.py --dataset-dir T1DiabetesGranada --patients LIB193263 LIB193669
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HYPO_THRESHOLD = 70
HYPER_THRESHOLD = 180
HORIZON_STEPS = 4

RAW_COLUMNS = ['Patient_ID', 'Measurement_date', 'Measurement_time', 'Measurement']
RAW_DTYPES = {'Patient_ID': str, 'Measurement_date': str, 'Measurement_time': str}


def glucose_class(values: pd.Series) -> pd.Series:
    classes = np.where(
        values < HYPO_THRESHOLD,
        'hypo',
        np.where(values > HYPER_THRESHOLD, 'hyper', 'normal'),
    )
    return pd.Series(classes, index=values.index, dtype=object).where(values.notna())


def label_patient(df: pd.DataFrame, horizon_steps: int = HORIZON_STEPS) -> pd.DataFrame:
    """Adds the label columns to the rows of one patient (in temporal order)."""
    labeled = df.copy().reset_index(drop=True)
    labeled['timestamp'] = labeled['Measurement_date'] + ' ' + labeled['Measurement_time']
    values = labeled['Measurement'].astype(float)

    labeled['current_class'] = glucose_class(values)
    future_value = values.shift(-horizon_steps)
    labeled['future_value_t'] = future_value
    labeled['target_point_t'] = glucose_class(future_value)

    future_hypo = pd.Series(False, index=labeled.index)
    future_hyper = pd.Series(False, index=labeled.index)
    for step in range(1, horizon_steps + 1):
        shifted = values.shift(-step)
        future_hypo |= shifted < HYPO_THRESHOLD
        future_hyper |= shifted > HYPER_THRESHOLD

    observable = future_value.notna()
    labeled['future_hypo_within_t'] = future_hypo.astype(float).where(observable)
    labeled['future_hyper_within_t'] = future_hyper.astype(float).where(observable)
    target = np.where(future_hypo, 'hypo', np.where(future_hyper, 'hyper', 'normal'))
    labeled['target_any_within_t'] = pd.Series(target, index=labeled.index, dtype=object).where(observable)
    return labeled


def split_measurements(
    measurements_path: Path,
    split_dir: Path,
    patients: set[str] | None = None,
    chunksize: int = 2_000_000,
) -> list[str]:
    """Writes the rows of each patient to split_dir/<Patient_ID>.csv, keeping the original order."""
    split_dir.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    for chunk in pd.read_csv(measurements_path, usecols=RAW_COLUMNS, dtype=RAW_DTYPES, chunksize=chunksize):
        if patients is not None:
            chunk = chunk[chunk['Patient_ID'].isin(patients)]
        for patient_id, rows in chunk.groupby('Patient_ID', sort=False):
            output_path = split_dir / f'{patient_id}.csv'
            rows[RAW_COLUMNS].to_csv(output_path, mode='a', header=patient_id not in written, index=False)
            written.add(patient_id)
    return sorted(written)


def label_split_files(split_dir: Path, labeled_dir: Path, patient_ids: list[str]) -> None:
    labeled_dir.mkdir(parents=True, exist_ok=True)
    for patient_id in patient_ids:
        df = pd.read_csv(split_dir / f'{patient_id}.csv', dtype=RAW_DTYPES)
        label_patient(df).to_csv(labeled_dir / f'{patient_id}.csv', index=False)


def ensure_empty(directory: Path, patient_ids: set[str] | None, overwrite: bool) -> None:
    if not directory.exists():
        return
    existing = [
        path for path in directory.glob('*.csv')
        if patient_ids is None or path.stem in patient_ids
    ]
    if existing and not overwrite:
        raise RuntimeError(f'{directory} already contains patient files; use --overwrite to replace them')
    for path in existing:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description='Split and label the T1DiabetesGranada glucose measurements.')
    parser.add_argument('--dataset-dir', default='T1DiabetesGranada', type=Path)
    parser.add_argument('--output-dir', default=None, type=Path, help='Defaults to --dataset-dir')
    parser.add_argument('--patients', nargs='*', default=None, help='Only process these Patient_IDs')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    output_dir = args.output_dir or args.dataset_dir
    split_dir = output_dir / 'split'
    labeled_dir = output_dir / 'split-labeled'
    patients = set(args.patients) if args.patients else None

    ensure_empty(split_dir, patients, args.overwrite)
    ensure_empty(labeled_dir, patients, args.overwrite)

    patient_ids = split_measurements(args.dataset_dir / 'original' / 'Glucose_measurements.csv', split_dir, patients)
    label_split_files(split_dir, labeled_dir, patient_ids)
    print(f'Processed {len(patient_ids)} patients into {split_dir} and {labeled_dir}')


if __name__ == '__main__':
    main()
