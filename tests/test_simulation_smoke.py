"""
End-to-end smoke tests of the simulator on a small synthetic cohort.
Run from the repository root with: python -m unittest discover -s tests -t .
"""
import glob
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests import _path  # noqa: F401

from LearningConfig import effective_local_epochs, experiment_folder_name, load_config, save_config
from main import prepare_output_dir, run_simulation
from preprocess import label_patient


def write_synthetic_cohort(directory: Path, patients: int = 9, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp('2018-01-01 08:00:00')
    for index in range(patients):
        start = base + pd.Timedelta(days=int(rng.integers(0, 60)))
        days = int(rng.integers(90, 170))
        timestamps = pd.date_range(start, periods=days * 96, freq='15min')
        level = 110 + 15 * index
        walk = np.cumsum(rng.normal(0, 4, len(timestamps)))
        walk -= np.linspace(0, walk[-1], len(timestamps))
        daily = 40 * np.sin(np.arange(len(timestamps)) * 2 * np.pi / 96)
        glucose = np.clip(level + daily + walk, 40, 400).round().astype(int)
        df = pd.DataFrame({
            'Patient_ID': f'SYN{index:03d}',
            'Measurement_date': timestamps.strftime('%Y-%m-%d'),
            'Measurement_time': timestamps.strftime('%H:%M:%S'),
            'Measurement': glucose,
        })
        label_patient(df).to_csv(directory / f'SYN{index:03d}.csv', index=False)


class SimulationSmokeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.data_dir = cls.root / 'patients'
        cls.data_dir.mkdir()
        write_synthetic_cohort(cls.data_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def run_experiment(self, **values):
        overrides = {
            'data_folder': str(self.data_dir),
            'data_export_path': str(self.root / 'results'),
            'number_of_hospitals': 3,
            'hidden_size': 8,
            'fl_global_rounds': 2,
            'fl_local_epochs': 1,
            'personalization_epochs': 1,
            'drift_bootstrap_months': 1,
            'time_based_interval_months': 1,
            'adwin_delta': 0.5,
            'adwin_min_evaluated_dts': 1,
            'adwin_drifted_dt_fraction_threshold': 0.1,
            'adwin_drifted_dta_fraction_threshold': 0.3,
            **values,
        }
        config = load_config(overrides=[f'{key}={self.toml(value)}' for key, value in overrides.items()])
        output_dir = Path(config.data_export_path) / experiment_folder_name(config)
        prepare_output_dir(output_dir, overwrite=True)
        save_config(config, output_dir)
        run_simulation(0, experiment_folder_name(config), config)
        return config, output_dir

    @staticmethod
    def toml(value):
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, dict):
            return '{ ' + ', '.join(f'{key} = {value[key]}' for key in value) + ' }'
        return str(value)

    def read_tests(self, output_dir: Path) -> pd.DataFrame:
        frames = []
        for path in sorted(glob.glob(str(output_dir / 'test_*.csv'))):
            frame = pd.read_csv(path)
            frame['timestamp'] = pd.Timestamp(Path(path).name[len('test_'):-len('-seed_0.csv')])
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def assert_common_outputs(self, output_dir: Path) -> pd.DataFrame:
        self.assertTrue((output_dir / 'config.json').exists())
        self.assertTrue((output_dir / 'retraining_log-seed_0.csv').exists())
        tests = self.read_tests(output_dir)
        # Every DT appears at most once per inference event
        self.assertFalse(tests.duplicated(['timestamp', 'dt_id']).any())
        self.assertGreater((tests['status'] == 'evaluated').sum(), 0)
        evaluated = tests[tests['status'] == 'evaluated']
        self.assertTrue(evaluated['precision'].between(0, 1).all())
        return tests

    def test_adwin_federated_with_personalization(self):
        config, output_dir = self.run_experiment(
            experiment='ADWINErrorHierarchicalRetrainingPolicy', personalization=True,
        )
        tests = self.assert_common_outputs(output_dir)
        # The shared model reaches every DT, also in DTAs that had no data at the first training
        self.assertEqual((tests['status'] == 'skipped_no_model').sum(), 0)
        log = pd.read_csv(output_dir / 'retraining_log-seed_0.csv')
        trained = log[log['trained']]
        self.assertTrue((trained['federated_epochs'] <= config.fl_global_rounds * config.fl_local_epochs).all())
        self.assertTrue((trained['personalization_epochs'] == config.personalization_epochs).all())
        self.assertTrue(list(output_dir.glob('training_*-PERS-seed_0.csv')))

    def test_time_based_federated(self):
        _, output_dir = self.run_experiment(experiment='RetrainAfterTime')
        tests = self.assert_common_outputs(output_dir)
        self.assertEqual((tests['status'] == 'skipped_no_model').sum(), 0)
        log = pd.read_csv(output_dir / 'retraining_log-seed_0.csv')
        self.assertGreater(log['retraining_index'].nunique(), 1)

    def test_adwin_local_only_per_dta(self):
        config, output_dir = self.run_experiment(
            experiment='ADWINErrorHierarchicalRetrainingPolicy',
            only_local_learning=True,
            local_retrain_trigger='dta',
            iid=True,
            hospital_patient_caps={'Hospital-0': 1},
        )
        self.assert_common_outputs(output_dir)
        mapping = pd.read_csv(output_dir / 'hospital_patient_mapping_seed_0.csv')
        self.assertEqual((mapping['hospital_id'] == 'Hospital-0').sum(), 1)
        self.assertEqual(len(mapping), 9)
        log = pd.read_csv(output_dir / 'retraining_log-seed_0.csv')
        trained = log[log['trained']]
        self.assertTrue((trained['local_epochs'] == effective_local_epochs(config)).all())
        self.assertTrue((trained['federated_epochs'] == 0).all())

    def test_refuses_non_empty_output_folder(self):
        output_dir = self.root / 'results' / 'not-empty'
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'test_old.csv').write_text('x\n1\n')
        with self.assertRaises(RuntimeError):
            prepare_output_dir(output_dir, overwrite=False)
        prepare_output_dir(output_dir, overwrite=True)
        self.assertEqual(list(output_dir.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
