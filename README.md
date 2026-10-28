# Digital Twin Aggregates for hierarchical and adaptive federated learning

Simulation code for the IEEE JBHI paper *Digital Twin Aggregates for Hierarchical
and Adaptive Collaborative Learning in Smart-Healthcare*. Patient-level DTs run
inference and monitor the error of the deployed model, hospital-level DTAs act
as federated learning clients, and an FL coordinator triggers federated
retraining (FedProx, optional local personalization) with a time-based or an
ADWIN-based hierarchical quorum policy.

## Requirements

Python 3.12 and [Poetry](https://python-poetry.org/):

```bash
poetry install
```

## Dataset

The T1DiabetesGranada dataset is not redistributed. Once access has been
granted, place the original files in `T1DiabetesGranada/original/` and build the
per-patient labeled files used by the simulator:

```bash
poetry run python src/preprocess.py --dataset-dir T1DiabetesGranada
```

This writes `T1DiabetesGranada/split/` and `T1DiabetesGranada/split-labeled/`.
The label of each measurement (`target_any_within_t`) is `hypo` if one of the
next 4 measurements (60 min) is below 70 mg/dL, otherwise `hyper` if one is
above 180 mg/dL, otherwise `normal`.

## Running an experiment

Every experiment is described by a TOML file with `LearningConfig` values
(see `src/LearningConfig.py` for all the options and their defaults):

```bash
poetry run python src/main.py --config configs/adwin-fedpers-noniid.toml
# values can be overridden from the command line (TOML syntax)
poetry run python src/main.py --config configs/adwin-fedpers-noniid.toml --set seeds=[0] --set mu=0.1
```

| Config | Setting |
| --- | --- |
| `configs/adwin-fedpers-noniid.toml` | ADWIN policy, FedProx + personalization, non-IID |
| `configs/adwin-federated-noniid.toml` | ADWIN policy, FedProx without personalization, non-IID |
| `configs/adwin-lowdata-fedpers.toml` | quantity skew (two hospitals with 5 patients), FedProx + personalization |
| `configs/adwin-lowdata-local.toml` | quantity skew, purely local learning with the same training budget |

Main options:

- `experiment`: `ADWINErrorHierarchicalRetrainingPolicy`, `RetrainAfterTime`
  (every `time_based_interval_months`) or `RetrainAfterPerformanceDrift`;
- `iid`, `hospital_patient_caps`, `number_of_hospitals`: partitioning;
- `mu` (FedProx, 0 = FedAvg), `fl_global_rounds`, `fl_local_epochs`,
  `personalization`, `personalization_epochs`;
- `adwin_delta`, `adwin_drifted_dt_fraction_threshold` (intra-institution
  quorum *p*), `adwin_drifted_dta_fraction_threshold` (inter-institution quorum
  *q*), `adwin_min_evaluated_dts`, `adwin_one_sided`, `adwin_sticky_dt_flags`;
- `only_local_learning`, `local_epochs`, `local_mu`, `local_retrain_trigger`:
  purely local baseline.

## Outputs

Results are written to `data/<experiment folder>/`, whose name is derived from
the configuration unless `experiment_name` is set. A run refuses to write into
a non-empty folder (use `--set overwrite=true` to replace it), so results of
different runs are never mixed. The folder contains:

- `config.json`: the full configuration and the git commit used;
- `hospital_patient_mapping_seed_<s>.csv`: patients of each hospital;
- `test_<time>-seed_<s>.csv`: one row per DT and inference event, with
  accuracy, macro precision/recall/F1 (over the classes that occur) and the
  confusion matrix (`cm_<true>_<pred>`);
- `training_<time>-hospital_<h>-<GR_k|PERS>-seed_<s>.csv`: training logs;
- `retraining_log-seed_<s>.csv`: training work of every retraining event
  (per DTA: training windows and epochs per phase);
- `ADWINErrorHierarchicalRetrainingPolicy_*.csv`: drift detections, DTA
  signals and retraining decisions of the ADWIN policy.

## Docker

```bash
docker compose up --build adwin-fedpers-noniid
```

The dataset and the results folder are mounted as volumes (see
`docker-compose.yml`).

## Tests

```bash
poetry run python -m unittest discover -s tests -t .
```
