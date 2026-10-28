import json
import subprocess
import tomllib
import torch
from dataclasses import dataclass, field, fields
from pathlib import Path


def get_current_device():
    device: str = 'cpu'
    if torch.accelerator.is_available():
        current_accelerator = torch.accelerator.current_accelerator()
        if current_accelerator is not None:
            device = current_accelerator.type
    return torch.device(device)


SUPPORTED_EXPERIMENTS = (
    'RetrainAfterTime',
    'RetrainAfterPerformanceDrift',
    'ADWINErrorHierarchicalRetrainingPolicy',
    'ADWINErrorDecentralizedRetrainingPolicy',
)

# Fields that describe the runtime environment rather than the experiment.
NON_SERIALIZABLE_FIELDS = {'device'}


@dataclass
class LearningConfig:
    ## Experiment
    experiment: str = 'ADWINErrorHierarchicalRetrainingPolicy'
    # Name of the output folder inside data_export_path. When None it is derived
    # from the configuration (see experiment_folder_name).
    experiment_name: str | None = None
    data_folder: str = 'T1DiabetesGranada/split-labeled'
    data_export_path: str = 'data'
    # Remove the content of a non-empty output folder instead of refusing to run
    overwrite: bool = False
    seeds: list[int] = field(default_factory=lambda: [0])
    iid: bool = False
    only_local_learning: bool = False
    # e.g. {'Hospital-0': 5, 'Hospital-1': 5} for the quantity-skew setting
    hospital_patient_caps: dict[str, int] = field(default_factory=dict)
    # Keep the data of patients that are no longer active in the DTA training
    # set (Definition 1). False reproduces the old behaviour (active DTs only).
    retain_inactive_patient_data: bool = True

    ## Model and local training
    layers: int = 1
    stride: int = 12
    dropout: float = 0.0
    hidden_size: int = 64
    batch_size: int = 256
    # Fraction of each patient's history used for training. 1.0 trains on all the
    # observable data; < 1.0 keeps the most recent rows as a (logged only)
    # temporal validation split.
    train_ratio: float = 1.0
    sequence_length: int = 12
    learning_rate: float = 1e-3
    label_column: str = 'target_any_within_t'
    # Number of future rows the label looks at (4 x 15 min = 60 min): a label is
    # used for training/evaluation only once those rows have been observed.
    label_horizon_steps: int = 4

    ## Time-based policy
    time_based_interval_months: int = 3
    retrain_after_time_inference_interval_days: int = 1

    ## Performance-drift policy
    drift_bootstrap_months: int = 3
    drift_inference_interval_days: int = 3
    drift_retraining_delay_days: int = 1
    drift_metric_name: str = 'f1_score'
    drift_degradation_threshold: float = 0.1
    drift_metric_floor: float | None = None
    drift_min_comparable_dts: int = 5
    drift_threshold_mode: str = 'relative'
    drift_higher_is_worse: bool = False
    degraded_dt_fraction_threshold: float = 0.2

    ## ADWIN hierarchical policy
    adwin_delta: float = 0.1
    adwin_inference_interval_days: int = 1
    adwin_drifted_dt_fraction_threshold: float = 0.1
    adwin_min_evaluated_dts: int = 5
    adwin_drifted_dta_fraction_threshold: float = 0.1
    adwin_min_evaluated_dtas: int = 1
    adwin_reset_after_retrain: bool = True

    ## FL
    fl_global_rounds: int = 8
    fl_local_epochs: int = 3
    number_of_hospitals: int = 10
    # FedProx proximal coefficient (mu = 0 is FedAvg)
    mu: float = 0.7
    # Local fine-tuning of the aggregated model at each DTA after the federated
    # rounds; the personalized model is deployed only to the DTs of that DTA.
    personalization: bool = False
    personalization_epochs: int = 3
    # Proximal coefficient towards the global model during personalization
    # (None = same as mu)
    personalization_mu: float | None = None

    device: torch.device = field(default_factory=get_current_device)


def _parse_value(raw: str):
    """Parses a CLI override value with TOML syntax, falling back to a plain string."""
    try:
        return tomllib.loads(f'value = {raw}')['value']
    except tomllib.TOMLDecodeError:
        return raw


def validate_config(config: LearningConfig) -> None:
    if config.experiment not in SUPPORTED_EXPERIMENTS:
        raise ValueError(
            f'Unsupported experiment: {config.experiment}. '
            f'Supported: {", ".join(SUPPORTED_EXPERIMENTS)}'
        )
    if not config.seeds:
        raise ValueError('At least one seed is required')
    if config.number_of_hospitals <= 0:
        raise ValueError('number_of_hospitals must be greater than zero')
    if config.mu < 0:
        raise ValueError('mu must be non-negative')
    if not 0 < config.train_ratio <= 1:
        raise ValueError('train_ratio must be in (0, 1]')
    if config.label_horizon_steps < 0:
        raise ValueError('label_horizon_steps must be non-negative')


def load_config(path: str | None = None, overrides: list[str] | None = None) -> LearningConfig:
    """
    Builds a LearningConfig from an optional TOML file and optional KEY=VALUE
    overrides (values use TOML syntax, e.g. iid=true, seeds=[0,1,2],
    hospital_patient_caps={Hospital-0=5}).
    """
    values = {}
    if path is not None:
        with open(path, 'rb') as config_file:
            values.update(tomllib.load(config_file))

    for override in overrides or []:
        key, separator, raw_value = override.partition('=')
        if not separator:
            raise ValueError(f'Invalid override {override!r}: expected KEY=VALUE')
        values[key.strip()] = _parse_value(raw_value.strip())

    known_fields = {f.name for f in fields(LearningConfig)} - NON_SERIALIZABLE_FIELDS
    unknown_fields = set(values) - known_fields
    if unknown_fields:
        raise ValueError(f'Unknown configuration keys: {", ".join(sorted(unknown_fields))}')

    config = LearningConfig(**values)
    validate_config(config)
    return config


def experiment_folder_name(config: LearningConfig) -> str:
    if config.experiment_name:
        return config.experiment_name

    parts = [config.experiment]
    if config.hospital_patient_caps:
        parts.append('LowDataHospitals')
    if config.only_local_learning:
        parts.append('Local')
    else:
        parts.append('FedPers' if config.personalization else 'Federated')
    parts.append('IID' if config.iid else 'NonIID')
    if not config.only_local_learning:
        parts.append(f'mu{config.mu:g}')
    return '-'.join(parts)


def config_to_dict(config: LearningConfig) -> dict:
    return {
        f.name: getattr(config, f.name)
        for f in fields(config)
        if f.name not in NON_SERIALIZABLE_FIELDS
    }


def git_revision() -> dict:
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--untracked-files=no'],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return {'commit': commit, 'dirty': bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': 'unknown', 'dirty': None}


def save_config(config: LearningConfig, output_dir: Path) -> None:
    payload = {
        'config': config_to_dict(config),
        'git': git_revision(),
        'device': str(config.device),
    }
    (output_dir / 'config.json').write_text(json.dumps(payload, indent=2, default=str))
