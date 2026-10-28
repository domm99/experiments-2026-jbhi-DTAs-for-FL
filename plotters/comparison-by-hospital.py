import argparse
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# =============================================================================
# EDITA QUI: lista degli esperimenti da confrontare
# =============================================================================
EXPERIMENTS = [
    "ADWINErrorHierarchicalRetrainingPolicy-FedPers-NonIID",
    "RetrainAfterTime-FedProx-NonIID"
    #"ADWINErrorHierarchicalRetrainingPolicy-Federated-IID"
    #"ADWINErrorHierarchicalRetrainingPolicy-Federated-NonIID"
    #"ADWINErrorHierarchicalRetrainingPolicy-LowDataHospitals-FedPers-IID",
    #"ADWINErrorHierarchicalRetrainingPolicy-LowDataHospitals-Local-IID",
    # "ADWINErrorHierarchicalRetrainingPolicy-LowDataHospitals-Federated-IID",
]

DATA_ROOT = Path("data")
CHARTS_ROOT = Path("charts/")
COMPARISON_NAME = "AdwinVSTimed"


# =============================================================================
# EDITA QUI: limiti asse x e campionamento
# =============================================================================

# Se None, non imposta limite sinistro.
TIME_STEP_XLIM_LEFT = 0

# Se None, non imposta limite destro.
# Attenzione: questo limite è sul time step DOPO il campionamento.
# Esempio: se hai 1400 punti e TIME_STEP_SAMPLE_EVERY = 3,
# il massimo time step sarà circa 466.
TIME_STEP_XLIM_RIGHT = 300

# Campionamento.
# 1 = usa tutti i punti
# 3 = usa un punto ogni 3, e poi reindicizza l'asse x: 0, 1, 2, ...
# 10 = usa un punto ogni 10, e poi reindicizza l'asse x: 0, 1, 2, ...
TIME_STEP_SAMPLE_EVERY = 3


TRAINING_FILE_RE = re.compile(
    r"^training_(?P<timestamp>.+)-hospital_(?P<hospital>.+?)-GR_(?P<global_round>\d+)-seed_(?P<seed>\d+)\.csv$"
)
TEST_FILE_RE = re.compile(r"^test_(?P<timestamp>.+)-seed_(?P<seed>\d+)\.csv$")

METADATA_COLUMNS = {
    "timestamp",
    "hospital",
    "global_round",
    "seed",
    "source_file",
    "epoch",
    "dt_id",
    "status",
    "num_points",
    "prediction_count",
    "prediction_correct_count",
    "prediction_error_count",
}


def load_patient_hospital_map(results_dir: Path, seed: int) -> dict[str, str]:
    mapping_path = results_dir / f"hospital_patient_mapping_seed_{seed}.csv"

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Hospital-patient mapping not found for seed {seed}: {mapping_path}"
        )

    mapping_df = pd.read_csv(mapping_path)
    required_columns = {"hospital_id", "patient_id"}
    missing_columns = required_columns - set(mapping_df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{mapping_path} is missing required column(s): {missing}")

    if mapping_df.empty:
        raise ValueError(f"{mapping_path} is empty")

    duplicated_patients = mapping_df["patient_id"].astype(str).duplicated(keep=False)

    if duplicated_patients.any():
        duplicates = sorted(
            mapping_df.loc[duplicated_patients, "patient_id"].astype(str).unique()
        )
        preview = ", ".join(duplicates[:8])
        raise ValueError(
            f"{mapping_path} contains duplicated patient_id values. "
            f"First duplicates: {preview}"
        )

    return {
        str(row.patient_id): str(row.hospital_id)
        for row in mapping_df[["patient_id", "hospital_id"]].itertuples(index=False)
    }


def parse_timestamp(value: str) -> pd.Timestamp:
    return pd.to_datetime(value, errors="raise")


def read_training_results(results_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    for path in sorted(results_dir.glob("training_*.csv")):
        match = TRAINING_FILE_RE.match(path.name)

        if match is None:
            continue

        metrics = pd.read_csv(path)

        if metrics.empty:
            continue

        if "epoch" in metrics.columns:
            selected_row = metrics.sort_values("epoch").iloc[-1].to_dict()
        else:
            selected_row = metrics.iloc[-1].to_dict()

        selected_row.update(
            {
                "timestamp": parse_timestamp(match.group("timestamp")),
                "hospital": match.group("hospital"),
                "global_round": int(match.group("global_round")),
                "seed": int(match.group("seed")),
                "source_file": str(path),
            }
        )

        rows.append(selected_row)

    if not rows:
        return pd.DataFrame()

    training_df = pd.DataFrame(rows)

    idx = (
        training_df.groupby(["timestamp", "hospital", "seed"])["global_round"]
        .idxmax()
        .dropna()
    )

    return (
        training_df.loc[idx]
        .sort_values(["timestamp", "hospital", "seed"])
        .reset_index(drop=True)
    )


def read_test_results(results_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    hospital_maps_by_seed: dict[int, dict[str, str]] = {}

    for path in sorted(results_dir.glob("test_*.csv")):
        match = TEST_FILE_RE.match(path.name)

        if match is None:
            continue

        seed = int(match.group("seed"))
        metrics = pd.read_csv(path)

        if metrics.empty:
            continue

        if seed not in hospital_maps_by_seed:
            hospital_maps_by_seed[seed] = load_patient_hospital_map(results_dir, seed)

        metrics["timestamp"] = parse_timestamp(match.group("timestamp"))
        metrics["seed"] = seed
        metrics["source_file"] = str(path)
        metrics["hospital"] = metrics["dt_id"].astype(str).map(hospital_maps_by_seed[seed])

        frames.append(metrics)

    if not frames:
        return pd.DataFrame()

    test_df = pd.concat(frames, ignore_index=True)

    missing_hospital = test_df["hospital"].isna()

    if missing_hospital.any():
        missing_ids = sorted(test_df.loc[missing_hospital, "dt_id"].astype(str).unique())
        preview = ", ".join(missing_ids[:8])
        raise ValueError(
            f"Could not map {len(missing_ids)} dt_id values to hospitals. "
            f"First missing ids: {preview}"
        )

    if "status" in test_df.columns:
        test_df = test_df[test_df["status"].eq("evaluated")].copy()

    return (
        test_df.sort_values(["timestamp", "hospital", "seed", "dt_id"])
        .reset_index(drop=True)
    )


def numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    metric_columns: list[str] = []

    for column in df.columns:
        if column in METADATA_COLUMNS:
            continue

        numeric_values = pd.to_numeric(df[column], errors="coerce")

        if numeric_values.notna().any():
            df[column] = numeric_values
            metric_columns.append(column)

    return metric_columns


def aggregate_by_hospital(df: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "hospital", *metric_columns])

    return (
        df.groupby(["timestamp", "hospital"], as_index=False)[metric_columns]
        .mean()
        .sort_values(["timestamp", "hospital"])
        .reset_index(drop=True)
    )


def aggregate_mean(df_by_hospital: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    if df_by_hospital.empty:
        return pd.DataFrame(columns=["timestamp", *metric_columns])

    return (
        df_by_hospital.groupby("timestamp", as_index=False)[metric_columns]
        .mean()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "metric"


def metric_label(metric: str) -> str:
    return metric.replace("_", " ").title()


def add_experiment_column(df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    df = df.copy()
    df["experiment"] = experiment_name
    return df


def validate_time_step_sample_every() -> None:
    if not isinstance(TIME_STEP_SAMPLE_EVERY, int):
        raise TypeError("TIME_STEP_SAMPLE_EVERY must be an integer")

    if TIME_STEP_SAMPLE_EVERY < 1:
        raise ValueError("TIME_STEP_SAMPLE_EVERY must be >= 1")


def rolling_smooth(
    df: pd.DataFrame,
    metric: str,
    smooth_window: int,
    smooth_min_periods: int,
) -> pd.DataFrame:
    plot_df = df[["timestamp", metric]].dropna().sort_values("timestamp").copy()

    if plot_df.empty:
        return plot_df

    # Rolling sul segnale completo, prima del campionamento.
    plot_df["smoothed"] = (
        plot_df[metric]
        .rolling(window=smooth_window, min_periods=smooth_min_periods)
        .mean()
    )

    plot_df = plot_df.dropna(subset=["smoothed"]).copy()

    # Campionamento: tiene un punto ogni TIME_STEP_SAMPLE_EVERY.
    # Se TIME_STEP_SAMPLE_EVERY = 3, tiene circa 1/3 dei punti.
    plot_df = plot_df.iloc[::TIME_STEP_SAMPLE_EVERY].copy()

    # IMPORTANTISSIMO:
    # il time_step viene creato DOPO il campionamento.
    # Quindi se prima avevi 1400 punti e campioni ogni 3,
    # ora l'asse x arriva circa a 466, non più a 1400.
    plot_df["time_step"] = range(len(plot_df))

    return plot_df


def apply_metric_ylim(ax, plotted_values: list[pd.Series]) -> None:
    if not plotted_values:
        return

    values = pd.concat(plotted_values).dropna()

    if not values.empty and values.min() >= 0 and values.max() <= 1:
        ax.set_ylim(-0.02, 1.02)


def beautify_exp_name(experiment_name: str) -> str:
    if "FedPers" in experiment_name:
        return "ADWIN-Based"
    elif "FedProx" in experiment_name:
        return "RetrainAfterTime"
        # return "Federated w/ personalization"
    elif "Local" in experiment_name:
        return "Local"
    elif "Federated" in experiment_name:
        return "IID"
    else:
        raise ValueError(f"Unknown experiment name: {experiment_name}")


def plot_smoothed_comparison(
    experiment_dfs: dict[str, pd.DataFrame],
    metric: str,
    output_path: Path,
    title: str,
    smooth_window: int,
    smooth_min_periods: int,
    palette: list,
) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.8))

    plotted_values: list[pd.Series] = []
    plotted_anything = False

    for index, (experiment_name, df) in enumerate(experiment_dfs.items()):
        if df.empty or metric not in df.columns:
            continue

        smooth_df = rolling_smooth(
            df=df,
            metric=metric,
            smooth_window=smooth_window,
            smooth_min_periods=smooth_min_periods,
        )

        if smooth_df.empty:
            continue

        color = palette[index % len(palette)]

        ax.plot(
            smooth_df["time_step"],
            smooth_df["smoothed"],
            linewidth=3,
            color=color,
            label=beautify_exp_name(experiment_name),
        )

        plotted_values.append(smooth_df["smoothed"])
        plotted_anything = True

    if not plotted_anything:
        plt.close(fig)
        return False

    # ax.set_title(title)

    ax.set_xlabel("Time Step", fontsize=22)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    if TIME_STEP_XLIM_LEFT is not None:
        ax.set_xlim(left=TIME_STEP_XLIM_LEFT)

    if TIME_STEP_XLIM_RIGHT is not None:
        ax.set_xlim(right=TIME_STEP_XLIM_RIGHT)

    ax.set_ylabel(metric_label(metric), fontsize=22)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, alpha=0.25)

    ax.legend(
        fontsize=20,
        title_fontsize=20,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        #ncol=len(experiment_dfs),
        ncol=1,
        frameon=True,
    )

    apply_metric_ylim(ax, plotted_values)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return True


def load_experiment_data(experiment_name: str) -> dict[str, pd.DataFrame | list[str]]:
    results_dir = DATA_ROOT / experiment_name

    if not results_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {results_dir}")

    training_df = read_training_results(results_dir)
    train_metrics = numeric_metric_columns(training_df)
    train_by_hospital = aggregate_by_hospital(training_df, train_metrics)
    train_mean = aggregate_mean(train_by_hospital, train_metrics)

    test_df = read_test_results(results_dir)
    test_metrics = numeric_metric_columns(test_df)
    test_by_hospital = aggregate_by_hospital(test_df, test_metrics)
    test_mean = aggregate_mean(test_by_hospital, test_metrics)

    return {
        "train_metrics": train_metrics,
        "train_by_hospital": add_experiment_column(train_by_hospital, experiment_name),
        "train_mean": add_experiment_column(train_mean, experiment_name),
        "test_metrics": test_metrics,
        "test_by_hospital": add_experiment_column(test_by_hospital, experiment_name),
        "test_mean": add_experiment_column(test_mean, experiment_name),
    }


def collect_all_metrics(
    loaded: dict[str, dict[str, pd.DataFrame | list[str]]],
    key: str,
) -> list[str]:
    all_metrics: set[str] = set()

    for data in loaded.values():
        metrics = data[key]

        if isinstance(metrics, list):
            all_metrics.update(metrics)

    return sorted(all_metrics)


def build_mean_dfs(
    loaded: dict[str, dict[str, pd.DataFrame | list[str]]],
    phase: str,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}

    key = f"{phase}_mean"

    for experiment_name, data in loaded.items():
        df = data[key]

        if isinstance(df, pd.DataFrame):
            result[experiment_name] = df

    return result


def build_hospital_dfs(
    loaded: dict[str, dict[str, pd.DataFrame | list[str]]],
    phase: str,
    hospital: str,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}

    key = f"{phase}_by_hospital"

    for experiment_name, data in loaded.items():
        df = data[key]

        if not isinstance(df, pd.DataFrame):
            continue

        if df.empty or "hospital" not in df.columns:
            continue

        hospital_df = df[df["hospital"].astype(str).eq(str(hospital))].copy()

        if not hospital_df.empty:
            result[experiment_name] = hospital_df

    return result


def collect_hospitals(
    loaded: dict[str, dict[str, pd.DataFrame | list[str]]],
    phase: str,
) -> list[str]:
    hospitals: set[str] = set()

    key = f"{phase}_by_hospital"

    for data in loaded.values():
        df = data[key]

        if isinstance(df, pd.DataFrame) and not df.empty and "hospital" in df.columns:
            hospitals.update(df["hospital"].astype(str).dropna().unique())

    return sorted(hospitals)


def export_comparison_charts(
    loaded: dict[str, dict[str, pd.DataFrame | list[str]]],
    output_dir: Path,
    smooth_window: int,
    smooth_min_periods: int,
) -> int:
    chart_count = 0

    palette = sns.color_palette("deep", n_colors=max(len(loaded), 1))

    for phase in ["train", "test"]:
        metric_key = f"{phase}_metrics"
        metrics = collect_all_metrics(loaded, metric_key)

        mean_dir = output_dir / phase / "mean"
        by_hospital_dir = output_dir / phase / "by_hospital"

        mean_dfs = build_mean_dfs(loaded, phase)

        for metric in metrics:
            written = plot_smoothed_comparison(
                experiment_dfs=mean_dfs,
                metric=metric,
                output_path=mean_dir / f"{safe_filename(metric)}_comparison_smoothed_time_step.png",
                title=f"{phase.title()} mean - {metric_label(metric)}",
                smooth_window=smooth_window,
                smooth_min_periods=smooth_min_periods,
                palette=palette,
            )

            if written:
                chart_count += 1

        hospitals = collect_hospitals(loaded, phase)

        for hospital in hospitals:
            hospital_dfs = build_hospital_dfs(loaded, phase, hospital)
            hospital_dir = by_hospital_dir / safe_filename(str(hospital))

            for metric in metrics:
                written = plot_smoothed_comparison(
                    experiment_dfs=hospital_dfs,
                    metric=metric,
                    output_path=hospital_dir / f"{safe_filename(metric)}_comparison_smoothed_time_step.png",
                    title=f"{phase.title()} {hospital} - {metric_label(metric)}",
                    smooth_window=smooth_window,
                    smooth_min_periods=smooth_min_periods,
                    palette=palette,
                )

                if written:
                    chart_count += 1

    return chart_count


def run(args: argparse.Namespace) -> None:
    validate_time_step_sample_every()

    if not EXPERIMENTS:
        raise ValueError("EXPERIMENTS is empty. Add at least one experiment name.")

    loaded: dict[str, dict[str, pd.DataFrame | list[str]]] = {}

    for experiment_name in EXPERIMENTS:
        print(f"Loading experiment: {experiment_name}")
        loaded[experiment_name] = load_experiment_data(experiment_name)

    output_dir = CHARTS_ROOT / COMPARISON_NAME

    chart_count = export_comparison_charts(
        loaded=loaded,
        output_dir=output_dir,
        smooth_window=args.smooth_window,
        smooth_min_periods=args.smooth_min_periods,
    )

    print()
    print(f"Experiments compared: {len(EXPERIMENTS)}")
    print(f"Charts written: {chart_count}")
    print(f"Rolling smooth window: {args.smooth_window}")
    print(f"Rolling smooth min periods: {args.smooth_min_periods}")
    print("X-axis: Time Step")
    print(f"Time step xlim left: {TIME_STEP_XLIM_LEFT}")
    print(f"Time step xlim right: {TIME_STEP_XLIM_RIGHT}")
    print(f"Time step sample every: {TIME_STEP_SAMPLE_EVERY}")
    print(f"Output directory: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare smoothed training/test metrics across multiple FL experiments using time step on x-axis."
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=30,
        help="Rolling window size used for smoothed charts, in number of observations.",
    )

    parser.add_argument(
        "--smooth-min-periods",
        type=int,
        default=5,
        help="Minimum number of observations required to compute a smoothed value.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())