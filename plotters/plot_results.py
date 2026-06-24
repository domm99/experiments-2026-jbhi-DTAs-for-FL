import argparse
import glob
import os
import random
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib.pyplot as plt
import pandas as pd


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


def balanced_split(xs: list[str], n: int) -> list[list[str]]:
    if n <= 0:
        raise ValueError("number_of_hospitals must be greater than zero")

    k, r = divmod(len(xs), n)
    return [
        xs[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)]
        for i in range(n)
    ]


def build_patient_hospital_map(
    patient_data_dir: Path,
    seed: int,
    number_of_hospitals: int,
) -> dict[str, str]:
    patient_files = glob.glob(str(patient_data_dir / "*.csv"))
    if not patient_files:
        raise FileNotFoundError(f"No patient CSV files found in {patient_data_dir}")

    patient_ids: list[str] = []
    for patient_file in patient_files:
        patient_df = pd.read_csv(patient_file, usecols=["Patient_ID"])
        if patient_df.empty:
            continue
        patient_ids.append(str(patient_df["Patient_ID"].iloc[0]))

    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    patient_groups = balanced_split(patient_ids, number_of_hospitals)

    return {
        patient_id: f"Hospital-{hospital_index}"
        for hospital_index, patients in enumerate(patient_groups)
        for patient_id in patients
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


def read_test_results(
    results_dir: Path,
    patient_data_dir: Path,
    number_of_hospitals: int,
) -> pd.DataFrame:
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
            hospital_maps_by_seed[seed] = build_patient_hospital_map(
                patient_data_dir=patient_data_dir,
                seed=seed,
                number_of_hospitals=number_of_hospitals,
            )

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

    return test_df.sort_values(["timestamp", "hospital", "seed", "dt_id"]).reset_index(drop=True)


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
    return metric.replace("_", " ")


def plot_metric(df: pd.DataFrame, metric: str, output_path: Path, title: str) -> bool:
    plot_df = df[["timestamp", metric]].dropna().sort_values("timestamp")
    if plot_df.empty:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    marker = None if len(plot_df) > 80 else "o"
    ax.plot(
        plot_df["timestamp"],
        plot_df[metric],
        marker=marker,
        markersize=3,
        linewidth=1.8,
        color="#2563eb",
    )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(metric_label(metric))
    ax.grid(True, alpha=0.25)

    values = plot_df[metric].dropna()
    if not values.empty and values.min() >= 0 and values.max() <= 1:
        ax.set_ylim(-0.02, 1.02)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def export_phase_charts(
    phase: str,
    df_by_hospital: pd.DataFrame,
    df_mean: pd.DataFrame,
    metric_columns: list[str],
    charts_dir: Path,
) -> int:
    chart_count = 0
    mean_dir = charts_dir / phase / "mean"
    by_hospital_dir = charts_dir / phase / "by_hospital"

    for directory in (mean_dir, by_hospital_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for metric in metric_columns:
        if plot_metric(
            df=df_mean,
            metric=metric,
            output_path=mean_dir / f"{safe_filename(metric)}.png",
            title=f"{phase.title()} mean - {metric_label(metric)}",
        ):
            chart_count += 1

    for hospital, hospital_df in df_by_hospital.groupby("hospital"):
        hospital_dir = by_hospital_dir / safe_filename(str(hospital))
        for metric in metric_columns:
            if plot_metric(
                df=hospital_df,
                metric=metric,
                output_path=hospital_dir / f"{safe_filename(metric)}.png",
                title=f"{phase.title()} {hospital} - {metric_label(metric)}",
            ):
                chart_count += 1

    return chart_count


def export_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def run(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    charts_dir = Path(args.charts_dir)
    patient_data_dir = Path(args.patient_data_dir)

    data_dir = charts_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    training_df = read_training_results(results_dir)
    train_metrics = numeric_metric_columns(training_df)
    train_by_hospital = aggregate_by_hospital(training_df, train_metrics)
    train_mean = aggregate_mean(train_by_hospital, train_metrics)

    export_csv(training_df, data_dir / "train_last_global_round.csv")
    export_csv(train_by_hospital, data_dir / "train_by_hospital.csv")
    export_csv(train_mean, data_dir / "train_mean.csv")
    train_chart_count = export_phase_charts(
        phase="train",
        df_by_hospital=train_by_hospital,
        df_mean=train_mean,
        metric_columns=train_metrics,
        charts_dir=charts_dir,
    )

    test_df = read_test_results(
        results_dir=results_dir,
        patient_data_dir=patient_data_dir,
        number_of_hospitals=args.number_of_hospitals,
    )
    test_metrics = numeric_metric_columns(test_df)
    test_by_hospital = aggregate_by_hospital(test_df, test_metrics)
    test_mean = aggregate_mean(test_by_hospital, test_metrics)

    export_csv(test_df, data_dir / "test_evaluated_rows.csv")
    export_csv(test_by_hospital, data_dir / "test_by_hospital.csv")
    export_csv(test_mean, data_dir / "test_mean.csv")
    test_chart_count = export_phase_charts(
        phase="test",
        df_by_hospital=test_by_hospital,
        df_mean=test_mean,
        metric_columns=test_metrics,
        charts_dir=charts_dir,
    )

    print(f"Training rows at last global round: {len(training_df)}")
    print(f"Training metrics plotted: {len(train_metrics)}")
    print(f"Training charts written: {train_chart_count}")
    print(f"Test evaluated rows: {len(test_df)}")
    print(f"Test metrics plotted: {len(test_metrics)}")
    print(f"Test charts written: {test_chart_count}")
    print(f"Charts directory: {charts_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training and test metrics exported by the FL experiment."
    )
    parser.add_argument("--results-dir", default="data/RetrainAfterTime")
    parser.add_argument("--charts-dir", default="charts")
    parser.add_argument("--patient-data-dir", default="T1DiabetesGranada/split-labeled")
    parser.add_argument("--number-of-hospitals", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
