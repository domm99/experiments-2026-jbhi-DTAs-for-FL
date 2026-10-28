from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


OUTPUT_COLUMNS = [
    "latency_days",
]

DEFAULT_EXPERIMENTS = {
    "adwin": "ADWINErrorHierarchicalRetrainingPolicy-FedPers-NonIID",
    "time": "RetrainAfterTime-FedProx-NonIID",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one row per scheduled ADWIN/DDM retraining trigger, "
            "with one raw latency value for downstream plots."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        type=Path,
        help="Directory containing experiment export folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        type=Path,
        help="Directory where delay CSV files will be written.",
    )
    parser.add_argument(
        "--include-unscheduled",
        action="store_true",
        help="Also include drift decisions that did not schedule a retraining event.",
    )
    parser.add_argument(
        "--include-all-experiments",
        action="store_true",
        help="Include every ADWIN/DDM experiment variant instead of only the decentralized policies.",
    )
    return parser.parse_args()


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def days_between(start: str, end: str) -> float:
    return (parse_timestamp(end) - parse_timestamp(start)).total_seconds() / 86400


def is_scheduled(row: dict[str, str]) -> bool:
    return row.get("train_event_scheduled", "True").lower() == "true"


def iter_detector_files(
    data_dir: Path,
    detector: str,
    include_all_experiments: bool,
) -> list[Path]:
    files = []
    for path in data_dir.glob(f"*/{detector.upper()}*events-seed_*.csv"):
        if "_dt_drift_events-" in path.name:
            continue
        if detector.upper() not in path.name:
            continue
        if (
            not include_all_experiments
            and path.parent.name != DEFAULT_EXPERIMENTS[detector]
        ):
            continue
        files.append(path)
    return sorted(files)


def build_delay_rows(
    data_dir: Path,
    detector: str,
    include_unscheduled: bool,
    include_all_experiments: bool,
) -> list[dict[str, str]]:
    rows = []
    for source_file in iter_detector_files(data_dir, detector, include_all_experiments):
        with source_file.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for source_row in reader:
                if not include_unscheduled and not is_scheduled(source_row):
                    continue

                detection_column = (
                    "detection_time"
                    if "detection_time" in source_row
                    else "timestep"
                )
                detection_time = source_row[detection_column]
                last_training_time = source_row["last_training_time"]

                rows.append(
                    {
                        "latency_days": f"{days_between(last_training_time, detection_time):.10g}",
                    }
                )
    return rows


def write_delay_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    for detector in ("adwin", "time"):
        rows = build_delay_rows(
            args.data_dir,
            detector,
            args.include_unscheduled,
            args.include_all_experiments,
        )
        output_path = args.output_dir / f"{detector}_retraining_delays.csv"
        write_delay_csv(output_path, rows)
        print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
