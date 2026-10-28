from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =========================
# CONFIG
# =========================

DATA_DIR = Path("T1DiabetesGranada/split-labeled")
OUTPUT_DIR = Path("charts/cumulative_instances")

PLOT_EVERY_N_STEPS = 3
X_AXIS_RIGHT_LIMIT = 300  # es. 180 per mostrare solo i primi 180 punti plottati


# =========================
# LOAD DATA
# =========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if PLOT_EVERY_N_STEPS <= 0:
    raise ValueError("PLOT_EVERY_N_STEPS must be greater than zero")

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR.resolve()}")

all_row_days = []
patient_intervals = []

for csv_path in csv_files:
    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        timestamps = pd.to_datetime(
            df["Measurement_date"] + " " + df["Measurement_time"],
            errors="coerce",
        )

    row_days = timestamps.dropna().dt.normalize()

    if row_days.empty:
        continue

    all_row_days.append(row_days)
    patient_intervals.append((row_days.min(), row_days.max()))

if not all_row_days:
    raise ValueError("No valid timestamps found in the dataset")


# =========================
# DAILY SIMULATION STEPS
# =========================

row_days = pd.concat(all_row_days, ignore_index=True)
first_day = min(start for start, _ in patient_intervals)
last_day = max(end for _, end in patient_intervals)

calendar_days = pd.date_range(first_day, last_day, freq="D")
active_day_mask = pd.Series(False, index=calendar_days)

for start_day, end_day in patient_intervals:
    active_day_mask |= (calendar_days >= start_day) & (calendar_days <= end_day)

active_days = active_day_mask[active_day_mask].index

daily_counts = row_days.value_counts().sort_index()
daily_counts = daily_counts.reindex(calendar_days, fill_value=0).cumsum()

summary = pd.DataFrame({
    "simulation_step": range(len(active_days)),
    "date": active_days,
    "cumulative_instances": daily_counts.reindex(active_days).to_numpy(),
})

sampled_summary = summary.iloc[::PLOT_EVERY_N_STEPS].copy()
sampled_summary["plot_step"] = range(len(sampled_summary))

summary_path = OUTPUT_DIR / "cumulative_instances_by_simulation_day.csv"
summary.to_csv(summary_path, index=False)


# =========================
# PLOT
# =========================

plt.figure(figsize=(7, 4))
plt.plot(
    sampled_summary["plot_step"],
    sampled_summary["cumulative_instances"],
    # marker="o",
    markersize=3,
    linewidth=3,
)

plt.yscale("log")
plt.xlabel(f"Time Step", fontsize=20)
plt.ylabel("Cumulative instances", fontsize=20)
plt.grid(alpha=0.3)
plt.tick_params(axis="both", labelsize=18)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)

if X_AXIS_RIGHT_LIMIT is not None:
    plt.xlim(left=0, right=X_AXIS_RIGHT_LIMIT)

plt.tight_layout()

plot_path = OUTPUT_DIR / "cumulative_instances_by_simulation_day.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Saved summary to: {summary_path}")
print(f"Saved plot to: {plot_path}")
print(f"Patients: {len(patient_intervals)}")
print(f"Simulation steps: {len(summary)}")
print(f"Plotted points: {len(sampled_summary)}")
print(f"Total instances: {int(summary['cumulative_instances'].iloc[-1])}")
