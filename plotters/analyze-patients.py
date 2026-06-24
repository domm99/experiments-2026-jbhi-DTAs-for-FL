from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================

DATA_DIR = Path("T1DiabetesGranada/split-labeled")
OUTPUT_DIR = Path("charts/label_distribution_analysis")

LABEL_COL = "target_any_within_t"

EXPECTED_LABELS = ["normal", "hyper", "hypo"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOAD PATIENT CSVs
# =========================

rows = []

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR.resolve()}")

for csv_path in csv_files:
    patient_id = csv_path.stem

    df = pd.read_csv(csv_path)

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Column '{LABEL_COL}' not found in {csv_path.name}. "
            f"Available columns: {list(df.columns)}"
        )

    labels = df[LABEL_COL].astype(str).str.strip().str.lower()

    counts = labels.value_counts().to_dict()

    total = len(labels)

    row = {
        "patient_id": patient_id,
        "total_samples": total,
    }

    for label in EXPECTED_LABELS:
        row[f"count_{label}"] = counts.get(label, 0)
        row[f"ratio_{label}"] = counts.get(label, 0) / total if total > 0 else 0.0

    row["event_count"] = row["count_hyper"] + row["count_hypo"]
    row["event_ratio"] = row["event_count"] / total if total > 0 else 0.0

    row["dominant_label"] = max(
        EXPECTED_LABELS,
        key=lambda label: row[f"count_{label}"]
    )

    rows.append(row)


summary = pd.DataFrame(rows)

summary = summary.sort_values("patient_id").reset_index(drop=True)

summary_path = OUTPUT_DIR / "patient_label_summary.csv"
summary.to_csv(summary_path, index=False)

print(f"Saved patient-level summary to: {summary_path}")
print()
print(summary.head())


# =========================
# GLOBAL DATASET DISTRIBUTION
# =========================

global_counts = {
    label: summary[f"count_{label}"].sum()
    for label in EXPECTED_LABELS
}

global_counts_df = pd.DataFrame({
    "label": list(global_counts.keys()),
    "count": list(global_counts.values())
})

global_counts_df["ratio"] = global_counts_df["count"] / global_counts_df["count"].sum()

print()
print("Global label distribution:")
print(global_counts_df)


plt.figure(figsize=(7, 5))
plt.bar(global_counts_df["label"], global_counts_df["count"])
plt.title("Global label distribution")
plt.xlabel("Label")
plt.ylabel("Number of samples")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "global_label_distribution_counts.png", dpi=300)
plt.close()


plt.figure(figsize=(7, 5))
plt.bar(global_counts_df["label"], global_counts_df["ratio"])
plt.title("Global label distribution")
plt.xlabel("Label")
plt.ylabel("Ratio")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "global_label_distribution_ratios.png", dpi=300)
plt.close()


# =========================
# PER-PATIENT LABEL RATIOS
# =========================

summary_sorted_by_event = summary.sort_values("event_ratio").reset_index(drop=True)

x = range(len(summary_sorted_by_event))

plt.figure(figsize=(14, 8))
bottom = [0] * len(summary_sorted_by_event)

for label in EXPECTED_LABELS:
    values = summary_sorted_by_event[f"ratio_{label}"]
    plt.bar(x, values, bottom=bottom, label=label, width=2)
    bottom = [b + v for b, v in zip(bottom, values)]

# plt.title("Per-patient label composition, sorted by event ratio")
plt.xlabel("Patients sorted by hyper/hypo event ratio", fontsize=30)
plt.xlim(0, 736)
plt.ylim(0,1)
plt.tick_params(axis="both", labelsize=26)
plt.ylabel("Label ratio", fontsize=30)
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.2),
    ncol=3,
    fontsize=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "per_patient_label_composition_sorted_by_event_ratio.png", dpi=300)
plt.close()


# =========================
# EVENT RATIO DISTRIBUTION
# =========================

plt.figure(figsize=(8, 5))
plt.hist(summary["event_ratio"], bins=30)
plt.title("Distribution of event ratio per patient")
plt.xlabel("Event ratio: hyper + hypo")
plt.ylabel("Number of patients")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "event_ratio_distribution_per_patient.png", dpi=300)
plt.close()


plt.figure(figsize=(8, 5))
plt.hist(summary["ratio_hyper"], bins=30, alpha=0.8, label="hyper")
plt.hist(summary["ratio_hypo"], bins=30, alpha=0.8, label="hypo")
plt.title("Distribution of hyper and hypo ratios per patient")
plt.xlabel("Per-patient label ratio")
plt.ylabel("Number of patients")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hyper_hypo_ratio_distribution_per_patient.png", dpi=300)
plt.close()


# =========================
# PATIENT SIZE DISTRIBUTION
# =========================

plt.figure(figsize=(8, 5))
plt.hist(summary["total_samples"], bins=30)
plt.title("Number of samples per patient")
plt.xlabel("Samples")
plt.ylabel("Number of patients")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "samples_per_patient_distribution.png", dpi=300)
plt.close()


# =========================
# SCATTER: SIZE VS EVENT RATIO
# =========================

plt.figure(figsize=(8, 6))
plt.scatter(summary["total_samples"], summary["event_ratio"])
plt.title("Patient size vs event ratio")
plt.xlabel("Number of samples")
plt.ylabel("Event ratio: hyper + hypo")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "patient_size_vs_event_ratio.png", dpi=300)
plt.close()


# =========================
# SCATTER: HYPER VS HYPO RATIO
# =========================

plt.figure(figsize=(8, 6))
plt.scatter(summary["ratio_hyper"], summary["ratio_hypo"])
plt.title("Hyper vs hypo ratio per patient")
plt.xlabel("Hyper ratio")
plt.ylabel("Hypo ratio")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hyper_vs_hypo_ratio_per_patient.png", dpi=300)
plt.close()


# =========================
# DOMINANT LABEL COUNTS
# =========================

dominant_counts = summary["dominant_label"].value_counts().reindex(EXPECTED_LABELS, fill_value=0)

plt.figure(figsize=(7, 5))
plt.bar(dominant_counts.index, dominant_counts.values)
plt.title("Number of patients by dominant label")
plt.xlabel("Dominant label")
plt.ylabel("Number of patients")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "patients_by_dominant_label.png", dpi=300)
plt.close()


# =========================
# CANDIDATE SPLIT DIAGNOSTICS
# =========================

print()
print("Patients by dominant label:")
print(dominant_counts)

print()
print("Event ratio summary:")
print(summary["event_ratio"].describe())

print()
print("Hyper ratio summary:")
print(summary["ratio_hyper"].describe())

print()
print("Hypo ratio summary:")
print(summary["ratio_hypo"].describe())

print()
print("Samples per patient summary:")
print(summary["total_samples"].describe())

print()
print(f"Charts saved in: {OUTPUT_DIR.resolve()}")