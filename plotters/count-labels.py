from pathlib import Path

import pandas as pd


# =========================
# CONFIG
# =========================

DATA_DIR = Path("T1DiabetesGranada/split-labeled")

LABEL_COL = "target_any_within_t"

TARGET_LABEL = "hypo"   # cambia in "hypo" se vuoi contare le ipo
N = 10                   # soglia: pazienti con meno di N label TARGET_LABEL


# =========================
# COUNT PATIENTS
# =========================

num_patients_less_than_n = 0
total_patients = 0

for csv_path in sorted(DATA_DIR.glob("*.csv")):
    total_patients += 1

    df = pd.read_csv(csv_path)

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Colonna '{LABEL_COL}' non trovata in {csv_path.name}. "
            f"Colonne disponibili: {list(df.columns)}"
        )

    labels = df[LABEL_COL].astype(str).str.strip().str.lower()

    label_count = (labels == TARGET_LABEL.lower()).sum()

    if label_count < N:
        num_patients_less_than_n += 1


print(
    f"Pazienti con meno di {N} label '{TARGET_LABEL}': "
    f"{num_patients_less_than_n} su {total_patients}"
)