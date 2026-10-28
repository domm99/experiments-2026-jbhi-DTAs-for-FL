#!/usr/bin/env python3

from pathlib import Path
from bisect import bisect_right
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIGURAZIONE
# =========================

PATIENTS_DIR = Path("T1DiabetesGranada/split-labeled")
RESULTS_DIR = Path("data/cost-estimation/")
OUTPUT_DIR = Path("charts/cost_analysis")

POLICIES = [
    "fedpers",
    "retrainaftertime"

]

POLICY_DISPLAY_ORDER = [
    "Federated w/ personalization",
    #"Local",
    "Time-Based"
]

FIXED_COST = 1.0
COST_PER_ROW = 0.00001
ALPHA = 1.0


# Esempio nome file:
# training_2018-11-04 13:39:00-seed_0.csv
TRAINING_DATE_PATTERN = re.compile(
    r"training_(\d{4}-\d{2}-\d{2}[ _]\d{2}:\d{2}:\d{2})"
)


def read_all_dataset_timestamps(patients_dir: Path) -> pd.Series:
    """
    Legge tutti i CSV paziente e raccoglie tutti i timestamp.

    Assunzione:
    - ogni CSV ha una colonna chiamata 'timestamp'
    - ogni riga rappresenta un dato disponibile dal proprio timestamp in poi

    Quindi, a una data t, i dati disponibili sono tutte le righe con:
    timestamp <= t
    """
    all_timestamps = []

    patient_files = sorted(patients_dir.glob("*.csv"))

    if not patient_files:
        raise FileNotFoundError(f"Nessun CSV trovato in: {patients_dir}")

    for csv_file in patient_files:
        df = pd.read_csv(csv_file)

        if "timestamp" not in df.columns:
            raise ValueError(f"Nel file {csv_file} manca la colonna 'timestamp'")

        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
        timestamps = timestamps.dropna()

        if len(timestamps) > 0:
            all_timestamps.append(timestamps)

    if not all_timestamps:
        raise ValueError("Non ho trovato nessun timestamp valido nei CSV paziente.")

    all_timestamps = pd.concat(all_timestamps)
    all_timestamps = all_timestamps.sort_values().reset_index(drop=True)

    return all_timestamps


def extract_training_time(filename: str) -> pd.Timestamp | None:
    """
    Estrae la data di training dal nome del file.

    Accetta sia:
    training_2018-11-04 13:39:00-seed_0.csv

    sia:
    training_2018-11-04_13:39:00-seed_0.csv
    """
    match = TRAINING_DATE_PATTERN.search(filename)

    if match is None:
        return None

    date_text = match.group(1).replace("_", " ")
    training_time = pd.to_datetime(date_text, errors="coerce")

    if pd.isna(training_time):
        return None

    return training_time


def count_rows_available_at_time(
    sorted_timestamps: pd.Series,
    training_time: pd.Timestamp,
) -> int:
    """
    Conta quante righe del dataset erano disponibili al momento del training.

    Siccome sorted_timestamps è ordinato, uso bisect_right:
    restituisce il numero di timestamp <= training_time.
    """
    return bisect_right(
        sorted_timestamps.values,
        training_time.to_datetime64(),
    )


def estimate_training_cost(
    n_rows: int,
    fixed_cost: float,
    cost_per_row: float,
    alpha: float,
) -> float:
    """
    Formula di costo usata:

        costo = fixed_cost + cost_per_row * n_rows ** alpha

    Dove:
    - fixed_cost: costo fisso di fare un retraining
    - cost_per_row: costo associato alla quantità di dati
    - n_rows: numero di righe disponibili al momento del training
    - alpha: esponente che controlla come cresce il costo

    Esempi:
    - alpha = 1.0  -> costo lineare nel numero di righe
    - alpha = 1.5  -> costo più che lineare
    - alpha = 2.0  -> costo quadratico
    """
    return fixed_cost + cost_per_row * (n_rows ** alpha)


def beautify_policy_name(policy_name: str) -> str:
    if 'fedpers' in policy_name:
        return 'Federated w/ personalization'
    elif 'local' in policy_name:
        return 'Local'
    elif 'time' in policy_name:
        return 'Time-Based'
    elif 'PerformanceDrift' in policy_name:
        return 'Performance-Based'
    elif 'ADWIN' in policy_name:
        return 'ADWIN-Based'
    elif 'DDM' in policy_name:
        return 'DDM-Based'
    else:
        return 'Unknown Policy'


def analyze_policy(
    policy_dir: Path,
    dataset_timestamps: pd.Series,
    fixed_cost: float,
    cost_per_row: float,
    alpha: float,
) -> tuple[list[dict], dict]:
    """
    Analizza una singola policy.

    Ritorna:
    - una lista con il dettaglio di ogni training
    - una riga summary per la policy
    """
    policy_name = policy_dir.name
    training_events = []

    for file_path in sorted(policy_dir.iterdir()):
        if not file_path.is_file():
            continue

        training_time = extract_training_time(file_path.name)

        if training_time is None:
            continue

        training_events.append((training_time, file_path.name))

    training_events.sort(key=lambda x: x[0])

    detail_rows = []
    total_cost = 0.0

    for i, (training_time, training_file) in enumerate(training_events, start=1):
        n_rows = count_rows_available_at_time(
            sorted_timestamps=dataset_timestamps,
            training_time=training_time,
        )

        training_cost = estimate_training_cost(
            n_rows=n_rows,
            fixed_cost=fixed_cost,
            cost_per_row=cost_per_row,
            alpha=alpha,
        )

        total_cost += training_cost

        detail_rows.append(
            {
                "policy": beautify_policy_name(policy_name),
                "training_number": i,
                "training_file": training_file,
                "training_time": training_time,
                "available_rows": n_rows,
                "training_cost": training_cost,
                "cumulative_policy_cost": total_cost,
            }
        )

    summary_row = {
        "policy": beautify_policy_name(policy_name),
        "n_trainings": len(training_events),
        "first_training_time": training_events[0][0] if training_events else None,
        "last_training_time": training_events[-1][0] if training_events else None,
        "total_cost": total_cost,
        "mean_cost_per_training": (
            total_cost / len(training_events) if training_events else 0.0
        ),
    }

    return detail_rows, summary_row


from matplotlib.patches import Patch


def plot_total_cost_barplot(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Crea un bar plot con:
    - barre separate
    - griglia orizzontale
    - bordi neri
    - legenda sotto al plot
    """
    plot_path = output_dir / "policy_total_cost_barplot.png"

    plot_df = summary_df.copy()

    policy_labels = {
        "RetrainAfterTime": "Time-based",
        "RetrainEachNDTsActivated": "Population-Based",
        "RetrainAfterPerformanceDrift": "Performance-Based",
        "ADWIN": "ADWIN-Based",
        "DDM": "DDM-Based",
    }

    plot_df["policy"] = plot_df["policy"].replace(policy_labels)

    # Ordine delle policy nel plot e nella legenda
    policy_order = POLICY_DISPLAY_ORDER

    # Colori seaborn
    colors = sns.color_palette("deep", n_colors=len(policy_order))
    color_map = dict(zip(policy_order, colors))

    fig, ax = plt.subplots(figsize=(8, 4.8))

    sns.barplot(
        data=plot_df,
        x="policy",
        y="total_cost",
        order=policy_order,
        palette=color_map,
        errorbar=None,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Total Estimated Cost", fontsize=18)

    # Nasconde i nomi sotto le barre
    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.tick_params(axis="y", labelsize=18)

    # Griglia orizzontale sotto le barre
    ax.yaxis.grid(True, linestyle="-", linewidth=0.8, alpha=0.35)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Bordi neri
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(2)

    # Legenda manuale sotto
    legend_handles = [
        Patch(facecolor=color_map[policy], label=policy)
        for policy in policy_order
    ]

    ax.legend(
        handles=legend_handles,
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=22,
    )

    # Lascia spazio alla legenda sotto
    fig.subplots_adjust(bottom=0.28)

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Bar plot salvato in:          {plot_path}")


def analyze_all_policies(
    patients_dir: Path,
    results_dir: Path,
    policy_names: list[str],
    output_dir: Path,
    fixed_cost: float,
    cost_per_row: float,
    alpha: float,
) -> None:
    """
    Funzione principale:
    - legge tutti i timestamp del dataset
    - analizza solo le policy indicate in policy_names
    - salva due CSV finali
    - salva un bar plot del costo totale per policy
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Leggo i timestamp del dataset...")
    dataset_timestamps = read_all_dataset_timestamps(patients_dir)

    print(f"Righe totali nel dataset completo: {len(dataset_timestamps)}")
    print(f"Primo timestamp: {dataset_timestamps.min()}")
    print(f"Ultimo timestamp: {dataset_timestamps.max()}")
    print()

    all_detail_rows = []
    summary_rows = []

    for policy_name in policy_names:
        policy_dir = results_dir / policy_name

        if not policy_dir.exists():
            print(f"[WARN] Policy saltata: cartella non trovata: {policy_dir}")
            continue

        if not policy_dir.is_dir():
            print(f"[WARN] Policy saltata: non è una cartella: {policy_dir}")
            continue

        print(f"Analizzo policy: {policy_name}")

        detail_rows, summary_row = analyze_policy(
            policy_dir=policy_dir,
            dataset_timestamps=dataset_timestamps,
            fixed_cost=fixed_cost,
            cost_per_row=cost_per_row,
            alpha=alpha,
        )

        all_detail_rows.extend(detail_rows)
        summary_rows.append(summary_row)

    if not summary_rows:
        raise ValueError("Nessuna policy valida analizzata.")

    details_df = pd.DataFrame(all_detail_rows)
    summary_df = pd.DataFrame(summary_rows)

    summary_df["policy"] = pd.Categorical(
        summary_df["policy"],
        categories=POLICY_DISPLAY_ORDER,
        ordered=True,
    )

    summary_df = summary_df.sort_values("policy")

    details_path = output_dir / "policy_training_cost_details.csv"
    summary_path = output_dir / "policy_cost_summary.csv"

    details_df.to_csv(details_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_total_cost_barplot(
        summary_df=summary_df,
        output_dir=output_dir,
    )

    print()
    print("Fatto.")
    print(f"Dettaglio training salvato in: {details_path}")
    print(f"Summary policy salvato in:    {summary_path}")
    print()
    print(summary_df.to_string(index=False))


def main():
    analyze_all_policies(
        patients_dir=PATIENTS_DIR,
        results_dir=RESULTS_DIR,
        policy_names=POLICIES,
        output_dir=OUTPUT_DIR,
        fixed_cost=FIXED_COST,
        cost_per_row=COST_PER_ROW,
        alpha=ALPHA,
    )


if __name__ == "__main__":
    main()