"""
Étape 2 : Chargement et structuration des annotations de référence
depuis le fichier Excel fourni (mmc1.xlsx)
"""

import pandas as pd
import json
from pathlib import Path


# Paramètres cibles à extraire pour les cellules solaires organiques ternaires
TARGET_PARAMETERS = [
    "donor_material",       # Matériau donneur (ex: PM6)
    "acceptor_1",           # Premier accepteur (ex: Y6)
    "acceptor_2",           # Deuxième accepteur (ex: PC71BM)
    "donor_ratio",          # Ratio donneur
    "acceptor_1_ratio",     # Ratio accepteur 1
    "acceptor_2_ratio",     # Ratio accepteur 2
    "solvent",              # Solvant (ex: chlorobenzene)
    "additive",             # Additif (ex: DIO, CN)
    "device_architecture",  # Architecture (inverted/conventional)
    "active_layer_thickness",  # Épaisseur couche active (nm)
    "pce",                  # Power Conversion Efficiency (%)
    "voc",                  # Tension de circuit ouvert (V)
    "jsc",                  # Densité de courant (mA/cm²)
    "ff",                   # Fill Factor (%)
    "annealing_temperature",# Température de recuit (°C)
    "spin_coating_speed",   # Vitesse de dépôt (rpm)
]


def load_reference_data(xlsx_path: str) -> pd.DataFrame:
    """Charge le fichier Excel de référence et normalise les colonnes."""
    df = pd.read_excel(xlsx_path, sheet_name=0)
    print(f"📊 Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"Colonnes disponibles :\n{list(df.columns)}")
    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les noms de colonnes et les types de données."""
    # Uniformisation des noms de colonnes
    df.columns = [col.strip().lower().replace(" ", "_").replace("/", "_") for col in df.columns]
    # Conversion des valeurs numériques
    numeric_cols = ["pce", "voc", "jsc", "ff", "active_layer_thickness"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def export_ground_truth(df: pd.DataFrame, output_path: str):
    """Exporte le jeu de référence en JSON pour comparaison avec les extractions LLM."""
    records = df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"✅ Ground truth exporté : {output_path} ({len(records)} entrées)")


if __name__ == "__main__":
    Path("data/annotations").mkdir(parents=True, exist_ok=True)
    df = load_reference_data("1-s2.0-S2095495624006089-mmc1.xlsx")
    df = normalize_dataframe(df)
    export_ground_truth(df, "data/annotations/ground_truth.json")
    print("\nAperçu des données :")
    print(df.head())