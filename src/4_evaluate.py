"""
Étape 4 : Évaluation des extractions LLM par rapport au ground truth
Métriques : Précision, Rappel, F1-Score (exact match + soft match)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import re


NUMERIC_PARAMETERS = ["pce", "voc", "jsc", "ff", "active_layer_thickness",
                       "donor_ratio", "acceptor_1_ratio", "acceptor_2_ratio",
                       "annealing_temperature", "spin_coating_speed"]
TEXT_PARAMETERS = ["donor_material", "acceptor_1", "acceptor_2", "solvent",
                   "additive", "device_architecture"]
TOLERANCE = 0.05  # 5% de tolérance pour les valeurs numériques


def normalize_text(value) -> str:
    """Normalise une valeur texte pour comparaison."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower().replace("-", "").replace(" ", "")


def normalize_numeric(value) -> Optional[float]:
    """Extrait et normalise une valeur numérique."""
    if value is None:
        return None
    try:
        # Gère les formats comme "18.5%" ou "18.5 %"
        cleaned = re.sub(r'[^\d.,]', '', str(value)).replace(',', '.')
        return float(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None


def exact_match(pred, truth) -> bool:
    """Vérifie la correspondance exacte (normalisée) pour les valeurs textuelles."""
    return normalize_text(pred) == normalize_text(truth) and normalize_text(truth) != ""


def soft_match_numeric(pred, truth, tolerance: float = TOLERANCE) -> bool:
    """Vérifie la correspondance numérique avec tolérance."""
    p = normalize_numeric(pred)
    t = normalize_numeric(truth)
    if p is None or t is None:
        return False
    if t == 0:
        return p == 0
    return abs(p - t) / abs(t) <= tolerance


def evaluate_extractions(extractions_path: str, ground_truth_path: str) -> pd.DataFrame:
    """Compare les extractions LLM au ground truth et calcule les métriques."""
    with open(extractions_path, "r", encoding="utf-8") as f:
        extractions = json.load(f)
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    print(f"📊 {len(extractions)} extractions vs {len(ground_truth)} références")

    results = []

    for param in TEXT_PARAMETERS + NUMERIC_PARAMETERS:
        tp = fp = fn = 0
        param_results = []

        for ext in extractions:
            source = ext.get("source_file", "")
            pred = ext.get(param)

            # Cherche la valeur de référence correspondante
            truth_entry = next(
                (g for g in ground_truth if param in g), None
            )
            if truth_entry is None:
                continue
            truth = truth_entry.get(param)

            is_numeric = param in NUMERIC_PARAMETERS
            if is_numeric:
                match = soft_match_numeric(pred, truth)
            else:
                match = exact_match(pred, truth)

            predicted_present = pred is not None and str(pred).strip() not in ("", "null", "None")
            truth_present = truth is not None and str(truth).strip() not in ("", "null", "None")

            if match and predicted_present:
                tp += 1
            elif predicted_present and not match:
                fp += 1
            elif truth_present and not predicted_present:
                fn += 1

            param_results.append({
                "parameter": param,
                "predicted": pred,
                "truth": truth,
                "match": match,
                "source": source
            })

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            "parameter": param,
            "type": "numeric" if param in NUMERIC_PARAMETERS else "text",
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        })

    df = pd.DataFrame(results)
    return df


def print_report(df: pd.DataFrame):
    """Affiche un rapport détaillé des métriques."""
    print("\n" + "="*70)
    print("📈 RAPPORT D'ÉVALUATION — EXTRACTION LLM")
    print("="*70)
    print(df.to_string(index=False))
    print("\n--- Moyennes globales ---")
    print(f"Précision moyenne  : {df['precision'].mean():.4f}")
    print(f"Rappel moyen       : {df['recall'].mean():.4f}")
    print(f"F1-Score moyen     : {df['f1_score'].mean():.4f}")
    print("\n--- Par type ---")
    for t in ["text", "numeric"]:
        sub = df[df["type"] == t]
        print(f"  [{t}] Précision={sub['precision'].mean():.3f}, "
              f"Rappel={sub['recall'].mean():.3f}, "
              f"F1={sub['f1_score'].mean():.3f}")


def plot_metrics(df: pd.DataFrame, output_path: str = "data/results/metrics.png"):
    """Génère un graphique des métriques par paramètre."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df["precision"], width, label="Précision", color="#2196F3")
    ax.bar(x,          df["recall"],   width, label="Rappel",    color="#4CAF50")
    ax.bar(x + width,  df["f1_score"], width, label="F1-Score",  color="#FF9800")

    ax.set_xlabel("Paramètre")
    ax.set_ylabel("Score")
    ax.set_title("Métriques d'évaluation de l'extraction LLM\n(Cellules Solaires Organiques Ternaires)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["parameter"], rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="Seuil 80%")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"📊 Graphique sauvegardé : {output_path}")
    plt.show()


if __name__ == "__main__":
    df = evaluate_extractions(
        "data/extractions/all_extractions.json",
        "data/annotations/ground_truth.json"
    )
    print_report(df)
    plot_metrics(df)
    df.to_csv("data/results/evaluation_report.csv", index=False)
    print("✅ Rapport sauvegardé : data/results/evaluation_report.csv")