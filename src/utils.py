"""Fonctions utilitaires partagées."""
import re
import numpy as np


def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 300) -> list:
    """Découpe le texte en chunks chevauchants."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def normalize_text(value) -> str:
    """Normalise une valeur texte pour comparaison."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower().replace("-", "").replace(" ", "")


def normalize_numeric(value):
    """Extrait et normalise une valeur numérique."""
    if value is None:
        return None
    try:
        cleaned = re.sub(r'[^\d.,]', '', str(value)).replace(',', '.')
        return float(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None