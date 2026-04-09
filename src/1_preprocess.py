"""
Étape 1 : Extraction et nettoyage du texte depuis PDF et DOCX
"""

import fitz  # PyMuPDF
import os
import re
from pathlib import Path
from docx import Document


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrait le texte d'un PDF page par page."""
    doc = fitz.open(pdf_path)
    full_text = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        full_text.append(f"--- Page {page_num + 1} ---\n{text}")
    doc.close()
    return "\n".join(full_text)


def extract_text_from_docx(docx_path: str) -> str:
    """Extrait le texte d'un fichier DOCX."""
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def clean_text(text: str) -> str:
    """Nettoie le texte extrait : supprime les caractères parasites."""
    # Supprime les retours chariot multiples
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Supprime les espaces en fin de ligne
    text = re.sub(r'[ \t]+\n', '\n', text)
    # Reconstruit les mots coupés en fin de ligne (ex: "effi-\nciency" -> "efficiency")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 300) -> list[str]:
    """
    Découpe le texte en chunks chevauchants pour respecter
    les limites de contexte des LLM.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def process_all_documents(input_dir: str, output_dir: str):
    """Traite tous les documents d'un dossier et sauvegarde le texte extrait."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in input_path.iterdir():
        if file.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(str(file))
        elif file.suffix.lower() == ".docx":
            text = extract_text_from_docx(str(file))
        else:
            continue

        text = clean_text(text)
        out_file = output_path / (file.stem + ".txt")
        out_file.write_text(text, encoding="utf-8")
        print(f"✅ Traité : {file.name} → {out_file.name}")


if __name__ == "__main__":
    process_all_documents("data/raw", "data/processed")