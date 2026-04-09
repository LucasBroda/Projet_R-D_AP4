"""
Étape 3 : Extraction automatique des paramètres via LLM
Supporte : OpenAI GPT-4o, Anthropic Claude, Mistral, LLaMA via Ollama
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# PROMPT SYSTÈME
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert in organic photovoltaics and materials science.
Your task is to extract experimental parameters from scientific articles about 
ternary organic solar cells (OSC).

Extract ONLY values explicitly mentioned in the text. 
If a value is not found, return null.
Always return a valid JSON object."""

EXTRACTION_PROMPT_TEMPLATE = """Extract the following parameters from this scientific text about ternary organic solar cells.

TEXT:
{text}

Extract these parameters and return a JSON object:
{{
  "donor_material": "main donor material (e.g. PM6, D18, PTQ10)",
  "acceptor_1": "primary non-fullerene acceptor (e.g. Y6, BTP-eC9)",
  "acceptor_2": "secondary acceptor or fullerene (e.g. PC71BM, IT-4F)",
  "donor_ratio": "donor weight ratio in blend (number only, e.g. 1.0)",
  "acceptor_1_ratio": "primary acceptor weight ratio (number only)",
  "acceptor_2_ratio": "secondary acceptor weight ratio (number only)",
  "solvent": "processing solvent (e.g. chlorobenzene, o-xylene)",
  "additive": "solvent additive if any (e.g. 1-CN, DIO, null if none)",
  "device_architecture": "inverted or conventional",
  "active_layer_thickness": "thickness in nm (number only)",
  "pce": "Power Conversion Efficiency in % (number only, best value)",
  "voc": "Open-circuit voltage in V (number only)",
  "jsc": "Short-circuit current density in mA/cm2 (number only)",
  "ff": "Fill Factor in % (number only)",
  "annealing_temperature": "thermal annealing temperature in °C (number only, null if none)",
  "spin_coating_speed": "spin coating speed in rpm (number only, null if none)"
}}

Rules:
- Return ONLY the JSON object, no extra text
- Use null for missing values
- Extract the BEST/OPTIMAL device performance values
- For ratios, use the format: 1:0.8:0.2 becomes donor=1.0, acc1=0.8, acc2=0.2"""


# ──────────────────────────────────────────────
# CLIENT OPENAI (GPT-4o, GPT-4-turbo)
# ──────────────────────────────────────────────

class OpenAIExtractor:
    """Extraction via OpenAI API (GPT-4o recommandé)."""

    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def extract(self, text: str) -> dict:
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(text=text[:8000])  # limite contexte
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # Déterministe pour extraction
            response_format={"type": "json_object"},  # Force JSON valide
        )
        return json.loads(response.choices[0].message.content)


# ──────────────────────────────────────────────
# CLIENT ANTHROPIC (Claude 3.5 Sonnet)
# ──────────────────────────────────────────────

class AnthropicExtractor:
    """Extraction via Anthropic Claude API."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def extract(self, text: str) -> dict:
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(text=text[:8000])
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text
        # Extraction du bloc JSON si présent dans la réponse
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError(f"Pas de JSON valide dans la réponse : {raw[:200]}")


# ──────────────────────────────────────────────
# CLIENT MISTRAL (via API officielle)
# ──────────────────────────────────────────────

class MistralExtractor:
    """Extraction via Mistral AI API (Mistral Large)."""

    def __init__(self, model: str = "mistral-large-latest"):
        from mistralai import Mistral
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.model = model

    def extract(self, text: str) -> dict:
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(text=text[:8000])
        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)


# ──────────────────────────────────────────────
# CLIENT LOCAL via OLLAMA (LLaMA3, Mistral local)
# ──────────────────────────────────────────────

class OllamaExtractor:
    """
    Extraction via Ollama (modèles locaux gratuits).
    Installer : https://ollama.com/
    Lancer : ollama run llama3.1:8b
    """

    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        import requests
        self.model = model
        self.base_url = base_url
        self.requests = requests

    def extract(self, text: str) -> dict:
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(text=text[:4000])  # modèles locaux = contexte limité
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        
        response = self.requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "format": "json",  # Force JSON output
                "options": {"temperature": 0}
            }
        )
        result = response.json()
        raw = result.get("response", "{}")
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {}


# ──────────────────────────────────────────────
# CLIENT HUGGINGFACE (modèles open-source via API)
# ──────────────────────────────────────────────

class HuggingFaceExtractor:
    """Extraction via HuggingFace Inference API."""

    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(
            model=model,
            token=os.getenv("HF_API_TOKEN")
        )
        self.model = model

    def extract(self, text: str) -> dict:
        prompt = f"{SYSTEM_PROMPT}\n\n{EXTRACTION_PROMPT_TEMPLATE.format(text=text[:3000])}"
        response = self.client.text_generation(
            prompt,
            max_new_tokens=1024,
            temperature=0.01,
            return_full_text=False
        )
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {}


# ──────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────

def run_extraction_pipeline(
    processed_dir: str,
    output_dir: str,
    extractor,
    delay: float = 1.0  # délai entre appels API pour respecter les rate limits
):
    """Lance l'extraction sur tous les textes prétraités."""
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    for txt_file in sorted(processed_path.glob("*.txt")):
        print(f"\n🔍 Traitement : {txt_file.name}")
        text = txt_file.read_text(encoding="utf-8")

        # Découpe en chunks si le texte est trop long
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from importlib import import_module        
        chunks = chunk_text(text, chunk_size=2500, overlap=200)
        print(f"   → {len(chunks)} chunk(s) à traiter")

        article_results = []
        for i, chunk in enumerate(chunks):
            try:
                extracted = extractor.extract(chunk)
                extracted["source_file"] = txt_file.name
                extracted["chunk_index"] = i
                article_results.append(extracted)
                print(f"   ✅ Chunk {i+1}/{len(chunks)} extrait")
                time.sleep(delay)
            except Exception as e:
                print(f"   ❌ Erreur chunk {i}: {e}")
                article_results.append({
                    "source_file": txt_file.name,
                    "chunk_index": i,
                    "error": str(e)
                })

        # Fusion des résultats du même article (garde les valeurs non-null)
        merged = merge_chunk_results(article_results)
        results.append(merged)

        # Sauvegarde intermédiaire
        out_file = output_path / (txt_file.stem + "_extracted.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

    # Sauvegarde globale
    global_output = output_path / "all_extractions.json"
    with open(global_output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Extraction terminée ! {len(results)} articles traités.")
    print(f"Résultats : {global_output}")
    return results

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

def merge_chunk_results(chunks: list[dict]) -> dict:
    """Fusionne les résultats de plusieurs chunks d'un même article."""
    merged = {}
    for chunk in chunks:
        for key, value in chunk.items():
            if key in ("chunk_index", "error"):
                continue
            # Garde la première valeur non-null trouvée
            if key not in merged or merged[key] is None:
                merged[key] = value
    return merged


# ──────────────────────────────────────────────
# POINT D'ENTRÉE
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extraction LLM de paramètres scientifiques")
    parser.add_argument("--model", choices=["openai", "anthropic", "mistral", "ollama", "huggingface"],
                        default="openai", help="LLM à utiliser")
    parser.add_argument("--input", default="data/processed", help="Dossier des textes traités")
    parser.add_argument("--output", default="data/extractions", help="Dossier de sortie")
    args = parser.parse_args()

    extractors = {
        "openai":       OpenAIExtractor(model="gpt-4o"),
        "anthropic":    AnthropicExtractor(model="claude-3-5-sonnet-20241022"),
        "mistral":      MistralExtractor(model="mistral-large-latest"),
        "ollama":       OllamaExtractor(model="llama3.1:8b"),
        "huggingface":  HuggingFaceExtractor(model="mistralai/Mistral-7B-Instruct-v0.3"),
    }

    extractor = extractors[args.model]
    print(f"🤖 Modèle sélectionné : {args.model}")
    run_extraction_pipeline(args.input, args.output, extractor)