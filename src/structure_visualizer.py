"""
Document structure visualization for Docling processed documents.
"""
from typing import List, Dict, Any, Optional
from collections import Counter
from pathlib import Path
import re
import pandas as pd
from docling_core.types.doc import DoclingDocument
import json

import requests
from streamlit.string_util import clean_text

class DocumentStructureVisualizer:
    """Extracts and organizes document structure from Docling documents."""

    def __init__(self, docling_document: DoclingDocument):
        self.doc = docling_document

    def get_full_text(self) -> str:
        """Export full Docling document to markdown text."""
        try:
            return self.doc.export_to_markdown()
        except Exception:
            texts = getattr(self.doc, "texts", [])
            return "\n".join(
                getattr(item, "text", "")
                for item in texts
                if getattr(item, "text", "")
            )
        
    def load_stopwords(self, stopwords_path: str = "stopwords-nl.txt") -> set:
        """Load Dutch stopwords from txt file."""
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / stopwords_path

        if not path.exists():
            print(f"Warning: stopwords file not found: {stopwords_path}")
            return set()

        with open(path, "r", encoding="utf-8") as file:
            return {
                line.strip().lower()
                for line in file
                if line.strip()
            }
        
    def clean_text(self, text: str) -> str:
        """Clean markdown text for analysis."""
        if not text:
            return ""

        text = re.sub(r".*<!-- image -->.*", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[#*_`>\[\]()+\-|]", " ", text)
        text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip().lower()

    def get_document_summary(
        self,
        keyword_count: int = 20,
        model_name: str = "bramvanroy/geitje-7b-ultra:Q5_K_M",
        ollama_url: str = "http://localhost:11434/api/chat"):
        """
        Generate document summary with GEITje via Ollama.
        """
        markdown_text = self.get_full_text()

        if not markdown_text or len(markdown_text.split()) < 50:
            return {
                "summary": "De tekst is te kort om automatisch samen te vatten."
            }

        prompt = f"""
Maak een samenvatting van de volgende documentinhoud in een lopende tekst, zonder opsommingen.

DOCUMENT:
{markdown_text}

SAMENVATTING:
"""

        try:
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": """
Je bent een assistent voor documentanalyse.
Vat documenten samen in neutraal en feitelijk Nederlands.

Regels:

* Geef uitsluitend een korte lopende tekst.
* Gebruik geen opsommingen, markdown, tabellen of letterlijke documentfragmenten.
* Geef geen interpretatie, advies, mening of conclusie.
* Gebruik geen positieve of overtuigende formuleringen.
* Beschrijf alleen wat expliciet in het document staat.
* Voeg geen informatie toe die niet in het document voorkomt.
* Benoem kort welke onderwerpen, onderdelen of thema’s in het document voorkomen.
* Houd de toon zakelijk en beschrijvend.
* Begin direct met de samenvatting.
* Gebruik tussen de 120 en 180 woorden.
"""
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "stream": True,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 600,
                },
            }

            response = requests.post(ollama_url, json=payload, stream=True, timeout=(30, 600))
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))

                    content = ""

                    if "message" in chunk:
                        content += chunk["message"].get("content", "")

                        if content:
                            yield content

        except Exception as e:
            yield f"Samenvatting kon niet worden gegenereerd via Ollama: {e}"

#        return {
#            "summary": summary_text if summary_text else "Geen samenvatting beschikbaar.",
#        }

    def get_term_frequency(
        self,
        stopwords_path: str = "stopwords-nl.txt",
        top_n: int = 30,
        min_frequency: int = 3,
    ) -> pd.DataFrame:
        """
        Calculate term frequency with stopwords removed.
        """
        stopwords = self.load_stopwords(stopwords_path)
        text = self.clean_text(self.get_full_text())

        words = [
            word for word in text.split()
            if word not in stopwords
            and len(word) > 2
            and not word.isnumeric()
        ]

        counts = Counter(words)

        results = [
            {"term": term, "frequency": freq}
            for term, freq in counts.most_common(top_n)
            if freq >= min_frequency
        ]

        return pd.DataFrame(results)

    def get_bigrams(
        self,
        stopwords_path: str = "stopwords-nl.txt",
        top_n: int = 30,
        min_frequency: int = 3,
    ) -> pd.DataFrame:
        """
        Calculate bigram frequency with stopwords removed.
        """
        stopwords = self.load_stopwords(stopwords_path)
        text = self.clean_text(self.get_full_text())

        words = [
            word for word in text.split()
            if word not in stopwords
            and len(word) > 2
            and not word.isnumeric()
        ]

        bigrams = [
            f"{words[i]} {words[i + 1]}"
            for i in range(len(words) - 1)
        ]

        counts = Counter(bigrams)

        results = [
            {"bigram": bigram, "frequency": freq}
            for bigram, freq in counts.most_common(top_n)
            if freq >= min_frequency
        ]

        return pd.DataFrame(results)

    def get_tables_info(self) -> List[Dict[str, Any]]:
        """
        Extract table information and convert to DataFrames.

        Returns:
            List of dictionaries with table metadata and DataFrame
        """
        tables_info = []

        if not hasattr(self.doc, 'tables') or not self.doc.tables:
            return tables_info

        for i, table in enumerate(self.doc.tables, 1):
            try:
                # Export table to DataFrame
                df = table.export_to_dataframe(doc=self.doc)

                # Get provenance
                prov = getattr(table, 'prov', [])
                page_no = prov[0].page_no if prov else None

                # Get caption if available
                caption_text = getattr(table, 'caption_text', None)
                caption = caption_text if caption_text and not callable(caption_text) else None

                tables_info.append({
                    'table_number': i,
                    'page': page_no,
                    'caption': caption,
                    'dataframe': df,
                    'shape': df.shape,
                    'is_empty': df.empty
                })

            except Exception as e:
                # Handle tables that can't be converted
                print(f"Warning: Could not process table {i}: {e}")
                continue

        return tables_info

    def get_pictures_info(self) -> List[Dict[str, Any]]:
        """
        Extract picture/image metadata and image data.

        Returns:
            List of dictionaries with picture information and PIL images
        """
        pictures_info = []

        if not hasattr(self.doc, 'pictures') or not self.doc.pictures:
            return pictures_info

        for i, pic in enumerate(self.doc.pictures, 1):
            prov = getattr(pic, 'prov', [])

            if prov:
                page_no = prov[0].page_no
                bbox = prov[0].bbox

                # Get caption if available
                caption_text = getattr(pic, 'caption_text', None)
                caption = caption_text if caption_text and not callable(caption_text) else None

                # Get PIL image if available
                pil_image = None
                try:
                    if hasattr(pic, 'image') and pic.image is not None:
                        if hasattr(pic.image, 'pil_image'):
                            pil_image = pic.image.pil_image
                except Exception as e:
                    print(f"Warning: Could not extract image {i}: {e}")

                pictures_info.append({
                    'picture_number': i,
                    'page': page_no,
                    'caption': caption,
                    'pil_image': pil_image,  # Add PIL image
                    'bounding_box': {
                        'left': bbox.l,
                        'top': bbox.t,
                        'right': bbox.r,
                        'bottom': bbox.b
                    } if bbox else None
                })

        return pictures_info

    def export_full_structure(self) -> Dict[str, Any]:
        """
        Export complete document structure.

        Returns:
            Dictionary containing all structure information
        """
        return {
            'summary': self.get_document_summary(),
            'term frequency': self.get_term_frequency(),
            'tables': self.get_tables_info(),
            'pictures': self.get_pictures_info()
        }