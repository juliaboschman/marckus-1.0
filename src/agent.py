"""
Simple RAG agent using Ollama.
No OpenAI, LangGraph or Transformers pipeline required.
"""

from annotated_types import doc
import requests
import json
from pathlib import Path

SYSTEM_PROMPT = """
Je bent een AI-assistent voor documentanalyse binnen een creatief marketingbureau.

Je taak is om vragen van gebruikers te beantwoorden op basis van aangeleverde documentfragmenten.

BELANGRIJKE REGELS:
- Gebruik uitsluitend informatie uit de aangeleverde documentfragmenten.
- Gebruik geen externe kennis, aannames of interpretaties.
- Als informatie ontbreekt, zeg dan duidelijk:
  "Deze informatie is niet terug te vinden in de aangeleverde fragmenten."
- Verzin nooit informatie.
- Geef korte, duidelijke en feitelijke antwoorden.
- Antwoord in correct, natuurlijk en professioneel Nederlands.
- Focus eerst direct op het beantwoorden van de vraag.
- Geef alleen extra uitleg wanneer dit relevant is voor het antwoord.
- Gebruik opsommingen alleen wanneer dit helpt bij de leesbaarheid.
- Gedraag je als ondersteunende assistent, niet als besluitvormer.
"""


class LocalRAGAgent:
    """Simple RAG pipeline: retrieve context -> ask Ollama model."""

    def __init__(
        self,
        vectorstore,
        model_name="llama3.2:3b",
        ollama_url="http://localhost:11434/api/chat",
        max_new_tokens=400,
        temperature=0.1,
        k=5,
    ):
        self.vectorstore = vectorstore
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.k = k

    def retrieve_context(self, question: str):
        """
        Retrieve relevant chunks.
        If a filename is mentioned in the question,
        only search within that document.
        """

        all_docs = self.vectorstore.get()["documents"]
        all_metadatas = self.vectorstore.get()["metadatas"]

        lower_question = question.lower()

        # Try to detect mentioned filenames
        mentioned_filename = None

        for metadata in all_metadatas:
            filename = metadata.get("filename", "")

            if not filename:
                continue

            filename_clean = Path(filename).name.lower()
            filename_without_ext = Path(filename).stem.lower()
            
            if filename_clean in lower_question or filename_without_ext in lower_question:
                mentioned_filename = filename
                break

        # If filename is mentioned -> filter retrieval
        if mentioned_filename:
            print(f"\nFiltering retrieval to: {mentioned_filename}\n")

            docs = self.vectorstore.similarity_search(
                question,
                k=self.k,
                filter={"filename": mentioned_filename}
            )

        else:
            docs = self.vectorstore.similarity_search(
                question,
                k=self.k
            )

        return docs

    def build_context(self, docs):
        """Build readable context from retrieved documents."""
        context_parts = []

        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", doc.metadata.get("source", "Onbekende bron"))
            content = doc.page_content.strip()[:1500]

            context_parts.append(
                f"[Fragment {i} uit bron: {source}]\n{content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def build_messages(self, question, docs):
        """Build Ollama chat messages."""
        context = self.build_context(docs)

        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
Je krijgt hieronder tekstfragmenten uit één of meerdere geüploade documenten.
Een fragment is niet hetzelfde als een volledig document.
Meerdere fragmenten kunnen uit hetzelfde document komen.
Deze documentfragmenten vormen jouw ENIGE bron van informatie.

BELANGRIJK:
- Gebruik alleen informatie uit deze fragmenten.
- Gebruik geen algemene kennis over bedrijven, marketing of documenten.
- Zeg niet dat je geen toegang hebt tot documenten.
- Als de informatie onvoldoende aanwezig is, antwoord dan exact:
"Deze informatie is niet terug te vinden in de aangeleverde fragmenten."

DOCUMENTFRAGMENTEN:
{context}

VRAAG VAN DE GEBRUIKER:
{question}

Geef een kort en duidelijk antwoord in het Nederlands, gebaseerd op de documentfragmenten.
""",
            },
        ]

    def invoke(self, question):
        """Generate answer for user question."""
        docs = self.retrieve_context(question)

#        print("\n=== RETRIEVED DOCUMENTS ===")

#        for i, doc in enumerate(docs, 1):
#           print(f"\n--- Document {i} ---")
#            print("Filename:", doc.metadata.get("filename"))
#            print(doc.page_content[:300])

#       print("\n===========================\n")

        if not docs:
            yield "Ik kan hierover geen informatie vinden in de aangeleverde documenten."
            return 

        messages = self.build_messages(question, docs)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
            },
        }

        response = requests.post(self.ollama_url, json=payload, stream=True, timeout=(30,600))
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            chunk = json.loads(line.decode("utf-8"))

            content = ""

            if "message" in chunk:
                content = chunk["message"].get("content", "")

            if content:
                yield content

        sources = sorted({
            doc.metadata.get("filename", doc.metadata.get("source", "Onbekende bron"))
            for doc in docs
        })

        yield "\n\n**Gebruikte bron(nen):** " + ", ".join(sources)


def create_documentation_agent(
    vectorstore,
    model_name="bramvanroy/geitje-7b-ultra:Q5_K_M",
):
    return LocalRAGAgent(
        vectorstore=vectorstore,
        model_name=model_name,
    )