"""
Vector store management for document storage and retrieval.

Deze versie gebruikt HuggingFace/SentenceTransformer embeddings in plaats van OpenAI embeddings.
Daardoor is geen OpenAI API-key nodig.
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from numpy import rint
from torch import chunk
import uuid

class VectorStoreManager:
    """Manages document chunking, embedding, and vector storage."""

    def __init__(
        self,
        embedding_model_name = "intfloat/multilingual-e5-small",
        chunk_size = 1500,
        chunk_overlap = 300,
        ):

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, 
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True})

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ".", " ", ""])

    def chunk_documents(self, documents):
        """
        Split documents into smaller chunks for better retrieval.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        print(f"✂️ Chunking {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)

        print("\n=== CREATED CHUNKS ===")

        for i, chunk in enumerate(chunks[:20], 1):
            print(f"\n--- Chunk {i} ---")
            print("Filename:", chunk.metadata.get("filename"))
            print(chunk.page_content)

        print("\n======================\n")

        clean_chunks = []

        for chunk in chunks:
            text = chunk.page_content.strip()

            if set(text) <= {"-", "|", " "}:
                continue

            clean_chunks.append(chunk)

        chunks = clean_chunks

        print(f"✅ Created {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks):
        """
        Create a Chroma vector store from document chunks.

        Args:
            chunks: List of document chunks

        Returns:
            Chroma vector store instance
        """
        print(f"🔢 Creating vector store with {len(chunks)} chunks...")

        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=f"documents_{uuid.uuid4().hex}"
            )
            print("✅ Vector store created successfully")
            return vectorstore

        except Exception as e:
            print(f"❌ Error creating vector store: {str(e)}")
            raise

    def search_similar(self, vectorstore, query, k=10):
        """
        Perform semantic similarity search.

        Args:
            vectorstore: The Chroma vector store
            query: Search query
            k: Number of results to return

        Returns:
            List of similar documents
        """
        try:
            results = vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"❌ Error searching vector store: {str(e)}")
            return []