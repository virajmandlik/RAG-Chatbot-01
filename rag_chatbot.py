# RAG Chatbot for Company Documents - Phase 1 Implementation
# This script provides the core backend logic for a command-line version of the RAG system.

import os
import glob
import time
from typing import List, Dict, Any

from dotenv import load_dotenv

# --- LLM and Vector DB Libraries ---
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

# --- Document Processing Libraries ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
# Load environment variables from a .env file
load_dotenv()

# Read config with safe defaults
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "company-files")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBEDDING_MODEL = os.getenv("GENAI_EMBEDDING_MODEL", "models/text-embedding-004")
GENERATIVE_MODEL = os.getenv("GENAI_GENERATIVE_MODEL", "models/gemini-1.5-flash-latest")

DOCUMENTS_DIR = "./documents"  # Ensure this folder exists and contains your files

# --- Initialize APIs ---
try:
    if not GOOGLE_API_KEY or not PINECONE_API_KEY:
        raise ValueError("Missing GOOGLE_API_KEY or PINECONE_API_KEY in environment.")

    genai.configure(api_key=GOOGLE_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
except Exception as e:
    print("\n🚨 ERROR: API keys not found or invalid.\n"
          "Please copy .env.example to .env and add your GOOGLE_API_KEY and PINECONE_API_KEY.\n"
          f"Details: {e}\n")
    raise SystemExit(1)


# --- Core Functions ---

def load_documents(directory_path: str):
    """Loads all supported documents from a specified directory."""
    print(f"📚 Loading documents from '{directory_path}'...")
    documents = []
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)

    for file_path in glob.glob(os.path.join(directory_path, "*.*")):
        fp_lower = file_path.lower()
        try:
            if fp_lower.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif fp_lower.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif fp_lower.endswith(".txt") or fp_lower.endswith(".md"):
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                print(f"  - Skipping unsupported file type: {file_path}")
                continue

            print(f"  - Loading {os.path.basename(file_path)}")
            documents.extend(loader.load())
        except Exception as e:
            print(f"  - Failed to load {file_path}: {e}")

    print(f"✅ Loaded {len(documents)} document pages/files.")
    return documents


def split_text_into_chunks(documents):
    """Splits loaded documents into smaller chunks for processing."""
    print("🔪 Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks.")
    return chunks


def get_or_create_pinecone_index(index_name: str):
    """Checks if a Pinecone index exists, and if not, creates it."""
    print(f"🌲 Checking for Pinecone index '{index_name}'...")
    existing = pc.list_indexes().names()
    if index_name not in existing:
        print("  - Index not found. Creating new index...")
        pc.create_index(
            name=index_name,
            dimension=768,  # Gemini embeddings have 768 dimensions
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        # Wait for readiness (simple delay). For production, poll describe_index.
        time.sleep(2)
        print("✅ Index created.")
    else:
        print("✅ Index already exists.")
    return pc.Index(index_name)


def _extract_embedding_values(result: Dict[str, Any]) -> List[float]:
    """Helper to normalize embedding result shapes across SDK versions."""
    # Common shapes:
    # {"embedding": [..floats..]} OR {"embedding": {"values": [..floats..]}}
    emb = result.get("embedding")
    if isinstance(emb, dict) and "values" in emb:
        return emb["values"]
    if isinstance(emb, list):
        return emb
    # Fallback: some SDKs return list directly
    if isinstance(result, list) and result and isinstance(result[0], (float, int)):
        return result
    raise ValueError("Unexpected embedding response shape from Google Generative AI SDK.")


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embeds a list of texts using Gemini embeddings, robust to SDK variations."""
    vectors: List[List[float]] = []
    for t in texts:
        res = genai.embed_content(model=EMBEDDING_MODEL, content=t)
        vectors.append(_extract_embedding_values(res))
    return vectors


def embed_and_upsert_chunks(index, chunks, batch_size: int = 64):
    """Generates embeddings for text chunks and upserts them into Pinecone."""
    print("🧠 Generating embeddings and upserting to Pinecone...")

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        texts_to_embed = [chunk.page_content for chunk in batch_chunks]

        print(f"  - Embedding batch {i // batch_size + 1} ({len(batch_chunks)} chunks)...")
        embeddings = embed_texts(texts_to_embed)

        vectors_to_upsert = []
        for j, chunk in enumerate(batch_chunks):
            source_name = os.path.basename(str(chunk.metadata.get("source", "unknown")))
            vector = {
                "id": f"chunk_{i + j}",
                "values": embeddings[j],
                "metadata": {
                    "text": chunk.page_content,
                    "source": source_name,
                },
            }
            vectors_to_upsert.append(vector)

        print(f"  - Upserting batch {i // batch_size + 1} to Pinecone...")
        index.upsert(vectors=vectors_to_upsert)

    print("✅ All chunks have been embedded and stored.")


def build_prompt(context: str, query: str) -> str:
    return (
        "You are a helpful HR assistant designed to answer employee questions based on company policies.\n\n"
        "Use the following information from the internal documents to provide a concise and accurate answer.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{query}\n\n"
        "If the answer cannot be found in the provided context, respond with \"I cannot find the answer in the available resources.\"\n\n"
        "ANSWER:"
    )


def get_rag_response(index, query: str) -> str:
    """Performs the RAG process: embed query, search, and generate answer."""
    print("\n🔎 Processing query with RAG...")

    # 1. Embed the user's query
    print("  - Embedding your query...")
    query_vec = embed_texts([query])[0]

    # 2. Query Pinecone for relevant context
    print("  - Searching for relevant documents...")
    query_results = index.query(vector=query_vec, top_k=3, include_metadata=True)

    matches = query_results.get("matches", []) if isinstance(query_results, dict) else getattr(query_results, "matches", [])

    # 3. Assemble the context
    context_parts: List[str] = []
    for m in matches:
        md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})
        text = md.get("text", "")
        if text:
            context_parts.append(text)

    context = "\n\n".join(context_parts)

    if not context.strip():
        return "I cannot find the answer in the available resources."

    # 4. Construct the prompt
    prompt = build_prompt(context, query)

    # 5. Generate the final answer
    print("  - Generating final answer...")
    model = genai.GenerativeModel(GENERATIVE_MODEL)
    response = model.generate_content(prompt)

    # Safely extract text
    try:
        return (response.text or "").strip() or "I cannot find the answer in the available resources."
    except Exception:
        return "I cannot find the answer in the available resources."


def main():
    """Main function to run the setup and chat loop."""
    print("--- RAG Chatbot Initialization ---")

    # Ensure the documents directory exists
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        print(
            f"Created a '{DOCUMENTS_DIR}' directory. Please add your PDF, DOCX, MD, or TXT files there and restart."
        )
        return

    # --- Setup Phase ---
    docs = load_documents(DOCUMENTS_DIR)
    if not docs:
        print(f"No documents found in '{DOCUMENTS_DIR}'. Please add files to start.")
        return

    chunks = split_text_into_chunks(docs)
    pinecone_index = get_or_create_pinecone_index(PINECONE_INDEX_NAME)

    # Optional: Clear the index before upserting new data to avoid duplicates
    # print("  - Clearing existing data from index...")
    # pinecone_index.delete(delete_all=True)

    embed_and_upsert_chunks(pinecone_index, chunks)

    print("\n--- Setup Complete. You can now ask questions. ---")

    # --- Chat Loop ---
    try:
        while True:
            user_query = input("\n👤 You: ")
            if user_query.lower().strip() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            if not user_query.strip():
                continue

            response = get_rag_response(pinecone_index, user_query)
            print(f"\n🤖 Assistant:\n{response}")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
