"""
Entry point for the SEC RAG pipeline.

Usage:
    python main.py

Requirements:
    - A .env file with OPENAI_API_KEY set
    - PDF files placed in the data/ directory

The pipeline will load your PDFs, embed them into a local Chroma vector
store, and let you ask questions about the documents via an OpenAI model.
"""


import logging
import os
import sys

from dotenv import load_dotenv

from src.rag_pipeline import (
    build_retriever,
    build_vector_store,
    chunk_documents,
    load_documents,
    run_rag,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = "data/"
PERSIST_DIR = "./db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 59
SEARCH_TYPE = "similarity"
TOP_K = 10
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_pipeline(data_dir: str = DATA_DIR, persist_dir: str = PERSIST_DIR):
    """Load, chunk, embed, and return a configured retriever."""
    logger.info("Loading documents from '%s'", data_dir)
    documents = load_documents(file_path=data_dir)

    logger.info("Chunking %d document(s)", len(documents))
    chunks = chunk_documents(
        documents=documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    logger.info("Building vector store at '%s'", persist_dir)
    vector_store = build_vector_store(chunks=chunks, persist_directory=persist_dir)

    retriever = build_retriever(
        vector_store=vector_store,
        search_type=SEARCH_TYPE,
        k=TOP_K,
    )
    return retriever


def ask(company_name: str, retriever) -> str:
    """Run a RAG query for a given company name."""
    question = (
        f"How much did {company_name} spend on Research and development "
        f"in 2025 and the previous years?"
    )
    logger.info("Running RAG query: %s", question)
    return run_rag(
        question=question,
        retriever=retriever,
        model=MODEL,
        temperature=TEMPERATURE,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    try:
        retriever = build_pipeline()
    except FileNotFoundError as e:
        logger.error("Could not load documents: %s", e)
        sys.exit(1)

    company_name = input("Enter the name of the company in your PDF file: ").strip()
    if not company_name:
        logger.error("Company name cannot be empty.")
        sys.exit(1)

    result = ask(company_name, retriever)
    print(f"\nResult: {result}\n")


if __name__ == "__main__":
    main()