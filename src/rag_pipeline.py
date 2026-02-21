"""RAG pipeline for querying PDF documents."""

import logging
import os
from typing import Any, List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Prompt ---

RAG_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "Given the following context, answer the question as accurately as possible.\n"
        "If you do not know the answer, say so. Do not make anything up.\n\n"
        "Question: {question}\n\n"
        "Context: {context}\n\n"
        "Answer:"
    ),
)


# --- Ingestion ---

def load_documents(file_path: str) -> List[Document]:
    """Load PDF documents from a file or directory.

    Args:
        file_path: Path to a single PDF or a directory of PDFs.

    Returns:
        List of loaded Document objects.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {file_path}")

    loader = (
        PyPDFLoader(str(path))
        if path.is_file()
        else DirectoryLoader(str(path), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    )
    documents = loader.load()
    logger.info("Loaded %d document(s) from '%s'", len(documents), file_path)
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Split documents into chunks for a vector store.

    Args:
        documents: Documents to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlapping characters between chunks.

    Returns:
        List of chunked Document objects.

    Raises:
        ValueError: If documents is empty or chunk params are invalid.
    """
    if not documents:
        raise ValueError("No documents provided")
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "\t", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunk(s)", len(chunks))
    return chunks


# --- Retriever ---

def build_vector_store(
    chunks: List[Document],
    persist_directory: str = "./db",
) -> Chroma:
    """Build and persist a Chroma vector store.

    Args:
        chunks: Chunked documents to embed and store.
        persist_directory: Directory to persist the vector store.

    Returns:
        Chroma vector store.
    """
    os.makedirs(persist_directory, exist_ok=True)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory,
    )
    logger.info("Vector store built with %d chunks", len(chunks))
    return vector_store


def build_retriever(
    vector_store: Chroma,
    search_type: str = "similarity",
    k: int,
) -> Any:
    """Create a retriever from a Chroma vector store.

    Args:
        vector_store: Chroma vector store to retrieve from.
        search_type: Retrieval strategy (e.g. "similarity", "mmr").
        k: Number of documents to retrieve.

    Returns:
        Configured retriever.
    """
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )


# --- RAG ---

def run_rag(
    question: str,
    retriever: Any,
    model: str,
    temperature: float,
) -> str:
    """Run a RAG query and return the LLM response.

    Args:
        question: Question to ask.
        retriever: Configured retriever to fetch context.
        model: OpenAI model to use.
        temperature: LLM temperature.

    Returns:
        LLM generated answer as a string.
    """
    llm = ChatOpenAI(model=model, temperature=temperature)
    context = retriever.invoke(question)
    chain = RAG_PROMPT | llm
    result = chain.invoke({"question": question, "context": context})
    logger.info("RAG query completed")
    return result.content