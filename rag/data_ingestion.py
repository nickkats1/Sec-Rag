from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List

import logging


logger = logging.getLogger(__name__)

# --- Load PDF documents ---

def load_documents(file_path: str) -> List[Document]:
    """load documents from file path.
    
    Args:
        file_path: path where file is located.
        
    Returns:
        documents: PDF documents loaded from file path.
    
    Raises:
        FileNotFoundError:
        - Raised if file is not to be found.
    """
    if file_path is not None:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    else:
        raise FileNotFoundError("no file to be found")



# --- Chunk Documents ---

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







