from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Chunk documents into smaller ingestion.

    Args:
        documents: documents loaded from PDF file.
        chunk_size: the size of documents manageable for retrieval.
        chunk_overlap: how much chunk size overlaps with original documents.

    Returns:
        documents chunked for retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
    )
    return splitter.split_documents(documents=documents)
