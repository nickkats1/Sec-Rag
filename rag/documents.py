from pathlib import Path

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document


def load_documents(file_path: str | Path) -> list[Document]:
    """Load a PDF as one Document per page.

    Args:
        file_path: where the PDF is located.

    Returns:
        One Document per page, each carrying its source and page metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"could not find file path: {file_path}")
    return PyPDFLoader(str(path)).load()


def document_key(document: Document) -> str:
    """Return a stable identity for a passage.

    Source and page are included because chunking a filing produces passages
    whose text repeats across pages; keying on text alone would silently merge
    them during fusion and miscount them during evaluation.

    Args:
        document: the passage to identify.

    Returns:
        Source, page and content joined by pipes, with missing metadata blank.
    """
    source = document.metadata.get("source", "")
    page = document.metadata.get("page", "")
    return f"{source}|{page}|{document.page_content}"
