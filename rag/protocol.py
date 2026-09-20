from typing import Protocol

from langchain_core.documents import Document


class Retriever(Protocol):
    """Minimal retrieval interface.

    Structural typing keeps the strategies free of a base class, so a
    hand-written stand-in satisfies it without subclassing.
    """

    def add_documents(self, documents: list[Document]) -> None:
        """Add passages to the index, keeping any added earlier.

        Args:
            documents: langchain Document objects to add.
        """
        ...

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Return the passages best matching a query.

        Args:
            query: search string.
            top_k: maximum number of results to return.

        Returns:
            Ranked passages, empty before anything is indexed.
        """
        ...


__all__ = ["Retriever"]
