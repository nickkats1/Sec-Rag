import re

import bm25s

from rag.documents import Document

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric terms.

    Sparse retrieval over 10-K text lives or dies on this: headings like
    ``Item 1A. Risk Factors`` only match a plain query once the casing and the
    punctuation are gone. Graph retrieval matches entity names with the same
    function so the two agree on what a term is.

    Args:
        text: string to split.

    Returns:
        The terms, in order; empty when the text holds no alphanumerics.
    """
    return _WORD.findall(text.lower())


class BM25Retriever:
    """Sparse retrieval scoring passages on the query terms they contain.

    Attributes:
        documents: every passage held, in insertion order.
    """

    def __init__(self) -> None:
        """Start with an empty index."""
        self.documents: list[Document] = []
        self._bm25: bm25s.BM25 | None = None

    def __len__(self) -> int:
        """Number of passages held.

        Returns:
            The passage count, zero before anything is added. A passage that
            tokenises to nothing is still held, and still counts.
        """
        return len(self.documents)

    def __repr__(self) -> str:
        """Summarise how much is held."""
        return f"BM25Retriever(documents={len(self.documents)})"

    def add_documents(self, documents: list[Document]) -> None:
        """Add passages to the index, keeping any added earlier.

        bm25s computes the corpus statistics at index time, so the index is
        rebuilt over the accumulated corpus rather than over the new passages
        alone.

        Args:
            documents: Document objects to add.
        """
        if not documents:
            return
        self.documents.extend(documents)
        corpus = [tokenize(doc.page_content) for doc in self.documents]
        if any(corpus):
            self._bm25 = bm25s.BM25()
            self._bm25.index(corpus, show_progress=False)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Return the passages scoring highest on the query's terms.

        Args:
            query: search string.
            top_k: maximum number of results to return.

        Returns:
            Ranked passages, best first, and only ones holding at least one
            query term. A corpus that matches nothing gives an empty list
            rather than an arbitrary ordering of unrelated passages.
        """
        terms = tokenize(query)
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)
        matched = [index for index in range(len(self.documents)) if scores[index] > 0]
        matched.sort(key=lambda index: -scores[index])
        return [self.documents[index] for index in matched[:top_k]]


__all__ = ["BM25Retriever", "tokenize"]
