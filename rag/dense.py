from collections.abc import Callable

import faiss
import numpy as np

from rag.documents import Document
from rag.embeddings import DEFAULT_BI_ENCODER, embed_texts


class DenseRetriever:
    """Semantic retrieval over a FAISS index of bi-encoder embeddings.

    Attributes:
        model_name: bi-encoder used for both passages and queries.
        embed: turns texts into L2-normalised vectors, so an inner product is
            already a cosine similarity.
        documents: every passage held, positionally aligned with the index.
        index: a flat inner-product FAISS index holding the vectors, which
            live here and nowhere else; None until the first passages arrive.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BI_ENCODER,
        embed: Callable[[list[str], str], np.ndarray] = embed_texts,
    ) -> None:
        """Start with an empty index.

        Args:
            model_name: HuggingFace bi-encoder identifier.
            embed: encoder to embed passages and queries with.
        """
        self.model_name = model_name
        self.embed = embed
        self.documents: list[Document] = []
        self.index: faiss.Index | None = None

    def __len__(self) -> int:
        """Number of passages held.

        Returns:
            The passage count, zero before anything is added.
        """
        return len(self.documents)

    def __repr__(self) -> str:
        """Summarise the bi-encoder in use and how much is held."""
        return (
            f"DenseRetriever(model_name={self.model_name!r}, "
            f"documents={len(self.documents)})"
        )

    def add_documents(self, documents: list[Document]) -> None:
        """Embed and index passages, keeping any added earlier.

        Args:
            documents: Document objects to add.
        """
        if not documents:
            return
        vectors = self.embed([doc.page_content for doc in documents], self.model_name)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.documents.extend(documents)

    def retrieve(self, query: str, top_k: int = 4) -> list[Document]:
        """Return the passages whose embeddings sit closest to the query's.

        Args:
            query: search string.
            top_k: maximum number of results to return.

        Returns:
            Ranked passages, best first, empty before anything is indexed. The
            -1 padding FAISS returns when fewer passages exist than top_k is
            dropped.
        """
        if self.index is None:
            return []
        query_vector = self.embed([query], self.model_name)
        _, positions = self.index.search(query_vector, min(top_k, len(self.documents)))
        return [self.documents[i] for i in positions[0] if i >= 0]
