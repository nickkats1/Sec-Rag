from typing import Any

from sentence_transformers import CrossEncoder

from rag.documents import Document
from rag.protocol import Retriever

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_cross_encoder(model_name: str = DEFAULT_CROSS_ENCODER) -> CrossEncoder:
    """Load a cross-encoder model, falling back to CPU when the GPU is full.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        A CrossEncoder instance.
    """
    try:
        return CrossEncoder(model_name)
    except RuntimeError:
        return CrossEncoder(model_name, device="cpu")


class RerankerRetriever:
    """Retrieval that re-scores another retriever's shortlist with a cross-encoder.

    A bi-encoder embeds the query and the passage apart and can only compare
    the two summaries. A cross-encoder reads them together, which is far more
    accurate and far too slow to run over a corpus, so it runs over a shortlist
    the base retriever produced.

    Attributes:
        base_retriever: supplies the shortlist to rerank.
        candidate_pool: shortlist size requested from the base retriever.
        model_name: HuggingFace cross-encoder identifier.
        cross_encoder: anything with a ``predict`` over (query, passage) pairs,
            loaded on the first retrieve when not injected.
    """

    def __init__(
        self,
        base_retriever: Retriever,
        candidate_pool: int = 30,
        model_name: str = DEFAULT_CROSS_ENCODER,
        cross_encoder: Any = None,
    ) -> None:
        """Initialise the wrapper.

        Args:
            base_retriever: supplies the shortlist to rerank.
            candidate_pool: shortlist size requested from the base retriever.
            model_name: HuggingFace cross-encoder identifier.
            cross_encoder: an already-built cross-encoder; when None the model
                named by ``model_name`` is loaded on the first retrieve.
        """
        self.base_retriever = base_retriever
        self.candidate_pool = candidate_pool
        self.model_name = model_name
        self.cross_encoder = cross_encoder

    def add_documents(self, documents: list[Document]) -> None:
        """Index passages into the base retriever.

        Args:
            documents: Document objects to add.
        """
        self.base_retriever.add_documents(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Return the shortlist reordered by cross-encoder relevance.

        Args:
            query: search string.
            top_k: maximum number of results to return.

        Returns:
            Ranked passages, empty when the base retriever found nothing.
        """
        candidates = self.base_retriever.retrieve(query, top_k=self.candidate_pool)
        if not candidates:
            return []
        if self.cross_encoder is None:
            self.cross_encoder = load_cross_encoder(self.model_name)
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: -pair[1])
        return [document for document, _ in ranked[:top_k]]


__all__ = ["DEFAULT_CROSS_ENCODER", "RerankerRetriever", "load_cross_encoder"]
