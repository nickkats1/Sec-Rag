from rag.bm25 import BM25Retriever
from rag.dense import DenseRetriever
from rag.documents import Document, document_key


class HybridRetriever:
    """Retrieval fusing BM25 and dense results by reciprocal rank.

    Attributes:
        bm25: the sparse half.
        dense: the dense half.
        bm25_weight: weight applied to the sparse half's contribution.
        dense_weight: weight applied to the dense half's contribution.
        candidates: number of results requested from each half before fusing.
        rrf_k: rank offset damping the influence of the very top ranks.
    """

    def __init__(
        self,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        candidates: int = 40,
        rrf_k: int = 60,
    ) -> None:
        """Initialise the fusion.

        Args:
            bm25: sparse retriever; a fresh BM25Retriever when omitted. Pass one
                that is already indexed to reuse it without re-indexing.
            dense: dense retriever; a fresh DenseRetriever when omitted.
            bm25_weight: weight applied to the sparse half's contribution.
            dense_weight: weight applied to the dense half's contribution.
            candidates: number of results requested from each half before fusing.
            rrf_k: rank offset damping the influence of the very top ranks.
        """
        self.bm25 = bm25 if bm25 is not None else BM25Retriever()
        self.dense = dense if dense is not None else DenseRetriever()
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.candidates = candidates
        self.rrf_k = rrf_k

    def add_documents(self, documents: list[Document]) -> None:
        """Index passages into both halves.

        Args:
            documents: Document objects to add.
        """
        self.bm25.add_documents(documents)
        self.dense.add_documents(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Return the passages both halves agree on most strongly.

        Args:
            query: search string.
            top_k: maximum number of results to return.

        Returns:
            Ranked passages, at most one per page. A page chunked into several
            passages keeps only its best-ranked one.
        """
        scores: dict[str, float] = {}
        found: dict[str, Document] = {}
        halves = (
            (self.bm25.retrieve(query, top_k=self.candidates), self.bm25_weight),
            (self.dense.retrieve(query, top_k=self.candidates), self.dense_weight),
        )
        for documents, weight in halves:
            for rank, document in enumerate(documents, start=1):
                key = document_key(document)
                found.setdefault(key, document)
                scores[key] = scores.get(key, 0.0) + weight / (self.rrf_k + rank)

        results: list[Document] = []
        seen_pages: set[tuple[str, str]] = set()
        for key in sorted(scores, key=lambda key: -scores[key]):
            document = found[key]
            page = (document.metadata.get("source", ""), document.metadata.get("page", ""))
            if page in seen_pages:
                continue
            seen_pages.add(page)
            results.append(document)
            if len(results) == top_k:
                break
        return results
