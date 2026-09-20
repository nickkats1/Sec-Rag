import pytest
from langchain_core.documents import Document

from rag.bm25 import BM25Retriever
from rag.reranker import RerankerRetriever

# ---------------------------------------------------------------------------
# Local doubles and fixtures
# ---------------------------------------------------------------------------


class ScriptedCrossEncoder:
    """Cross-encoder double scoring pairs in the order they arrive.

    Attributes:
        scores: the relevance scores handed back, positionally.
        pairs: the (query, passage) pairs the reranker asked about.
    """

    def __init__(self, *scores):
        self.scores = list(scores)
        self.pairs = []

    def predict(self, pairs):
        self.pairs = list(pairs)
        return self.scores[: len(self.pairs)]


@pytest.fixture
def candidates() -> list[Document]:
    """Three Documents a base retriever can hand to the reranker."""
    return [
        Document(page_content="alpha", metadata={"source": "a.pdf"}),
        Document(page_content="beta", metadata={"source": "b.pdf"}),
        Document(page_content="gamma", metadata={"source": "c.pdf"}),
    ]


@pytest.fixture
def scored_base(recording_retriever, candidates):
    """A base retriever holding the candidates, and a model scoring them 0.1/0.9/0.5."""

    def build(**kwargs):
        base = recording_retriever(candidates)
        model = ScriptedCrossEncoder(0.1, 0.9, 0.5)
        return (
            base,
            model,
            RerankerRetriever(base_retriever=base, cross_encoder=model, **kwargs),
        )

    return build


# ---------------------------------------------------------------------------
# TestRerankerRetriever
# ---------------------------------------------------------------------------


class TestRerankerRetriever:
    """Tests for RerankerRetriever (cross-encoder reranking wrapper)."""

    def test_reorders_candidates_by_score(self, scored_base):
        """The shortlist comes back ordered by cross-encoder score, not base rank."""
        _, _, retriever = scored_base()
        result = retriever.retrieve("query", top_k=3)
        assert [doc.page_content for doc in result] == ["beta", "gamma", "alpha"]

    def test_top_k_truncates_after_reranking(self, scored_base):
        """Truncation drops the lowest-scored passage, not the last one retrieved."""
        _, _, retriever = scored_base()
        result = retriever.retrieve("query", top_k=2)
        assert [doc.page_content for doc in result] == ["beta", "gamma"]

    def test_asks_the_base_for_candidate_pool_not_top_k(self, scored_base):
        """The base is asked for the whole shortlist, however small top_k is."""
        base, _, retriever = scored_base(candidate_pool=7)
        retriever.retrieve("query", top_k=1)
        assert base.calls == [("query", 7)]

    def test_scores_every_candidate_against_the_query(self, scored_base):
        """Each passage is paired with the query exactly once, in base order."""
        _, model, retriever = scored_base()
        retriever.retrieve("risks?")
        assert model.pairs == [
            ("risks?", "alpha"),
            ("risks?", "beta"),
            ("risks?", "gamma"),
        ]

    def test_no_candidates_returns_empty(self, recording_retriever):
        """An empty shortlist returns early, without loading or calling a model."""
        retriever = RerankerRetriever(
            base_retriever=recording_retriever(),
            cross_encoder=ScriptedCrossEncoder(),
        )
        assert retriever.retrieve("query") == []

    def test_add_documents_delegates_to_base(self, recording_retriever, sample_docs):
        """Indexing passes straight through to the base retriever."""
        base = recording_retriever()
        RerankerRetriever(base_retriever=base).add_documents(sample_docs)
        assert base.documents == sample_docs

    def test_default_model_reranks(self, sample_docs):
        """With no injected model the named cross-encoder is loaded and used."""
        retriever = RerankerRetriever(base_retriever=BM25Retriever(), candidate_pool=3)
        retriever.add_documents(sample_docs)
        result = retriever.retrieve("What financial disclosures are filed?", top_k=1)
        assert result[0].page_content == "SEC filings contain financial disclosures."
