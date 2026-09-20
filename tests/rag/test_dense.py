from langchain_core.documents import Document

from rag.dense import DenseRetriever

# ---------------------------------------------------------------------------
# TestDenseRetriever
# ---------------------------------------------------------------------------


class TestDenseRetriever:
    """Tests for DenseRetriever."""

    def test_retrieve_before_add_returns_empty(self):
        """Retrieval before indexing returns an empty list."""
        retriever = DenseRetriever()
        assert retriever.retrieve("financial") == []

    def test_retrieve_returns_list(self, sample_docs):
        """Result after indexing is a list."""
        retriever = DenseRetriever()
        retriever.add_documents(sample_docs)
        assert isinstance(retriever.retrieve("financial"), list)

    def test_retrieve_returns_document_objects(self, sample_docs):
        """Every returned element is a langchain Document."""
        retriever = DenseRetriever()
        retriever.add_documents(sample_docs)
        result = retriever.retrieve("financial")
        assert all(isinstance(doc, Document) for doc in result)

    def test_top_k_limits_results(self, sample_docs):
        """Result length does not exceed top_k."""
        retriever = DenseRetriever()
        retriever.add_documents(sample_docs)
        assert len(retriever.retrieve("financial", top_k=2)) <= 2

    def test_add_documents_appends(self, sample_docs):
        """A second add_documents call keeps the Documents from the first."""
        retriever = DenseRetriever()
        retriever.add_documents(sample_docs[:1])
        retriever.add_documents(sample_docs[1:])
        assert len(retriever.retrieve("financial", top_k=10)) == 3

    def test_add_no_documents_leaves_the_index_empty(self):
        """Adding an empty list is a no-op rather than an error."""
        retriever = DenseRetriever()
        retriever.add_documents([])
        assert retriever.retrieve("financial") == []

    def test_does_not_duplicate_the_faiss_vectors(self, sample_docs):
        """Vectors live only in the FAISS index, not in a parallel matrix."""
        retriever = DenseRetriever()
        retriever.add_documents(sample_docs)
        assert not hasattr(retriever, "_embeddings")
