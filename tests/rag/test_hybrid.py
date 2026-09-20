from langchain_core.documents import Document

from rag.dense import DenseRetriever
from rag.hybrid import HybridRetriever

# ---------------------------------------------------------------------------
# TestHybridRetriever
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    """Tests for HybridRetriever (BM25 + Dense via RRF fusion)."""

    def test_retrieve_returns_list(self, sample_docs):
        """Result is a list."""
        retriever = HybridRetriever()
        retriever.add_documents(sample_docs)
        assert isinstance(retriever.retrieve("financial"), list)

    def test_retrieve_returns_document_objects(self, sample_docs):
        """Every returned element is a langchain Document."""
        retriever = HybridRetriever()
        retriever.add_documents(sample_docs)
        result = retriever.retrieve("financial")
        assert all(isinstance(doc, Document) for doc in result)

    def test_retrieve_non_empty(self, sample_docs):
        """Results are non-empty after indexing."""
        retriever = HybridRetriever()
        retriever.add_documents(sample_docs)
        assert len(retriever.retrieve("fox", top_k=3)) > 0

    def test_no_duplicate_documents(self, sample_docs):
        """Fused results contain no duplicate Documents."""
        retriever = HybridRetriever()
        retriever.add_documents(sample_docs)
        result = retriever.retrieve("fox risk financial", top_k=3)
        contents = [doc.page_content for doc in result]
        assert len(contents) == len(set(contents))

    def test_uses_injected_sub_retrievers(self, sample_docs):
        """A sub-retriever indexed beforehand is queried without re-indexing."""
        dense = DenseRetriever()
        dense.add_documents(sample_docs)
        retriever = HybridRetriever(dense=dense)
        assert len(retriever.retrieve("financial", top_k=1)) == 1

    def test_same_text_on_different_pages_is_not_collapsed(self, repeated_text_docs):
        """Identical text on two pages yields two results, not one."""
        retriever = HybridRetriever()
        retriever.add_documents(repeated_text_docs)
        result = retriever.retrieve("risk factors", top_k=5)
        assert len(result) == 2

    def test_one_chunk_per_page(self, same_page_docs):
        """Two chunks from the same page collapse to the better-ranked one."""
        retriever = HybridRetriever()
        retriever.add_documents(same_page_docs)
        result = retriever.retrieve("risk factors", top_k=5)
        assert [doc.metadata["page"] for doc in result] == [1, 2]
        assert result[0].page_content == "Risk factors include market volatility."
