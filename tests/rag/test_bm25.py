import pytest
from langchain_core.documents import Document

from rag.bm25 import BM25Retriever, tokenize

# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def heading_docs() -> list[Document]:
    """Documents whose text uses 10-K heading capitalization and punctuation.

    The target heading is deliberately last: with a tokenizer that fails to
    match it, every score ties at zero and insertion order would return the
    wrong document. Three documents are the minimum that give the matched
    terms a non-zero BM25 IDF.
    """
    return [
        Document(
            page_content="Item 2. Properties",
            metadata={"source": "10k.pdf", "page": 9},
        ),
        Document(
            page_content="Item 3. Legal Proceedings",
            metadata={"source": "10k.pdf", "page": 11},
        ),
        Document(
            page_content="Item 1A. Risk Factors",
            metadata={"source": "10k.pdf", "page": 7},
        ),
    ]


# ---------------------------------------------------------------------------
# TestTokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    """Tests for the tokenizer BM25 and graph retrieval share."""

    def test_splits_on_punctuation_and_lowercases(self):
        """A 10-K heading tokenizes to its bare lowercase terms."""
        assert tokenize("Item 1A. Risk Factors") == ["item", "1a", "risk", "factors"]

    def test_discards_non_alphanumeric_text(self):
        """Text with no alphanumeric characters yields no tokens."""
        assert tokenize("--- ... ---") == []

    def test_empty_string_yields_no_tokens(self):
        """The empty string yields no tokens."""
        assert tokenize("") == []


# ---------------------------------------------------------------------------
# TestBM25Retriever
# ---------------------------------------------------------------------------


class TestBM25Retriever:
    """Tests for BM25Retriever."""

    def test_retrieve_before_add_returns_empty(self):
        """Retrieval before indexing returns an empty list."""
        retriever = BM25Retriever()
        assert retriever.retrieve("fox") == []

    def test_retrieve_returns_list(self, sample_docs):
        """Result after indexing is a list."""
        retriever = BM25Retriever()
        retriever.add_documents(sample_docs)
        assert isinstance(retriever.retrieve("fox"), list)

    def test_retrieve_returns_document_objects(self, sample_docs):
        """Every returned element is a langchain Document."""
        retriever = BM25Retriever()
        retriever.add_documents(sample_docs)
        result = retriever.retrieve("fox")
        assert all(isinstance(doc, Document) for doc in result)

    def test_retrieve_returns_relevant_doc(self, sample_docs):
        """Top-1 result for 'fox' contains the word 'fox'."""
        retriever = BM25Retriever()
        retriever.add_documents(sample_docs)
        result = retriever.retrieve("fox", top_k=1)
        assert "fox" in result[0].page_content.lower()

    def test_top_k_limits_results(self, sample_docs):
        """Result length does not exceed top_k."""
        retriever = BM25Retriever()
        retriever.add_documents(sample_docs)
        assert len(retriever.retrieve("fox", top_k=2)) <= 2

    def test_retrieve_ignores_case_and_punctuation(self, heading_docs):
        """A lowercase query matches a capitalized, punctuated 10-K heading."""
        retriever = BM25Retriever()
        retriever.add_documents(heading_docs)
        result = retriever.retrieve("risk factors", top_k=1)
        assert result[0].page_content == "Item 1A. Risk Factors"

    def test_add_documents_appends(self, sample_docs):
        """A second add_documents call keeps the Documents from the first."""
        retriever = BM25Retriever()
        retriever.add_documents(sample_docs[:1])
        retriever.add_documents(sample_docs[1:])
        assert len(retriever.retrieve("fox financial risk", top_k=10)) == 3

    def test_empty_query_returns_empty(self, sample_docs):
        """A query with no alphanumeric terms returns nothing rather than raising."""
        retriever = BM25Retriever()
        retriever.add_documents(sample_docs)
        assert retriever.retrieve("???") == []

    def test_documents_without_terms_leave_index_empty(self):
        """A corpus that tokenizes to nothing does not build a broken index."""
        retriever = BM25Retriever()
        retriever.add_documents([Document(page_content="...", metadata={})])
        assert retriever.retrieve("risk") == []

    def test_top_k_above_corpus_size_returns_all(self, sample_docs):
        """Asking for more passages than are indexed returns every one of them."""
        retriever = BM25Retriever()
        retriever.add_documents(sample_docs)
        assert len(retriever.retrieve("fox financial risk", top_k=50)) == 3
