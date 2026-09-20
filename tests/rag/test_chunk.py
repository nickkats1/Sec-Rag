import pytest
from langchain_core.documents import Document

from rag.chunk import chunk_documents

# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def short_document():
    """One Document whose content is short enough to test chunk boundaries."""
    return Document(
        page_content="alpha beta gamma delta epsilon",
        metadata={"source": "dummy.txt", "page": 0},
    )


@pytest.fixture
def multi_sentence_documents():
    """Multiple Documents with distinct content and metadata."""
    return [
        Document(page_content="First sentence here.", metadata={"source": "a.txt"}),
        Document(page_content="Second sentence here.", metadata={"source": "b.txt"}),
    ]


@pytest.fixture
def two_sentence_document():
    """Two sentences joined by ". " with no newline, so separators alone decide breaks."""
    return Document(
        page_content="First idea ends here. Second idea starts here.",
        metadata={"source": "dummy.txt", "page": 0},
    )


@pytest.fixture
def empty_document_list():
    """Edge case: no documents passed to chunk_documents."""
    return []


# ---------------------------------------------------------------------------
# TestChunkDocuments
# ---------------------------------------------------------------------------


class TestChunkDocuments:
    """Tests for chunk_documents(documents, chunk_size, chunk_overlap, separators)."""

    def test_returns_list(self, short_document):
        """Result is always a list."""
        result = chunk_documents([short_document], chunk_size=10, chunk_overlap=0)
        assert isinstance(result, list)

    def test_returns_document_objects(self, short_document):
        """Every returned element is a Document."""
        result = chunk_documents([short_document], chunk_size=10, chunk_overlap=0)
        assert all(isinstance(chunk, Document) for chunk in result)

    def test_chunk_count_exceeds_input(self, short_document):
        """Long text split with a small chunk_size yields more chunks than input docs."""
        result = chunk_documents([short_document], chunk_size=8, chunk_overlap=0)
        assert len(result) > 1

    def test_chunks_respect_chunk_size(self, short_document):
        """No chunk's page_content exceeds chunk_size characters."""
        chunk_size = 10
        result = chunk_documents([short_document], chunk_size=chunk_size, chunk_overlap=0)
        assert all(len(chunk.page_content) <= chunk_size for chunk in result)

    def test_overlap_produces_shared_content(self):
        """With overlap > 0, consecutive chunks share a character sequence.

        Uses a space-free string so the splitter can overlap at any character
        position rather than being forced to a word boundary.
        """
        doc = Document(page_content="abcdefghijklmnop", metadata={"source": "x"})
        result = chunk_documents([doc], chunk_size=8, chunk_overlap=4)
        assert len(result) >= 2
        first_end = result[0].page_content[-4:]
        second_start = result[1].page_content[:4]
        assert first_end == second_start

    def test_metadata_preserved_after_chunking(self, short_document):
        """Source metadata from the original Document carries through to chunks."""
        result = chunk_documents([short_document], chunk_size=10, chunk_overlap=0)
        for chunk in result:
            assert chunk.metadata.get("source") == "dummy.txt"

    def test_multiple_input_documents(self, multi_sentence_documents):
        """Chunks from multiple Documents are all returned in one flat list."""
        result = chunk_documents(
            multi_sentence_documents, chunk_size=100, chunk_overlap=0
        )
        sources = {chunk.metadata["source"] for chunk in result}
        assert sources == {"a.txt", "b.txt"}

    def test_empty_input_returns_empty_list(self, empty_document_list):
        """Empty input yields an empty list without raising."""
        result = chunk_documents(empty_document_list, chunk_size=50, chunk_overlap=0)
        assert result == []

    def test_default_separators_break_between_sentences(self, two_sentence_document):
        """The default list prefers ". " over " ", so a break lands after a sentence.

        The separator is kept at the start of the following chunk, and the same text
        split on the space-only list cuts "Second" away from "idea" instead.
        """
        default = chunk_documents([two_sentence_document], chunk_size=30, chunk_overlap=0)
        assert [chunk.page_content for chunk in default] == [
            "First idea ends here",
            ". Second idea starts here.",
        ]
