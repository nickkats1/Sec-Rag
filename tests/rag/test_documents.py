import pytest
from langchain_core.documents import Document

from rag.documents import load_documents

# ---------------------------------------------------------------------------
# TestLoadDocuments
# ---------------------------------------------------------------------------


class TestLoadDocuments:
    """Tests for load_documents(file_path)."""

    def test_returns_list(self, pdf_file):
        """Result is a list."""
        result = load_documents(str(pdf_file))
        assert isinstance(result, list)

    def test_returns_document_objects(self, pdf_file):
        """Every element is a langchain Document."""
        result = load_documents(str(pdf_file))
        assert all(isinstance(doc, Document) for doc in result)

    def test_one_document_per_page(self, pdf_file):
        """Single-page PDF produces exactly one Document."""
        result = load_documents(str(pdf_file))
        assert len(result) == 1

    def test_document_has_source_metadata(self, pdf_file):
        """Each Document carries a 'source' key in its metadata."""
        result = load_documents(str(pdf_file))
        for doc in result:
            assert "source" in doc.metadata

    def test_source_metadata_matches_path(self, pdf_file):
        """The 'source' metadata value equals the file path that was loaded."""
        result = load_documents(str(pdf_file))
        assert result[0].metadata["source"] == str(pdf_file)

    def test_missing_file_raises(self, tmp_path):
        """A path that does not exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_documents(str(tmp_path / "absent.txt"))
