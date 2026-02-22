"""Unit tests for src/rag_pipeline.py"""

import os
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.rag_pipeline import (
    load_documents,
    chunk_documents,
    build_vector_store,
    build_retriever,
    run_rag,
)

# ---------------------------------------------------------------------------
# load_documents
# ---------------------------------------------------------------------------

class TestLoadDocuments:

    def test_raises_file_not_found_for_missing_path(self):
        """Should raise FileNotFoundError when path does not exist."""
        with pytest.raises(FileNotFoundError, match="Path does not exist"):
            load_documents("/nonexistent/path/to/file.pdf")

    def test_load_single_pdf_file(self, pdf_file):
        """Should load documents from a single PDF file."""
        docs = load_documents(str(pdf_file))
        assert isinstance(docs, list)

    def test_load_from_directory(self, pdf_directory):
        """Should load documents from a directory of PDFs."""
        docs = load_documents(str(pdf_directory))
        assert isinstance(docs, list)

    def test_uses_pypdf_loader_for_file(self, pdf_file):
        """Should use PyPDFLoader when given a file path."""
        with patch("src.rag_pipeline.PyPDFLoader") as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.load.return_value = [Document(page_content="test", metadata={})]
            mock_loader_cls.return_value = mock_loader

            docs = load_documents(str(pdf_file))

            mock_loader_cls.assert_called_once_with(str(pdf_file))
            mock_loader.load.assert_called_once()
            assert len(docs) == 1

    def test_uses_directory_loader_for_dir(self, pdf_directory):
        """Should use DirectoryLoader when given a directory path."""
        with patch("src.rag_pipeline.DirectoryLoader") as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.load.return_value = [Document(page_content="test", metadata={})]
            mock_loader_cls.return_value = mock_loader

            docs = load_documents(str(pdf_directory))

            mock_loader_cls.assert_called_once()
            mock_loader.load.assert_called_once()
            assert len(docs) == 1


# ---------------------------------------------------------------------------
# chunk_documents
# ---------------------------------------------------------------------------

class TestChunkDocuments:

    def test_raises_on_empty_documents(self):
        """Should raise ValueError when given an empty list."""
        with pytest.raises(ValueError, match="No documents provided"):
            chunk_documents([], chunk_size=500, chunk_overlap=50)

    def test_raises_when_overlap_gte_chunk_size(self, sample_documents):
        """Should raise ValueError when overlap >= chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_documents(sample_documents, chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_documents(sample_documents, chunk_size=100, chunk_overlap=200)

    def test_returns_list_of_documents(self, sample_documents):
        """Should return a non-empty list of Document objects."""
        chunks = chunk_documents(sample_documents, chunk_size=200, chunk_overlap=20)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)

    def test_chunks_are_within_size(self, sample_documents):
        """Each chunk should be at most chunk_size characters long."""
        chunk_size = 100
        chunks = chunk_documents(sample_documents, chunk_size=chunk_size, chunk_overlap=10)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size


# ---------------------------------------------------------------------------
# build_vector_store
# ---------------------------------------------------------------------------

class TestBuildVectorStore:

    def test_builds_and_returns_chroma(self, sample_documents, tmp_path):
        """Should call Chroma.from_documents and return the result."""
        with patch("src.rag_pipeline.Chroma.from_documents") as mock_from_docs, \
             patch("src.rag_pipeline.OpenAIEmbeddings"):
            mock_store = MagicMock()
            mock_from_docs.return_value = mock_store

            result = build_vector_store(sample_documents, persist_directory=str(tmp_path))

            mock_from_docs.assert_called_once()
            assert result is mock_store

    def test_creates_persist_directory(self, sample_documents, tmp_path):
        """Should create the persist directory if it does not exist."""
        persist_dir = str(tmp_path / "new_db")
        with patch("src.rag_pipeline.Chroma.from_documents") as mock_from_docs, \
             patch("src.rag_pipeline.OpenAIEmbeddings"):
            mock_from_docs.return_value = MagicMock()
            build_vector_store(sample_documents, persist_directory=persist_dir)

        assert os.path.isdir(persist_dir)


# ---------------------------------------------------------------------------
# build_retriever
# ---------------------------------------------------------------------------

class TestBuildRetriever:

    def test_calls_as_retriever(self):
        """Should call as_retriever on the vector store with the right kwargs."""
        mock_store = MagicMock()
        mock_retriever = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever

        result = build_retriever(mock_store, search_type="similarity", k=4)

        mock_store.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 4},
        )
        assert result is mock_retriever

    def test_passes_mmr_search_type(self):
        """Should forward the mmr search_type correctly."""
        mock_store = MagicMock()
        build_retriever(mock_store, search_type="mmr", k=3)

        mock_store.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={"k": 3},
        )


# ---------------------------------------------------------------------------
# run_rag
# ---------------------------------------------------------------------------

class TestRunRag:
    def test_llm_initialized_with_correct_params(self):
        """Should initialize ChatOpenAI with the given model and temperature."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        mock_result = MagicMock()
        mock_result.content = "answer"

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result

        with patch("src.rag_pipeline.ChatOpenAI") as mock_llm_cls, \
             patch("src.rag_pipeline.RAG_PROMPT.__or__", return_value=mock_chain):
            run_rag("q", mock_retriever, model="gpt-3.5-turbo", temperature=0.3)

        mock_llm_cls.assert_called_once_with(model="gpt-3.5-turbo", temperature=0.3)




    
    
    
    
 
    
    
    
    
    
    
    
    
    
    

    
