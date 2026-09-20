import pytest
from langchain_core.documents import Document

from rag.bm25 import BM25Retriever
from rag.hyde import HydeRetriever
from rag.prompts import HYDE_PROMPT

# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def generator(scripted_generator):
    """A generator returning a passage about risk factors."""
    return scripted_generator(default="Risk factors include market volatility.")


# ---------------------------------------------------------------------------
# TestHydeRetriever
# ---------------------------------------------------------------------------


class TestHydeRetriever:
    """Tests for HydeRetriever."""

    def test_searches_with_hypothetical_not_query(self, generator, recording_retriever):
        """The base retriever receives the generated passage, not the query."""
        base = recording_retriever()
        HydeRetriever(base_retriever=base, generator=generator).retrieve("What risks?")
        assert base.calls == [("Risk factors include market volatility.", 5)]

    def test_prompt_embeds_the_question(self, generator, recording_retriever):
        """The question is interpolated into the prompt sent to the generator."""
        base = recording_retriever()
        HydeRetriever(base_retriever=base, generator=generator).retrieve("What risks?")
        assert "What risks?" in generator.prompts[0]

    def test_uses_default_prompt(self, generator, recording_retriever):
        """The default 10-K prompt is used when none is supplied."""
        base = recording_retriever()
        HydeRetriever(base_retriever=base, generator=generator).retrieve("What risks?")
        assert generator.prompts[0] == HYDE_PROMPT.format(question="What risks?")

    def test_custom_prompt_is_used(self, generator, recording_retriever):
        """A supplied prompt template replaces the default."""
        base = recording_retriever()
        retriever = HydeRetriever(
            base_retriever=base,
            generator=generator,
            prompt="Answer: {question}",
        )
        retriever.retrieve("What risks?")
        assert generator.prompts[0] == "Answer: What risks?"

    def test_generates_once_per_query(self, generator, recording_retriever):
        """A single hypothetical document is drafted per query."""
        base = recording_retriever()
        HydeRetriever(base_retriever=base, generator=generator).retrieve("What risks?")
        assert len(generator.prompts) == 1

    def test_top_k_is_forwarded(self, generator, recording_retriever):
        """top_k reaches the base retriever unchanged."""
        base = recording_retriever()
        retriever = HydeRetriever(base_retriever=base, generator=generator)
        retriever.retrieve("What risks?", top_k=3)
        assert [top_k for _, top_k in base.calls] == [3]

    def test_add_documents_delegates_to_base(
        self, generator, paged_docs, recording_retriever
    ):
        """Indexing passes straight through to the base retriever."""
        base = recording_retriever()
        retriever = HydeRetriever(base_retriever=base, generator=generator)
        retriever.add_documents(paged_docs)
        assert base.documents == paged_docs

    def test_retrieve_returns_document_objects(self, generator, paged_docs):
        """Every returned element is a langchain Document."""
        retriever = HydeRetriever(base_retriever=BM25Retriever(), generator=generator)
        retriever.add_documents(paged_docs)
        result = retriever.retrieve("What risks?")
        assert all(isinstance(doc, Document) for doc in result)

    def test_hypothetical_passage_finds_the_matching_document(
        self, generator, paged_docs
    ):
        """A query sharing no words with the target still retrieves it."""
        retriever = HydeRetriever(base_retriever=BM25Retriever(), generator=generator)
        retriever.add_documents(paged_docs)
        result = retriever.retrieve("What could go wrong?", top_k=1)
        assert result[0].page_content == "Risk factors include market volatility."

    def test_blank_draft_falls_back_to_the_query(
        self, scripted_generator, recording_retriever
    ):
        """An empty draft means the query itself is searched, not an empty string."""
        base = recording_retriever()
        generator = scripted_generator(default="   ")
        HydeRetriever(base_retriever=base, generator=generator).retrieve("What risks?")
        assert base.calls == [("What risks?", 5)]
