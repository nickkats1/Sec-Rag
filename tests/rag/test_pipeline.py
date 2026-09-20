import pytest
from langchain_core.documents import Document

from rag.bm25 import BM25Retriever
from rag.pipeline import Answer, RAGPipeline

# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def paged_docs() -> list[Document]:
    """Three Documents from the same filing on different pages."""
    return [
        Document(
            page_content="Risk factors include market volatility.",
            metadata={"source": "10k.pdf", "page": 7},
        ),
        Document(
            page_content="Revenue grew twelve percent year over year.",
            metadata={"source": "10k.pdf", "page": 3},
        ),
        Document(
            page_content="The company operates four data centers.",
            metadata={"source": "10k.pdf", "page": 5},
        ),
    ]


# ---------------------------------------------------------------------------
# TestRAGPipeline
# ---------------------------------------------------------------------------


class TestRAGPipeline:
    """Tests for RAGPipeline."""

    def test_returns_an_answer(self, scripted_generator, paged_docs, recording_retriever):
        """The result is an Answer carrying the generated text."""
        generator = scripted_generator(default="generated answer")
        pipeline = RAGPipeline(recording_retriever(paged_docs), generator, top_k=3)
        result = pipeline.answer("what are the risks?")
        assert isinstance(result, Answer)
        assert result.text == "generated answer"

    def test_calls_the_generator_exactly_once(
        self, scripted_generator, paged_docs, recording_retriever
    ):
        """One question costs one generation call."""
        generator = scripted_generator(default="generated answer")
        RAGPipeline(recording_retriever(paged_docs), generator, top_k=3).answer("risks?")
        assert len(generator.prompts) == 1

    def test_prompt_contains_the_question(
        self, scripted_generator, paged_docs, recording_retriever
    ):
        """The question reaches the model verbatim."""
        generator = scripted_generator(default="generated answer")
        RAGPipeline(recording_retriever(paged_docs), generator, top_k=3).answer(
            "what risks?"
        )
        assert "what risks?" in generator.prompts[0]

    def test_prompt_contains_every_retrieved_passage(
        self, scripted_generator, paged_docs, recording_retriever
    ):
        """Every retrieved passage is present in the context block."""
        generator = scripted_generator(default="generated answer")
        RAGPipeline(recording_retriever(paged_docs), generator, top_k=3).answer("risks?")
        prompt = generator.prompts[0]
        assert all(doc.page_content in prompt for doc in paged_docs)

    def test_prompt_cites_source_and_page(
        self, scripted_generator, paged_docs, recording_retriever
    ):
        """Citation labels name the passage's source and page."""
        generator = scripted_generator(default="generated answer")
        RAGPipeline(recording_retriever(paged_docs), generator, top_k=3).answer("risks?")
        prompt = generator.prompts[0]
        assert all(
            f"[{doc.metadata['source']} p{doc.metadata['page']}]" in prompt
            for doc in paged_docs
        )

    def test_prompt_sends_each_passage_once(
        self, scripted_generator, paged_docs, recording_retriever
    ):
        """The citation label must not repeat the passage it labels."""
        generator = scripted_generator(default="generated answer")
        RAGPipeline(recording_retriever(paged_docs), generator, top_k=3).answer("risks?")
        prompt = generator.prompts[0]
        assert all(prompt.count(doc.page_content) == 1 for doc in paged_docs)

    def test_top_k_reaches_the_retriever(
        self, scripted_generator, paged_docs, recording_retriever
    ):
        """The configured top_k is what the retriever is asked for."""
        retriever = recording_retriever(paged_docs)
        RAGPipeline(
            retriever, scripted_generator(default="generated answer"), top_k=2
        ).answer("risks?")
        assert retriever.calls == [("risks?", 2)]

    def test_documents_are_the_retrieved_ones(
        self, scripted_generator, paged_docs, recording_retriever
    ):
        """The Answer carries exactly what the retriever returned."""
        pipeline = RAGPipeline(
            recording_retriever(paged_docs),
            scripted_generator(default="generated answer"),
            top_k=2,
        )
        assert pipeline.answer("risks?").documents == paged_docs[:2]

    def test_works_with_a_real_retriever(self, scripted_generator, paged_docs):
        """The pipeline composes with a concrete retriever end to end."""
        retriever = BM25Retriever()
        retriever.add_documents(paged_docs)
        generator = scripted_generator(default="generated answer")
        result = RAGPipeline(retriever, generator, top_k=1).answer("data centers")
        assert len(result.documents) == 1
        assert "data centers" in result.documents[0].page_content

    def test_no_documents_still_generates(self, scripted_generator, recording_retriever):
        """An empty retrieval still produces an answer rather than raising."""
        generator = scripted_generator(default="generated answer")
        result = RAGPipeline(recording_retriever([]), generator, top_k=3).answer("risks?")
        assert result.documents == []
        assert result.text == "generated answer"
