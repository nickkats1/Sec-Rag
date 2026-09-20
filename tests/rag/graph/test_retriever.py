import pytest
from langchain_core.documents import Document

from rag.graph import GraphRetriever

# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus() -> list[Document]:
    """Three Documents matching the triples scripted for the fake generator."""
    return [
        Document(page_content="Alphabet designs TPUs.", metadata={"page": 1}),
        Document(page_content="TPUs power data centers.", metadata={"page": 2}),
        Document(page_content="Acme sells widgets.", metadata={"page": 3}),
    ]


@pytest.fixture
def corpus_generator(scripted_generator, triple_json):
    """A generator scripted with one triple per document in ``corpus``."""
    return scripted_generator(
        triple_json("Alphabet", "designs", "TPUs"),
        triple_json("TPUs", "power", "data centers"),
        triple_json("Acme", "sells", "widgets"),
        default="[]",
    )


# ---------------------------------------------------------------------------
# TestGraphRetriever
# ---------------------------------------------------------------------------


class TestGraphRetriever:
    """Tests for GraphRetriever."""

    def test_retrieve_before_add_returns_empty(self, corpus_generator):
        """Retrieval before indexing returns an empty list."""
        assert GraphRetriever(generator=corpus_generator).retrieve("alphabet") == []

    def test_query_naming_no_entity_returns_empty(self, corpus_generator, corpus):
        """A query mentioning nothing in the graph retrieves nothing."""
        retriever = GraphRetriever(generator=corpus_generator)
        retriever.add_documents(corpus)
        assert retriever.retrieve("unrelated question") == []

    def test_extracts_once_per_document(self, corpus_generator, corpus):
        """Indexing makes exactly one generation call per document."""
        GraphRetriever(generator=corpus_generator).add_documents(corpus)
        assert len(corpus_generator.prompts) == 3

    def test_retrieve_returns_document_objects(self, corpus_generator, corpus):
        """Every returned element is a langchain Document."""
        retriever = GraphRetriever(generator=corpus_generator)
        retriever.add_documents(corpus)
        result = retriever.retrieve("what does alphabet make?")
        assert all(isinstance(doc, Document) for doc in result)

    def test_matches_entity_case_insensitively(self, corpus_generator, corpus):
        """A lowercase query matches an entity extracted in capitals."""
        retriever = GraphRetriever(generator=corpus_generator)
        retriever.add_documents(corpus)
        result = retriever.retrieve("tell me about alphabet", top_k=1)
        assert result[0].page_content == "Alphabet designs TPUs."

    def test_traversal_reaches_a_document_without_the_query_term(
        self, corpus_generator, corpus
    ):
        """A one-hop neighbour surfaces a document that never names Alphabet."""
        retriever = GraphRetriever(generator=corpus_generator, hops=1)
        retriever.add_documents(corpus)
        contents = [doc.page_content for doc in retriever.retrieve("alphabet")]
        assert "TPUs power data centers." in contents

    def test_zero_hops_stays_on_the_matched_document(self, corpus_generator, corpus):
        """With no expansion only the directly matching document is returned."""
        retriever = GraphRetriever(generator=corpus_generator, hops=0)
        retriever.add_documents(corpus)
        contents = [doc.page_content for doc in retriever.retrieve("alphabet")]
        assert contents == ["Alphabet designs TPUs."]

    def test_top_k_limits_results(self, corpus_generator, corpus):
        """Result length does not exceed top_k."""
        retriever = GraphRetriever(generator=corpus_generator, hops=2)
        retriever.add_documents(corpus)
        assert len(retriever.retrieve("alphabet", top_k=1)) <= 1

    def test_add_documents_appends(self, corpus_generator, corpus):
        """A second add_documents call keeps the Documents from the first."""
        retriever = GraphRetriever(generator=corpus_generator)
        retriever.add_documents(corpus[:1])
        retriever.add_documents(corpus[1:])
        assert len(retriever.retrieve("alphabet tpus acme", top_k=10)) == 3

    def test_short_entity_does_not_match_inside_a_word(
        self, scripted_generator, triple_json
    ):
        """A two-letter entity does not match a query that merely contains it."""
        generator = scripted_generator(triple_json("AI", "powers", "search"))
        retriever = GraphRetriever(generator=generator)
        retriever.add_documents([Document(page_content="AI powers search.")])
        assert retriever.retrieve("explain the chairman") == []

    def test_entity_does_not_match_a_longer_word_containing_it(
        self, scripted_generator, triple_json
    ):
        """An entity is not matched inside a longer token."""
        generator = scripted_generator(triple_json("TPU", "powers", "compute"))
        retriever = GraphRetriever(generator=generator)
        retriever.add_documents([Document(page_content="TPU powers compute.")])
        assert retriever.retrieve("what is a tpuservice") == []

    def test_empty_entity_matches_nothing(self, scripted_generator, triple_json):
        """An empty extracted entity does not match every query."""
        generator = scripted_generator(triple_json("", "relates to", ""))
        retriever = GraphRetriever(generator=generator)
        retriever.add_documents([Document(page_content="Nothing in particular.")])
        assert retriever.retrieve("any query at all") == []

    def test_multi_word_entity_matches_a_contiguous_run(self, corpus_generator, corpus):
        """A two-word entity matches when both words appear in order."""
        retriever = GraphRetriever(generator=corpus_generator, hops=0)
        retriever.add_documents(corpus)
        contents = [
            doc.page_content for doc in retriever.retrieve("how many data centers")
        ]
        assert "TPUs power data centers." in contents

    def test_multi_word_entity_does_not_match_reordered_words(
        self, corpus_generator, corpus
    ):
        """The same two words in the wrong order do not match."""
        retriever = GraphRetriever(generator=corpus_generator, hops=0)
        retriever.add_documents(corpus)
        assert retriever.retrieve("centers data") == []

    def test_punctuation_around_the_entity_still_matches(self, corpus_generator, corpus):
        """Trailing punctuation in the query does not prevent a match."""
        retriever = GraphRetriever(generator=corpus_generator, hops=0)
        retriever.add_documents(corpus)
        contents = [doc.page_content for doc in retriever.retrieve("what is Alphabet?")]
        assert contents == ["Alphabet designs TPUs."]
