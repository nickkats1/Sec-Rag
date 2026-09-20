import pytest
from langchain_core.documents import Document

from rag.embeddings import embed_texts
from rag.graph import GraphRetriever
from rag.graph.resolution import EntityResolver


class CountingEmbedder:
    """Wraps the real encoder and counts how many times it is called."""

    def __init__(self):
        self.calls = 0

    def __call__(self, texts, model_name):
        self.calls += 1
        return embed_texts(texts, model_name)

# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def resolver() -> EntityResolver:
    """A resolver at the default threshold."""
    return EntityResolver()


@pytest.fixture
def embedder() -> CountingEmbedder:
    """The real encoder behind a call counter."""
    return CountingEmbedder()


# ---------------------------------------------------------------------------
# TestEntityResolver
# ---------------------------------------------------------------------------


class TestEntityResolver:
    """Tests for EntityResolver."""

    def test_first_mention_becomes_its_own_key(self, resolver):
        """An unseen mention is registered as a canonical key."""
        assert resolver.resolve("Alphabet Corporation") == "alphabet corporation"

    def test_casing_collapses_without_embedding(self, resolver):
        """A mention differing only by case resolves to the same key."""
        first = resolver.resolve("Alphabet Corporation")
        assert resolver.resolve("alphabet corporation") == first

    def test_alias_resolves_to_the_first_spelling(self, resolver):
        """A near-identical mention resolves onto the established key."""
        canonical = resolver.resolve("Alphabet Corporation")
        assert resolver.resolve("Alphabet Corp.") == canonical

    def test_unrelated_entities_stay_separate(self, resolver):
        """Mentions of different things keep different keys."""
        assert resolver.resolve("Alphabet") != resolver.resolve("Acme Widgets")

    def test_threshold_above_one_never_merges(self):
        """An unreachable threshold makes every distinct mention canonical."""
        strict = EntityResolver(threshold=1.01)
        assert strict.resolve("Alphabet Corporation") != strict.resolve("Alphabet Corp.")

    def test_empty_mention_resolves_to_empty(self, resolver):
        """A blank mention is returned unchanged rather than embedded."""
        assert resolver.resolve("   ") == ""

    def test_repeated_mention_is_stable(self, resolver):
        """Resolving the same mention twice yields the same key."""
        assert resolver.resolve("Acme") == resolver.resolve("Acme")

    def test_prepare_embeds_once_and_resolves_the_same(self, resolver, embedder):
        """Prepared mentions resolve as before, from a single embedding call."""
        prepared = EntityResolver(embed=embedder)
        mentions = ["Alphabet Corporation", "Alphabet Corp.", "Acme Widgets"]
        prepared.prepare(mentions)
        assert embedder.calls == 1
        assert [prepared.resolve(m) for m in mentions] == [
            resolver.resolve(m) for m in mentions
        ]
        assert embedder.calls == 1


# ---------------------------------------------------------------------------
# TestGraphRetrieverResolution
# ---------------------------------------------------------------------------


class TestGraphRetrieverResolution:
    """Tests for entity resolution wired into GraphRetriever."""

    def test_alias_spellings_surface_both_documents(
        self, scripted_generator, triple_json
    ):
        """Two documents naming one company differently both answer one query."""
        generator = scripted_generator(
            triple_json("Alphabet Corporation", "designs", "TPUs"),
            triple_json("Alphabet Corp.", "reports", "revenue"),
        )
        retriever = GraphRetriever(generator=generator, hops=0)
        retriever.add_documents(
            [
                Document(page_content="Alphabet Corporation designs TPUs."),
                Document(page_content="Alphabet Corp. reports revenue."),
            ]
        )
        results = retriever.retrieve("alphabet corporation", top_k=5)
        assert len(results) == 2

    def test_resolution_can_be_disabled_by_threshold(
        self, scripted_generator, triple_json
    ):
        """An unreachable threshold keeps the alias spellings apart."""
        generator = scripted_generator(
            triple_json("Alphabet Corporation", "designs", "TPUs"),
            triple_json("Alphabet Corp.", "reports", "revenue"),
        )
        retriever = GraphRetriever(
            generator=generator, hops=0, resolver=EntityResolver(threshold=1.01)
        )
        retriever.add_documents(
            [
                Document(page_content="Alphabet Corporation designs TPUs."),
                Document(page_content="Alphabet Corp. reports revenue."),
            ]
        )
        assert len(retriever.retrieve("alphabet corporation", top_k=5)) == 1

    def test_unrelated_entities_are_not_merged(self, scripted_generator, triple_json):
        """Resolution does not collapse genuinely different companies."""
        generator = scripted_generator(
            triple_json("Alphabet", "designs", "TPUs"),
            triple_json("Acme Widgets", "sells", "widgets"),
        )
        retriever = GraphRetriever(generator=generator, hops=0)
        retriever.add_documents(
            [
                Document(page_content="Alphabet designs TPUs."),
                Document(page_content="Acme Widgets sells widgets."),
            ]
        )
        results = retriever.retrieve("alphabet", top_k=5)
        assert [doc.page_content for doc in results] == ["Alphabet designs TPUs."]
