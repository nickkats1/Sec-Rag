from collections import Counter

import networkx as nx

from rag.bm25 import tokenize
from rag.documents import Document
from rag.graph.extraction import Triple, parse_triples
from rag.graph.resolution import EntityResolver
from rag.llm import Generator
from rag.prompts import TRIPLE_PROMPT


def _contains_run(terms: list[str], run: list[str]) -> bool:
    """Report whether a term sequence appears contiguously in another.

    Matching whole terms rather than substrings is what keeps the entity "AI" out
    of "chairman" and "TPU" out of "tpuservice"; requiring them contiguous and in
    order is what keeps "data centers" from matching "centers data".

    Args:
        terms: the sequence to search.
        run: the sequence to find; an empty run never matches.

    Returns:
        True when run appears as a contiguous slice of terms.
    """
    if not run:
        return False
    return any(
        terms[start : start + len(run)] == run
        for start in range(len(terms) - len(run) + 1)
    )


class GraphRetriever:
    """Retriever that answers from an entity graph built over the corpus.

    The graph is undirected on purpose: "Alphabet designs TPUs" should be
    reachable when the question starts from TPUs just as much as when it starts
    from Alphabet.

    Attributes:
        generator: extracts triples from each passage at indexing time.
        hops: how far to walk out from the entities the query names.
        resolver: collapses alias spellings onto one key.
        graph: entities as nodes, stated relations as edges.
        documents: every indexed passage, in insertion order.
    """

    def __init__(
        self,
        generator: Generator,
        hops: int = 2,
        resolver: EntityResolver | None = None,
    ) -> None:
        """Start with an empty graph.

        Args:
            generator: extracts triples from each passage at indexing time.
            hops: how far to walk out from the entities the query names.
            resolver: alias resolver; a default one when omitted.
        """
        self.generator = generator
        self.hops = hops
        self.resolver = resolver if resolver is not None else EntityResolver()
        self.graph = nx.Graph()
        self.documents: list[Document] = []
        self._mentions: dict[str, set[int]] = {}

    def add_documents(self, documents: list[Document]) -> None:
        """Extract a graph from passages and index them, keeping earlier ones.

        Args:
            documents: Document objects to add.
        """
        prompts = [TRIPLE_PROMPT.format(text=doc.page_content) for doc in documents]
        replies = self.generator.generate_batch(prompts)
        extracted = [parse_triples(reply) for reply in replies]
        self.resolver.prepare(
            [name for triples in extracted for t in triples for name in (t.subject, t.object)]
        )
        start = len(self.documents)
        for offset, triples in enumerate(extracted):
            for triple in triples:
                self._add_triple(triple, start + offset)
        self.documents.extend(documents)

    def _add_triple(self, triple: Triple, doc_index: int) -> None:
        """Link a triple's resolved entities and record where it was stated.

        Args:
            triple: a fact extracted from that document.
            doc_index: position of the document in the corpus.
        """
        subject = self.resolver.resolve(triple.subject)
        obj = self.resolver.resolve(triple.object)
        self.graph.add_edge(subject, obj, relation=triple.relation)
        for entity in (subject, obj):
            self._mentions.setdefault(entity, set()).add(doc_index)

    def _entities_named_in(self, query: str) -> set[str]:
        """Find the graph entities the query mentions.

        Args:
            query: search string.

        Returns:
            The matching entity keys, empty when the query names none.
        """
        terms = tokenize(query)
        return {
            entity
            for entity in self.graph.nodes
            if _contains_run(terms, tokenize(entity))
        }

    def _rank(self, entities: set[str]) -> list[int]:
        """Order documents by how many of the reached entities each mentions.

        A document counts once per entity however many times that entity was
        extracted from it, so a passage repeating one name does not outrank a
        passage that genuinely covers several. Ties keep corpus order: Python
        randomizes string hashing, so set order is not reproducible.

        Args:
            entities: normalized keys to look up.

        Returns:
            Corpus positions, most mentions first.
        """
        counts: Counter[int] = Counter()
        for entity in entities:
            counts.update(self._mentions.get(entity, set()))
        return sorted(counts, key=lambda index: (-counts[index], index))

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Return the passages the query's entities lead to.

        Args:
            query: search string.
            top_k: maximum number of results to return.

        Returns:
            Ranked passages, empty when the query names no known entity.
        """
        reached: set[str] = set()
        for entity in self._entities_named_in(query):
            reached |= set(
                nx.single_source_shortest_path_length(
                    self.graph, entity, cutoff=self.hops
                )
            )
        return [self.documents[index] for index in self._rank(reached)[:top_k]]


__all__ = ["GraphRetriever"]
