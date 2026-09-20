from collections.abc import Callable

import numpy as np

from rag.embeddings import DEFAULT_BI_ENCODER, embed_texts
from rag.graph.extraction import normalise


class EntityResolver:
    """Maps entity mentions onto canonical keys, merging near-identical ones.

    A filing names the same company as "Alphabet Inc.", "Alphabet Inc" and
    "Alphabet" on three consecutive pages. Keyed literally those are three
    unconnected nodes, and a traversal that should have crossed between them
    stops short.

    Attributes:
        threshold: cosine similarity at or above which a mention joins an
            existing key. A value above 1.0 is unreachable and disables merging.
        model_name: bi-encoder used to compare mentions.
        embed: turns mentions into L2-normalised vectors.
    """

    def __init__(
        self,
        threshold: float = 0.75,
        model_name: str = DEFAULT_BI_ENCODER,
        embed: Callable[[list[str], str], np.ndarray] = embed_texts,
    ) -> None:
        """Start with no known entities.

        Args:
            threshold: cosine similarity at or above which a mention joins an
                existing key.
            model_name: bi-encoder used to compare mentions.
            embed: encoder to compare mentions with.
        """
        self.threshold = threshold
        self.model_name = model_name
        self.embed = embed
        self._keys: list[str] = []
        self._vectors: np.ndarray | None = None
        self._canonical: dict[str, str] = {}
        self._pending: dict[str, np.ndarray] = {}

    def prepare(self, mentions: list[str]) -> None:
        """Embed every unseen mention in one call, ahead of resolving them.

        Resolving embeds one mention at a time, and a corpus produces thousands
        of them; one batched encode is far cheaper than thousands of single ones.

        Args:
            mentions: the entities as they appeared in the source.
        """
        keys = {
            key
            for key in map(normalise, mentions)
            if key and key not in self._canonical and key not in self._pending
        }
        if not keys:
            return
        ordered = sorted(keys)
        for key, vector in zip(ordered, self.embed(ordered, self.model_name)):
            self._pending[key] = vector[np.newaxis, :]

    def resolve(self, mention: str) -> str:
        """Return the canonical key for a mention, registering it if it is new.

        Every spelling seen is remembered, not just the ones that became keys. A
        filing repeats an alias on page after page, and without that each repeat
        would be embedded again to reach the same answer.

        Args:
            mention: the entity as it appeared in the source.

        Returns:
            An established key when the mention is close enough to one, the
            mention's own normalised form otherwise.
        """
        key = normalise(mention)
        if not key or key in self._canonical:
            return self._canonical.get(key, key)

        vector = self._pending.pop(key, None)
        if vector is None:
            vector = self.embed([key], self.model_name)
        if self._vectors is not None:
            similarities = self._vectors @ vector[0]
            closest = int(similarities.argmax())
            if similarities[closest] >= self.threshold:
                self._canonical[key] = self._keys[closest]
                return self._canonical[key]

        self._keys.append(key)
        self._vectors = (
            vector if self._vectors is None else np.vstack([self._vectors, vector])
        )
        self._canonical[key] = key
        return key


__all__ = ["EntityResolver"]
