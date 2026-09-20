import json
import re
from dataclasses import dataclass

from rag.llm import Generator
from rag.prompts import TRIPLE_PROMPT

_WHITESPACE = re.compile(r"\s+")
_REQUIRED_KEYS = ("subject", "relation", "object")


def normalise(text: str) -> str:
    """Reduce an entity mention to the key the graph stores it under.

    Casing and irregular spacing are the two ways the same company is written
    twice in one filing, so both are removed before anything is keyed on.

    Args:
        text: the mention as it appeared in the source.

    Returns:
        The lowercase mention with runs of whitespace collapsed to one space.
    """
    return _WHITESPACE.sub(" ", text).strip().lower()


@dataclass(frozen=True)
class Triple:
    """One stated fact: a subject, the relation, and the object.

    Attributes:
        subject: the entity the fact is about.
        relation: what the subject does to or has with the object.
        object: the entity on the other end.
    """

    subject: str
    relation: str
    object: str


def _strip_code_fence(reply: str) -> str:
    """Remove a markdown code fence the model may have wrapped its JSON in.

    Args:
        reply: raw model output.

    Returns:
        The reply with an opening fence and any closing fence removed.
    """
    text = reply.strip()
    if not text.startswith("```"):
        return text
    without_opening = text.split("\n", 1)[-1]
    closing = without_opening.rfind("```")
    return without_opening[:closing] if closing != -1 else without_opening


def _to_triple(entry: object) -> Triple | None:
    """Convert one decoded JSON entry into a Triple.

    Args:
        entry: one element of the decoded JSON list.

    Returns:
        The Triple, or None unless the entry is an object whose three keys are
        all present and all strings. Numbers are the common case: a filing is
        full of them, and one reaching ``normalise`` raises rather than being
        keyed.
    """
    if not isinstance(entry, dict) or not all(
        isinstance(entry.get(key), str) for key in _REQUIRED_KEYS
    ):
        return None
    return Triple(entry["subject"], entry["relation"], entry["object"])


def parse_triples(reply: str) -> list[Triple]:
    """Read the triples out of a model's reply to ``TRIPLE_PROMPT``.

    A reply that is fenced, truncated, shaped wrong, or not JSON at all yields no
    triples rather than an exception, because one uncooperative passage must not
    abort the indexing of a whole filing.

    Args:
        reply: raw model output.

    Returns:
        The triples the model reported, empty if it reported none or its reply
        could not be parsed.
    """
    try:
        entries = json.loads(_strip_code_fence(reply))
    except json.JSONDecodeError:
        return []
    if not isinstance(entries, list):
        return []
    return [triple for triple in map(_to_triple, entries) if triple is not None]


def extract_triples(text: str, generator: Generator) -> list[Triple]:
    """Ask a generator for the relationships stated in a passage.

    Args:
        text: the passage to read.
        generator: the model that reads it.

    Returns:
        The triples the model reported, empty if it reported none or its reply
        could not be parsed.
    """
    return parse_triples(generator.generate(TRIPLE_PROMPT.format(text=text)))


__all__ = ["Triple", "extract_triples", "normalise", "parse_triples"]
