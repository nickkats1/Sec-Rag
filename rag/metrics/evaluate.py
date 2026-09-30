from collections.abc import Sequence
from dataclasses import dataclass

from rag.metrics.retrieval import (
    average_precision,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from rag.protocol import Retriever


@dataclass
class EvalExample:
    """One question and the pages that answer it.

    Attributes:
        question: the query handed to the retriever.
        relevant_pages: page numbers known to hold the answer.
    """

    question: str
    relevant_pages: set[int]


@dataclass
class EvalResult:
    """Averaged metrics plus the per-question rows they were averaged from.

    Attributes:
        aggregate: metric name to mean score over every example.
        per_example: one row of scores per example, in input order.
    """

    aggregate: dict[str, float]
    per_example: list[dict[str, float]]


def retrieved_pages(retriever: Retriever, query: str, k: int) -> list[int]:
    """Return the distinct pages of the top k passages, in rank order.

    Args:
        retriever: anything with ``retrieve(query, top_k)``.
        query: the question to retrieve for.
        k: how many passages to fetch before collapsing them to pages.
    """
    pages: list[int] = []
    for doc in retriever.retrieve(query, top_k=k):
        page = doc.metadata["page"]
        if page not in pages:
            pages.append(page)
    return pages


def score_rag(retrieved: list[int], relevant: set[int], k: int) -> dict[str, float]:
    """Score one ranked page list against the pages known to be relevant."""
    return {
        "hit": hit_rate_at_k(retrieved, relevant, k),
        "precision": precision_at_k(retrieved, relevant, k),
        "recall": recall_at_k(retrieved, relevant, k),
        "mrr": mean_reciprocal_rank(retrieved, relevant),
        "ndcg": ndcg_at_k(retrieved, relevant, k),
        "ap": average_precision(retrieved, relevant),
    }


def evaluate_retriever(
    retriever: Retriever, examples: Sequence[EvalExample], k: int
) -> EvalResult:
    """Score a retriever on every example and average the scores.

    Args:
        retriever: anything with ``retrieve(query, top_k)``.
        examples: questions paired with their relevant pages.
        k: cutoff for the @k metrics.

    Returns:
        Per-example scores and their means, keyed like ``hit@5`` and ``map``.
    """
    per_example = [
        score_rag(retrieved_pages(retriever, ex.question, k), ex.relevant_pages, k)
        for ex in examples
    ]
    names = {
        "hit": f"hit@{k}",
        "precision": f"precision@{k}",
        "recall": f"recall@{k}",
        "mrr": "mrr",
        "ndcg": f"ndcg@{k}",
        "ap": "map",
    }
    aggregate = {
        name: sum(row[metric] for row in per_example) / len(per_example)
        for metric, name in names.items()
    }
    return EvalResult(aggregate=aggregate, per_example=per_example)
