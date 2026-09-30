import math
from collections.abc import Hashable, Iterable, Sequence


def hit_rate_at_k(
    retrieved: Sequence[Hashable], relevant: Iterable[Hashable], k: int
) -> float:
    """Return 1.0 if any relevant id appears in the top k, else 0.0.

    Args:
        retrieved: ranked ids returned by a retriever.
        relevant: ids that are relevant to the query.
        k: cutoff.
    """
    relevant_set = set(relevant)
    return float(any(doc_id in relevant_set for doc_id in retrieved[:k]))


def precision_at_k(
    retrieved: Sequence[Hashable], relevant: Iterable[Hashable], k: int
) -> float:
    """Fraction of the k slots filled with a relevant id.

    Divides by k rather than by how many ids came back, so a retriever that
    returns fewer than k results is charged for the empty slots.

    Args:
        retrieved: ranked ids returned by a retriever.
        relevant: ids that are relevant to the query.
        k: cutoff.
    """
    relevant_set = set(relevant)
    return sum(doc_id in relevant_set for doc_id in retrieved[:k]) / k


def recall_at_k(
    retrieved: Sequence[Hashable], relevant: Iterable[Hashable], k: int
) -> float:
    """Fraction of relevant ids that appear in the top k.

    Args:
        retrieved: ranked ids returned by a retriever.
        relevant: ids that are relevant to the query.
        k: cutoff.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(doc_id in relevant_set for doc_id in retrieved[:k])
    return hits / len(relevant_set)


def mean_reciprocal_rank(
    retrieved: Sequence[Hashable], relevant: Iterable[Hashable]
) -> float:
    """Reciprocal rank of the first relevant id, or 0.0 if none appears.

    Args:
        retrieved: ranked ids returned by a retriever.
        relevant: ids that are relevant to the query.
    """
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def average_precision(
    retrieved: Sequence[Hashable], relevant: Iterable[Hashable]
) -> float:
    """Mean of precision at every rank where a relevant id appears.

    Divides by the number of relevant ids, so relevant ids that never show up
    still count against the score.

    Args:
        retrieved: ranked ids returned by a retriever.
        relevant: ids that are relevant to the query.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            hits += 1
            total += hits / rank
    return total / len(relevant_set)


def mean_average_precision(
    batch_retrieved: Sequence[Sequence[Hashable]],
    batch_relevant: Sequence[Iterable[Hashable]],
) -> float:
    """Mean of average_precision across a batch of queries.

    Args:
        batch_retrieved: one ranked id list per query.
        batch_relevant: one relevant id collection per query, in the same order.

    Raises:
        ValueError: if the two batches differ in length.
    """
    if len(batch_retrieved) != len(batch_relevant):
        raise ValueError("batch_retrieved and batch_relevant must align in length")
    if not batch_retrieved:
        return 0.0
    scores = [
        average_precision(retrieved, relevant)
        for retrieved, relevant in zip(batch_retrieved, batch_relevant, strict=True)
    ]
    return sum(scores) / len(scores)


def ndcg_at_k(
    retrieved: Sequence[Hashable], relevant: Iterable[Hashable], k: int
) -> float:
    """Normalised discounted cumulative gain at k with binary relevance.

    Each relevant id at rank r contributes 1 / log2(r + 1); the sum is divided
    by the best sum possible given how many relevant ids exist.

    Args:
        retrieved: ranked ids returned by a retriever.
        relevant: ids that are relevant to the query.
        k: cutoff.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


__all__ = [
    "average_precision",
    "hit_rate_at_k",
    "mean_average_precision",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
