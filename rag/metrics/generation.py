from collections.abc import Sequence

import numpy as np

from rag.embeddings import DEFAULT_BI_ENCODER, embed_texts


def answer_similarity(
    predicted: str,
    reference: str,
    model_name: str = DEFAULT_BI_ENCODER,
) -> float:
    """Cosine similarity between the generated answer and the reference answer.

    Args:
        predicted: the generated answer.
        reference: the gold answer.
        model_name: bi-encoder used to embed both strings.

    Returns:
        Cosine in [-1, 1], or 0.0 if either string is blank.
    """
    if not predicted.strip() or not reference.strip():
        return 0.0
    embeddings = embed_texts([predicted, reference], model_name=model_name)
    return float(embeddings[0] @ embeddings[1])


def embedding_faithfulness(
    predicted: str,
    contexts: Sequence[str],
    model_name: str = DEFAULT_BI_ENCODER,
) -> float:
    """Highest cosine between the answer and any context it was generated from.

    A grounded answer sits close to at least one passage the model was shown.

    Args:
        predicted: the generated answer.
        contexts: passages handed to the model as context.
        model_name: bi-encoder used to embed everything.

    Returns:
        max over contexts of cosine(answer, context), or 0.0 if either is empty.
    """
    if not predicted.strip() or not contexts:
        return 0.0
    answer_embedding = embed_texts([predicted], model_name=model_name)[0]
    context_embeddings = embed_texts(list(contexts), model_name=model_name)
    return float(np.max(context_embeddings @ answer_embedding))


def context_precision(
    reference: str,
    contexts: Sequence[str],
    threshold: float = 0.5,
    model_name: str = DEFAULT_BI_ENCODER,
) -> float:
    """Fraction of contexts whose cosine to the reference reaches the threshold.

    Args:
        reference: the gold answer or evidence passage.
        contexts: passages a retriever returned.
        threshold: cosine at or above which a context counts as relevant.
        model_name: bi-encoder used to embed everything.

    Returns:
        relevant contexts / total contexts in [0, 1], or 0.0 if either is empty.
    """
    if not reference.strip() or not contexts:
        return 0.0
    reference_embedding = embed_texts([reference], model_name=model_name)[0]
    context_embeddings = embed_texts(list(contexts), model_name=model_name)
    return float(np.mean(context_embeddings @ reference_embedding >= threshold))


__all__ = [
    "answer_similarity",
    "context_precision",
    "embedding_faithfulness",
]
