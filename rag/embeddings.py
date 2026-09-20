from functools import cache

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


@cache
def load_bi_encoder(model_name: str = DEFAULT_BI_ENCODER) -> SentenceTransformer:
    """Load a bi-encoder model, cached per model name.

    The cache is what makes repeated retrieval affordable: without it every call
    re-reads the weights from disk. It falls back to CPU when the GPU is full.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        A cached SentenceTransformer instance.
    """
    try:
        return SentenceTransformer(model_name)
    except RuntimeError:
        return SentenceTransformer(model_name, device="cpu")


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_BI_ENCODER,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed a list of texts with a bi-encoder.

    Args:
        texts: strings to encode.
        model_name: HuggingFace model identifier.
        batch_size: encoding batch size.

    Returns:
        Float32 array of shape (n, embedding_dim), L2-normalised, so a dot
        product between two rows is already their cosine similarity.
    """
    model = load_bi_encoder(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


__all__ = [
    "DEFAULT_BI_ENCODER",
    "embed_texts",
    "load_bi_encoder",
]
