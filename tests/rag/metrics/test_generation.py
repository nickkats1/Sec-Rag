import pytest

from rag.metrics.generation import (
    answer_similarity,
    context_precision,
    embedding_faithfulness,
)

# ---------------------------------------------------------------------------
# TestGenerationMetrics
# ---------------------------------------------------------------------------


class TestGenerationMetrics:
    """Tests for the embedding-based scores in rag.metrics.generation."""

    def test_identical_answers_score_one(self):
        """The same string embedded twice has cosine 1.0."""
        text = "Revenue grew because of advertising."
        assert answer_similarity(text, text) == pytest.approx(1.0, abs=1e-5)

    def test_different_answers_score_below_one(self):
        """Unrelated answers land strictly below a perfect match."""
        score = answer_similarity("Revenue grew.", "The fox jumped over the dog.")
        assert score < 0.9

    def test_blank_answer_scores_zero(self):
        """A blank string returns 0.0 without embedding anything."""
        assert answer_similarity("   ", "Revenue grew.") == 0.0

    def test_answer_copied_from_a_context_is_faithful(self):
        """An answer equal to one context reaches cosine 1.0 on that context."""
        contexts = ["Risk factors include market volatility.", "Properties owned."]
        score = embedding_faithfulness(contexts[0], contexts)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_faithfulness_with_no_contexts_is_zero(self):
        """No contexts to compare against scores 0.0."""
        assert embedding_faithfulness("Revenue grew.", []) == 0.0

    def test_context_precision_counts_every_context_at_zero_threshold(self):
        """Every context clears a threshold of 0.0, so precision is 1.0."""
        contexts = ["Revenue grew.", "The fox jumped."]
        assert context_precision("Revenue grew.", contexts, threshold=0.0) == 1.0

    def test_context_precision_counts_no_context_above_one(self):
        """No cosine exceeds 1.0, so a threshold of 1.01 gives 0.0."""
        contexts = ["Revenue grew.", "The fox jumped."]
        assert context_precision("Revenue grew.", contexts, threshold=1.01) == 0.0

    def test_context_precision_with_blank_reference_is_zero(self):
        """A blank reference returns 0.0 without embedding anything."""
        assert context_precision("", ["Revenue grew."]) == 0.0
