import pytest

from rag.metrics.retrieval import (
    average_precision,
    hit_rate_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# TestRetrievalMetrics
# ---------------------------------------------------------------------------


class TestRetrievalMetrics:
    """Tests for the ranking metrics in rag.metrics.retrieval."""

    def test_hit_rate_finds_a_relevant_id_in_the_top_k(self, ranked_ids):
        """Any relevant id inside the cutoff scores 1.0."""
        retrieved, relevant = ranked_ids
        assert hit_rate_at_k(retrieved, relevant, k=1) == 1.0

    def test_hit_rate_misses_when_relevant_ids_sit_below_k(self):
        """A relevant id past the cutoff does not count."""
        assert hit_rate_at_k(["x", "a"], {"a"}, k=1) == 0.0

    def test_precision_divides_by_k(self, ranked_ids):
        """Two of four slots are relevant, so precision@4 is 0.5."""
        retrieved, relevant = ranked_ids
        assert precision_at_k(retrieved, relevant, k=4) == 0.5

    def test_precision_charges_for_empty_slots(self):
        """One relevant id with k=4 scores 0.25 even though only one id came back."""
        assert precision_at_k(["a"], {"a"}, k=4) == 0.25

    def test_recall_divides_by_relevant_count(self, ranked_ids):
        """Two of three relevant ids appear, so recall@4 is 2/3."""
        retrieved, relevant = ranked_ids
        assert recall_at_k(retrieved, relevant, k=4) == pytest.approx(2 / 3)

    def test_recall_with_no_relevant_ids_is_zero(self):
        """An empty relevant set scores 0.0 rather than dividing by zero."""
        assert recall_at_k(["a"], set(), k=1) == 0.0

    def test_mrr_uses_the_first_relevant_rank(self):
        """The first relevant id at rank 2 gives 0.5."""
        assert mean_reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5

    def test_mrr_with_no_hit_is_zero(self):
        """No relevant id anywhere scores 0.0."""
        assert mean_reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_average_precision_counts_missing_relevant_ids(self, ranked_ids):
        """Hits at ranks 1 and 3 give (1/1 + 2/3) / 3 with three relevant ids."""
        retrieved, relevant = ranked_ids
        assert average_precision(retrieved, relevant) == pytest.approx((1 + 2 / 3) / 3)

    def test_map_averages_over_queries(self):
        """A perfect query and a failed query average to 0.5."""
        batch_retrieved = [["a"], ["x"]]
        batch_relevant = [{"a"}, {"a"}]
        assert mean_average_precision(batch_retrieved, batch_relevant) == 0.5

    def test_map_rejects_misaligned_batches(self):
        """Batches of different length raise ValueError."""
        with pytest.raises(ValueError, match="align"):
            mean_average_precision([["a"]], [])

    def test_map_of_empty_batch_is_zero(self):
        """No queries scores 0.0."""
        assert mean_average_precision([], []) == 0.0

    def test_ndcg_of_a_perfect_ranking_is_one(self):
        """All relevant ids first gives 1.0."""
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == pytest.approx(1.0)

    def test_ndcg_discounts_relevant_ids_lower_down(self, ranked_ids):
        """Hits at ranks 1 and 3 score below a perfect ranking but above zero."""
        retrieved, relevant = ranked_ids
        assert 0.0 < ndcg_at_k(retrieved, relevant, k=4) < 1.0

    def test_ndcg_with_no_relevant_ids_is_zero(self):
        """An empty relevant set scores 0.0 rather than dividing by zero."""
        assert ndcg_at_k(["a"], set(), k=1) == 0.0
