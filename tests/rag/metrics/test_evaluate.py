from rag.metrics.evaluate import evaluate_retriever, retrieved_pages


class TestRetrievedPages:
    """Tests for collapsing ranked passages into ranked pages."""

    def test_drops_repeated_pages_and_keeps_rank_order(
        self, recording_retriever, same_page_docs
    ):
        """Two chunks from page 1 followed by page 2 give [1, 2]."""
        retriever = recording_retriever(same_page_docs)
        assert retrieved_pages(retriever, "risk", k=3) == [1, 2]

    def test_passes_k_through_as_top_k(self, recording_retriever, paged_docs):
        """The retriever is asked for exactly k passages."""
        retriever = recording_retriever(paged_docs)
        retrieved_pages(retriever, "fox", k=2)
        assert retriever.calls == [("fox", 2)]


class TestEvaluateRetriever:
    """Tests for scoring a retriever over a small test set."""

    def test_scores_each_example_in_order(
        self, recording_retriever, paged_docs, eval_examples
    ):
        """Page 1 sits at rank 1 of 2 for the first question; the second misses."""
        result = evaluate_retriever(recording_retriever(paged_docs), eval_examples, k=2)
        first, missing = result.per_example
        assert first == {
            "hit": 1.0,
            "precision": 0.5,
            "recall": 1.0,
            "mrr": 1.0,
            "ndcg": 1.0,
            "ap": 1.0,
        }
        assert set(missing.values()) == {0.0}

    def test_aggregate_averages_over_examples(
        self, recording_retriever, paged_docs, eval_examples
    ):
        """One hit and one miss average to 0.5."""
        result = evaluate_retriever(recording_retriever(paged_docs), eval_examples, k=2)
        assert result.aggregate["hit@2"] == 0.5
        assert result.aggregate["map"] == 0.5

    def test_aggregate_keys_name_the_cutoff(
        self, recording_retriever, paged_docs, eval_examples
    ):
        """The @k metrics carry k in their name; mrr and map do not."""
        result = evaluate_retriever(recording_retriever(paged_docs), eval_examples, k=2)
        assert set(result.aggregate) == {
            "hit@2",
            "precision@2",
            "recall@2",
            "mrr",
            "ndcg@2",
            "map",
        }
