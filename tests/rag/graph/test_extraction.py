from rag.graph import Triple, extract_triples, parse_triples

# ---------------------------------------------------------------------------
# TestExtractTriples
# ---------------------------------------------------------------------------


class TestExtractTriples:
    """Tests for extract_triples()."""

    def test_parses_a_json_list(self, scripted_generator, triple_json):
        """A well-formed JSON list becomes Triple objects."""
        generator = scripted_generator(triple_json("Alphabet", "designs", "TPUs"))
        expected = [Triple("Alphabet", "designs", "TPUs")]
        assert extract_triples("text", generator) == expected

    def test_strips_markdown_code_fence(self, scripted_generator, triple_json):
        """A fenced reply is unwrapped before parsing."""
        fenced = "```json\n" + triple_json("A", "r", "B") + "\n```"
        generator = scripted_generator(fenced)
        assert extract_triples("text", generator) == [Triple("A", "r", "B")]

    def test_strips_an_unclosed_code_fence(self, scripted_generator, triple_json):
        """A reply opening a fence but never closing it still parses."""
        fenced = "```json\n" + triple_json("A", "r", "B")
        generator = scripted_generator(fenced)
        assert extract_triples("text", generator) == [Triple("A", "r", "B")]

    def test_malformed_json_yields_no_triples(self, scripted_generator):
        """Unparseable output is discarded rather than raising."""
        assert extract_triples("text", scripted_generator("not json at all")) == []

    def test_non_list_json_yields_no_triples(self, scripted_generator):
        """A JSON object instead of a list yields nothing."""
        assert extract_triples("text", scripted_generator('{"subject": "A"}')) == []

    def test_entries_missing_keys_are_skipped(self, scripted_generator):
        """Objects lacking the required keys are dropped."""
        reply = '[{"subject": "A"}, {"subject": "A", "relation": "r", "object": "B"}]'
        generator = scripted_generator(reply)
        assert extract_triples("text", generator) == [Triple("A", "r", "B")]

    def test_empty_list_yields_no_triples(self, scripted_generator):
        """An empty list is a valid answer meaning no facts."""
        assert extract_triples("text", scripted_generator("[]")) == []


# ---------------------------------------------------------------------------
# TestParseTriples
# ---------------------------------------------------------------------------


class TestParseTriples:
    """Tests for parse_triples()."""

    def test_parses_a_reply_without_a_generator(self, triple_json):
        """A reply already in hand parses to the same triples."""
        assert parse_triples(triple_json("A", "r", "B")) == [Triple("A", "r", "B")]
