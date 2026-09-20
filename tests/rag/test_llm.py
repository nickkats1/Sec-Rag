import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage

from rag.llm import LangChainGenerator


@pytest.fixture
def chat_model() -> FakeListChatModel:
    """A langchain chat model replying with fixed text, no network involved."""
    return FakeListChatModel(responses=["  Risk factors include market volatility.  "])


@pytest.fixture
def generator(chat_model) -> LangChainGenerator:
    """The generator under test, wrapping the fake chat model."""
    return LangChainGenerator(chat_model)


class TestLangChainGenerator:
    """Tests for LangChainGenerator."""

    def test_returns_reply_content_as_text(self, generator):
        """The chat model's reply content comes back as a plain string."""
        reply = generator.generate("What risks?")
        assert reply == "Risk factors include market volatility."

    def test_strips_surrounding_whitespace(self, generator):
        """Leading and trailing whitespace around the reply is removed."""
        reply = generator.generate("What risks?")
        assert reply == reply.strip()

    def test_wraps_the_given_chat_model(self, generator, chat_model):
        """The model handed in is the one used for generation."""
        assert generator.llm is chat_model

    def test_replies_follow_the_scripted_order(self):
        """Each call consumes the next scripted reply."""
        generator = LangChainGenerator(FakeListChatModel(responses=["one", "two"]))
        assert [generator.generate("a"), generator.generate("b")] == ["one", "two"]

    def test_batch_replies_are_stripped_and_in_order(self):
        """A batch returns one stripped reply per prompt, in prompt order."""
        generator = LangChainGenerator(FakeListChatModel(responses=[" one ", "two "]))
        assert generator.generate_batch(["a", "b"]) == ["one", "two"]

    def test_keeps_only_text_blocks(self):
        """A thinking model's reply comes back as the answer text alone."""
        reply = AIMessage(
            content=[
                {"type": "thinking", "thinking": "", "signature": "abc"},
                {"type": "text", "text": "Risk factors include market volatility."},
            ]
        )
        generator = LangChainGenerator(GenericFakeChatModel(messages=iter([reply])))
        reply = generator.generate("What risks?")
        assert reply == "Risk factors include market volatility."
