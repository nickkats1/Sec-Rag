from typing import Protocol

from langchain_core.language_models import BaseChatModel, LanguageModelInput

DEFAULT_CHAT_MODEL = "gpt-4o-mini"


class Generator(Protocol):
    """Minimal text-generation interface.

    Structural typing keeps the retrievers and the pipeline free of any
    particular backend: anything with a ``generate`` method satisfies it,
    including the scripted doubles the tests use.
    """

    def generate(self, prompt: str) -> str:
        """Return the model's reply to a single prompt.

        Args:
            prompt: the full prompt, already formatted.

        Returns:
            The generated text.
        """
        ...

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Return the replies to several prompts, in the same order.

        Args:
            prompts: the full prompts, already formatted.

        Returns:
            One reply per prompt.
        """
        ...


class LangChainGenerator:
    """Generator backed by any langchain chat model.

    The model is built by the caller, so the same class serves OpenAI in the
    notebooks and a fake chat model in the tests.

    Attributes:
        llm: the langchain chat model that writes the replies.
        max_concurrency: how many prompts a batch sends at once.
    """

    def __init__(self, llm: BaseChatModel, max_concurrency: int = 8) -> None:
        """Wrap a chat model.

        Args:
            llm: any langchain chat model, for example ``ChatOpenAI``.
            max_concurrency: how many prompts a batch sends at once.
        """
        self.llm = llm
        self.max_concurrency = max_concurrency

    def generate(self, prompt: str) -> str:
        """Send the prompt as one user message and return the reply text.

        Only text blocks are kept, so a model that thinks before answering
        still hands back just the answer.

        Args:
            prompt: the full prompt, already formatted.

        Returns:
            The reply content, stripped of surrounding whitespace.
        """
        return self.llm.invoke(prompt).text.strip()

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Send prompts concurrently and return the reply texts in prompt order.

        Indexing a corpus is one call per passage, and waiting for each reply
        before sending the next one is what made that take tens of minutes.

        Args:
            prompts: the full prompts, already formatted.

        Returns:
            One stripped reply per prompt.
        """
        inputs: list[LanguageModelInput] = list(prompts)
        replies = self.llm.batch(inputs, config={"max_concurrency": self.max_concurrency})
        return [reply.text.strip() for reply in replies]


__all__ = ["DEFAULT_CHAT_MODEL", "Generator", "LangChainGenerator"]
