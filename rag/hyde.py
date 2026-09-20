from rag.documents import Document
from rag.llm import Generator
from rag.prompts import HYDE_PROMPT
from rag.protocol import Retriever


class HydeRetriever:
    """Retrieval against an LLM-drafted hypothetical passage.

    Attributes:
        base_retriever: runs the search the draft is handed to.
        generator: drafts the hypothetical passage.
        prompt: template with a ``{question}`` placeholder.
    """

    def __init__(
        self,
        base_retriever: Retriever,
        generator: Generator,
        prompt: str = HYDE_PROMPT,
    ) -> None:
        """Initialise the wrapper.

        Args:
            base_retriever: runs the search the draft is handed to.
            generator: drafts the hypothetical passage.
            prompt: template with a ``{question}`` placeholder.
        """
        self.base_retriever = base_retriever
        self.generator = generator
        self.prompt = prompt

    def add_documents(self, documents: list[Document]) -> None:
        """Index passages into the base retriever.

        Args:
            documents: Document objects to add.
        """
        self.base_retriever.add_documents(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Draft a passage for the query and search with it.

        Args:
            query: the question to answer.
            top_k: maximum number of results to return.

        Returns:
            Whatever the base retriever returns for the draft. A refused or
            filtered completion comes back blank, and searching on an empty
            string ranks the corpus arbitrarily rather than merely worse, so a
            blank draft falls back to the query itself.
        """
        draft = self.generator.generate(self.prompt.format(question=query))
        return self.base_retriever.retrieve(draft.strip() or query, top_k=top_k)
