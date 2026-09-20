from dataclasses import dataclass

from langchain_core.documents import Document

from rag.llm import Generator
from rag.prompts import ANSWER_PROMPT
from rag.protocol import Retriever


def citation(document: Document) -> str:
    """Label a passage for the model to cite.

    Args:
        document: the passage to label.

    Returns:
        Source and page, with missing metadata shown as a question mark.
    """
    source = document.metadata.get("source", "?")
    page = document.metadata.get("page", "?")
    return f"{source} p{page}"


@dataclass
class Answer:
    """A generated answer and the passages it was grounded in.

    Attributes:
        text: the generated answer.
        documents: the retrieved passages, in rank order.
    """

    text: str
    documents: list[Document]


class RAGPipeline:
    """Answers questions by retrieving passages and generating over them.

    Passages are labelled with their source and page rather than with
    :func:`rag.documents.document_key`, which holds the passage's whole text and
    would send every passage to the model twice.

    Attributes:
        retriever: supplies the passages to ground the answer in.
        generator: writes the answer.
        top_k: number of passages to retrieve per question.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        top_k: int,
    ) -> None:
        """Initialise the pipeline.

        Args:
            retriever: supplies the passages to ground the answer in.
            generator: writes the answer.
            top_k: number of passages to retrieve per question.
        """
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def answer(self, query: str) -> Answer:
        """Retrieve passages for a query and generate a grounded answer.

        Args:
            query: the question to answer.

        Returns:
            The generated answer and the passages it was grounded in.
        """
        documents = self.retriever.retrieve(query, top_k=self.top_k)
        context = "\n\n".join(
            f"[{citation(doc)}] {doc.page_content}" for doc in documents
        )
        prompt = ANSWER_PROMPT.format(context=context, question=query)
        return Answer(text=self.generator.generate(prompt), documents=documents)
