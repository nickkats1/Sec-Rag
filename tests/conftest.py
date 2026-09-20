import json

import pytest
from langchain_core.documents import Document
from pypdf import PdfWriter


class ScriptedGenerator:
    """Generator double handing back canned replies, so no test calls a model.

    Scripted replies are returned in order; once they run out, every further
    call returns ``default``. A test wanting one fixed reply forever passes
    only ``default``.

    Attributes:
        replies: the replies still to be handed back, in order.
        default: returned once the scripted replies run out.
        prompts: every prompt the generator was given, in order.
    """

    def __init__(self, *replies, default=""):
        self.replies = list(replies)
        self.default = default
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else self.default

    def generate_batch(self, prompts):
        return [self.generate(prompt) for prompt in prompts]


@pytest.fixture
def scripted_generator():
    """The generator double itself, for tests to script and construct."""
    return ScriptedGenerator


class RecordingRetriever:
    """Retriever double recording every call, so a wrapper's behaviour is visible.

    Satisfies the two-method retriever interface without indexing anything, which
    is what the wrapper strategies need from the thing they wrap.

    Attributes:
        documents: the Documents handed back from every retrieve call.
        calls: the ``(query, top_k)`` pairs received, in call order.
    """

    def __init__(self, documents=None):
        self.documents = list(documents) if documents else []
        self.calls = []

    def add_documents(self, documents):
        self.documents.extend(documents)

    def retrieve(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.documents[:top_k]


@pytest.fixture
def recording_retriever():
    """The retriever double itself, for tests to construct."""
    return RecordingRetriever


@pytest.fixture
def paged_docs() -> list[Document]:
    """Three Documents covering different topics, numbered by page."""
    return [
        Document(page_content="The quick brown fox jumps.", metadata={"page": 1}),
        Document(page_content="Properties and facilities owned.", metadata={"page": 2}),
        Document(
            page_content="Risk factors include market volatility.",
            metadata={"page": 3},
        ),
    ]


@pytest.fixture
def triple_json():
    """Render one triple the way the extraction prompt asks the model to."""

    def render(subject, relation, obj):
        return json.dumps([{"subject": subject, "relation": relation, "object": obj}])

    return render


@pytest.fixture
def pdf_file(tmp_path):
    """A single minimal one-page PDF file."""
    pdf_path = tmp_path / "test.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as f:
        writer.write(f)
    return pdf_path


@pytest.fixture
def sample_docs() -> list[Document]:
    """Three distinct Documents covering different topics."""
    return [
        Document(
            page_content="The quick brown fox jumps over the lazy dog.",
            metadata={"source": "a.pdf"},
        ),
        Document(
            page_content="SEC filings contain financial disclosures.",
            metadata={"source": "b.pdf"},
        ),
        Document(
            page_content="Risk factors include market volatility.",
            metadata={"source": "c.pdf"},
        ),
    ]


@pytest.fixture
def repeated_text_docs() -> list[Document]:
    """Two Documents with identical text but different page numbers."""
    return [
        Document(
            page_content="Risk factors include market volatility.",
            metadata={"source": "10k.pdf", "page": 1},
        ),
        Document(
            page_content="Risk factors include market volatility.",
            metadata={"source": "10k.pdf", "page": 2},
        ),
    ]


@pytest.fixture
def same_page_docs() -> list[Document]:
    """Two chunks from one page and a third chunk from another."""
    return [
        Document(
            page_content="Risk factors include market volatility.",
            metadata={"source": "10k.pdf", "page": 1},
        ),
        Document(
            page_content="Risk factors also include currency swings.",
            metadata={"source": "10k.pdf", "page": 1},
        ),
        Document(
            page_content="Properties and facilities owned.",
            metadata={"source": "10k.pdf", "page": 2},
        ),
    ]
