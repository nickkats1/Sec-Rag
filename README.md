# SEC RAG

[![CI](https://github.com/nickkats1/Sec-Rag/actions/workflows/ci.yaml/badge.svg)](https://github.com/nickkats1/Sec-Rag/actions/workflows/ci.yaml)

Retrieval-augmented generation over SEC 10-K filings. Several retrievers
(BM25, dense, hybrid, HyDE, graph) share one interface so they can be swapped
under the same pipeline.

## Install

```bash
pip install -e ".[dev]"
```

Generation uses OpenAI through `langchain-openai`. Put `OPENAI_API_KEY`
in a `.env` file. BM25, dense and hybrid retrieval work without a key; HyDE,
graph and answer generation need one.

## Data

Filings are not included. The notebooks expect a PDF at `data/google_10K.pdf`;
`data/` is gitignored, so add your own.

## Usage

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag.chunk import chunk_documents
from rag.documents import load_documents
from rag.hybrid import HybridRetriever
from rag.llm import DEFAULT_CHAT_MODEL, LangChainGenerator
from rag.pipeline import RAGPipeline

load_dotenv()

chunks = chunk_documents(load_documents("data/google_10K.pdf"))

retriever = HybridRetriever()
retriever.add_documents(chunks)

generator = LangChainGenerator(ChatOpenAI(model=DEFAULT_CHAT_MODEL, max_tokens=512))
answer = RAGPipeline(retriever, generator, top_k=5).answer("What are the principal risk factors?")

print(answer.text)
for document in answer.documents:
    print(document.metadata["source"], document.page_content[:200])
```

## Retrievers

Every retriever has `add_documents(documents)` and `retrieve(query, top_k)`.

| Class | Module | How it ranks |
| --- | --- | --- |
| `BM25Retriever` | `rag.bm25` | keyword scoring with bm25s |
| `DenseRetriever` | `rag.dense` | bi-encoder embeddings in FAISS |
| `HybridRetriever` | `rag.hybrid` | reciprocal rank fusion of BM25 and dense |
| `HydeRetriever` | `rag.hyde` | asks the model for a draft answer, then searches with it |
| `GraphRetriever` | `rag.graph` | entity graph built from model-extracted triples |

`HydeRetriever` and `GraphRetriever` take a `generator`:

```python
from rag.dense import DenseRetriever
from rag.graph import GraphRetriever
from rag.hyde import HydeRetriever

hyde = HydeRetriever(base_retriever=DenseRetriever(), generator=generator)
graph = GraphRetriever(generator=generator, hops=3)
```

## Notebooks

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks/
```

`bm25` and `dense` run offline. `hyde` calls OpenAI.

## Development

```bash
pytest
```

## License

MIT
