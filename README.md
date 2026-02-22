# SEC RAG Pipeline

A simple Retrieval-Augmented Generation (RAG) pipeline for querying SEC filing PDFs using LangChain and OpenAI.

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/nickkats1/Sec-Rag.git
cd Sec-Rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Create a `.env` file in the root of the project:
```
OPENAI_API_KEY=YOUR_API_KEY
```

### 4. Add your PDF files
Drop one or more SEC filing PDFs into the `data/` directory:
```
data/
└── apple_10k_2025.pdf
```

### 5. Run the pipeline
```bash
python main.py
```

You will be prompted to enter a company name, and the pipeline will answer questions about R&D spending from your documents.

## Project Structure

```
Sec-Rag/
├── data/                   # Put your PDF files here
├── db/                     # Auto-created: Chroma vector store
├── src/
│   └── rag_pipeline.py     # Core pipeline logic
├── tests/
│   ├── conftest.py         # Shared fixtures
│   └── test_rag_pipeline.py
├── main.py                 # Entry point — run this
├── .env                    # Not committed — add your API key here
└── requirements.txt
```

## Running Tests
```bash
pytest tests/
```







