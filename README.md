# SEC RAG Pipeline

A simple Retrieval-Augmented Generation (RAG) pipeline for querying SEC filing PDFs using LangChain OpenAI, Grog, and Google AI. This project is designed to quickly extract insights from SEC filings(or other PDF files) by asking natural language questions.

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
API_KEY=YOUR_API_KEY
```

### 4. Add your PDF files

Drop one or more SEC filing PDFs into the `data/` directory:

```
data/
├── 10-K_CompanyA.pdf
├── 10-Q_CompanyB.pdf
```

### 5. Run the pipeline

You will be prompted to ask a question. Try to keep it relevant to the content of the PDFs you added.

Try to be precised and specific in your question to get the best results!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
