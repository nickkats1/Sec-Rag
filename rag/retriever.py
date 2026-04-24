from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Any

def get_retriever(documents: List[Document], k: int, search_type: str) -> Any:
    """
    Get FAISS retriever.
    """
    if not documents:
        raise ValueError("must have valid documents")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": k, "search_type": search_type})
    return retriever
