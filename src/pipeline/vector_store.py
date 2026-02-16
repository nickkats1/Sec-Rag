"""Module to get vector store and retriever"""

import os

from src.pipeline.data_ingestion import DataIngestion

from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings


from langchain_community.vectorstores.chroma import Chroma


from typing import Any

@dataclass
class VectorStore:
    """Utility class to load Chroma VectorDB and then use chroma as retriever.
    
    
    
    Attributes
    ----------
    file_path: str
        File path where PDF file to be loaded and chunked is located.
        
    Methods
    -------
    load_vector_store():
        Creates Chromadb Vector Store consisting of persist directory, and chunks from
        DataIngestion module.

    load_retriever():
        vector_store as retriever.
    """
    
    def __init__(
        self,
        file_path: str,
    ):
        """Initialize VectorStore Instance.

        Args:
            file_path: file path where PDF files are located.
        """
        self.file_path = file_path

        
    def load_vector_store(
        self,
        persist_directory: str,
        chunks: int
    ) -> Chroma:
        """Create Vector Store with persist directory and chunks from DataIngestion.

        Args:
            persist_directory: path to save vector store data locally.
            chunks: Chunks from DataIngestion module.
            
        Returns:
            vector_store: Chromadb vector store with chunks and OpenAIEmbeddings.
        """
        
        try:
            persist_directory = "./db"
            os.makedirs(persist_directory, exist_ok=True)
            
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=OpenAIEmbeddings(),
                persist_directory=persist_directory
            )
            return vector_store
        except Exception as e:
            print(f"Could Not Retrieve Vector Store: {e}")
            return None
            
    def load_retriever(
        self,
        vector_store,
        search_type: str,
        k: int
    ) -> Any:
        """Cast 'load_vector_store' method as retriever.

        Args:
            vector_store: vector database from 'load_vector_store' method.
            search_type: query for semantically similar documents.
            k: Number of results to be returned.
        
        Returns:
            Any: vector_store as a retriever.
        
        Raises:
            NameError: Raised if vector_store is not defined.
        
        """
        if vector_store is None:
            raise NameError("Vector Store can not be found!")
        
        
        retriever = vector_store.as_retriever(
            search_type=search_type,
            kwargs={"k", k}
        )
        
        return retriever
    
    
        
            

        

        
            
        
                
            




