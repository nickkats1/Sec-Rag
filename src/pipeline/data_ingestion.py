"""Load data and convert to documents"""

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


from typing import List


class DataIngestion:
    """Utility class to load documents and chunk documents for Vector db
    
    
    Attributes
    ----------
    file_path: str
        The file path where data to be loaded and chunked is located.
        
    Methods
    -------
    load_documents():
        Loads documents from file path and returns as document loader from langchain library.
    chunk():
        RecursiveCharacterTextSplitter applied to documents given: chunk_size and chunk_overlap
        returns chunked documents
    """

    
    
    
    def __init__(self,file_path: str):
        """Initialize DataIngestion instance.
        
        Args:
            file_path: Where PDF document(s) to be loaded and chunked is located.
        """
        
        self.file_path = file_path

    def load_documents(self) -> List[Document]:
        """Loads PDF files from given directory of PDF file and returns documents.
        
        Returns:
            documents
        """
        try:
            documents = []
            loader = DirectoryLoader(
                path=self.file_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            # load documents
            documents.extend(loader.load())
            print(f"Length of documents: {len(documents)}")
            return documents
        except Exception as e:
            print(f"Could not find path for documents: {e}")
            return []
    
    
    def chunk(
        self,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Document]:
        """Chunk documents into smaller pieces for VectorStore.
        
        Args:
            chunk_size: maximum size of a chunk.
            chunk_overlap: Target overlap between chunks.
            
        Returns:
            documents with RecursiveCharacterSplitter applied.
        
        Raises:
            FileNotFoundError: If files where documents should be
            cannot be found.
        """
                
        documents = self.load_documents()
        
        if not documents:
            raise FileNotFoundError("No documents found")
        
        # text splitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n","\n\n","\t"]
        )
        
        chunks = text_splitter.split_documents(documents=documents)
        print(f"Size of chunks: {len(chunks)}")
        
        return chunks
            
            




