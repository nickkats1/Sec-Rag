import pytest


from rag.data_ingestion import load_documents, chunk_documents

class TestDataIngestion:
    """test data ingestion"""
    
    def test_load_pdf(self, pdf_file):
        """test if pdf file is loaded"""
        assert len(load_documents(str(pdf_file))) > 0
        
    def test_raises_error(self):
        """test if FileNotFoundError is raised"""
        with pytest.raises(FileNotFoundError):
            load_documents(file_path=None)
            
    def test_chunks_documents(self, pdf_file):
        """test if documents are chunked"""
        docs = load_documents(pdf_file)
        
        chunks = chunk_documents(docs, chunk_size=20, chunk_overlap=0)
        
        assert len(chunks) < len(docs)
        

