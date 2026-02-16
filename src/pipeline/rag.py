"""Module to perform rag"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.pipeline.vector_store import VectorStore



class Rag:
    """Performs RAG.
    
    
    Attributes
    ----------
    model: str
        The version of gpt to be used with ChatOpenAI
    temperature: int
        To make the responses more consistent by setting it to 0.
        
    Methods
    -------
    prompt_rag():
        retriever invokes question for context, context and question is fed as input in to prompt template,
        prompt template is prompted, result is returned as llm generated string.
    """
    
    def __init__(self, model: str, temperature: int):
        """Initialize Rag"""
        self.llm = ChatOpenAI(temperature=temperature, model=model)
        
    def prompt_rag(
        self,
        company_name: str,
        retriever: VectorStore
    ) -> str:
        """Performs Retrieval Augmented Generation (RAG) for a given company.
        
        Args:
            company_name: Name of company document to perform RAG on.
            retriever: Instance of VectorStore.
            
        Returns:
            result: Generated llm response containing answer to the question.
            
        Raises:
            NotImplementedError: Raised if 'invoke' method from VectorStore class is not implemented.
        """
        if retriever is None:
            raise NotImplementedError("Retriever must be implemented to have a 'invoke' method")
        
        
        # formulate the question
        
        question = (
            f"How much did {company_name} spend on Research and"
            f"development in 2025 and previous years?"
        )
        
        # invoke retriever to get context
        
        context = retriever.invoke(question)
        
        
        # prompt template
        
        prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            Given the following context, your task is to answer the question the best you can.
            If you do not know the answer, just say you do not know. Do not make anything up.
            
            Question: {question}
            
            Context: {context}
            
            Answer"""
        )
        
        # chain llm and prompt
        
        chain = prompt | self.llm
        
        result = chain.invoke({"context": context, "question": question}).content
        
        return result
        
        
        

        
        
        
