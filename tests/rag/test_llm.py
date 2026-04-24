from unittest.mock import patch, MagicMock
from rag.llm import LLM


def test_openai_provider():
    with patch("rag.llm.ChatOpenAI") as mock_openai:
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance
        llm = LLM(api_key="fake-key")
        result = llm.get_llm("openai", "gpt-4", 0.0)
        mock_openai.assert_called_once_with(model="gpt-4", temperature=0.0, openai_api_key="fake-key")
        
        
        assert result == mock_instance
        
    
        


def test_google_provider():
    """test google llm returns values"""
    with patch("rag.llm.ChatGoogleGenerativeAI") as mock_google:
        mock_instance = MagicMock()
        mock_google.return_value = mock_instance
        
        llm = LLM(api_key="fake-key")
        
        result = llm.get_llm("google", "gemini-2.0-flash", temperature=0.0)
        
        mock_google.assert_called_once_with(model="gemini-2.0-flash", temperature=0.0,google_api_key="fake-key")
        

        assert result == mock_instance




                
        
    