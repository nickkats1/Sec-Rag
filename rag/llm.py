from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from enum import Enum



class Providers(Enum):
    OPENAI:str = "openai"
    GROQ:str = "groq"
    GOOGLE:str = "google"

class LLM:
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("Must provide an API key")
        self.api_key = api_key
        load_dotenv()

    def get_llm(self, provider: str, model_name: str, temperature: float):
        provider = provider.lower()
        if provider == Providers.OPENAI.value:
            return ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=self.api_key)
        elif provider == Providers.GROQ.value:
            return ChatGroq(model=model_name, temperature=temperature, groq_api_key=self.api_key)
        elif provider == Providers.GOOGLE.value:
            return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")