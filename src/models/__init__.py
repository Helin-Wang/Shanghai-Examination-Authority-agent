import streamlit as st
from src.models.M3eEmbedding import M3eEmbeddings
from src.utils.config import config
from langchain_openai import ChatOpenAI

def init_llm(llm_model, api_key, base_url, temperature=0.7):
    """
    Initialize the LLM with proper error handling.
    
    Args:
        llm_model (str): The model name to use
        api_key (str): OpenAI API key
        base_url (str): Base URL for the API
        temperature (float): Temperature parameter for response generation
        
    Returns:
        ChatOpenAI: Initialized LLM instance
        
    Raises:
        ValueError: If required parameters are missing
    """
    if not api_key:
        raise ValueError("API key is required")
    if not base_url:
        raise ValueError("Base URL is required")
    if not llm_model:
        raise ValueError("Model name is required")
        
    try:
        llm = ChatOpenAI(
            model=llm_model, 
            openai_api_key=api_key, 
            openai_api_base=base_url,
            temperature=temperature,
            streaming=True
        )
        return llm
    except Exception as e:
        raise Exception(f"Failed to initialize LLM: {str(e)}")
    
