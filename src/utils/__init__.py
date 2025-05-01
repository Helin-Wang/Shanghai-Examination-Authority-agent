from src.utils.conversations import Conversation
from src.utils.config import config
from src.utils.database import connect_database
from src.retriever.ChromaDBRetriever import ChromaDBRetriever
from src.retriever.FAISSRetriever import FAISSRetriever
import streamlit as st
from src.retriever.BM25ZhRetriever import BM25ZhRetriever

def summarize(llm, history):
    prompt = """你是一个专业的总结助手，能将用户的历史对话总结成10字左右的句子，下面是对话，请进行总结，只给出最终的总结，不要给出任何解释。
    对话：
    {history}
    总结：
    """.format(history='\n'.join([str(conversation) for conversation in history]))

    summary = llm.invoke(prompt).content.replace("总结：", "")
    return summary

def init_session_state():
    if "llm" not in st.session_state:
        from src.models import init_llm
        st.session_state["llm"] = init_llm(
            config.LLM_MODEL, 
            config.OPENAI_API_KEY, 
            config.BASE_URL
        )
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = FAISSRetriever(
            config.FAISS_DB_PATH,
            config.EXAM_TYPE_LIST,
            config.RETRIEVER_TOP_K
        )  
        # st.session_state["retriever"] = ChromaDBRetriever(
        #     config.CHROMA_DB_PATH,
        #     config.EXAM_TYPE_LIST,
        #     config.CHROMA_DB_NAME,
        #     config.RETRIEVER_TOP_K
        # )
    if "kg_retriever" not in st.session_state:
        st.session_state["kg_retriever"] = FAISSRetriever(
            config.FAISS_KG_DB_PATH,
            config.EXAM_TYPE_LIST,
            config.RETRIEVER_TOP_K
        )
        # st.session_state["kg_retriever"] = ChromaDBRetriever(
        #     config.CHROMA_KG_DB_PATH,
        #     config.EXAM_TYPE_LIST,
        #     config.CHROMA_KG_DB_NAME,
        #     config.RETRIEVER_TOP_K
        # )
        
    if 'bm25_retriever' not in st.session_state:
        st.session_state["bm25_retriever"] = BM25ZhRetriever(config.BM25_DB_PATH, config.RETRIEVER_TOP_K)
    
    if 'kg_bm25_retriever' not in st.session_state:
        st.session_state["kg_bm25_retriever"] = BM25ZhRetriever(config.BM25_KG_PATH, config.RETRIEVER_TOP_K)

    
    # if "conn" not in st.session_state:
    #     st.session_state["conn"], st.session_state["cursor"] = connect_database()