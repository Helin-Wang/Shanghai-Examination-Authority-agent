from src.agents.state import AgentState
import streamlit as st
from langgraph.graph import StateGraph
from src.utils.config import config
from src.retriever import retrieved_docs_to_str
from src.retriever import rewrite_query, query2qapair, query2triplets, query2passage


def call_llm_with_docs(state):
    """ 结合 RAG 结果进行 LLM 推理 """
    query = state.user_input
    docs = state.retrieved_docs  # 获取检索到的文档
    triplets = state.retrieved_triplets  # 获取检索到的知识图谱 triplet
    
    print(docs)
    print(triplets)
    
    # 生成 LLM Prompt
    prompt = f"""
    你是一个智能助理，回答以下问题：
    
    **问题**：{query}

    **相关文档**：
    {retrieved_docs_to_str(docs)}
    {retrieved_docs_to_str(triplets)}

    请根据提供的文档回答问题。
    """
    print("\n****** prompt ******\n")
    print(prompt)
    print("\n****** prompt ******\n")
    state.final_answer = st.session_state["llm"].invoke(prompt).content
    return state

def retrieve_node(state):
    """根据查询从不同数据库中检索相关文档"""
    # qa_pair
    query = state.qapair_query # 获取查询
    top_k = config.RETRIEVER_TOP_K  # 从配置中获取返回文档数量
    retrieved_docs = st.session_state["retriever"].retrieve('类型', query)
    state.retrieved_docs = retrieved_docs
    
    # triplets
    triplets_query_list = state.triplets_query_list # 获取查询
    for triplets_query in triplets_query_list:
        retrieved_triplets = st.session_state["kg_retriever"].retrieve('类型', triplets_query)
        state.retrieved_triplets.extend(retrieved_triplets)
    
    print(f"****** retrieved_docs: {state.retrieved_docs} ******")
    print(f"****** retrieved_triplets: {state.retrieved_triplets} ******")
    
    return state


def query_rewrite_node(state):
    """重写查询"""
    query = state.user_input # 获取查询
    revised_query = rewrite_query(st.session_state["llm"], query)
    state.revised_query = revised_query
    return state

def query2passage(state):
    """将查询转换为文本段落"""
    query = state.revised_query # 获取查询
    passage = query2passage(query)
    state.passage = passage
    return state

def query2docs_node(state):
    """将查询转换为不同形式的文档"""
    query = state.user_input # 获取查询
    qapair_query = query2qapair(st.session_state["llm"], query)
    triplets_query_list = query2triplets(st.session_state["llm"], query)
    state.qapair_query = qapair_query
    state.triplets_query_list = triplets_query_list
    
    print(f"****** qapair_query: {qapair_query} ******")
    print(f"****** triplets_query_list: {triplets_query_list} ******")
    return state


