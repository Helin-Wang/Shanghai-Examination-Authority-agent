from src.agents.state import AgentState
import streamlit as st
from langgraph.graph import StateGraph
from src.utils.config import config
from src.retriever import retrieved_docs_to_str
from src.retriever import rewrite_query, query2qapair, query2triplets, query2passage, get_exam_types, remove_duplicate_docs, rerank_docs, rerank_triplets, ask_for_more_info
from src.utils.conversations import Conversation, Role
import concurrent.futures
from src.utils.config import Config
from langchain_openai import ChatOpenAI
def call_llm_with_docs(state):
    """ 结合 RAG 结果进行 LLM 推理 """
    query = state.revised_query
    docs = state.reranked_docs  # 获取检索到的文档
    triplets = state.reranked_triplets  # 获取检索到的知识图谱 triplet
    history = [str(conversation) for conversation in state.history]
    
    # 生成 LLM Prompt
    prompt = f"""你是一个上海教育局智能助理，专门解答上海相关考试的问题。为了帮助你，我会给你提供一些参考文档，希望能让你更清晰、准确地回答问题：
    
    **问题**：{query}

    **相关文档**：
    {retrieved_docs_to_str(docs)}
    {retrieved_docs_to_str(triplets)}

    请根据提供的文档回答问题。
    """
    history.append({"role": "user", "content": prompt})
    print("\n****** prompt ******\n")
    print(prompt)
    print("\n****** prompt ******\n")
    # state.final_answer = st.session_state["llm"].invoke(prompt).content
    state.final_answer = st.session_state["llm"](history).content
    return state

def retrieve_node(state):
    """根据查询从不同数据库中检索相关文档"""
    
    query = state.qapair_query
    triplets_query_list = state.triplets_query_list
    
    # FAISS Retriever
    print("****** FAISS ******")
    # qa_pair
    retrieved_docs = st.session_state["retriever"].retrieve(query)
    # filter
    # state.retrieved_docs = [doc for doc in retrieved_docs if state.exam_types == [] or doc.metadata['type'] in state.exam_types]
    print(f"****** retrieved_docs: {state.retrieved_docs} ******")
    
    # triplets
    retrieved_triplets_list = []
    for triplets_query in triplets_query_list:
        retrieved_triplets = st.session_state["kg_retriever"].retrieve_with_score(triplets_query)
        # filter
        # retrieved_triplets = [doc for doc in retrieved_triplets if state.exam_types == [] or doc[0].metadata['type'] in state.exam_types]
        retrieved_triplets_list.extend(retrieved_triplets)
    state.retrieved_triplets = remove_duplicate_docs(retrieved_triplets_list)
    print(f"****** retrieved_triplets: {state.retrieved_triplets} ******")
    
    # BM25 Retriever
    print("****** BM25 ******")
    bm25_retrieved_docs = st.session_state["bm25_retriever"].invoke(query)
    # filter
    # bm25_retrieved_docs = [doc for doc in bm25_retrieved_docs if state.exam_types == [] or doc[0].metadata['type'] in state.exam_types]
    state.bm25_retrieved_docs = remove_duplicate_docs(bm25_retrieved_docs)
    print(f"****** retrieved_docs: {state.bm25_retrieved_docs} ******")
    # triplets
    bm25_retrieved_triplets_list = []
    for triplets_query in triplets_query_list:
        bm25_retrieved_triplets = st.session_state["kg_bm25_retriever"].invoke(triplets_query)
        # filter
        # bm25_retrieved_triplets = [doc for doc in bm25_retrieved_triplets if state.exam_types == [] or doc[0].metadata['type'] in state.exam_types]
        bm25_retrieved_triplets_list.extend(bm25_retrieved_triplets)
    state.bm25_retrieved_triplets = remove_duplicate_docs(bm25_retrieved_triplets_list)
    print(f"****** retrieved_triplets: {state.bm25_retrieved_triplets} ******")
    
    return state


def query_rewrite_node(state):
    """重写查询"""
    history = state.history # 获取历史对话
    user_input = state.user_input
    llm = ChatOpenAI(
            model=Config.LLM_MODEL, 
            openai_api_key=Config.OPENAI_API_KEY, 
            openai_api_base=Config.BASE_URL,
            streaming=True
        )
    revised_query = rewrite_query(llm, history, user_input)
    state.revised_query = revised_query
    return state


def qapair_task(llm, query, result_dict):
    result_dict["qapair_query"] = query2qapair(llm, query)
def triplets_task(llm, query, result_dict):
    result_dict["triplets_query_list"] = query2triplets(llm, query) 
def query2docs_node(state):
    """将查询转换为不同形式的文档"""
    query = state.revised_query # 获取查询
    # TODO：并行处理！！！！
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     future_qapair = executor.submit(query2qapair, query)
    #     future_triplets = executor.submit(query2triplets, query)

    #     state.qapair_query = future_qapair.result()
    #     state.triplets_query_list = future_triplets.result()
        
    state.qapair_query = query2qapair(query)
    state.triplets_query_list = query2triplets(query)

    return state

def exam_type_node(state):
    """根据查询确定相关考试类型"""
    query = state.revised_query # 获取查询
    llm = ChatOpenAI(
            model=Config.LLM_MODEL, 
            openai_api_key=Config.OPENAI_API_KEY, 
            openai_api_base=Config.BASE_URL,
            streaming=True
        )
    exam_types = get_exam_types(llm, query)
    state.exam_types = exam_types
    print(f"****** exam_types: {exam_types} ******")
    return state

def rerank_docs_node(state):
    """重排序文档"""
    query = state.qapair_query # 获取查询
    docs = state.retrieved_docs + state.bm25_retrieved_docs # 获取文档
    # remove duplicate
    seen = set()
    unique_docs = []
    for doc in docs:
        key = doc.metadata["id"]  # 唯一性判断键
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    reranked_docs = rerank_docs(unique_docs, query)
    state.reranked_docs = reranked_docs[:config.RERANK_TOP_K]
    return state

def rerank_triplets_node(state):
    """重排序三元组"""
    triplets = state.retrieved_triplets + state.bm25_retrieved_triplets # 获取三元组
    query_list = state.triplets_query_list # 获取查询
    # remove duplicate
    seen = set()
    unique_docs = []
    for doc in triplets:
        key = (doc.page_content.replace("(", "").replace(")", "").replace(",", "").replace(" ", ""), doc.metadata['type'])
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    reranked_triplets = rerank_triplets(unique_docs, query_list)
    state.reranked_triplets = reranked_triplets[:config.KG_RERANK_TOP_K]
    return state

def reask_node(state):
    query = state.user_input
    llm = ChatOpenAI(
            model=Config.LLM_MODEL, 
            openai_api_key=Config.OPENAI_API_KEY, 
            openai_api_base=Config.BASE_URL,
            streaming=True
        )
    state.final_answer = ask_for_more_info(llm, query)
    return state




