from langgraph.graph import StateGraph
from src.agents.node import call_llm_with_docs, exam_type_node, retrieve_node, query_rewrite_node, query2docs_node, rerank_docs_node, rerank_triplets_node, reask_node
from src.agents.state import AgentState
from langgraph.graph import START, END

def reask_condition(state):
    return "reask" if not state.exam_types else "query2docs"

def create_graph():
    graph = StateGraph(AgentState)
    
    # Add nodes for different processing steps
    graph.add_node("llm_with_docs", call_llm_with_docs)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("query2docs", query2docs_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("exam_type", exam_type_node)
    graph.add_node("rerank_docs", rerank_docs_node)
    graph.add_node("rerank_triplets", rerank_triplets_node)
    graph.add_node("reask", reask_node)
    
    graph.add_edge(START, "query_rewrite")
    graph.add_edge("query_rewrite", "exam_type")
    graph.add_conditional_edges("exam_type", reask_condition)
    
    # reask
    graph.add_edge("reask", END)
    
    # query2docs
    graph.add_edge("query2docs", "retrieve")
    graph.add_edge("retrieve", "rerank_docs")
    graph.add_edge("rerank_docs", "rerank_triplets")
    graph.add_edge("rerank_triplets", "llm_with_docs")
    graph.add_edge("llm_with_docs", END)
    
    return graph