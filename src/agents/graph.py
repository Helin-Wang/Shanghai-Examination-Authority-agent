from langgraph.graph import StateGraph
from src.agents.node import call_llm_with_docs, retrieve_node, query_rewrite_node, query2docs_node
from src.agents.state import AgentState
from langgraph.graph import START, END

# 需要一个合并（join）逻辑，确保 node3 在 node1 和 node2 之后执行
def final_answer_generation_condition(state):
    """只有当 retrieve 都完成时，才触发"""
    return state.retrieved_docs and state.retrieved_triplets

def create_graph():
    graph = StateGraph(AgentState)
    
    # Add nodes for different processing steps
    graph.add_node("llm_with_docs", call_llm_with_docs)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("query2docs", query2docs_node)
    graph.add_node("retrieve", retrieve_node)
    
    #graph.add_edge(START, "query_rewrite")
    graph.add_edge(START, "query2docs")
    graph.add_edge("query2docs", "retrieve")
    graph.add_edge("retrieve", "llm_with_docs")
    # Final step
    graph.add_edge("llm_with_docs", END)
    
    return graph