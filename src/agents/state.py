from typing import List, Tuple, Dict, Optional, Union
from pydantic import BaseModel
from langchain.schema import Document
import numpy as np
from src.utils.conversations import Conversation
class AgentState(BaseModel):
    user_input: str  # 用户原始输入
    final_answer: Optional[str] = None  # 最终 LLM 输出

    revised_query: str = ""
    qapair_query: str = ""
    passage_query: str = ""
    triplets_query_list: List[str] = []

    retrieved_docs: List[Document] = []  # 检索到的文档
    retrieved_triplets: List[Document] = []  # 检索到的知识图谱 triplet
    bm25_retrieved_docs: List[Document] = []
    bm25_retrieved_triplets: List[Document] = []
    
    reranked_docs: List[Document] = []
    reranked_triplets: List[Document] = []
    
    exam_types: List[str] = []  # 相关考试类型（如果适用）
    
    history: List[Conversation] = []

    # is_query_clear: bool = False  # 用户指令是否清晰
    # requires_rag: bool = False  # 是否需要 RAG

    
    class Config:
        arbitrary_types_allowed = True
    