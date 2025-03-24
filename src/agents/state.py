from typing import List, Tuple, Dict, Optional, Union
from pydantic import BaseModel
from langchain.schema import Document
import numpy as np

class AgentState(BaseModel):
    user_input: str  # 用户原始输入
    final_answer: Optional[str] = None  # 最终 LLM 输出

    revised_query: str = ""
    qapair_query: str = ""
    passage_query: str = ""
    triplets_query_list: List[str] = []

    retrieved_docs: List[Tuple[Document, np.float32]] = []  # 检索到的文档
    retrieved_triplets: List[Tuple[Document, np.float32]] = []  # 检索到的知识图谱 triplet
    
    

    # is_query_clear: bool = False  # 用户指令是否清晰
    # requires_rag: bool = False  # 是否需要 RAG
    # exam_types: List[str] = []  # 相关考试类型（如果适用）
    
    # reformulated_queries: List[Tuple[str, str]] = []  # 适用于相似度检索的 query 形式 [(考试类型, query)]
    # kg_queries: List[Tuple[str, str]] = []  # 适用于知识图谱的 query 形式 [(考试类型, query)]
    
    # split_docs: List[List[str]] = []  # 若文档过多，分组后的 docs
    # intermediate_answers: List[str] = []  # LLM 生成的中间答案
    
    class Config:
        arbitrary_types_allowed = True
    