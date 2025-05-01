from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import os
import pandas as pd
from pathlib import Path
import ast

class FAISSRetriever():
    def __init__(self, faiss_path: str, exam_type_list: list, k=20, embedding_model=None):
        """
        初始化 FAISS 检索器
        :param faiss_path: FAISS 向量存储的路径
        :param k: 返回的最相似文档数量（默认5）
        :param embedding_model: 可选的嵌入模型，默认为 M3eEmbeddings
        """

        from src.models.M3eEmbedding import M3eEmbeddings
        self.embedding_model = embedding_model if embedding_model is not None else M3eEmbeddings()
        self.k = k
        self.exam_type_list = exam_type_list
        self.faiss_path = faiss_path
        
        self.db = FAISS.load_local(self.faiss_path, self.embedding_model, allow_dangerous_deserialization=True)
        
    def save_vectorstore(self, filepath: str):
        df = pd.read_csv(filepath)
        print(df.head())
        filename = Path(filepath).stem
        
        docs = [
            Document(
                page_content=f'{row["问答对"]}',
                metadata={
                    "id": str(row["index"]),
                    "source": f'{filename}',
                    "type": f'{row["部分"]}_{row["考试类型"]}'
                }
            )
            for _, row in df.iterrows()
        ]   
        self.db = FAISS.from_documents(docs, embedding=self.embedding_model)
        # 保存每个考试类型的FAISS索引到对应目录
        self.db.save_local(self.faiss_path)
        
    def save_triplets(self, filepath: str):
        df = pd.read_csv(filepath)
        print(df.head())
        filename = Path(filepath).stem
     
        # 三元组去重
        seen = set()
        unique_docs = []
        for index, row in df.iterrows():
            key = (row['三元组'], f'{row["部分"]}_{row["考试类型"]}')
            if key not in seen:
                seen.add(key)
                unique_docs.append(Document(
                    page_content=f"{row['三元组'].replace('(','').replace(')','').replace(',','').replace(' ','')}",
                    metadata={
                        "id": str(index), 
                        "source": f'{filename}',
                        "type": key[1]  # 直接使用 key[1]，保证一致性
                    }
                ))
            else:
                print(f"重复的三元组: {row['三元组']}")
            
        self.db = FAISS.from_documents(unique_docs, embedding=self.embedding_model)
        self.db.save_local(self.faiss_path)
        
    def retrieve(self, query, k=None):
        """
        根据考试类型和查询内容进行检索
        :param exam_type: 要检索的考试类型（例如：'高考'、'托福'等）
        :param query: 查询字符串
        :return: 最相似的k条结果
        """
        # 获取查询的嵌入表示（embedding）
        query_embedding = self.embedding_model.embed_documents([query])[0]  # 获取查询的embedding
        results = self.db.similarity_search_by_vector(query_embedding, k=self.k)
        
        return results
    
    def retrieve_with_score(self, query, k=None):
        """
        根据考试类型和查询内容进行检索
        :param exam_type: 要检索的考试类型（例如：'高考'、'托福'等）
        :param query: 查询字符串
        :return: 最相似的k条结果
        """
        # 获取查询的嵌入表示（embedding）
        query_embedding = self.embedding_model.embed_documents([query])[0]  # 获取查询的embedding
        results = self.db.similarity_search_with_score_by_vector(query_embedding, k=self.k)
        
        return results

    def save_md_vectorstore(self, filepath: str):
        df = pd.read_csv(filepath)
        print(df.head())
        print(df.columns)
        
        
        docs = []
        for index, row in df.iterrows():
            docs.append(Document(
                page_content=row['文本内容'],
                metadata={
                    "id": str(index), 
                    "source": row['文件名'],
                    "type": row['部分']
                }
            ))
        self.db = FAISS.from_documents(docs, embedding=self.embedding_model)
        self.db.save_local(self.faiss_path)
