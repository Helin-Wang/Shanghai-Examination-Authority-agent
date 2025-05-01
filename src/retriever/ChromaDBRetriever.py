from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import os
import pandas as pd
from pathlib import Path
import numpy as np
import chromadb
import ast

class ChromaDBRetriever():
    def __init__(self, chroma_path: str, exam_type_list: list, collection_name: str, k=5, embedding_model=None):
        """
        初始化 ChromaDB 检索器
        :param chroma_path: ChromaDB 持久化存储路径
        :param k: 返回的最相似文档数量（默认5）
        :param embedding_model: 可选的嵌入模型，默认为 M3eEmbeddings
        """
        
        from src.models.M3eEmbedding import M3eEmbeddings
        self.embedding_model = embedding_model if embedding_model is not None else M3eEmbeddings()
        self.chroma_path = chroma_path
        self.k = k
        self.exam_type_list = exam_type_list
        
        # 创建一个 ChromaDB 客户端
        self.client = chromadb.PersistentClient(self.chroma_path)
        # 创建一个 collection（相当于数据库表）
        #self.client.delete_collection(collection_name)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    
    def save_vectorstore(self, filepath: str):
        df = pd.read_csv(filepath)
        print(df.head())
        filename = Path(filepath).stem
        
        self.collection.add(
            ids=[str(i) for i in df["index"].tolist()],  # 文档的唯一 ID
            documents=df['问答对'].tolist(),
            metadatas=[{"source": filename, "type":f'{row["部分"]}_{row["考试类型"]}'} for _, row in df.iterrows()],  # 可选元数据
            embeddings=self.embedding_model.embed_documents(df['问答对'].tolist())
        )
    
    def save_triplets(self, filepath: str):
        df = pd.read_csv(filepath)
        print(df.head())
        filename = Path(filepath).stem
        
        # Convert string representation of triplets to list
        df['筛选后三元组'] = df['筛选后三元组'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
        
        # Use list comprehension for better readability and performance
        # triplet_data = [(f"{str(row['index'])}_{triplet_index}", triplet, {"source": filename, "type":f'{row["部分"]}_{row["考试类型"]}'})
        #                 for _, row in df.iterrows() 
        #                 for triplet_index, triplet in enumerate(row['筛选后三元组'])]
        seen = set()
        triplet_data = []

        for _, row in df.iterrows():
            for triplet_index, triplet in enumerate(row['筛选后三元组']):
                key = (triplet, f'{row["部分"]}_{row["考试类型"]}')  # 作为去重依据
                if key not in seen:
                    seen.add(key)
                    triplet_data.append((f"{str(row['index'])}_{triplet_index}", triplet, {"source": filename, "type": key[1]}))

        
        
        if triplet_data:  # Check if there are any triplets
            ids, documents, metadatas = zip(*triplet_data)
            
            # Get embeddings for all documents at once
            embeddings = self.embedding_model.embed_documents(list(documents))
            
            # Add all data to collection in one batch
            self.collection.add(
                ids=list(ids),
                documents=list(documents), 
                metadatas=list(metadatas),
                embeddings=embeddings
            )
    
    def retrieve(self, query, k=None):
        """
        根据考试类型和查询内容进行检索
        :param exam_type: 要检索的考试类型（例如：'高考'、'托福'等）
        :param query: 查询字符串
        :return: 最相似的 k 条结果
        """
        k = k or self.k
        
        # 进行查询
        query_embeddings = self.embedding_model.embed_text([query])
        results = self.collection.query(query_embeddings=query_embeddings, n_results=k)
           
        structured_data = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            metadata['id'] = results['ids'][0][i]
            doc = Document(
                page_content=results['documents'][0][i],
                metadata=metadata
            )
            structured_data.append(doc)

        return structured_data
    
    def save_md_vectorstore(self, filepath: str):
        df = pd.read_csv(filepath)
        print(df.head())

        self.collection.add(
            ids=[str(index) for index, row in df.iterrows()],  # 文档的唯一 ID
            documents=df['文本内容'].tolist(),
            metadatas=[{"source": row['文件名'], "type":row["部分"]} for _, row in df.iterrows()],  # 可选元数据
            embeddings=self.embedding_model.embed_documents(df['文本内容'].tolist())
        )