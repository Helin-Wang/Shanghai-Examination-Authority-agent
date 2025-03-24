from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import os
from src.models.M3eEmbedding import M3eEmbeddings

class FAISSRetriever():
    def __init__(self, faiss_path: str, exam_type_list: list, k=5, embedding_model=None):
        """
        初始化 FAISS 检索器
        :param faiss_path: FAISS 向量存储的路径
        :param k: 返回的最相似文档数量（默认5）
        :param embedding_model: 可选的嵌入模型，默认为 M3eEmbeddings
        """
  
        self.embedding_model = embedding_model if embedding_model is not None else M3eEmbeddings()
        self.k = k
        self.exam_type_list = exam_type_list
        
        # 加载 FAISS 索引
        db_dict = dict()
        db_dict['全部'] = FAISS.load_local(os.path.join(faiss_path, '全部'), self.embedding_model, allow_dangerous_deserialization=True)
        for exam_type in self.exam_type_list:
            db_dict[exam_type] = FAISS.load_local(os.path.join(faiss_path, exam_type), self.embedding_model, allow_dangerous_deserialization=True)
        self.db_dict = db_dict
        
    def retrieve(self, exam_type, query, k=None):
        """
        根据考试类型和查询内容进行检索
        :param exam_type: 要检索的考试类型（例如：'高考'、'托福'等）
        :param query: 查询字符串
        :return: 最相似的k条结果
        """
        # 获取查询的嵌入表示（embedding）
        query_embedding = self.embedding_model.embed_documents([query])[0]  # 获取查询的embedding
        
        # 如果 exam_type 不在预定义类型中，使用全部类型的索引
        if exam_type not in self.exam_type_list:
            results = self.db_dict['全部'].similarity_search_with_score_by_vector(query_embedding, k=self.k)
        else:
            # 如果 exam_type 在 EXAM_TYPE 中，则在对应的索引中搜索
            results = self.db_dict[exam_type].similarity_search_with_score_by_vector(query_embedding, k=self.k)
        
        return results
