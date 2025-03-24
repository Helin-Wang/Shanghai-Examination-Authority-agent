from langchain_community.vectorstores import Milvus
from src.models.M3eEmbedding import M3eEmbeddings

class MilvusRetriever():
    def __init__(self, milvus_host: str, milvus_port: str, collection_list: list, k=5, embedding_model=None):
        """
        初始化 Milvus 检索器
        :param milvus_host: Milvus 服务器地址
        :param milvus_port: Milvus 服务器端口
        :param collection_list: Milvus 中的集合（相当于不同类型的考试数据存储）
        :param k: 返回的最相似文档数量（默认5）
        :param embedding_model: 可选的嵌入模型，默认为 M3eEmbeddings
        """
  
        self.embedding_model = embedding_model if embedding_model is not None else M3eEmbeddings()
        self.k = k
        self.collection_list = collection_list
        
        # 连接 Milvus 并加载索引
        db_dict = dict()
        db_dict['全部'] = Milvus(collection_name='全部', connection_args={"host": milvus_host, "port": milvus_port})
        for collection in self.collection_list:
            db_dict[collection] = Milvus(collection_name=collection, connection_args={"host": milvus_host, "port": milvus_port})
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
        
        # 如果 exam_type 不在预定义类型中，使用 '全部' 这个集合进行检索
        if exam_type not in self.collection_list:
            results = self.db_dict['全部'].similarity_search_with_score_by_vector(query_embedding, k=self.k)
        else:
            results = self.db_dict[exam_type].similarity_search_with_score_by_vector(query_embedding, k=self.k)
        
        return results
