import jieba
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
import pickle

class BM25ZhRetriever(BM25Retriever):
    def __init__(self, filepath, k=10):
        """
        BM25 中文检索器
        :param docs: 文档列表，每个文档是字符串
        :param k: 返回前 k 个最相关的文档
        """
        # 预处理文档：分词 + 转换格式
        with open(filepath, "rb") as f:
            docs = pickle.load(f)

        # 构建 BM25 检索器
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = k

        # 直接将 `retriever` 的属性复制给 `self`
        self.__dict__.update(retriever.__dict__)


    def invoke(self, query):
        """
        进行 BM25 检索
        :param query: 查询字符串
        :return: 相关文档列表（已去掉分词空格）
        """
        # 🔹 分词
        query = " ".join(jieba.cut(query))

        # 进行检索
        results = super().invoke(query) # 调用原来的父类的方法
        scores = self.vectorizer.get_scores(query).tolist()
        scores.sort(reverse=True)

        doc_score_pairs = list(zip(results, scores[:self.k]))

        # 处理输出：去掉空格，恢复原文
        clean_results = [
            (Document(page_content=doc.page_content.replace(" ", ""), metadata=doc.metadata), score)
            for doc, score in doc_score_pairs
        ]
        return clean_results
