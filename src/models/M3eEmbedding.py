from sentence_transformers import SentenceTransformer
from src.utils.config import config

class M3eEmbeddings:
    def __init__(self):
        # 加载嵌入模型（如您之前的 m3e-base）
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

    def embed_documents(self, documents):
        """
        实现 LangChain 所需的 embed_documents 方法，返回文档的嵌入向量；
        此处的document已经提取过page_content，输入其实就是string列表
        """
        embeddings = self.model.encode(documents)  # 使用 embed 方法生成嵌入向量
        return embeddings
