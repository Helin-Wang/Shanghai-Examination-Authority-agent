import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

class Config:
    """配置管理类，统一存储 API Key、数据库路径等"""
    
    # LLM 配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", 'sk-GJkAcNd5eZ9jMQNOD372E79bE9464526A7C80b99545b7548')
    BASE_URL = os.getenv("BASE_URL", 'https://api.xty.app/v1',)
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')

    # Embedding配置
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "./src/models/m3e-base")
    RERANK_MODEL = os.getenv("RERANK_MODEL", "./src/models/bge-reranker-base")
    
    # 知识库 & 向量数据库
    FAISS_DB_PATH = os.getenv("FAISS_DB_PATH", "./data/faiss/qapair")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chromadb/qapair")
    
    FAISS_KG_DB_PATH = os.getenv("FAISS_KG_DB_PATH", "./data/faiss/kg")
    CHROMA_KG_DB_PATH = os.getenv("CHROMA_KG_DB_PATH", "./data/chromadb/kg")
    
    FAISS_MD_DB_PATH = os.getenv("FAISS_MD_DB_PATH", "./data/faiss/passage")
    CHROMA_MD_DB_PATH = os.getenv("CHROMA_MD_DB_PATH", "./data/chromadb/passage")
    
    CHROMA_DB_NAME = os.getenv("CHROMA_DB_NAME", "Shanghai_Examination_Authority")
    CHROMA_KG_DB_NAME = os.getenv("CHROMA_KG_DB_NAME", "Shanghai_Examination_Authority_KG")
    CHROMA_MD_DB_NAME = os.getenv("CHROMA_MD_DB_NAME", "Shanghai_Examination_Authority_MD")
    #KG_DB_URI = os.getenv("KG_DB_URI", "bolt://localhost:7687")
    #KG_DB_USER = os.getenv("KG_DB_USER", "neo4j")
    #KG_DB_PASSWORD = os.getenv("KG_DB_PASSWORD", "your-neo4j-password")

    BM25_DB_PATH = os.getenv("BM25_DB_PATH", './data/bm25/qapair.pkl')
    BM25_KG_PATH = os.getenv("BM25_KG_PATH", './data/bm25/kg.pkl')

    # RAG 检索配置
    EXAM_TYPE_LIST = ['高考学考_秋考', '高考学考_春考', '高考学考_艺术类统一考试', '高考学考_体育类统一考试', '高考学考_三校生高考', '高考学考_专科自主招生', '高考学考_高中学业水平考试', '高考学考_中职校学业水平考试', '高考学考_其他考试—专升本考试', '高考学考_其他考试—普通高校联合招收华侨港澳台考试', '中考中招_中考中招', '研考成考_研究生招生考试', '研考成考_成人高考', '研考成考_同等学力人员申请硕士学位外国语水平和学科综合水平全国统一考试', '自学考试_自学考试', '证书考试_全国大学英语四、六级考试（CET)', '证书考试_全国中小学教师资格考试笔试', '证书考试_全国计算机等级考试（NCRE)', '证书考试_上海市高校信息技术水平考试']
    RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", 20))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 5))
    KG_RERANK_TOP_K = int(os.getenv("KG_RERANK_TOP_K", 10))
    SIMILARITY_THRESHOLD = 0.7
    KG_SIMILARITY_THRESHOLD = 0.7

    # Prompt 配置
    PROMPT_TEMPLATE_PATH = os.getenv("PROMPT_TEMPLATE_PATH", "./prompts")

    # 其他
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# 允许外部代码直接访问 Config 类
config = Config()
