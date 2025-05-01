from src.models.BgeReranker import BgeReranker

if __name__ == '__main__':
    reranker = BgeReranker()
    query = '数学'
    text_list = ['语文','高等数学','天气']
    print(reranker.rerank(query, text_list))
    