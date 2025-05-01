from typing import List, Tuple
import numpy as np
from langchain.schema import Document
from pathlib import Path
from openai import OpenAI
import os
import re
import pandas as pd
from tqdm import tqdm
import streamlit as st
from src.utils.conversations import Conversation
from src.utils.config import Config
from src.models.BgeReranker import BgeReranker
from langchain_openai import ChatOpenAI
from rapidfuzz import fuzz, process
import ast
import pickle
import jieba

# api_key_v3 = '73688f34-7502-421b-a188-040f460d5eb1'
api_key_r1 = 'ebe4d4b6-00ae-4ea7-9890-9356d6a29570'
os.environ["OPENAI_API_BASE"] = 'https://ark.cn-beijing.volces.com/api/v3'
os.environ["OPENAI_API_KEY"] = api_key_r1
client = OpenAI(
  api_key=api_key_r1, 
  base_url = 'https://ark.cn-beijing.volces.com/api/v3'
)


def extract_triplets_by_llm(part, section, qapair):
    def extract_triple_from_ans(s):
        pattern = r'@@(.*?)@@'  # 使用非贪婪匹配
        matches = re.findall(pattern, s)
        
        if not matches:
            return []
        
        if matches[0].strip() == '无':
            return []
        triples = matches[0].split('、')
        return triples
    
    prompt = f"""你是一个知识图谱专家，我正在制作关于考试信息的知识图谱，我将给你一段文字，请依次完成下面几个步骤
（1）首先识别其中所有的实体，不同实体之间用顿号（、）分隔

（2）然后探索你抽取到的这些实体之间的关系，用三元组的形式表示。注意，你只能寻找上一步你得到的这些实体之间的关系，不可以加入新的实体。

（3）请你一步一步思考，并且提供你的想法；在给出你的回答之前，反复思考你给出的三元组本身是否正确且没有歧义

（4）将最终的三元组列用@@隔开，三元组之间用顿号（、）隔开，即最后一行将是@@（实体1，关联，实体2）、（实体1，关联，实体2）...@@形式；若没有可抽取的三元组关系，则返回无


文本：
这是一个关于{part}-{section}的问答对
{qapair}
"""
    llm = ChatOpenAI(
        model=Config.LLM_MODEL, 
        openai_api_key=Config.OPENAI_API_KEY, 
        openai_api_base=Config.BASE_URL,
    )
    response = llm.invoke(prompt)
    answer = response.content.strip()
    
    return extract_triple_from_ans(answer)

def filter_triplets_by_llm(part, section, qapair, triplet):
    prompt = f"""你是一个知识图谱优化助手。你将获得一个文本片段和与该文本相关的一个知识图谱三元组。由于三元组可能存在不完整、不清晰或冗余的情况，请你按照以下步骤进行优化，并在每一步提供详细的推理过程。

### **优化步骤**
1. **分析三元组的完整性**：
   - 如果三元组中的实体或关系缺失（如 "苹果公司 - 创始人 - 史蒂夫"），请基于文本补全信息，使其完整。
   
2. **检查关系是否清晰**：
   - 如果某个关系的表达不够准确或模糊（如 "苹果公司 - 由 - 史蒂夫·乔布斯"），请改为更合适的表述（如 "苹果公司 - 创始人 - 史蒂夫·乔布斯"）。

3. **判断是否冗余或无意义**：
   - 如果三元组的内容没有实际意义（如 "苹果公司 - 是 - 公司"），或者表达过于泛泛（如 "苹果公司 - 存在于 - 世界"），请删除该三元组，并返回 "无"。

4. **格式化并返回最终结果**：
   - 保持优化后的三元组格式与输入相同，如果有调整，请输出优化后的版本；如果三元组已经足够清晰，则保持原样返回。

---

### **示例 1**
**文本**：
苹果公司（Apple Inc.）由史蒂夫·乔布斯创立，总部位于美国加利福尼亚州库比蒂诺。

**输入三元组**：
(苹果公司, 创始人, 史蒂夫)

**思考过程**：
1. 该三元组的信息不完整，“史蒂夫” 应该是“史蒂夫·乔布斯”。
2. 补全后的三元组应为 (苹果公司, 创始人, 史蒂夫·乔布斯)。

**输出**：
(苹果公司, 创始人, 史蒂夫·乔布斯)

---

### **示例 2**
**文本**：
苹果公司（Apple Inc.）由史蒂夫·乔布斯创立，总部位于美国加利福尼亚州库比蒂诺。

**输入三元组**：
(苹果公司, 是, 公司)

**思考过程**：
1. 该三元组表达的信息过于泛泛，没有提供新的知识。
2. “苹果公司是公司” 这类表述显然没有信息增量，因此应删除。

**输出**：
无

---

### **任务**
请根据上述优化步骤，对以下文本和三元组进行分析，并提供 **思考过程** 以及 **最终优化后的三元组**。

**文本**：
这是一个关于 {part}-{section} 的问答对。
{text}

**输入三元组**：
{triplet}

**思考过程**：
（请一步步推理）

**输出**：
（优化后的三元组 或 "无"）
"""
    llm = ChatOpenAI(
        model=Config.LLM_MODEL, 
        openai_api_key=Config.OPENAI_API_KEY, 
        openai_api_base=Config.BASE_URL,
    )
    response = llm.invoke(prompt)
    answer = response.content.strip().split('\n')[-1]
    
    return answer

def init_bm25retriever(filepath, db_filepath):
    df = pd.read_csv(filepath)
    print(df.head())
    filename = Path(filepath).stem
        
    docs = [
        Document(
            page_content=" ".join(jieba.cut(row["问答对"])),
            metadata={
                "id": str(row["index"]),
                "source": f'{filename}',
                "type": f'{row["部分"]}_{row["考试类型"]}'
            }
        )
        for _, row in df.iterrows()
    ]   
    with open(db_filepath, "wb") as f:
        pickle.dump(docs, f)

def init_bm25retriever_triplets(filepath, db_filepath):
    df = pd.read_csv(filepath)
    print(df.head())
    filename = Path(filepath).stem

    # 三元组去重
    seen = set()
    unique_docs = []
    for index, row in df.iterrows():
        triplet = row['三元组'].replace('(','').replace(')','').replace(',','').replace(' ','')
        key = (triplet, f'{row["部分"]}_{row["考试类型"]}')
        if key not in seen:
            seen.add(key)
            unique_docs.append(Document(
            page_content=" ".join(jieba.cut(triplet)),
                metadata={
                    "id": str(index), 
                    "source": f'{filename}',
                    "type": key[1]  # 直接使用 key[1]，保证一致性
                }
        ))
        else:
            print(f"重复的三元组: {triplet}")

    with open(db_filepath, "wb") as f:
        pickle.dump(unique_docs, f)
    

def retrieved_docs_to_str(retrieved_docs: List[Document]):
    """将检索到的文档转换为字符串"""
    return "\n".join([f"【{doc.metadata['type']}】{doc.page_content}" for doc in retrieved_docs])

def query2qapair(query: str) -> List[str]:
    """将查询转换为文档/问答对，便于在知识库中搜索"""
    prompt = f"""Write a qa pair that answers the given query to Shanghai Examination Authority. Do not return any other content.
    
Query: 考生如果被春季招生院校预录取了，是否可以取消预录取参与后续的秋考？
Passage: 问：考生如果被春季招生院校预录取了，是否可以取消预录取参与后续的秋考？答：不能。根据市教委文件规定，预录取考生（含列入候补资格名单并最终被预录取的考生）无论是否与院校确认录取，一律不得参加后续专科层次依法自主招生、秋季统一高考等。

Query: 考生可以填写几个综合评价志愿？
Passage: 问：考生可以填写几个综合评价志愿？答：综评实行平行志愿。考生最多可以填报4个院校专业组志愿，每个院校专业组内最多可填报4个专业志愿。
    
Query: 本市户籍考生、非本市高中阶段学校毕业生中的应届毕业生2024年在沪报考普通高校条件是什么？
Passage: 问：本市户籍考生、非本市高中阶段学校毕业生中的应届毕业生2024年在沪报考普通高校条件是什么？答：要求：高中阶段学校毕业或具有同等学历；有关证明：身份证，户籍证明，学籍所在学校证明；备注：

Query: {query}
Passage: 
    """
    # client = OpenAI(
    #       api_key='ebe4d4b6-00ae-4ea7-9890-9356d6a29570', 
    #       base_url = 'https://ark.cn-beijing.volces.com/api/v3'
    #     )
    # response = client.chat.completions.create(
    #     model="deepseek-r1-250120",
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    # return response.choices[0].message.content.replace("Passage: ", "")
    llm = ChatOpenAI(
            model=Config.LLM_MODEL, 
            openai_api_key=Config.OPENAI_API_KEY, 
            openai_api_base=Config.BASE_URL,
        )
    response = llm.invoke(prompt)
    return response.content.replace("Passage: ", "").strip()

def extract_triplets(triplets_str: str) -> List[str]:
    """从答案中提取三元组列表"""
    pattern = r"\(([^)]+)\)"
    matches = re.findall(pattern, triplets_str)
    result = [f"({item})" for item in matches]
    return result

def query2triplets(query: str) -> List[str]:
    """将查询转换为三元组格式"""
    prompt = f"""Write structured triplets that accurately answer the given query to Shanghai Examination Authority. Do not return any other content.
    
Query: 参加初中学业水平考试需要携带身份证吗？
Triplets: [(参加初中学业水平考试, 不需要, 携带身份证), (参加初中学业水平考试, 可以选择, 携带身份证)]

Query: 报考军队院校的条件？
Triplets: [(报考部队院校, 需要条件, 参加普通高等学校招生全国统一考试成绩达到录取标准), (报考部队院校, 婚姻状况要求, 未婚), (报考部队院校, 需要条件, 高中阶段体质测试成绩及格以上), (报考部队院校, 需要通过, 政治考核), (报考部队院校, 要求, 面试合格), (报考部队院校, 需通过, 体格检查合格)]

Query: {query}
Triplets: 
    """
    # response = client.chat.completions.create(
    #     model="deepseek-r1-250120",
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    # return response.choices[0].message.content.replace("Triplets: ", "")
    llm = ChatOpenAI(
            model=Config.LLM_MODEL, 
            openai_api_key=Config.OPENAI_API_KEY, 
            openai_api_base=Config.BASE_URL,
        )
    response = llm.invoke(prompt)
    return extract_triplets(response.content.replace("Triplets: ", ""))

def query2passage(llm, query: str) -> List[str]:
    """将查询转换为文本段落格式"""
    prompt = f"""Write a passage that answers the given query to Shanghai Examination Authority. Do not return any other content.
    
Query: 中招的市级优秀体育学生申请条件是什么？
Passage: 具有2024年本市中招报名资格并完成报名，且具备以下条件之一的学生，可填写《2024年上海市高中阶段学校市级优秀体育学生资格确认报名表》（以下简称《报名表》，见附件1），并经毕业学校公示5个工作日。\n（一）自2021年9月1日起获得市教育、体育行政部门认可，并与招生学校项目对口的市级及以上体育比赛集体项目前6名的主力队员或个人项目前5名的学生（《2024年上海市市级优秀体育学生市级体育赛事认定目录》，见附件2）。\n（二）自2021年9月1日起获得本市由各区教育、体育行政部门举办并与招生学校项目对口的区级体育比赛集体项目第1名的主力队员或个人项目第1名的学生（须填写《2024年上海市高中阶段学校市级优秀体育学生区级赛事报考资格认定表》，见附件3），并经毕业学校报区教育、体育行政部门同意）。

Query: 高考英语总分多少分？
Passage: 统一文化考试科目及计分办法\n统一文化考试科目为语文、数学、外语3门科目。语文、数学每科目总分150分。外语科目考试分为笔试和听说测试，笔试分值为115分，听说测试分值为35分，总分150分；外语科目的考试语种分设英语、俄语、日语、法语、德语、西班牙语6种，由报考学生任选1种。统一文化考试成绩总分450分。\n根据本市高考改革相关规定，统一高考外语科目考试实行一年两考，考试时间分别为1月和6月。其中，1月的外语科目考试即为2025年春季考试外语科目考试。

Query: {query}
Passage: 
    """
    response = llm.invoke(prompt)
    return response.content.replace("Passage: ", "").strip()

def rewrite_query(llm, history: List[Conversation], user_input: str) -> str:
    """重写查询"""

    prompt = """你是一位专业从事信息查询语义解析与优化的语言工程师，逻辑严谨、注重细节、善于结构化思考。

    下面是一位用户和上海地区考试院助手的对话，请基于用户的所有历史对话（包括过去的提问和本次的新输入），提炼本次询问的核心信息，并对查询问题进行语义解析与优化。确保查询的准确性、完整性、相关性，并去除无效信息。请将优化后的查询问题返回，不要输出任何解释说明。

    History: {}
    Current Query: {}
    Revised Query: """.format(
        '\n'.join([str(conversation) for conversation in history]),
        user_input
    )

    # response = client.chat.completions.create(
    #     model="deepseek-r1-250120",
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    # return response.choices[0].message.content.replace("Revised Query: ", "")
    response = llm.invoke(prompt)
    return response.content.replace("Revised Query: ", "").strip()

def extract_exam_type(response: str) -> List[str]:
    """提取考试类型"""
    response = response.strip()  # 清理多余空格
    if response == "无":
        return []
    return response.split("; ")

def get_exam_types(llm, query: str) -> List[str]:
    prompt = f"""请根据用户输入判断其涉及的考试类型，并严格按照以下格式返回：
    
    1. 如果匹配到单个考试类型，直接返回该类型（如：高考学考_秋考）。
    2. 如果匹配到多个考试类型，用";"分隔（如：高考学考_秋考; 研考成考_研究生招生考试）。
    3. 如果不属于任何类型，返回“无”。
    
    允许的考试类型列表：
    {", ".join(Config.EXAM_TYPE_LIST)}

    用户输入："{query}"  
    你的回答（必须符合上述格式）："""
    response = llm.invoke(prompt)
    return extract_exam_type(response.content.replace("你的回答（必须符合上述格式）：", "").strip())

def get_exam_types_by_keywords(query: str) -> List[str]:
    keyword_dict = {
        '高考学考_秋考': ['秋考', '高考', '秋季统考', "高三考试", '普通高校招生统一考试'],
        '高考学考_春考': ['春考', '春季高考', '春季统考', "春季招生"],
        '高考学考_艺术类统一考试': ['艺术类', '艺考', '艺术统考', '艺术', '艺术类统一考试'],
        '高考学考_体育类统一考试': ['体育类', '体育考试', '体育统考', '体考', '体育', '体育类统一考试', "体育招生", "体育生高考"],
        '高考学考_三校生高考': ['三校生', '职校高考', "三校生高考", "中专升本", "职校高考", "职业学校升学", "技校高考", "职高高考", "对口升学"],
        '高考学考_专科自主招生': ["专科自主招生", "专科招生", "高职自主招生", "专科提前批", "高职单招", "单独招生", "单招"],
        '高考学考_高中学业水平考试': ['学业水平考试', '会考', '高中会考', "合格考", "高中合格考试", "高中学业水平合格性考试"],
        '高考学考_中职校学业水平考试': ['中职', '中职学考'],
        '高考学考_其他考试—专升本考试': ['专升本', "升本考试", "高职升本科"],
        '高考学考_其他考试—普通高校联合招收华侨港澳台考试': ['华侨生', "侨港澳联考", "华侨联考",'普通高校联合招收华侨港澳台考试'],
        '中考中招_中考中招': ['中考', '中招', '初中升学', "初升高", "初中毕业考试", "初三考试"],
        '研考成考_研究生招生考试': ['考研', '研究生', '研招', '硕士'],
        '研考成考_成人高考': ['成人高考', '成考', "成教", "成人学历"],
        '研考成考_同等学力人员申请硕士学位外国语水平和学科综合水平全国统一考试': ["同等学力申硕", '在职研究生统考', '同等学力', '申请硕士', '全国统考', '同等学力人员申请硕士学位外国语水平和学科综合水平全国统一考试'],
        '自学考试_自学考试': ['自考', '自学考试'],
        '证书考试_全国大学英语四、六级考试（CET)': ['四六级', '英语四级', '英语六级', 'CET', "大学英语考试", '全国大学英语四、六级考试（CET)', '四级', '六级', '四、六级', "cet4", "cet6"],
        '证书考试_全国中小学教师资格考试笔试': ['教师证', '教师资格', '教师资格证', '教资', '教资笔试','全国中小学教师资格考试笔试'],    
        '证书考试_全国计算机等级考试（NCRE)': ['计算机等级考试', 'NCRE', '计算机', '一级', '二级', '三级', '四级'],
        '证书考试_上海市高校信息技术水平考试': ['信息技术水平考试', '上海信息考试', '信息技术', "上海高校IT考试", '上信考', '计算机'],
    }
    
    exam_type_list = []
    for category, keywords in keyword_dict.items():
        for keyword in keywords:
            if keyword in query and category not in exam_type_list:
                exam_type_list.append(category)
   
    return exam_type_list

def remove_duplicate_docs(docs):
    """
    移除 page_content 和 metadata["type"] 相同的重复 Document，仅保留一个。
    
    :param docs: List[(Document, score)]，待去重的文档列表
    :return: List[Document]，去重后的文档列表
    """
    docs = [doc[0] for doc in sorted(docs, key=lambda x: x[1])] # FAISS相似度计算越低越好
    
    seen = set()
    unique_docs = []

    for doc in docs:
        if len(unique_docs) == Config.RETRIEVER_TOP_K:
            break
        key = (doc.page_content, doc.metadata["type"])  # 唯一性判断键
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs[:Config.RETRIEVER_TOP_K]

def rerank_docs(docs, query, similarity_threshold=Config.SIMILARITY_THRESHOLD, rerank_model=None):
    """
    对文档进行重排序
    
    :param docs: List[Document]，待重排序的文档列表
    :param query: str，查询语句
    """
    if rerank_model is None:
        rerank_model = BgeReranker() # TODO: 存在st.session_state中
    docs_text = [doc.page_content for doc in docs]
    score_list = rerank_model.compute_score(query, docs_text)
    sorted_indices = np.argsort(score_list)[::-1]

    return [docs[i] for i in sorted_indices if score_list[i] > similarity_threshold]

def rerank_triplets(triplets, query_list, similarity_threshold=0, rerank_model=None):
    """
    对三元组进行重排序
    """
    if rerank_model is None:
        rerank_model = BgeReranker()
    triplets_text = [doc.page_content for doc in triplets]
    
    max_score_list = []
    for triplet in triplets_text:
        score_list = rerank_model.compute_score(triplet, query_list)
        max_score_list.append(max(score_list))

    sorted_indices = np.argsort(max_score_list)[::-1]
    return [triplets[i] for i in sorted_indices if max_score_list[i] > similarity_threshold]

def rerank_triplets_with_multiple_threshold(triplets, query_list, similarity_threshold_list=[], rerank_model=None):
    """
    对三元组进行重排序
    """
    if rerank_model is None:
        rerank_model = BgeReranker()
    triplets_text = [doc.page_content for doc in triplets]
    
    max_score_list = []
    for triplet in triplets_text:
        score_list = rerank_model.compute_score(triplet, query_list)
        max_score_list.append(max(score_list))

    retrieved_id_list = []
    sorted_indices = np.argsort(max_score_list)[::-1]
    for similarity_threshold in similarity_threshold_list:
        retrieved_ids = [triplets[i].metadata['id'] for i in sorted_indices if max_score_list[i] > similarity_threshold]
        retrieved_id_list.append(retrieved_ids)
    return retrieved_id_list

def ask_for_more_info(llm, query:str) -> str:
    prompt = f"""你是上海教育局的智能助手，专门解答有关上海地区各类考试的问题（包括：{", ".join(Config.EXAM_TYPE_LIST)}）。
    
    这里有一个用户的提问：{query}
    但很不幸，用户的提问中未能明确指出具体的考试类型，因此请按以下要求生成回答：

    1. 如果用户没有在提问，比如只是在问好或表达感谢，那么直接生成回答即可，否则:
    2. 明确告知用户：根据目前提供的信息，无法判断其所指的具体考试类型。
    3. 友好地邀请用户提供更多的上下文信息或具体说明其所询问的考试。
    4. 保持语气温和、礼貌并乐于助人，表达你愿意继续协助用户。

    请基于上述要求生成最终回应内容，不用包含其他内容。

    Response:
    """
    response = llm.invoke(prompt)
    return response.content.replace("Response:", "").strip()