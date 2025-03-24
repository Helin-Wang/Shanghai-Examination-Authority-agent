from typing import List, Tuple
import numpy as np
from langchain.schema import Document

from openai import OpenAI
import os
import re
import pandas as pd
from tqdm import tqdm
import streamlit as st

# api_key_v3 = '73688f34-7502-421b-a188-040f460d5eb1'
# api_key_r1 = 'ebe4d4b6-00ae-4ea7-9890-9356d6a29570'
# os.environ["OPENAI_API_BASE"] = 'https://ark.cn-beijing.volces.com/api/v3'
# os.environ["OPENAI_API_KEY"] = api_key_r1
# client = OpenAI(
#   api_key=api_key_r1, 
#   base_url = 'https://ark.cn-beijing.volces.com/api/v3'
# )


def retrieved_docs_to_str(retrieved_docs: List[Tuple[Document, np.float32]]):
    """将检索到的文档转换为字符串"""
    return "\n".join([doc.page_content for doc, _ in retrieved_docs])

def query2qapair(llm, query: str) -> List[str]:
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
    response = llm.invoke(prompt)
    return response.content.replace("Passage: ", "").strip()

def extract_triplets(triplets_str: str) -> List[str]:
    """从答案中提取三元组列表"""
    pattern = r"\(([^)]+)\)"
    matches = re.findall(pattern, triplets_str)
    result = [f"({item})" for item in matches]
    return result

def query2triplets(llm, query: str) -> List[str]:
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
    response = llm.invoke(prompt)
    return extract_triplets(response.content.replace("Triplets: ", ""))

def query2passage(query: str) -> List[str]:
    """将查询转换为文本段落格式"""
    prompt = f"""Write a passage that answers the given query to Shanghai Examination Authority. Do not return any other content.
    
Query: 中招的市级优秀体育学生申请条件是什么？
Passage: 具有2024年本市中招报名资格并完成报名，且具备以下条件之一的学生，可填写《2024年上海市高中阶段学校市级优秀体育学生资格确认报名表》（以下简称《报名表》，见附件1），并经毕业学校公示5个工作日。\n（一）自2021年9月1日起获得市教育、体育行政部门认可，并与招生学校项目对口的市级及以上体育比赛集体项目前6名的主力队员或个人项目前5名的学生（《2024年上海市市级优秀体育学生市级体育赛事认定目录》，见附件2）。\n（二）自2021年9月1日起获得本市由各区教育、体育行政部门举办并与招生学校项目对口的区级体育比赛集体项目第1名的主力队员或个人项目第1名的学生（须填写《2024年上海市高中阶段学校市级优秀体育学生区级赛事报考资格认定表》，见附件3），并经毕业学校报区教育、体育行政部门同意）。

Query: 高考英语总分多少分？
Passage: 统一文化考试科目及计分办法\n统一文化考试科目为语文、数学、外语3门科目。语文、数学每科目总分150分。外语科目考试分为笔试和听说测试，笔试分值为115分，听说测试分值为35分，总分150分；外语科目的考试语种分设英语、俄语、日语、法语、德语、西班牙语6种，由报考学生任选1种。统一文化考试成绩总分450分。\n根据本市高考改革相关规定，统一高考外语科目考试实行一年两考，考试时间分别为1月和6月。其中，1月的外语科目考试即为2025年春季考试外语科目考试。

Query: {query}
Passage: 
    """
    response = client.chat.completions.create(
        model="deepseek-r1-250120",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.replace("Passage: ", "").strip()


def rewrite_query(llm, query: str) -> str:
    """重写查询"""
    prompt = f"""你是一位专业从事信息查询语义解析与优化的语言工程师，逻辑严谨、注重细节、善于结构化思考。
    
    请对以下对向上海地区考试院的提问进行语义解析与优化，确保查询的准确性、完整性、相关性，并去除无效信息。请将优化后的查询结果返回，不要输出任何解释说明。
        
Query: {query}
Revised Query: 
    """
    # response = client.chat.completions.create(
    #     model="deepseek-r1-250120",
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    # return response.choices[0].message.content.replace("Revised Query: ", "")
    response = llm.invoke(prompt)
    return response.content.replace("Revised Query: ", "").strip()

