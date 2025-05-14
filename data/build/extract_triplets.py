import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import re
import os
import argparse

client = OpenAI(
  api_key="", # Your API Key 
  base_url = "", # Your Base Url
)

prompt = """你是一个知识图谱专家。我正在制作关于考试信息的知识图谱，我将给你一段文字，请依次完成下面几个步骤：
（1）首先识别其中所有的实体，不同实体之间用顿号（、）分隔

（2）然后探索你抽取到的这些实体之间的关系，用三元组的形式表示。注意，你只能寻找上一步你得到的这些实体之间的关系，不可以加入新的实体。

（3）请你一步一步思考，并且提供你的想法；在给出你的回答之前，反复思考你给出的三元组本身是否正确且没有歧义

（4）将最终的三元组列用@@隔开，三元组之间用顿号（、）隔开，即最后一行将是@@（实体1，关联，实体2）、（实体1，关联，实体2）...@@形式；若没有可抽取的三元组关系，则返回无


文本：
这是一个关于PART-SECTION-SUBSECTION的问答对
TEXT
"""

def extract_triplet_by_llm(part, section, subsection, text):
    def extract_triplet_from_ans(s):
        pattern = r'@@(.*?)@@'  # 使用非贪婪匹配
        matches = re.findall(pattern, s)
        
        if not matches:
            return []
        
        if matches[0].strip() == '无':
            return []
        triplets = matches[0].split('、')
        return triplets

    # 让LLM直接生成
    completion = client.chat.completions.create(
        model="deepseek-r1-250120",
        messages=[
        {"role":"user","content":prompt.replace("TEXT", text).replace("SUBSECTION", subsection).replace("SECTION", section).replace("PART", part)}])
    
    answer = completion.choices[0].message.content.strip()
    
    return extract_triplet_from_ans(answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从问答对中抽取三元组')
    parser.add_argument('--input_path', type=str, required=True, help='输入CSV路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出CSV路径')
    args = parser.parse_args()
    
    # 加载数据库
    # filepath = './考试院faq第1版第7.00稿（打印版）-修改20250508 - 高考 一到六.csv'
    dataset = pd.read_csv(parser.input_path)
    print(dataset.head())
    
    triples_list = []
    try:
        for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
            triples_list.append(extract_triplet_by_llm(row["部分"], row["考试类型"], row["问题类型"], row["问答对"]))
            print(triples_list[-1])
    except Exception as e:
        print(f"Error: {e}")
    finally:     
        dataset["三元组"] = triples_list
        dataset.to_csv(parser.output, index=False)
