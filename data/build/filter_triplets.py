from openai import OpenAI
import yaml
from docx import Document
import re
import pandas as pd
from tqdm import tqdm
import ast
import os
import argparse

client = OpenAI(
  api_key="", # Your API Key 
  base_url = "", # Your Base Url
)

filter_prompt = """你是一个知识图谱优化助手。你将获得一个文本片段和与该文本相关的一个知识图谱三元组。由于三元组可能存在不完整、不清晰或冗余的情况，请你按照以下步骤进行优化，并在每一步提供详细的推理过程。

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
这是一个关于 PART - SECTION - SUBSECTION 的问答对。TEXT

**输入三元组**：
TRIPLET

**思考过程**：
（请一步步推理）

**输出**：
（优化后的三元组 或 "无"）
"""

def filter_triplet_by_llm(part, section, subsection, text, triplet):
    completion = client.chat.completions.create(
        model="deepseek-r1-250120",
        messages=[
        {"role":"user","content":filter_prompt.replace("TEXT", text).replace("TRIPLET", triplet).replace("SUBSECTION", subsection).replace("SECTION", section).replace("PART", part)}])

    answer = completion.choices[0].message.content.strip()
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='清洗与优化知识图谱三元组')
    parser.add_argument('--input_path', type=str, required=True, help='输入CSV路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出CSV路径')
    args = parser.parse_args()

    #filepath = './考试院faq第1版第7.00稿（打印版）-修改20250508 - 高考 一到六_triplets.csv'
    df = pd.read_csv(args.input_path)
    print(df.head())
    
    new_triplets = []
    try:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            triplet_list = ast.literal_eval(row['三元组'])
            new_triplet_list = []
            for triplet in triplet_list:
                response = filter_triplet_by_llm(row["部分"], row["考试类型"], row["问题类型"], row['问答对'], triplet)
                output = response.strip().split('\n')[-1]
                if output.strip() == '无':
                    continue
                else:
                    new_triplet_list.append(output)
                print(new_triplet_list)
            new_triplets.append(new_triplet_list)
    except Exception as e:
        print(f"Error: {e}")
    finally:     
        df["修改后三元组"] = new_triplets
        #df.to_csv('./考试院faq第1版第7.00稿（打印版）-修改20250508 - 高考 一到六_modified_triplets.csv', index=False)
        df.to_csv(args.output_path, index=False)
