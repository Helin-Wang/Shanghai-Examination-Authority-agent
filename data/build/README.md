# 📄 三元组提取

- 本部分旨在从结构化的 Word 文档中提取问答对数据，并进一步用于知识图谱构建等任务。整体流程包括：

    - 加载 Word 文档并提取结构化问答对（load_docx.py）

    - 利用大语言模型抽取三元组（extract_triplets.py）

    - 筛选、清洗抽取出的三元组数据（filter_triplets.py）

## 🧩 脚本说明

```python
python load_docx.py --input_path {输入word文件路径} --output_path {输出csv文件路径}
python extract_triplets.py --input_path {输入csv文件路径} --output_path {输出csv文件路径}
python filter_triplets.py --input_path {输入csv文件路径} --output_path {输出csv文件路径}
```

### load_docx.py：Word 文档 → 问答对 CSV

- 自动识别 Word 文档中的二级（Heading 2）和三级标题（Heading 3）

- 将每个问答对（以“问：”为开头）归类到所属的章节结构中（如：部分 / 考试类型 / 问题类型）

- 将处理后的数据保存为 CSV 文件

### extract_triplets.py：问答对 → 初步三元组

- 该脚本调用 LLM从问答对中抽取结构化的三元组信息，输出格式为：（实体1，关系，实体2）

- 输出为原始 CSV 加上新增的 “三元组” 列（包含抽取结果）

### filter_triplets.py：三元组清洗优化

- 该脚本借助 LLM 判断三元组是否完整、清晰、有意义

- 删除冗余、无效三元组，输出新的优化版三元组

- 结果写入csv中新的列“修改后三元组”





