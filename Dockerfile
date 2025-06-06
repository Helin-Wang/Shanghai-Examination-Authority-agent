# 选择官方 Python 3.10.16 基础镜像
FROM python:3.10.16

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器内
COPY . /app

# 安装依赖（如果用 pip）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Expose Streamlit 8601 端口
EXPOSE 8701

# 运行 Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8701", "--server.runOnSave=True"]