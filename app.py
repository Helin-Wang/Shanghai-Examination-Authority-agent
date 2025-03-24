from src.utils.config import config
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from src.models import init_llm
import streamlit as st
from src.utils.conversations import Conversation, Role
import torch
torch.classes.__path__ = []
from src.agents.state import AgentState
from src.agents.graph import create_graph
from src.retriever.FAISSRetriever import FAISSRetriever

def init_session_state():
    if "llm" not in st.session_state:
        st.session_state["llm"] = init_llm(
            config.LLM_MODEL, 
            config.OPENAI_API_KEY, 
            config.BASE_URL
        )
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = FAISSRetriever(
            config.VECTOR_DB_PATH,
            config.EXAM_TYPE_LIST,
            config.RETRIEVER_TOP_K
        )  
    if "kg_retriever" not in st.session_state:
        st.session_state["kg_retriever"] = FAISSRetriever(
            config.KG_DB_PATH,
            config.EXAM_TYPE_LIST,
            config.RETRIEVER_TOP_K
        )
        
        

def main():
    st.set_page_config(page_title="上海教育局智能助手", layout="wide")

    with st.sidebar:
        st.title("上海教育局智能助手")
        st.write("This is a simple chat application built with Streamlit.")
    
    # Initialize LLM if not already in session state
    init_session_state()
    
    # 创建一个聊天历史存储，并在WebUI中动态显示
    # placeholder = st.empty()
    # with placeholder.container():
    #     if 'chat_history' not in st.session_state:
    #         st.session_state['chat_history'] = []
    #     history: List[Conversation] = st.session_state['chat_history']
        
    #     for conversation in history:
    #         conversation.show()
    
    if st.session_state.get("llm"):
        workflow = create_graph()
        
        if prompt_text := st.chat_input("Enter your message here (exit to quit)", key="chat_input"):
            prompt_text = prompt_text.strip()
            
            if prompt_text.lower() == "exit":
                st.stop()
                
            # 储存并显示用户输入
            conversation = Conversation(role=Role.USER, content=prompt_text)
            # history.append(conversation)  # 在对话历史中添加对话
            conversation.show()           # 展示当前会话，即用户输入内容
            
            graph = workflow.compile()
            input_state = AgentState(user_input=prompt_text)
            
            
            # 创建占位符，用于动态更新界面内容：希望在加载的过程中assistant的avatar已经显示
            placeholder = st.empty()      # 创建占位符
            message_placeholder = placeholder.chat_message(name="assistant", avatar="assistant") # 显示AI助手的信息
            markdown_placeholder = message_placeholder.empty() # 先创建一个空的占位符
            
            print(st.session_state)
            output = graph.invoke(input_state)
            full_response = markdown_placeholder.write_stream(chunk for chunk in output['final_answer'])
            conversation = Conversation(role=Role.ASSISTANT, content=full_response)
            # history.append(conversation) 
            
if __name__ == "__main__":
    main()
