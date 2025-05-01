from src.utils.config import config
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from src.models import init_llm
import streamlit as st
from src.utils.conversations import Conversation, Role, str2Role
import torch
torch.classes.__path__ = []
from src.agents.state import AgentState
from src.agents.graph import create_graph
from src.retriever.FAISSRetriever import FAISSRetriever
from src.retriever.ChromaDBRetriever import ChromaDBRetriever
from typing import List
from src.utils.database import connect_database, insert_data, update_data, get_all_data
from src.utils import summarize
from src.utils import init_session_state

def main():
    # Initialize LLM if not already in session state
    init_session_state()
    
    st.set_page_config(page_title="上海教育局智能助手", layout="wide")

    with st.sidebar:
        st.title("上海教育局智能助手")
        st.markdown("<hr>", unsafe_allow_html=True)
        # st.write("This is a simple chat application built with Streamlit.")
        # 获取所有对话历史
        # all_conversations = get_all_data(st.session_state["conn"], st.session_state["cursor"])
        # print(all_conversations)
        # for conv in all_conversations:
        #     if st.button(f"{conv[4]}", use_container_width=True,key=f"button_{conv[0]}"):
        #         # Load the selected conversation history
        #         st.session_state['chat_history'] = [Conversation(str2Role(item[0]['role']), item[0]['content']) for item in conv[3]]  # Assuming conv[3] contains the conversation list
        #         st.session_state['id'] = conv[0]
        
    
    # 创建一个聊天历史存储，并在WebUI中动态显示
    placeholder = st.empty()
    with placeholder.container():
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        history: List[Conversation] = st.session_state['chat_history']
        
        for conversation in history:
            conversation.show()
    
    # 创建工作流，处理新的一轮用户输入
    workflow = create_graph()
        
    if prompt_text := st.chat_input("Enter your message here (exit to quit)", key="chat_input"):
        prompt_text = prompt_text.strip()
            
        if prompt_text.lower() == "exit":
            st.stop()
            
        graph = workflow.compile()
        input_state = AgentState(history=history, user_input=prompt_text)
            
        # 储存并显示用户输入
        conversation = Conversation(role=Role.USER, content=prompt_text)
        history.append(conversation)  # 在对话历史中添加对话
        conversation.show()           # 展示当前会话，即用户输入内容
            
            
        # 创建占位符，用于动态更新界面内容：希望在加载的过程中assistant的avatar已经显示
        placeholder = st.empty()      # 创建占位符
        message_placeholder = placeholder.chat_message(name="assistant", avatar="assistant") # 显示AI助手的信息
        markdown_placeholder = message_placeholder.empty() # 先创建一个空的占位符
            
        output = graph.invoke(input_state)
        full_response = markdown_placeholder.write_stream(chunk for chunk in output['final_answer'])
        conversation = Conversation(role=Role.ASSISTANT, content=full_response)
        history.append(conversation) 
        st.session_state['chat_history'] = history
        
        # TODO: st.session_state.pop('id', None)完成多轮对话切换
        # if 'id' not in st.session_state:
        #     summary = summarize(st.session_state['llm'], history)
        #     st.session_state['id'] = insert_data(st.session_state["conn"], st.session_state["cursor"], "test", history, summary)
        # else:
        #     update_data(st.session_state["conn"], st.session_state["cursor"], st.session_state['id'], history)
            
if __name__ == "__main__":
    main()
