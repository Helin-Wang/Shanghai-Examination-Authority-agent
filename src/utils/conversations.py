from enum import auto, Enum
from typing import Optional, Union, List, Dict, Any
import streamlit as st
from dataclasses import dataclass
from streamlit.delta_generator import DeltaGenerator


class Role(Enum):
    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()

    def __str__(self) -> str:
        """
        通过match-case语句返回每个角色对应的字符串表示
        在聊天系统中，通过这些字符串来标识信息的发送者
        """
        match self:
            case Role.SYSTEM:
                return "<|system|>"
            case Role.USER:
                return "<|user|>"
            case Role.ASSISTANT:
                return "<|assistant|>"
    
    def get_message(self) -> Optional[st.chat_message]:
        """
        根据角色返回一个Streamlit聊天消息对象
        st.chat_message是Streamlit中用于创建聊天消息的API，它可以自定义消息的名字和头像。
        """
        match self:
            case Role.SYSTEM:
                return None
            case Role.USER:
                return st.chat_message(name="user", avatar="user")
            case Role.ASSISTANT:
                return st.chat_message(name="assistant", avatar="assistant")

@dataclass
class Conversation:
    role: Role
    content: str
    
    def __str__(self) -> str:
        """
        将Conversation实例转换成字符串，用于日志记录
        """
        match self.role:
            case Role.SYSTEM | Role.USER | Role.ASSISTANT:
                return f'{self.role}\n{self.content}'
    
    def get_text(self) -> str:
        """
        返回Conversation中的文本
        """
        return self.content
    
    def show(self, placeholder: DeltaGenerator | None = None) -> None:
        """
        在Streamlit UI中显示消息。
        placeholder 是一个可选参数，类型是 DeltaGenerator。
        """
        if placeholder:
            message = placeholder
        else:
            message = self.role.get_message()
            
        message.write(self.content)
    
    def show_stream(self, placeholder: DeltaGenerator | None = None) -> None:
        """
        支持流式输出。
        """
        if placeholder:
            message = placeholder
        else:
            message = self.role.get_message()
        
        message.write_stream(chunk for chunk in self.content)
            
    def to_dict(self) -> List[Dict[str, str]]:
        """
        将Conversation对象转换为字典格式，适用于数据存储或API通信。
        """
        convers = []
        if isinstance(self.content, (list, tuple)):
            for c in self.content:
                convers.append({"role": f"{self.role}", "content": f"{c}"})
        else:
            convers.append({"role": f"{self.role}", "content": f"{self.content}"})
        return convers
    
    