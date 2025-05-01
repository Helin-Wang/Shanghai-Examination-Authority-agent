import psycopg2
from src.utils.conversations import Conversation
from typing import List
import json
def connect_database():
    # 连接数据库
    conn = psycopg2.connect(
        dbname="rag_agent",
        user="postgres",
        password="Fearless0215",
        host="localhost",
        port="5433"
    )
    cursor = conn.cursor()
    return conn, cursor

def create_table(conn, cursor):
# 创建 conversation_history 表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversation_history (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(50) NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        history JSONB NOT NULL,
        summary TEXT
        );
""")
    conn.commit()

def insert_data(conn, cursor, user_id, history: List[Conversation], summary: str):
    cursor.execute("""
    INSERT INTO conversation_history (user_id, history, summary)
    VALUES (%s, %s::jsonb, %s)
    RETURNING *;
    """, (user_id, json.dumps([conversation.to_dict() for conversation in history]), summary))
    conn.commit()
    return cursor.fetchone()[0]

def update_data(conn, cursor, id, history: List[Conversation]):
    cursor.execute("""
    UPDATE conversation_history
    SET history = %s::jsonb, timestamp = NOW()
    WHERE id = %s
    """, (json.dumps([conversation.to_dict() for conversation in history]), id))
    conn.commit()

def delete_data(conn, cursor, id):
    cursor.execute("""
    DELETE FROM conversation_history
    WHERE id = %s
    """, (id,))
    conn.commit()
    
def get_all_data(conn, cursor):
    cursor.execute("""
    SELECT * FROM conversation_history
    ORDER BY timestamp DESC
    """)
    conn.commit()
    return cursor.fetchall()


