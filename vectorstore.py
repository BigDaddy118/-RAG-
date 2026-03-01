# 创建向量数据库
import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def create_vectorstore(docs, persist_directory="./chroma_db"):
    embeddings = OllamaEmbeddings(model="snowflake-arctic-embed")  # 或你选择的模型
    
    # 检查是否已存在向量库
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        # 加载已有库
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        # 添加新文档（避免重复添加，可以自行判断，但简单起见直接添加）
        vectorstore.add_documents(docs)
        print(f"已有向量库，新增 {len(docs)} 个文档块")
    else:
        # 新建库
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"新建向量数据库，包含 {len(docs)} 个文档块")
    
    vectorstore.persist()
    return vectorstore