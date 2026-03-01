# 主程序入口
# main.py
from document_loader import load_and_split_docs
from vectorstore import create_vectorstore
from qa_chain import setup_qa_chain
from gradio_ui import create_ui

# 全局变量 qa_chain 现在直接定义在此文件中
qa_chain = None

if __name__ == "__main__":
    print("="*50)
    print("开始初始化本地知识库系统...")
    print("="*50)

    # 1. 加载并分割文档
    print("\n[1/4] 正在加载文档...")
    docs = load_and_split_docs()
    if not docs:
        print("程序终止。")
        exit()

    # 2. 创建向量数据库
    print("\n[2/4] 正在创建向量数据库...")
    vectorstore = create_vectorstore(docs)

    # 3. 设置问答链
    print("\n[3/4] 正在启动问答引擎...")
    qa_chain = setup_qa_chain(vectorstore)  # 直接赋值给本文件的全局变量

    # 4. 启动 Gradio 界面（将 qa_chain 作为参数传入）
    print("\n[4/4] 正在启动聊天界面...")
    create_ui(qa_chain)  # 注意：这里传递了 qa_chain