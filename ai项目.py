import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import gradio as gr

# 定义全局变量占位
qa_chain = None

def load_and_split_docs(directory="./docs"):
    """
    加载指定目录下的所有 PDF 和 TXT 文档，并将其分割成文本块。
    """
    # 如果目录不存在，则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"请将你的文档放入 {directory} 文件夹后再运行。")
        return []

    documents = []
    # 遍历目录中的文件
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            print(f"已加载PDF文件: {filename}")
        elif filename.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
            print(f"已加载TXT文件: {filename}")

    if not documents:
        print("未找到任何文档。请确保 ./docs 文件夹内有 PDF 或 TXT 文件。")
        return []

    # 文本分割器：将长文档切成小块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,      # 每个文本块的最大字符数
        chunk_overlap=50,    # 块与块之间的重叠字符数，避免语义断裂
        length_function=len,
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"文档已切分为 {len(split_docs)} 个文本块")
    return split_docs

def create_vectorstore(docs):
    """
    将文档块向量化，并存入 Chroma 向量数据库。
    """
    if not docs:
        return None

    # 使用 Ollama 的嵌入模型（请确保模型已通过 ollama pull 下载）
    embeddings = OllamaEmbeddings(model="snowflake-arctic-embed")  # 与你的模型名一致

    # 创建向量数据库，并自动将文档存入
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"  # 数据库保存路径
    )
    # 持久化数据库（保存到磁盘）
    vectorstore.persist()
    print(f"向量数据库创建完成，已保存至 ./chroma_db 目录")
    return vectorstore

def setup_qa_chain(vectorstore):
    """
    设置 RAG 问答链：结合检索器和 LLM。
    """
    if not vectorstore:
        return None

    # 加载 Ollama 上的大语言模型（请确保已下载）
    llm = OllamaLLM(model="qwen2.5:1.5b", temperature=0.2)

    # 自定义提示词模板，引导模型仅基于检索内容回答
    prompt_template = """你是一个专业的文档问答助手。请仅根据以下提供的“上下文信息”来回答用户的问题。
如果你在上下文中找不到答案，或者问题与上下文无关，请直接说“根据提供的文档，我无法回答这个问题”。不要编造答案。

上下文：
{context}

问题：{question}

请用中文回答："""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 构建检索式问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 简单地将所有检索到的文档块合并送入 LLM
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}  # 每次检索返回最相关的 3 个文档块
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True  # 返回参考源文档（可选）
    )
    print("问答引擎已启动。")
    return qa_chain

def answer_question(message, history):
    """
    Gradio 聊天处理函数：接收用户消息，返回模型回答。
    """
    global qa_chain  # 声明使用全局变量
    if not message:
        return "请输入问题。"

    # 调用问答链
    result = qa_chain.invoke({"query": message})
    answer = result['result']

    # 如果你想在界面上显示参考文档，可以在这里添加逻辑（本例略）
    return answer

def launch_interface(qa_chain):
    """
    启动 Gradio 聊天界面。
    """
    if not qa_chain:
        print("问答链未正确初始化，无法启动界面。")
        return

    gr.ChatInterface(
        fn=answer_question,
        title="📚 本地知识库问答助手",
        description="上传你的文档到 `./docs` 文件夹，然后问我关于它们的问题！所有处理都在本地进行，保护数据隐私。"
    ).launch()

# 主程序入口
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
    qa_chain = setup_qa_chain(vectorstore)

    # 4. 启动 Gradio 界面
    print("\n[4/4] 正在启动聊天界面...")
    launch_interface(qa_chain)