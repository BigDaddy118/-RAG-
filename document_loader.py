# 文档加载与分割
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_docs(directory="./docs"):
    """
    加载指定目录下的所有 PDF 和 TXT 文档，并将其分割成文本块。
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"请将你的文档放入 {directory} 文件夹后再运行。")
        return []

    documents = []
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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"文档已切分为 {len(split_docs)} 个文本块")
    return split_docs