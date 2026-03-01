# 构建问答链
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def setup_qa_chain(vectorstore):
    """
    设置 RAG 问答链：结合检索器和 LLM。
    """
    if not vectorstore:
        return None

    llm = OllamaLLM(model="qwen2.5:1.5b", temperature=0.5)
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

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    print("问答引擎已启动。")
    return qa_chain