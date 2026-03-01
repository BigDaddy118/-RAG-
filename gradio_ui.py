# Gradio 界面
import gradio as gr
# 不再需要 import config

def create_ui(qa_chain):  # 新增参数 qa_chain
    """
    启动 Gradio 聊天界面，使用传入的 qa_chain 问答链。
    """
    if not qa_chain:
        print("问答链未正确初始化，无法启动界面。")
        return

    def answer_question(message, history):
        if not message:
            return "请输入问题。"
        result = qa_chain.invoke({"query": message})
        return result['result']

    gr.ChatInterface(
        fn=answer_question,
        title="📚 本地知识库问答助手",
        description="上传你的文档到 `./docs` 文件夹，然后问我关于它们的问题！所有处理都在本地进行，保护数据隐私。"
    ).launch()