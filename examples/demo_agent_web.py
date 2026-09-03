import gradio as gr
from src.agent.agent_runner import SimpleAgent

# 全局Agent实例
agent = SimpleAgent(max_loop=5)

def chat_handler(user_query: str, chat_history):
    """
    gradio聊天回调函数
    user_query: 用户输入文本
    chat_history: gradio对话历史 [(user_msg, bot_msg), ...]
    """
    if not user_query or user_query.strip() == "":
        return "", chat_history

    try:
        answer = agent.run(user_query.strip())
        chat_history.append((user_query, answer))
        return "", chat_history
    except Exception as e:
        err_msg = f"程序异常：{str(e)}"
        chat_history.append((user_query, err_msg))
        return "", chat_history


def clear_session():
    """清空Agent记忆，开启全新会话"""
    agent.clear_memory()
    return None  # 清空聊天窗口

# 构建网页UI
with gr.Blocks(title="RAG‑Agent知识库助手") as demo:
    gr.Markdown("# 🤖 RAG‑Agent 本地知识库智能体")
    gr.Markdown("功能：本地PDF/TXT知识库检索 + 多轮工具调用（计算器）")

    chatbot = gr.Chatbot(height=550, bubble_full_width=False)
    msg_input = gr.Textbox(label="请输入你的问题", placeholder="文档里面参数A乘以参数B等于多少？")

    with gr.Row():
        submit_btn = gr.Button("发送", variant="primary")
        clear_btn = gr.Button("清空会话")

    # 绑定事件
    submit_btn.click(
        fn=chat_handler,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot]
    )
    msg_input.submit(
        fn=chat_handler,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot]
    )
    clear_btn.click(fn=clear_session, outputs=[chatbot])


if __name__ == "__main__":
    print("🌐 Web服务启动：http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860)
