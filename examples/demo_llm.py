import os
import sys

# 把项目根目录加入模块搜索路径，让我们能导入config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME
from openai import OpenAI


def main():
    # 初始化大模型客户端
    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )

    print("===== 大模型简单调用测试 =====")
    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是AI应用开发助手，回答简洁简短。"},
            {"role": "user", "content": "简单讲一下什么是RAG？"}
        ],
        temperature=0.3
    )
    answer = response.choices[0].message.content
    print(f"模型回答：\n{answer}")


if __name__ == "__main__":
    main()