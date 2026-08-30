import json
from src.llm.llm_client import DeepSeekLLMClient
from src.agent.tools import TOOL_LIST, TOOL_FUNC_MAP

# Agent系统提示词，约束大模型输出格式
AGENT_SYSTEM_PROMPT = """
你是智能Agent，可以使用工具解决用户问题。
可用工具列表：{tools_info}

强制规则：
1. 凡是和本项目、git、代码、目录结构、项目文档相关的问题，**必须调用rag_search工具查询本地知识库，禁止直接回答**。
2. 如果不确定问题是否在本地知识库，也优先调用rag_search。
3. 只有rag_search返回知识库没有相关内容之后，才允许使用你自身的知识作答。

输出严格JSON格式，禁止输出其他文字！格式二选一：

【不需要调用工具，直接回答用户】
{{
  "thought": "思考过程",
  "use_tool": false,
  "answer": "直接回答用户的答案"
}}

【需要调用工具】
{{
  "thought": "思考过程，说明为什么调用工具",
  "use_tool": true,
  "tool_name": "工具名字",
  "tool_args": {{参数键值对}}
}}
""".format(tools_info=json.dumps(TOOL_LIST, ensure_ascii=False))



class SimpleAgent:
    def __init__(self):
        self.llm = DeepSeekLLMClient()

    def run(self, user_question: str) -> str:
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_question}
        ]

        # 1.大模型做第一次思考，输出JSON
        resp_text = self.llm.chat(prompt=json.dumps(messages, ensure_ascii=False))
        # 清洗，去掉markdown ```json标记
        resp_text = resp_text.replace("```json", "").replace("```", "").strip()
        output = json.loads(resp_text)

        # 分支1：不调用工具，直接返回答案
        if not output["use_tool"]:
            return output["answer"]

        # 分支2：调用工具
        tool_name = output["tool_name"]
        tool_args = output["tool_args"]
        print(f"\n>>> Agent调用工具：{tool_name}, 参数:{tool_args}")

        func = TOOL_FUNC_MAP[tool_name]
        tool_result = func(**tool_args)
        print(f">>> 工具返回结果：{tool_result}")

        # 把【用户原始问题 + 工具执行结果】再丢给大模型，生成最终回答
        final_prompt = f"""
用户问题：{user_question}
工具调用返回结果：{tool_result}
根据工具结果整理成自然语言回答用户。
"""
        final_ans = self.llm.chat(prompt=final_prompt)
        return final_ans


if __name__ == "__main__":
    agent = SimpleAgent()
    print("====Agent启动，输入你的问题====")

    q1 = "如何把本地代码推送到GitHub？"
    ans1 = agent.run(q1)
    print(f"\n【最终回答】{ans1}\n")

    q2 = "789 + 211等于多少"
    ans2 = agent.run(q2)
    print(f"\n【最终回答】{ans2}\n")

    q3 = "什么是Redis"
    ans3 = agent.run(q3)
    print(f"\n【最终回答】{ans3}\n")

    q4 = "项目src目录下面有哪些模块"
    ans4 = agent.run(q4)
    print(f"\n【最终回答】{ans4}\n")


