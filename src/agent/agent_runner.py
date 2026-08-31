import json
from src.llm.llm_client import DeepSeekLLMClient
from src.agent.tools import TOOL_LIST, TOOL_FUNC_MAP

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
        # 消息记忆，保存完整对话历史
        self.messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ]

    def run(self, user_question: str) -> str:
        # 用户消息加入记忆
        self.messages.append({"role": "user", "content": user_question})

        resp_text = self.llm.chat(prompt=json.dumps(self.messages, ensure_ascii=False))
        resp_text = resp_text.replace("```json", "").replace("```", "").strip()
        output = json.loads(resp_text)

        if not output["use_tool"]:
            ans = output["answer"]
            self.messages.append({"role": "assistant", "content": ans})
            return ans

        # 执行工具
        tool_name = output["tool_name"]
        tool_args = output["tool_args"]
        print(f"\n>>> Agent调用工具：{tool_name}, 参数:{tool_args}")
        func = TOOL_FUNC_MAP[tool_name]
        tool_result = func(**tool_args)
        print(f">>> 工具返回结果：{tool_result}")

        # 工具结果，单独发一次性prompt，强制输出自然语言，禁止JSON
        final_prompt = f"""
用户原始提问：{user_question}
工具名称：{tool_name}
工具返回结果：{tool_result}

任务：根据以上信息，直接输出自然语言回答用户。
⚠️禁止输出JSON，禁止输出thought、use_tool等字段，只输出人类可读的回答文本。
"""
        final_ans = self.llm.chat(prompt=final_prompt)
        self.messages.append({"role":"assistant","content":final_ans})
        return final_ans

    def clear_memory(self):
        """清空记忆，开启新会话"""
        self.messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ]


if __name__ == "__main__":
    agent = SimpleAgent()
    print("==== Agent多轮对话聊天，输入quit退出 ====")
    while True:
        question = input("\n你：")
        if question.strip().lower() == "quit":
            print("Agent会话结束")
            break
        reply = agent.run(question)
        print(f"Agent：{reply}")
