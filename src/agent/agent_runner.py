import json
from src.llm.llm_client import DeepSeekLLMClient
from src.agent.tools import TOOL_LIST, TOOL_FUNC_MAP

AGENT_SYSTEM_PROMPT = """你是智能Agent，可以**多次循环调用工具**解决用户问题。
可用工具列表：{tools_info}

强制规则：
1. 凡是和本项目、git、代码、目录结构、项目文档相关的问题，必须调用rag_search工具查询本地知识库，禁止直接回答。
2. 如果不确定问题是否在本地知识库，也优先调用rag_search。
3. rag_search拿到结果后，如果还需要计算、查询，可以继续调用其他工具，不要直接编造答案。
4. **当需要做数学计算（加减乘除）时，禁止模型自己心算，必须调用calculator工具。即使你已经看到数字，也不能直接给出计算结果，一定要使用calculator工具完成运算。**
5. 只有获取全部足够信息之后，才直接输出最终回答。

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
    def __init__(self, max_loop=5):
        self.llm = DeepSeekLLMClient()
        self.max_loop = max_loop  # 最大工具循环轮次，防止死循环
        self.messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ]

    def run(self, user_question: str) -> str:
        # 用户输入加入上下文
        self.messages.append({"role": "user", "content": user_question})

        # ========== 循环工具调用核心 ==========
        for step in range(self.max_loop):
            print(f"\n----- 🧠 Agent 第 {step+1} 轮思考 -----")
            # 请求LLM输出JSON决策
            resp_text = self.llm.chat(prompt=json.dumps(self.messages, ensure_ascii=False))
            resp_text = resp_text.replace("```json", "").replace("```", "").strip()
            output = json.loads(resp_text)

            # 分支1：不需要工具，直接返回答案，循环结束
            if not output["use_tool"]:
                ans = output["answer"]
                self.messages.append({"role": "assistant", "content": ans})
                return ans

            # 分支2：需要调用工具
            tool_name = output["tool_name"]
            tool_args = output["tool_args"]
            print(f">>> Agent调用工具：{tool_name}, 参数:{tool_args}")
            func = TOOL_FUNC_MAP[tool_name]
            tool_result = func(**tool_args)
            print(f">>> 工具返回结果：{tool_result}")

            # ✅关键：把【Agent思考+工具返回结果】追加进对话历史，不直接生成答案！
            # 模拟assistant输出了工具调用的思考
            self.messages.append({
                "role": "assistant",
                "content": json.dumps(output, ensure_ascii=False)
            })
            # 把工具返回作为user侧消息喂回去，让LLM下一轮能读到工具结果
            self.messages.append({
                "role": "user",
                "content": f"【工具{tool_name}返回结果】：{tool_result}"
            })

            # ⭐循环继续！回到for循环开头，LLM读取工具返回，判断是否继续调用工具

        # 循环耗尽，达到最大轮次保护
        return f"(警告：达到最大思考轮次{self.max_loop})，无法继续处理，请简化问题。"

    def clear_memory(self):
        """清空记忆，开启新会话"""
        self.messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ]


if __name__ == "__main__":
    agent = SimpleAgent(max_loop=5)
    print("==== Agent循环链式工具对话，输入quit退出 ====")
    while True:
        question = input("\n你：")
        if question.strip().lower() == "quit":
            print("Agent会话结束")
            break
        reply = agent.run(question)
        print(f"Agent：{reply}")
