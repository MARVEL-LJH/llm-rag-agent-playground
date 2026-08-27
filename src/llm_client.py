from openai import OpenAI
from config.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME


class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
        )
        self.model_name = LLM_MODEL_NAME

    def chat(self, prompt: str, system_prompt: str = "你是一个有用的助手", temperature=0.3):
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return resp.choices[0].message.content


if __name__ == "__main__":
    # 本地测试
    llm = LLMClient()
    res = llm.chat("简单介绍RAG技术")
    print(res)