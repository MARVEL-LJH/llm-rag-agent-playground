import requests
from config.config import DEEPSEEK_API_KEY

DEEPSEEK_CHAT_URL = "https://api.deepseek.com/v1/chat/completions"

class DeepSeekLLMClient:
    def __init__(self):
        self.api_key = DEEPSEEK_API_KEY

    def chat(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        resp = requests.post(DEEPSEEK_CHAT_URL, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
