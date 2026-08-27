from src.vector_store import load_vector_db
from src.llm_client import LLMClient


def rag_query(question: str, top_k: int = 3):
    # 1.加载FAISS向量库
    db = load_vector_db()
    # 2.检索相关文档片段
    docs = db.similarity_search(question, k=top_k)
    context_list = [doc.page_content for doc in docs]
    context_text = "\n---\n".join(context_list)

    # 3.构造RAG的用户prompt
    user_prompt = f"""请仔细阅读下面【参考上下文】，依据上下文内容回答用户问题。
答案必须来自上下文，如果上下文完全没有相关内容，才回答不知道。

【参考上下文】
{context_text}

【用户问题】
{question}
"""
    # 4.实例化客户端，调用chat方法
    llm_client = LLMClient()
    answer = llm_client.chat(prompt=user_prompt)

    return {
        "question": question,
        "retrieve_context": context_list,
        "answer": answer
    }


if __name__ == "__main__":
    # 修改为你的真实问题
    result = rag_query("如何把本地代码推送到GitHub？", top_k=4)

    print("====检索到的知识库片段====")
    for chunk in result["retrieve_context"]:
        print(chunk)
        print("-" * 60)

    print("\n====RAG回答结果====")
    print(result["answer"])