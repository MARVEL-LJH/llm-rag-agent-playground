from src.vector_store import load_vector_db
from src.llm.llm_client import DeepSeekLLMClient


def rag_query(question: str, top_k: int = 3):
    # 1.加载向量库
    db = load_vector_db()
    # 2.检索相关文档片段
    docs = db.similarity_search(question, k=top_k)
    context_list = [doc.page_content for doc in docs]
    context_text = "\n---\n".join(context_list)

    # 3.构造RAG的用户prompt
    # 构造RAG提示词，把检索到的上下文塞进去
    rag_prompt = f"""请基于下面参考资料回答用户问题，不要编造资料以外的内容。
    参考资料：
{context_text}

【用户问题】
{question}
"""
    # 4.调用DeepSeek
    llm = DeepSeekLLMClient()
    answer = llm.chat(rag_prompt)

    return {
        "question": question,
        "retrieve_context": context_list,
        "answer": answer
    }


if __name__ == "__main__":
    result = rag_query("什么是Redis？", top_k=4)

    print("====检索到的知识库片段====")
    for chunk in result["retrieve_context"]:
        print(chunk)
        print("-" * 60)

    print("\n====RAG回答结果====")
    print(result["answer"])
77