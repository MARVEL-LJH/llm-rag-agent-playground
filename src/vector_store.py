# 【必须放在所有import最上方】设置HF全局离线模式
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.rag.loader import load_directory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "./chroma_db"
# ========== 关键：本地模型路径，完全离线加载 ==========
LOCAL_EMBED_MODEL_PATH = "./hf_models/all-MiniLM-L6-v2"

# 全局唯一embedding实例，全部函数统一复用这一个
embeddings = HuggingFaceEmbeddings(
    model_name=LOCAL_EMBED_MODEL_PATH,   # 指向本地文件夹！不是在线名字
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    # ❗删除 local_files_only=True！langchain_huggingface不支持该顶层参数
)


def load_vector_db():
    """加载已经保存好的向量库，给rag_qa调用"""
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    return db


def main():
    raw_docs = load_directory("./data")
    if not raw_docs:
        print("警告：没有加载到任何文档！")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_texts = []
    all_metas = []
    for item in raw_docs:
        chunks = splitter.split_text(item["content"])
        for chunk in chunks:
            all_texts.append(chunk)
            all_metas.append({"source": item["source"]})

    print(f"文档切分完成，一共 {len(all_texts)} 个片段")

    db = Chroma.from_texts(
        texts=all_texts,
        metadatas=all_metas,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print(f"向量库构建完成，保存路径：{PERSIST_DIR}")


if __name__ == "__main__":
    main()
