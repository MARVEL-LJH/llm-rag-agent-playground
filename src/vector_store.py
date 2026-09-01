import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from config.config import VECTOR_DB_PATH
from src.simple_embedding import SimpleEmbedding
import sys
from pathlib import Path
# vector_store.py 在 src/文件夹下，只需要两层parent
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))


def build_vector_db(txt_file_path: str):
    """加载txt文档，切分，构建FAISS向量库"""
    loader = TextLoader(txt_file_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"文档切分完成，一共 {len(split_docs)} 个片段")

    embeddings = SimpleEmbedding()
    db = FAISS.from_documents(documents=split_docs, embedding=embeddings)

    # FAISS保存本地
    db.save_local(VECTOR_DB_PATH)
    print(f"向量库构建完成，保存路径：{VECTOR_DB_PATH}")
    return db


def load_vector_db():
    """加载本地FAISS向量库"""
    embeddings = SimpleEmbedding()
    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db


if __name__ == "__main__":
    txt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plaintext.txt")
    build_vector_db(txt_path)