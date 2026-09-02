from src.rag.loader import load_directory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = "./chroma_db"

def load_vector_db():
    """加载已经保存好的向量库，给rag_qa调用"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_texts(
        texts=all_texts,
        metadatas=all_metas,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print(f"向量库构建完成，保存路径：{PERSIST_DIR}")

if __name__ == "__main__":
    main()
