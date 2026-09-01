import os
from pypdf import PdfReader


def load_txt(file_path: str) -> str:
    """读取txt文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    """读取pdf文件，逐页提取文本"""
    reader = PdfReader(file_path)
    text_list = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_list.append(page_text)
    return "\n".join(text_list)


def load_single_file(file_path: str) -> str:
    """根据文件后缀自动选择加载方式"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return load_txt(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    else:
        print(f"⚠️ 不支持的文件格式，跳过：{file_path}")
        return ""


def load_directory(dir_path: str) -> list:
    """
    加载整个目录下所有txt和pdf文件
    返回：[{"source": 文件名, "content": 文本内容}, ...]
    """
    documents = []
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if not os.path.isfile(file_path):
            continue
        content = load_single_file(file_path)
        if content.strip():
            documents.append({
                "source": filename,
                "content": content
            })
            print(f"✅ 已加载：{filename}")
    return documents


if __name__ == "__main__":
    # 测试：加载data目录
    docs = load_directory("./data")
    print(f"\n共加载 {len(docs)} 个文档")
    for d in docs:
        print(f" - {d['source']}，长度：{len(d['content'])} 字符")
