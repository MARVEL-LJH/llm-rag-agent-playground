from huggingface_hub import snapshot_download

# 将 all‑MiniLM‑L6‑v2完整下载到项目 ./hf_models 文件夹
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir="./hf_models/all-MiniLM-L6-v2",
    local_dir_use_symlinks=False,
    resume_download=True
)
