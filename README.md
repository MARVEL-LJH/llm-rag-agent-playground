# llm-rag-agent-playground

大模型应用实战：RAG检索增强生成、Agent智能体、LLM调用实践项目

基础RAG检索增强生成工程模板，实现文档向量化、检索、大模型问答完整链路。

## 项目特性
- 完整RAG链路：文档加载、文本切分、FAISS向量库、检索增强问答
- 自定义简易Embedding，无需torch重型依赖，快速验证流程
- 兼容OpenAI接口的大模型客户端封装
- `.gitignore`已配置：忽略虚拟环境、向量库缓存、IDE配置、密钥文件

## 环境部署
### 1. 克隆仓库
```bash
git clone https://github.com/你的GitHub用户名/llm-rag-agent-playground.git
cd llm-rag-agent-playground
```

### 2. 创建虚拟环境

```
# windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 配置大模型密钥

> 
> ⚠️ **不要将真实密钥提交到 GitHub 仓库**

在`config`目录下新建 `config.py`，复制下面模板，填入你的大模型信息：

```
# config/config.py
LLM_API_KEY = "你的api‑key"
LLM_BASE_URL = "大模型接口地址，例如 https://api.deepseek.com/v1"
LLM_MODEL_NAME = "模型名称，例如 deepseek‑chat"
```

### 4. 运行项目

1. **构建向量库**

```
python src\vector_store.py
```

会读取本地文档，生成 FAISS 向量索引，输出到`chroma_db`目录。

2. **启动 RAG 问答**

```
python src\rag_qa.py
```

修改代码内的问题即可进行知识库问答。

## 项目目录说明

```
llm‑rag‑agent‑playground/
├── config/                 # 配置目录，自行新建config.py填入密钥
├── examples/                # 示例脚本
├── src/
│   ├── llm_client.py        # OpenAI兼容大模型封装
│   ├── simple_embedding.py  # 自定义Embedding
│   ├── vector_store.py      # 文档切分、向量库构建加载
│   └── rag_qa.py            # RAG问答主逻辑
├── data/                    # 存放本地知识库txt/md文档
├── chroma_db/               # 向量数据库缓存，git已忽略
├── requirements.txt         # python依赖清单
└── .gitignore               # git忽略规则
```

## 当前局限

- 内置简易哈希 Embedding 仅用于流程验证，语义检索能力弱；
- 如需更好检索效果，可以接入`sentence‑transformers`真实语义向量。

## 后续扩展方向

- 支持 PDF 等更多文档格式加载
- 替换高性能 Embedding 模型
- 增加多轮对话记忆
- 开发 Agent 智能体工具调用

```

### 提交前核对两件事
1. 本地`config/config.py`（带真实密钥）**不要出现在git变更列表里**，.gitignore已经配置忽略它；
2. `requirements.txt`里面，如果暂时不用语义向量，把`sentence‑transformers`那一行删掉，避免别人安装大量重型依赖。

写完保存README，确认已经勾选到git变更列表，跟着刚才的commit信息提交，再Push origin推送到远程。

推送完打开GitHub网页就可以看到完整仓库文档。
```