import datetime
from src.rag_qa import rag_query

# 工具1：获取当前时间
def get_current_time() -> str:
    """获取系统当前时间"""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

# 工具2：计算器
def calculator(a: float, b: float, op: str) -> str:
    """
    计算两个数字运算
    :param a: 数字1
    :param b: 数字2
    :param op: 运算符，支持 + - * /
    """
    if op == "+":
        res = a + b
    elif op == "-":
        res = a - b
    elif op == "*":
        res = a * b
    elif op == "/":
        if b == 0:
            return "错误：除数不能为0"
        res = a / b
    else:
        return f"不支持运算符{op}"
    return f"计算结果：{a} {op} {b} = {res}"

# ==========新增工具3：RAG知识库检索==========
def rag_search(query: str):
    """
    查询本地项目知识库，获取项目相关文档内容
    :param query: 用户的问题
    """
    result = rag_query(query, top_k=3)
    # 把检索出来的上下文片段拼接返回给Agent
    context = "\n".join(result["retrieve_context"])
    return f"知识库检索结果：\n{context}"


TOOL_LIST = [
    {
        "name": "get_current_time",
        "desc": "调用获取系统当前时间，不需要参数"
    },
    {
        "name": "calculator",
        "desc": "做数学计算，参数：a数字，b数字，op运算符，op只能是+、-、*、/"
    },
    {
        "name": "rag_search",
        "desc": "查询本地项目知识库，询问git、项目目录结构等项目相关问题时调用，参数query：用户问题字符串"
    }
]

TOOL_FUNC_MAP = {
    "get_current_time": get_current_time,
    "calculator": calculator,
    "rag_search": rag_search
}
