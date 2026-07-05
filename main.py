import os
import sys

# 修复 Windows 终端中文编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
import requests
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


load_dotenv()

BOCHA_API_KEY = os.getenv("BOCHA_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-chat")


# 定义 Bocha Web Search 工具
@tool
def bocha_websearch_tool(query: str, count: int = 10) -> str:
    """
    使用 Bocha API 进行网页搜索。
    """
    url = "https://api.bocha.cn/v1/web-search"
    headers = {
        "Authorization": f"Bearer {BOCHA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "freshness": "noLimit",
        "summary": "true",
        "count": count,
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"

    data = response.json()

    webpages = data.get("data", {}).get("webPages", {}).get("value", [])
    if not webpages:
        return "未找到相关网页"

    formatted_results = []
    for webpage in webpages:
        title = webpage.get("name", "无标题")
        url = webpage.get("url", "无链接")
        snippet = webpage.get("snippet", "无摘要")
        formatted_results.append(f"标题: {title}\n链接: {url}\n摘要: {snippet}\n")
    return "\n".join(formatted_results)


tools = [bocha_websearch_tool]

# 使用 DeepSeek API（OpenAI 兼容接口）
llm = ChatOpenAI(
    model=DEFAULT_MODEL,
    openai_api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.7,
)

# 创建 Agent
agent = create_agent(model=llm, tools=tools)

# 运行
user_query = input("请输入您的查询: ")
result = agent.invoke({"messages": [("user", user_query)]})
print(result["messages"][-1].content)
