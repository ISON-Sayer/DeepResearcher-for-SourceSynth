import os
import sys
import re
from datetime import datetime
from pathlib import Path

# 修复 Windows 终端中文编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# 配置
# ============================================================
load_dotenv()

BOCHA_API_KEY = os.getenv("BOCHA_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-chat")

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 工具定义
# ============================================================

@tool
def bocha_websearch_tool(query: str, count: int = 20) -> str:
    """
    使用 Bocha API 进行网页搜索，返回网页标题、链接、摘要和发布日期。
    适合获取广泛的信息来源，每次返回最多 20 条结果。

    参数:
        query: 搜索关键词
        count: 返回结果数量（默认 20，最大 50）
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
        "count": min(count, 50),
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
    except requests.RequestException as e:
        return f"搜索请求失败: {e}"

    if response.status_code != 200:
        return f"搜索 API 错误: {response.status_code} - {response.text[:500]}"

    data = response.json()
    webpages = data.get("data", {}).get("webPages", {}).get("value", [])

    if not webpages:
        return "未找到相关网页，建议尝试不同的关键词。"

    formatted = []
    for i, page in enumerate(webpages, 1):
        title = page.get("name", "无标题")
        url = page.get("url", "")
        snippet = page.get("snippet", "无摘要")
        date = page.get("datePublished", "")[:10] if page.get("datePublished") else ""
        site = page.get("siteName", "")
        meta = f"{site} | {date}" if site or date else ""
        formatted.append(
            f"[{i}] {title}\n"
            f"    URL: {url}\n"
            f"    {meta}\n"
            f"    摘要: {snippet}\n"
        )

    return "\n".join(formatted)


@tool
def bocha_aisearch_tool(query: str, freshness: str = "noLimit") -> str:
    """
    使用 Bocha AI 搜索 API，返回 AI 生成的直接答案和引用来源。
    特别适合天气、股票、百科、汇率等结构化信息查询。

    参数:
        query: 搜索关键词
        freshness: 时间范围，可选 "oneDay"/"oneWeek"/"oneMonth"/"noLimit"
    """
    url = "https://api.bocha.cn/v1/ai-search"
    headers = {
        "Authorization": f"Bearer {BOCHA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "freshness": freshness,
        "answer": True,
        "count": 10,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
    except requests.RequestException as e:
        return f"AI 搜索请求失败: {e}"

    if response.status_code != 200:
        return f"AI 搜索 API 错误: {response.status_code} - {response.text[:500]}"

    data = response.json()
    answer = data.get("data", {}).get("answer", "")
    webpages = data.get("data", {}).get("webPages", {}).get("value", [])

    result_parts = []
    if answer:
        result_parts.append(f"【AI 生成答案】\n{answer}\n")

    if webpages:
        result_parts.append("【参考来源】")
        for i, page in enumerate(webpages, 1):
            title = page.get("name", "无标题")
            url = page.get("url", "")
            snippet = page.get("snippet", "")
            result_parts.append(f"[{i}] {title}\n    URL: {url}\n    摘要: {snippet}\n")

    if not result_parts:
        return "AI 搜索未返回结果，建议使用 bocha_websearch_tool 进行网页搜索。"

    return "\n".join(result_parts)


@tool
def fetch_webpage_tool(url: str) -> str:
    """
    抓取指定网页的正文内容，用于深度阅读关键来源。

    参数:
        url: 要抓取的网页链接
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
    except requests.RequestException as e:
        return f"网页抓取失败: {e}"

    soup = BeautifulSoup(response.text, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "noscript", "iframe", "form", "button", "input"]):
        tag.decompose()

    title = ""
    if soup.title:
        title = soup.title.get_text(strip=True)

    content_selectors = [
        "article", "main", '[role="main"]',
        ".article-content", ".post-content", ".entry-content",
        ".content", "#content", ".article", ".post",
    ]
    body = None
    for selector in content_selectors:
        body = soup.select_one(selector)
        if body:
            break

    if not body:
        body = soup.body

    if not body:
        return f"【{title}】\n\n无法提取页面正文内容。"

    text = body.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)

    max_chars = 6000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n... [内容已截断，访问原文获取完整信息]"

    return f"【{title}】\n来源: {url}\n\n{text}"


# ============================================================
# 工具列表
# ============================================================
tools = [bocha_websearch_tool, bocha_aisearch_tool, fetch_webpage_tool]


# ============================================================
# System Prompt
# ============================================================
RESEARCH_SYSTEM_PROMPT = """\
你是一个专业的资料研究助手，擅长进行深度主题研究。对于用户提出的研究主题，请严格遵循以下流程：

## 研究流程

1. **分析主题**：理解用户的真实意图，将主题拆解为 2-3 个子问题或不同搜索角度。

2. **多角度搜索**：
   - 使用 bocha_websearch_tool 从不同角度搜索，至少执行 2-3 轮搜索，确保信息全面
   - 如果涉及天气、股票、百科、汇率等结构化信息，优先使用 bocha_aisearch_tool
   - 每轮搜索结果带有 [序号] 编号，请在后续引用时使用这些编号

3. **筛选去重**：合并多轮搜索结果后，去除内容重复的来源，优先选择权威网站（政府、机构、知名媒体）。

4. **深度阅读**：对 2-3 个最关键或信息量最大的来源使用 fetch_webpage_tool 获取详细内容，提取关键数据和观点来支撑报告。

5. **撰写研究报告**，严格按以下格式输出：

---

## 📊 研究报告：{主题}

### 🔑 核心发现
- 发现1 [用搜索结果的序号引用，如 1]
- 发现2 [2]
- 发现3 [3]

### 📋 详细分析

#### {小标题1}
详细展开第一方面的内容，每个事实性陈述后标注来源编号，如「武汉今日气温 30°C，空气质量良好 [1]」。确保引用的内容真实来源于搜索结果，不要编造。

#### {小标题2}
详细展开第二方面的内容，同样标注来源。

### 📚 参考来源
[1] 网页标题 - 网页链接
[2] 网页标题 - 网页链接
...

---

## 注意事项
- **必须引用来源**：每个关键事实和数字后面都要标注 [N] 引用编号
- **如实报告**：信息不足时如实说明，绝不要编造数据和事实
- **来源质量**：引用至少 3 个不同的可靠来源
- **语言风格**：专业、客观、简洁，使用中文撰写
- **分清工具**：需要结构化数据用 bocha_aisearch_tool，需要网页列表用 bocha_websearch_tool，需要深度阅读用 fetch_webpage_tool
"""


# ============================================================
# LLM 和 Agent 初始化
# ============================================================
llm = ChatOpenAI(
    model=DEFAULT_MODEL,
    openai_api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.7,
)

memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=RESEARCH_SYSTEM_PROMPT,
    checkpointer=memory,
)


# ============================================================
# 辅助函数
# ============================================================
def sanitize_filename(name: str) -> str:
    """清理字符串使其可作为文件名"""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace(" ", "_")
    return name[:40]


def save_report(topic: str, content: str) -> str:
    """保存研究报告到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = sanitize_filename(topic)
    filename = f"{safe_topic}_{timestamp}.md"
    filepath = REPORTS_DIR / filename
    filepath.write_text(content, encoding="utf-8")
    return str(filepath)


# ============================================================
# 主循环
# ============================================================
def main():
    config = {"configurable": {"thread_id": "research-session"}}

    print("=" * 60)
    print("   📚 资料研究助手")
    print("   支持多轮搜索 · 深度阅读 · 引用标注 · 追问")
    print("   输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    print()

    while True:
        try:
            user_query = input("🔍 请输入研究主题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if user_query.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        if not user_query:
            continue

        print(f"\n⏳ 正在研究「{user_query}」...\n")

        try:
            result = agent.invoke(
                {"messages": [("user", user_query)]},
                config=config,
            )
        except Exception as e:
            print(f"❌ 研究过程出错: {e}")
            continue

        final_message = result["messages"][-1]
        content = final_message.content if hasattr(final_message, "content") else str(final_message)

        print("\n" + "─" * 60)
        print(content)
        print("─" * 60 + "\n")

        filepath = save_report(user_query, content)
        print(f"📁 报告已保存: {filepath}\n")


if __name__ == "__main__":
    main()
