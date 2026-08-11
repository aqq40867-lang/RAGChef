"""Interactive terminal Q&A for RAGChef.

Lets you exercise the RAG pipeline (rag.SimpleRAG) directly from the
terminal, without starting the FastAPI server or loading the Chrome
extension -- handy for quick manual testing.

Usage (from server/, with DEEPSEEK_API_KEY set in .env or the environment):

    python cli.py
"""

import logging
import os
import sys

from rag import RAGConfigError, SimpleRAG

# SimpleRAG logs its own INFO-level progress (index cache hit/miss, etc.);
# keep everything else quiet so it doesn't clutter the interactive prompt.
logging.basicConfig(level=logging.WARNING)

RECIPES_PATH = os.path.join(os.path.dirname(__file__), "data", "recipes")

EXIT_COMMANDS = {"exit", "quit", "q", ":q"}

BANNER = """
============================================================
🍽️  RAGChef - 菜谱问答系统 - 交互式问答  🍽️
============================================================
💡 用英文提问，例如 "recommend a few easy vegetarian dishes"
   或 "how do I make dumplings?"（知识库和回答均为英文）
   输入 exit / quit 退出
"""


def main() -> None:
    print(BANNER)

    print("正在加载知识库与向量索引...")
    try:
        rag = SimpleRAG(RECIPES_PATH)
    except RAGConfigError as e:
        print(f"❌ 初始化失败：{e}")
        sys.exit(1)

    print(f"✅ 已加载 {len(rag.documents)} 道菜谱，向量索引就绪！")
    print("✅ 系统初始化完成！\n")

    while True:
        try:
            question = input("您的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("再见！")
            break

        print()
        answer = rag.ask(question)
        print(answer)
        print()


if __name__ == "__main__":
    main()
