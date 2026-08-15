"""Evaluates SimpleRAG.query_router() classification accuracy against real DeepSeek calls.

Background: query_router() classifies each question into list/detail/general,
which ask() uses to pick a generation mode. It has never been checked against
real LLM output -- server/tests/ only mocks DeepSeek, which proves the
routing *code* (dispatch, fallback-to-general-on-failure) works, not that the
*classification* itself is accurate. See DEVLOG.md, "未来升级空间" #1.

This script runs a hand-labeled question set (router_questions.json) through
the real query_router() and reports:
  - overall accuracy
  - a confusion matrix (expected route -> predicted route)
  - the list of misclassified questions

Usage:
    cd server
    python eval/eval_router.py

Requires a real DEEPSEEK_API_KEY in server/.env (server/tests/conftest.py's
placeholder key would make every call fail and silently fall back to
"general" for all 30 questions, which is not a meaningful signal).

Implementation note: query_router() only depends on self.client/self.model
(via SimpleRAG._complete) -- it never touches self.documents/embedder/
vector_index/bm25. So instead of calling SimpleRAG(data_path), which would
also load the sentence-transformers embedding model and build/load the
FAISS + BM25 indexes (slow, and irrelevant to what's being measured here),
this script builds a SimpleRAG instance via __new__ and sets only
.client/.model directly. If query_router() ever starts reading another
instance attribute, this shortcut needs to grow the same attribute.

# 中文说明:
# 这是一个独立的评测脚本,专门评估 rag.py 里 query_router()(问题路由分类器)
# 在真实 DeepSeek 调用下的准确率。server/tests/ 里的单元测试全部 mock 掉了
# DeepSeek,只能证明"路由这段代码逻辑没写错"(比如调用失败会兜底成
# general),没法证明"分类结果本身准不准"——这个脚本就是补这一块的评测。
#
# 用法: 在 server 目录下跑 `python eval/eval_router.py`,需要 .env 里配置
# 真实的 DEEPSEEK_API_KEY(不能用测试用的占位 key,否则每次调用都失败,
# 全部兜底成 general,测出来的"准确率"没有意义)。
#
# 实现上的小技巧: query_router() 只依赖 self.client/self.model,完全不需要
# 加载菜谱、embedding 模型、FAISS/BM25 索引这些重量级的初始化步骤。所以这
# 里没有走正常的 SimpleRAG(data_path) 构造流程,而是用 __new__ 跳过 __init__,
# 手动只设置 client/model 这两个属性,评测跑起来快很多。
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(SERVER_DIR / ".env")

from openai import OpenAI  # noqa: E402

from rag import DEEPSEEK_BASE_URL, DEFAULT_MODEL, SimpleRAG, _VALID_ROUTES  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent / "router_questions.json"
RESULTS_PATH = Path(__file__).resolve().parent / "router_eval_results.json"


def build_router_only_rag() -> SimpleRAG:
    """Builds a SimpleRAG instance with just enough state to run query_router().

    Skips __init__ (and therefore the recipe loading / embedding / FAISS /
    BM25 setup __init__ normally does) since none of that is needed to
    exercise query_router() -- see module docstring.
    """
    # 中文: 用 __new__ 绕过 __init__,只手动装配 query_router() 真正需要的两个
    # 属性(client、model),省掉加载知识库和建索引的开销。
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "test-key-for-unit-tests":
        raise SystemExit(
            "DEEPSEEK_API_KEY is missing or is the tests/ placeholder key in "
            "server/.env -- set a real key before running this eval, "
            "otherwise every call fails and silently falls back to 'general', "
            "which isn't a real accuracy signal."
        )
    instance = SimpleRAG.__new__(SimpleRAG)
    instance.model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
    instance.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    return instance


def main() -> None:
    # 中文: 读入手工标注的题库(router_questions.json,每条是 question + 期望
    # 的 expected 路由),逐条跑真实的 query_router(),统计准确率、
    # 混淆矩阵(期望路由 x 实际预测路由),并把分类错的题目单独列出来,
    # 最后把完整结果写成 router_eval_results.json 存档。
    with open(DATASET_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    rag = build_router_only_rag()

    results = []
    correct = 0
    confusion: Counter[tuple[str, str]] = Counter()  # (expected, predicted) -> count

    for case in cases:
        question = case["question"]
        expected = case["expected"]
        predicted = rag.query_router(question)
        is_correct = predicted == expected
        correct += is_correct
        confusion[(expected, predicted)] += 1
        results.append(
            {
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
            }
        )
        mark = "OK" if is_correct else "X "
        print(f"[{mark}] expected={expected:8} predicted={predicted:8} | {question}")

    total = len(cases)
    accuracy = correct / total if total else 0.0

    print("\n" + "=" * 70)
    print(f"Accuracy: {correct}/{total} = {accuracy:.1%}")

    # 中文: 打印一个 expected(行) x predicted(列)的混淆矩阵,方便一眼看出
    # 是哪两类之间容易被搞混(比如 detail 常被误判成 general)。
    print("\nConfusion matrix (rows=expected, cols=predicted):")
    routes = sorted(_VALID_ROUTES)
    header = " " * 10 + "".join(f"{r:>10}" for r in routes)
    print(header)
    for expected in routes:
        row = "".join(f"{confusion.get((expected, r), 0):>10}" for r in routes)
        print(f"{expected:10}{row}")

    misclassified = [r for r in results if not r["correct"]]
    if misclassified:
        print(f"\nMisclassified ({len(misclassified)}):")
        for r in misclassified:
            print(f"  expected={r['expected']:8} got={r['predicted']:8} | {r['question']}")
    else:
        print("\nNo misclassifications.")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": rag.model,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "confusion_matrix": {f"{e}->{p}": c for (e, p), c in confusion.items()},
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
