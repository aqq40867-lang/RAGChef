"""Evaluates SimpleRAG retrieval quality (Recall@k / Hit@k / MRR) on a hand-labeled set.

Background: DEVLOG.md "未来升级空间" #5 / #10 -- retrieval had only been checked
by unit tests ("does a known question find a known recipe"), never measured.
RRF k=60 was copied from the reference implementation and the hybrid
(vector + BM25) setup was never compared against either retriever alone.

This script runs retrieval_questions.json (hand-labeled: every question lists
ALL recipes that should be retrieved) through SimpleRAG._rank_documents() and
reports, for several configurations:
  - Recall@k : relevant recipes found in top k / all relevant recipes
  - Hit@k    : 1 if at least one relevant recipe is in top k
  - MRR      : 1 / rank of the first relevant recipe (0 if not in top 10)
plus a per-question-type breakdown and the failures of the online config.

Only retrieval runs -- no DeepSeek call is made, so it costs nothing.

Usage:
    cd server
    python eval/eval_retrieval.py            # summary tables
    python eval/eval_retrieval.py --verbose  # also print every miss

Results are also written to eval/retrieval_eval_results.json.

# 中文说明:
# 检索召回率评估脚本。标注集 retrieval_questions.json 里每道题都列出了
# "应该被检索到的全部菜谱"(文件名去掉 .md),脚本调用 _rank_documents()
# 拿到 top-k 结果后计算 Recall@k / Hit@k / MRR。
# 对比的配置:
#   1. 线上配置: hybrid(向量+BM25, RRF k=60)+ 关键词自动过滤
#   2. 消融: 纯向量 / 纯 BM25 / hybrid 关闭自动过滤
#   3. RRF k 参数扫描
# 只跑检索、不调用 LLM, 不花 API 费用。SimpleRAG.__init__ 会校验
# DEEPSEEK_API_KEY, 没配置时这里塞一个占位值 —— 检索全程用不到它。
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))
os.environ.setdefault("DEEPSEEK_API_KEY", "eval-placeholder-not-used")

from rag import SimpleRAG  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVAL_DIR / "retrieval_questions.json"
RESULTS_PATH = EVAL_DIR / "retrieval_eval_results.json"
KS = (1, 2, 5, 10)
MAX_K = max(KS)

# (label, kwargs for _rank_documents)
ONLINE = ("hybrid + filters (online, RRF k=60)", {"retriever": "hybrid"})
ABLATIONS = [
    ONLINE,
    ("hybrid, no filters", {"retriever": "hybrid", "use_filters": False}),
    ("vector only + filters", {"retriever": "vector"}),
    ("vector only, no filters", {"retriever": "vector", "use_filters": False}),
    ("bm25 only + filters", {"retriever": "bm25"}),
    ("bm25 only, no filters", {"retriever": "bm25", "use_filters": False}),
]
RRF_SWEEP = (1, 5, 10, 30, 60, 100, 200)


def slug(dish_name: str) -> str:
    """Recipe.dish_name is the file stem with '-' -> ' '; labels use the file stem."""
    return dish_name.replace(" ", "-")


def evaluate(rag: SimpleRAG, questions: list[dict], **kwargs) -> dict:
    per_question = []
    for item in questions:
        relevant = set(item["relevant"])
        # 每个 k 单独调用一次 _rank_documents(top_k=k):候选池大小
        # max(top_k*5, 10) 随 top_k 变化,所以 top_k=10 结果的前 2 名
        # 不一定等于线上 top_k=2 的结果。这样测出来的 @k 和线上行为一致。
        got_at = {
            k: [slug(d.dish_name) for d in rag._rank_documents(item["q"], k, **kwargs)]
            for k in KS
        }
        got = got_at[MAX_K]
        first = next((i + 1 for i, g in enumerate(got) if g in relevant), None)
        row = {
            "id": item["id"],
            "type": item["type"],
            "q": item["q"],
            "relevant": item["relevant"],
            "retrieved@2": got_at[2],
            "retrieved": got,
            "first_rank": first,
        }
        for k in KS:
            hits = len(relevant & set(got_at[k]))
            row[f"recall@{k}"] = hits / len(relevant)
            row[f"hit@{k}"] = 1.0 if hits else 0.0
        row["mrr"] = 1.0 / first if first else 0.0
        per_question.append(row)
    return {"summary": aggregate(per_question), "per_question": per_question}


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n": n}
    for k in KS:
        out[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / n
        out[f"hit@{k}"] = sum(r[f"hit@{k}"] for r in rows) / n
    out["mrr"] = sum(r["mrr"] for r in rows) / n
    return out


def fmt_row(label: str, s: dict, width: int = 38) -> str:
    cells = [f"{s[f'recall@{k}']:6.1%}" for k in KS]
    cells += [f"{s['hit@2']:6.1%}", f"{s['hit@5']:6.1%}", f"{s['mrr']:.3f}"]
    return f"{label:<{width}}" + "  ".join(cells)


def header(first: str, width: int = 38) -> str:
    cols = [f"R@{k}".rjust(6) for k in KS] + ["Hit@2", " Hit@5", "  MRR"]
    return f"{first:<{width}}" + "  ".join(c.rjust(6) for c in cols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="print every miss")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    rag = SimpleRAG(str(SERVER_DIR / "data" / "recipes"))

    known = {slug(d.dish_name) for d in rag.documents}
    unknown = {r for q in questions for r in q["relevant"]} - known
    if unknown:
        sys.exit(f"Labels reference recipes that don't exist: {sorted(unknown)}")

    results = {"questions": len(questions), "ablations": {}, "rrf_sweep": {}}

    print(f"\n{len(questions)} questions, {len(rag.documents)} recipes, "
          f"{len(rag.child_chunks)} chunks\n")
    print("== Retriever / filter ablation ==")
    print(header("config"))
    for label, kwargs in ABLATIONS:
        res = evaluate(rag, questions, **kwargs)
        results["ablations"][label] = res
        print(fmt_row(label, res["summary"]))

    print("\n== RRF k sweep (hybrid + filters) ==")
    print(header("rrf_k"))
    for k in RRF_SWEEP:
        res = evaluate(rag, questions, retriever="hybrid", rrf_k=k)
        results["rrf_sweep"][str(k)] = res["summary"]
        print(fmt_row(f"k={k}", res["summary"]))

    online = results["ablations"][ONLINE[0]]["per_question"]
    print(f"\n== Online config by question type ==")
    print(header("type"))
    by_type = defaultdict(list)
    for row in online:
        by_type[row["type"]].append(row)
    results["online_by_type"] = {}
    for t, rows in by_type.items():
        s = aggregate(rows)
        results["online_by_type"][t] = s
        print(fmt_row(f"{t} (n={len(rows)})", s))

    misses = [r for r in online if r["hit@2"] == 0]
    print(f"\n== Online config: {len(misses)} questions with no relevant recipe in top 2 ==")
    for r in misses if args.verbose else misses[:15]:
        print(f"[{r['id']}] {r['q']}")
        print(f"     want: {r['relevant']}")
        print(f"     got@2 : {r['retrieved@2']}   (top10 first hit: {r['first_rank']})")

    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull results written to {RESULTS_PATH.relative_to(SERVER_DIR)}")


if __name__ == "__main__":
    main()
