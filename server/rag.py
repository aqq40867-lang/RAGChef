"""RAGChef 的核心 RAG(检索增强生成)逻辑 —— 轻量版,不依赖 LangChain。

整体流程:
1. 加载: 递归读取知识库目录下的每个 Markdown 文件,生成一个 Recipe(父文档),
   并补全分类/菜名/难度等元数据。
2. 分块: 把每篇 Recipe 按 "#"/"##"/"###" 标题切成父块(整篇菜谱)+
   若干子块(标题/食材/步骤等)。
3. 建索引: 用 sentence-transformers 直接加载 BGE 句向量模型,把子块编码成向量
   后交给 faiss-cpu 的 IndexFlatIP 建索引(缓存到磁盘,避免每次启动都重新
   embedding);同时用 rank_bm25 的 BM25Okapi 建一份关键词索引,复用同一套
   分词器。
4. 检索: 对每个问题,先按分类/难度(显式传入或从问题文本推断)缩小候选范围,
   再分别用向量检索(FAISS)和关键词检索(BM25)产出子块排名,
   通过 Reciprocal Rank Fusion(RRF)融合成统一分数;然后按"子块所属的
   父文档"聚合分数,把父文档(菜谱)排序后返回。
5. 路由: 先用免费的关键词判断(_infer_route)把问题分成 list/detail/general
   三类,判不出来再退回一次 LLM 调用(_classify_and_rewrite)同时完成分类和
   问题改写;关键词判成 detail 的问题仍需单独调用 query_rewrite 做改写,
   list 问题则原样使用。所有 prompt 都是纯字符串模板,用 str.format() 填充。
6. 生成: 按路由类型分发到对应生成模式——纯格式化的菜名列表、结构化的分步骤回答,
   或普通的问答——都要求 DeepSeek(通过 openai 官方 SDK 的 OpenAI 客户端,
   base_url 指向 DeepSeek 的 OpenAI 兼容接口)严格依据检索到的内容作答
   ("grounding"),避免模型编造信息。

# 中文说明: 为什么不用 LangChain?
# 这个项目规模小、流程固定,LangChain 在这里只是给 embedding 模型、向量库、
# BM25、LLM 客户端各包了一层薄封装,没有用到它的 Chain/Agent 之类的重逻辑。
# 直接调用 sentence-transformers / faiss-cpu / rank_bm25 / openai 这些底层库,
# 少一层抽象、少几十个用不上的间接依赖(比如 langgraph、langsmith 遥测、
# SQLAlchemy),也不用跟着 LangChain 的版本节奏走。
# 父子分块、以及"按父文档聚合子块 RRF 分数"这一步是本项目特有的策略
# (_rrf_fuse / _rank_documents):需要每个子块的分数去做"父文档累计打分 +
# 按命中小节数排序 + 去重",所以继续手写,不用某个检索框架的现成组件。
# 路由关键词预判、难度/分类的关键词推断:同样是本项目的业务逻辑,继续手写。
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
)
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

logger = logging.getLogger("ragchef")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

# Embedding Model 为什么选这个？
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Reciprocal Rank Fusion
# 兼顾语义和关键词检索;常数与 LangChain EnsembleRetriever 的默认值(c=60)一致。
RRF_K = 60

# 回答策略
# query_router() 把问题分成以下三类,ask() 据此选择生成模式。
ROUTE_LIST = "list" # 推荐
ROUTE_DETAIL = "detail" # 详细做法
ROUTE_GENERAL = "general" # 一般问答
_VALID_ROUTES = {ROUTE_LIST, ROUTE_DETAIL, ROUTE_GENERAL}

# "list" 类问题5个候选菜可选,
LIST_MODE_TOP_K = 5

# 检索不到相关内容时返回给用户的提示文案。
NO_RESULTS_MESSAGE = "No relevant information was found in the current recipe library."

# LLM 调用失败(鉴权/网络/超时/API 错误)时返回给用户的提示文案。
LLM_UNAVAILABLE_MESSAGE = (
    "Sorry, the recipe assistant is temporarily unavailable. Please try again in a moment."
)

# 把菜谱文件所在目录(server/data/recipes/<category>/)映射成人类可读的分类名,
# 与知识库的目录结构一一对应,如 server/data/recipes/meat_dish/kung-pao-chicken.md。
CATEGORY_LABELS = {
    "meat_dish": "Meat",
    "vegetable_dish": "Vegetable",
    "soup": "Soup",
    "dessert": "Dessert",
    "staple": "Staple",
    "aquatic": "Seafood",
    "breakfast": "Breakfast",
    "other_dish": "Other",
}

# 用于从问题文本推断分类/难度过滤条件的关键词触发词(见 _infer_filters)。
# 刻意保守,只用没有歧义的短语,避免问题里恰好带某个菜名、又碰巧命中关键词,
# 结果被误伤过滤掉。
_CATEGORY_KEYWORDS = {
    "Vegetable": ("vegetarian", "vegan", "meatless"),
    "Dessert": ("dessert", "sweet treat"),
    "Soup": ("soup",),
    "Breakfast": ("breakfast",),
}
_DIFFICULTY_KEYWORDS = {
    "Easy": ("easy", "simple", "beginner", "quick"),
    "Hard": ("difficult", "advanced", "complex"),
}

# _infer_route() 用来免调用 LLM 就判断问题是 list 还是 detail 的关键词触发词。
# 同样刻意保守(与 _CATEGORY_KEYWORDS/_DIFFICULTY_KEYWORDS 同一思路):
# 只用无歧义的短语,判不出来的问题会落到 LLM 分类(见 query_router /
# _classify_and_rewrite),而不是冒险分错。
_LIST_ROUTE_TRIGGERS = (
    "recommend",
    "suggest",
    "a few",
    "some dishes",
    "some recipes",
    "what dishes",
    "what recipes",
    "what soups",
    "what desserts",
    "give me a list",
    "give me some options",
    "what can i cook",
    "what should i cook",
)
_DETAIL_ROUTE_TRIGGERS = (
    "how do i make",
    "how do you make",
    "how to make",
    "how do i cook",
    "how to cook",
    "how do i prepare",
    "recipe for",
    "ingredients for",
    "ingredients do i need for",
    "steps for",
)

# 用于切分的所有标题级别,从父文档边界(#)到支持的最细子块粒度(###)。
_HEADING_RE = re.compile(r"(?m)^#{1,3}[ \t]+.+$")

# 步骤用数字编号("1."、"2." ...);源数据没有显式的难度评级,
# 所以用步骤数粗略估算难度。
_STEP_RE = re.compile(r"(?m)^\d+\.\s")

# BM25 分词器:转小写后提取字母数字单词,不做词干化/去停用词——
# 知识库规模小且用词集中,这样已经够用。传给 BM25Okapi 前先用它对语料和查询
# 分好词(BM25Okapi 本身不做分词,只接受 token 列表)。
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# LLM 回复有时会把 JSON 包一层 ```json ... ``` 代码块,这里剥掉围栏再解析;
# 剥不出代码块就当作回复本身是裸 JSON。
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_json_reply(text: str) -> dict:
    """从 LLM 回复里解析出 JSON 对象,兼容裸 JSON 和 ```json ... ``` 代码块两种形式。"""
    match = _JSON_FENCE_RE.search(text)
    candidate = match.group(1) if match else text.strip()
    return json.loads(candidate)


# ---------------------------------------------------------------------------
# Prompt 模板:所有跟 LLM 交互的 prompt 都用模板构建,而不是直接拼 f-string——
# 好处是模板本身可以被单独测试/复用,变量名也更清晰。用纯字符串 + str.format()
# 就够了,不需要专门的模板类。
# ---------------------------------------------------------------------------

ROUTER_PROMPT = """Classify the user's question into exactly one category. Reply with only the category name, nothing else.

Categories:
- list: the user wants a list of dish suggestions or names (e.g. "recommend a few vegetarian dishes", "what soups do you have?")
- detail: the user wants to know how to make a specific dish (e.g. "how do I make Kung Pao Chicken?", "what ingredients do I need for dumplings?")
- general: anything else (e.g. "what's the difference between a casserole and a hotpot?", "what does al dente mean?")

User question: {question}

Category:"""

REWRITE_PROMPT = """Decide whether this recipe-assistant question needs rewriting to be more specific before searching a recipe database.

Rules:
- If the question is already specific (names a dish, an ingredient, or a clear request), return it unchanged.
- If the question is vague (e.g. "give me something to cook", "suggest a meal"), rewrite it to be more specific and search-friendly, keeping the original intent and preferring simple, easy-to-make dishes when nothing else is specified.
- Reply with only the final question text, nothing else.

Question: {question}

Rewritten question:"""

CLASSIFY_AND_REWRITE_PROMPT = """You are helping a recipe assistant prepare to answer a question. Do two things:

1. Classify the question into exactly one category:
   - list: the user wants a list of dish suggestions or names (e.g. "recommend a few vegetarian dishes", "what soups do you have?")
   - detail: the user wants to know how to make a specific dish (e.g. "how do I make Kung Pao Chicken?")
   - general: anything else (e.g. "what's the difference between a casserole and a hotpot?")
2. Produce a version of the question rewritten to be more specific and search-friendly: if the question is already specific (names a dish, an ingredient, or a clear request), repeat it unchanged; if it's vague (e.g. "give me something to cook"), rewrite it to be more specific, keeping the original intent and preferring simple, easy-to-make dishes when nothing else is specified.

Reply with ONLY a JSON object, no other text, in exactly this shape:
{{"route": "list", "rewritten": "..."}}

User question: {question}"""

STEP_BY_STEP_PROMPT = """You are a professional recipe assistant. Answer the question using ONLY the recipe content below.
If the answer is not contained in the content, reply exactly: "{no_results_message}"

Structure your answer in Markdown with exactly these sections:
## Overview
A one-to-two sentence introduction to the dish.
## Ingredients
A bullet list of ingredients.
## Steps
A numbered list of steps.
## Tips
One or two practical tips, if the content supports any; omit this section otherwise.

Recipe content:
{context}

User question:
{question}
"""

BASIC_ANSWER_PROMPT = """You are a professional recipe assistant.

Answer the question using ONLY the recipe content provided below.
If the answer is not contained in the content, reply exactly: "{no_results_message}"

Recipe content:
{context}

User question:
{question}

Respond naturally and concisely in English.
"""


class RAGConfigError(RuntimeError):
    """SimpleRAG 初始化失败时抛出的异常。

    覆盖配置类问题,比如缺少 API key 或知识库为空。app.py 让这个异常在启动阶段
    直接往外抛,这样配置错误会导致启动失败,而不是带病运行、每个请求都返回 500。
    """


@dataclass
class Recipe:
    """父文档:一整篇菜谱及其元数据。

    Attributes:
        text: 菜谱的完整 Markdown 内容,包含 "# 标题" 和所有 "## "/"### " 小节。
        dish_name: 菜名,取自文件名。
        category: 菜谱分类,取自所在的父目录名(如 "meat_dish" -> "Meat");
            如果文件不在已知分类目录下,回退为 "Other"。
        difficulty: 难度估算("Easy"/"Medium"/"Hard"),根据 "## Steps"
            小节里的步骤数推算。
    """

    text: str
    dish_name: str = ""
    category: str = "Other"
    difficulty: str = "Unknown"


class SimpleRAG:
    """加载一次菜谱知识库,之后持续提供检索 + 生成服务"""

    def __init__(self, data_path: str):
        """初始化 RAG 流程:校验配置,加载并索引菜谱。

        流程:校验 DEEPSEEK_API_KEY -> 加载菜谱 -> 按标题切成父/子两级 ->
        加载 sentence-transformers 的 BGE 向量模型 -> 构建/加载 faiss-cpu
        向量索引(有磁盘缓存)-> 构建 BM25Okapi 关键词索引(不缓存,每次启动都
        重建)-> 创建 openai 的 OpenAI 客户端(base_url 指向 DeepSeek)。这是
        全类里最重的一次性初始化。

        Args:
            data_path: 存放菜谱 Markdown 文件的目录(每篇菜谱一个文件,
                递归查找;见 load_documents())。

        Raises:
            RAGConfigError: 未设置 DEEPSEEK_API_KEY、data_path 不存在,
                或该目录下找不到任何菜谱时抛出。
        """
        # 提前校验API key
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RAGConfigError(
                "DEEPSEEK_API_KEY is not set. Add it to server/.env or the environment "
                "before starting the server."
            )

        # 检查食谱文档
        self.documents = self.load_documents(data_path)
        if not self.documents:
            raise RAGConfigError(
                f"No recipes found under {data_path}. The knowledge base is empty."
            )

        # 父/子切分
        self.child_chunks, self.child_to_parent = self._build_child_chunks(self.documents)

        # 第三步 加载向量模型(sentence-transformers 直接加载 BGE 模型,
        # 编码时统一传 normalize_embeddings=True)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # 加载/构建向量库(faiss-cpu 原生 IndexFlatIP)
        self.vector_store = self._build_or_load_vector_store(data_path)

        # 构建 BM25 索引(rank_bm25 的 BM25Okapi,复用同一套分词器;不缓存,
        # 每次启动都重建——语料规模小,重建很快)
        tokenized_chunks = [self._tokenize(chunk) for chunk in self.child_chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        # 初始化 LLM 客户端(openai 官方 SDK 的 OpenAI 客户端,base_url 指向
        # DeepSeek 的 OpenAI 兼容接口;model/temperature 都在每次调用时显式
        # 传入,见 _complete/_raw_stream_complete)。
        self.model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
        self.llm = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    # 第一步 加载文档
    def load_documents(self, data_path: str) -> list[Recipe]:
        """递归读取 data_path 下的每个 Markdown 文件,生成对应的 Recipe"""
        root = Path(data_path)
        if not root.is_dir():
            raise RAGConfigError(
                f"Recipe knowledge base directory not found at {data_path}"
            )

        documents = []
        for md_file in sorted(root.rglob("*.md")):
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                continue
            recipe = Recipe(text=content)
            self._enhance_metadata(recipe, md_file)
            documents.append(recipe)

        return documents

    def _enhance_metadata(self, recipe: Recipe, path: Path) -> None:
        """原地补全 Recipe 的 category/dish_name/difficulty。

        Args:
            recipe: 要补全的 Recipe(原地修改)。
            path: 菜谱的加载路径,用于推断 dish_name(文件名)和
                category(父目录名)。
        """
        # 菜名:取文件名,把连字符换回空格("kung-pao-chicken" -> "kung pao chicken"),
        # 作为 "# 标题" 缺失时的兜底。
        recipe.dish_name = path.stem.replace("-", " ")

        # 分类:从所在父目录推断,如 .../recipes/meat_dish/kung-pao-chicken.md
        # -> "meat_dish" -> "Meat"。
        recipe.category = CATEGORY_LABELS.get(path.parent.name, "Other")

        # 难度:根据 "## Steps" 小节里的步骤数估算
        # (源数据没有现成的难度标注可以直接解析)。
        steps_match = re.search(
            r"(?m)^##[ \t]+Steps\s*$(.*?)(?=^#{1,2}[ \t]|\Z)", recipe.text, re.DOTALL
        )
        step_count = len(_STEP_RE.findall(steps_match.group(1))) if steps_match else 0
        if step_count == 0:
            recipe.difficulty = "Unknown"
        elif step_count <= 3:
            recipe.difficulty = "Easy"
        elif step_count == 4:
            recipe.difficulty = "Medium"
        else:
            recipe.difficulty = "Hard"

    # 第二步 chunk
    def _build_child_chunks(
        self, documents: list[Recipe]
    ) -> tuple[list[str], list[int]]:
        """按 #/##/### 标题把每篇菜谱切成父块+子块。

        Returns:
            (child_chunks, child_to_parent) 二元组:child_chunks 是所有子块的
            文本,按顺序 0..N-1 排列;child_to_parent 是等长的平行列表,
            child_to_parent[i] 是第 i 个子块所属的父 Recipe 在
            SimpleRAG.documents 里的下标。两者靠下标对齐,不需要额外的包装
            对象来携带这个映射。
        """
        child_chunks: list[str] = []
        child_to_parent: list[int] = []

        for parent_index, recipe in enumerate(documents):
            for section in self._split_by_headings(recipe.text):
                child_chunks.append(section)
                child_to_parent.append(parent_index)

        return child_chunks, child_to_parent

    @staticmethod
    def _split_by_headings(text: str) -> list[str]:
        """按 "#"/"##"/"###" 标题把文本切成多段,每个标题对应一段"""
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            stripped = text.strip()
            return [stripped] if stripped else []

        sections = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)
        return sections

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """BM25 用的分词器:转小写、提取字母数字单词。

        BM25Okapi 本身不做分词,只接受 token 列表,所以语料和查询都要先经过
        这个函数(见 __init__ 里建索引、_bm25_search 里查询)。
        """
        return _TOKEN_RE.findall(text.lower())

    # 第四步 入库
    def _build_or_load_vector_store(self, data_path: str) -> "faiss.Index":
        """构建/加载 child_chunks 的 faiss 向量索引(原生 faiss-cpu)。

        子块是按顺序 0..N-1 加入索引的,所以 faiss 检索返回的行号本身就是
        chunk_index,不需要额外维护一份 id 映射/docstore。
        """
        index_dir = Path(data_path).parent / "vector_index"
        index_path = index_dir / "index.faiss"
        meta_path = index_dir / "meta.json"

        content_hash = hashlib.sha256(
            "\n".join(self.child_chunks).encode("utf-8")
        ).hexdigest()

        if index_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
            if (
                meta.get("content_hash") == content_hash
                and meta.get("model") == EMBEDDING_MODEL_NAME
            ):
                try:
                    logger.info("Loading cached vector index from %s", index_dir)
                    return faiss.read_index(str(index_path))
                except RuntimeError:
                    # 缓存文件损坏,或来自旧版索引格式 —— 兜底重建,
                    # 而不是让启动直接崩溃。
                    logger.warning(
                        "Vector index cache at %s could not be loaded; rebuilding.",
                        index_dir,
                        exc_info=True,
                    )
            else:
                logger.info("Vector index cache at %s is stale; rebuilding.", index_dir)

        logger.info(
            "Building vector index for %d chunks with %s...",
            len(self.child_chunks),
            EMBEDDING_MODEL_NAME,
        )
        vectors = self.embedder.encode(
            self.child_chunks, normalize_embeddings=True, show_progress_bar=False
        )
        vectors = np.asarray(vectors, dtype="float32")

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        meta_path.write_text(
            json.dumps(
                {
                    "content_hash": content_hash,
                    "model": EMBEDDING_MODEL_NAME,
                    "chunk_count": len(self.child_chunks),
                }
            ),
            encoding="utf-8",
        )
        return index

    # 线上 第二步·向量检索(粗排)
    def _vector_search(
        self, question: str, k: int, allowed: set[int] | None = None
    ) -> list[int]:
        """按向量余弦相似度返回最多 k 个子块下标。

        通过 faiss 索引的 search() 对整个索引排序(k 传入语料库全量大小,
        拿到完整排名),再在 Python 里按 allowed 过滤——faiss 本身不理解我们的
        分类/难度元数据过滤,所以策略与旧版一致:先在整个索引里搜索,再过滤,
        而不是把过滤下推到 ANN 搜索本身。

        Args:
            question: 查询文本。
            k: 最多返回的子块下标数量。
            allowed: 如果给定,结果只保留在这个集合中的子块下标
                (用于分类/难度过滤)。

        Returns:
            按相关性从高到低排序的子块下标列表。
        """
        total = self.vector_store.ntotal
        if total == 0 or k <= 0:
            return []
        query_vec = self.embedder.encode([question], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype="float32")
        _, ids = self.vector_store.search(query_vec, total)
        ranked = [int(i) for i in ids[0] if i != -1]
        if allowed is not None:
            ranked = [i for i in ranked if i in allowed]
        return ranked[:k]

    def _bm25_search(
        self, question: str, k: int, allowed: set[int] | None = None
    ) -> list[int]:
        """按 BM25 分数返回最多 k 个子块下标。

        BM25Okapi.get_scores() 直接返回语料库全量的分数数组,在 Python 里按
        分数排序后再按 allowed 过滤,与 _vector_search 的过滤策略保持一致。

        Args:
            question: 查询文本。
            k: 最多返回的子块下标数量。
            allowed: 如果给定,结果只保留在这个集合中的子块下标
                (用于分类/难度过滤)。

        Returns:
            按相关性从高到低排序的子块下标列表。
        """
        if k <= 0 or not self.child_chunks:
            return []
        scores = self.bm25.get_scores(self._tokenize(question))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if allowed is not None:
            ranked = [i for i in ranked if i in allowed]
        return ranked[:k]

    # 线上 第四步 rerank精排
    @staticmethod
    def _rrf_fuse(*ranked_lists: list[int], k: int = RRF_K) -> dict[int, float]:
        """Reciprocal Rank Fusion:把多路排名列表融合成一个分数字典。

        某个下标在一路排名里排第 rank 名,就贡献 1/(k + rank + 1) 分;
        如果同一个下标出现在多路排名里,分数会累加。这样不需要对齐向量分数和
        BM25 分数的量纲,也能公平地融合两路结果。公式与 LangChain
        EnsembleRetriever.weighted_reciprocal_rank 一致(默认常数同为 60),
        这里保留自实现是因为后续还需要每个子块的分数去做父文档聚合,
        而不只是拿到融合后的文档列表(见模块顶部说明)。

        Args:
            *ranked_lists: 一个或多个下标列表,每个列表内部已按某种检索器
                从好到差排序。
            k: RRF 的平滑常数(见 RRF_K)。

        Returns:
            字典,把每个在任意一路出现过的下标映射到它的累加 RRF 分数
            (越高越相关);完全没出现过的下标不在结果里。
        """
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, index in enumerate(ranked):
                scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank + 1)
        return scores

    def _infer_filters(self, question: str) -> tuple[str | None, str | None]:
        """用关键词启发式地从问题文本推断分类/难度过滤条件。

        刻意保守:只在命中相当无歧义的短语时才触发(见 _CATEGORY_KEYWORDS/
        _DIFFICULTY_KEYWORDS),这样提到具体菜名的问题不会因为恰好带了某个
        触发词就被误伤过滤掉。

        Args:
            question: 用户的自然语言问题。

        Returns:
            (category, difficulty) 二元组;两者都可能是 None,表示没有命中。
        """
        lowered = question.lower()

        category = next(
            (
                label
                for label, keywords in _CATEGORY_KEYWORDS.items()
                if any(keyword in lowered for keyword in keywords)
            ),
            None,
        )
        difficulty = next(
            (
                label
                for label, keywords in _DIFFICULTY_KEYWORDS.items()
                if any(keyword in lowered for keyword in keywords)
            ),
            None,
        )
        return category, difficulty

    # 第一步 query处理
    def _infer_route(self, question: str) -> str | None:
        """用关键词判断问题是 list 还是 detail,不调用 LLM。

        作为 query_router 前面的一道廉价预判:大多数日常问题都带有明确的
        信号词("recommend""how do I make" 等),本地就能判断,从而完全省掉
        一次 DeepSeek 往返。同样刻意保守(与 _infer_filters 同一思路)——
        判不出来的问题会返回 None,落到 LLM(见 ask() / _classify_and_rewrite),
        而不是冒险把一个 general 问题分错类。

        Args:
            question: 用户的自然语言问题。

        Returns:
            命中触发短语则返回 ROUTE_LIST 或 ROUTE_DETAIL,否则返回 None。
            永远不会返回 ROUTE_GENERAL——general 问题没有可靠的关键词特征,
            一律交给 LLM 判断。
        """
        lowered = question.lower()
        if any(trigger in lowered for trigger in _LIST_ROUTE_TRIGGERS):
            return ROUTE_LIST
        if any(trigger in lowered for trigger in _DETAIL_ROUTE_TRIGGERS):
            return ROUTE_DETAIL
        return None

    def retrieve(
        self,
        question: str,
        top_k: int = 2,
        category: str | None = None,
        difficulty: str | None = None,
    ) -> list[str]:
        """返回与问题最相关的 top_k 篇父菜谱正文。

        对 _rank_documents() 的简单封装,供只需要菜谱文本(比如拼接 grounding
        prompt)而不需要完整 Recipe 元数据的调用方使用。检索算法本身见
        _rank_documents()。

        Args:
            question: 用户的自然语言问题。
            top_k: 最多返回的父菜谱数量。
            category: 可选的显式分类过滤,覆盖从问题文本推断出的结果。
            difficulty: 可选的显式难度过滤,覆盖从问题文本推断出的结果。

        Returns:
            最多 top_k 篇父菜谱正文,按相关性从高到低排序。
        """
        return [
            doc.text
            for doc in self._rank_documents(
                question, top_k, category=category, difficulty=difficulty
            )
        ]

    # 第四步 检索
    def _rank_documents(
        self,
        question: str,
        top_k: int = 2,
        category: str | None = None,
        difficulty: str | None = None,
    ) -> list[Recipe]:
        """返回与问题最相关的 top_k 篇父 Recipe 对象"""
        top_k = min(top_k, len(self.documents))
        if top_k == 0:
            return []

        if category is None and difficulty is None:
            category, difficulty = self._infer_filters(question)

        allowed_chunks = None
        if category or difficulty:
            allowed_parents = {
                i
                for i, doc in enumerate(self.documents)
                if (category is None or doc.category == category)
                and (difficulty is None or doc.difficulty == difficulty)
            }
            if allowed_parents:
                allowed_chunks = {
                    i for i, p in enumerate(self.child_to_parent) if p in allowed_parents
                }
            # 如果过滤条件一篇菜谱都没匹配上,就悄悄回退到不加过滤的搜索
            # (allowed_chunks 保持 None),而不是直接返回空结果。

        pool_size = len(allowed_chunks) if allowed_chunks is not None else len(self.child_chunks)
        vector_ranked = self._vector_search(question, k=pool_size, allowed=allowed_chunks)
        bm25_ranked = self._bm25_search(question, k=pool_size, allowed=allowed_chunks)

        fused_scores = self._rrf_fuse(vector_ranked, bm25_ranked)
        if not fused_scores:
            return []
        ranked_child_indices = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)

        # 融合分数最高的一批候选子块。大小相对 top_k 设置(下限 10),
        # 这样即便 top_k 较大,也有足够候选从中找出 top_k 篇不同的父菜谱,
        # 不需要扫描整个(过滤后的)语料库。
        candidate_pool = min(len(ranked_child_indices), max(top_k * 5, 10))

        score: dict[int, float] = {}
        hits: dict[int, int] = {}
        ordered_parents: list[int] = []  # 按候选中首次出现的顺序记录
        for child_index in ranked_child_indices[:candidate_pool]:
            parent_index = self.child_to_parent[child_index]
            if parent_index not in score:
                ordered_parents.append(parent_index)
            score[parent_index] = score.get(parent_index, 0.0) + fused_scores[child_index]
            hits[parent_index] = hits.get(parent_index, 0) + 1

        # 先按总 RRF 分数排序(奖励多处强匹配),分数接近时再按命中次数打破平局。
        ranked_parents = sorted(
            ordered_parents, key=lambda p: (score[p], hits[p]), reverse=True
        )

        # 如果候选列表凑不够 top_k 篇不同父菜谱,就从完整的融合排名里
        # 继续补足剩余名额,确保即便候选池里没那么多篇,retrieve() 依然
        # 尽量凑够 top_k。
        if len(ranked_parents) < top_k:
            seen = set(ranked_parents)
            for child_index in ranked_child_indices:
                parent_index = self.child_to_parent[child_index]
                if parent_index not in seen:
                    ranked_parents.append(parent_index)
                    seen.add(parent_index)
                if len(ranked_parents) == top_k:
                    break

        selected = ranked_parents[:top_k]
        return [self.documents[i] for i in selected]

    def _complete(self, prompt: str, temperature: float = 0.3) -> str | None:
        """发一次单轮对话请求给 DeepSeek(通过 openai 官方 SDK 的 OpenAI 客户端),
        吞掉常见的调用失败异常。

        把这个模块调用 DeepSeek 的 4 处地方(query_router、query_rewrite,
        以及两种基于 prompt 的生成模式)共用的 try/except 逻辑集中在这里,
        这样各处只需要处理"拿到 None 该怎么办",不用各写一遍相同的
        except 代码块。

        Args:
            prompt: 作为单条用户消息发送的完整 prompt(纯文本,包装成
                chat.completions.create() 的一条 user 角色消息)。
            temperature: 采样温度——分类/改写这类需要稳定、字面结果的调用
                用接近 0 的低值;开放式生成用更高的值。每次调用都显式传入。

        Returns:
            模型的回复文本;调用失败(鉴权/网络/超时/API 错误)或返回空内容
            则返回 None。调用方自行决定 None 对自己意味着什么、要不要记日志。
        """
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
        except AuthenticationError:
            logger.error("DeepSeek authentication failed - check DEEPSEEK_API_KEY.")
            return None
        except (APIConnectionError, APITimeoutError):
            logger.error("DeepSeek API connection/timeout error.")
            return None
        except APIError:
            logger.exception("DeepSeek API returned an error.")
            return None

        content = response.choices[0].message.content
        return content.strip() if content else None

    def query_router(self, question: str) -> str:
        """把问题分类为 "list"、"detail" 或 "general"。

        - list: 用户想要一批菜品建议/名字(如"推荐几个素菜")
        - detail: 用户想知道某道具体菜怎么做(如"宫保鸡丁怎么做?")
        - general: 其他情况(如"砂锅和火锅有什么区别?")

        Args:
            question: 用户的自然语言问题。

        Returns:
            ROUTE_LIST/ROUTE_DETAIL/ROUTE_GENERAL 之一。分类调用失败或返回
            意料之外的结果时,兜底为 ROUTE_GENERAL(最保守的默认值——
            走普通问答)。
        """
        prompt = ROUTER_PROMPT.format(question=question)
        route = self._complete(prompt, temperature=0)
        if route:
            route = route.strip().lower()
        return route if route in _VALID_ROUTES else ROUTE_GENERAL

    def query_rewrite(self, question: str) -> str:
        """把含糊的问题改写成更具体、更适合检索的问法。

        让 LLM 自己判断是否需要改写:已经明确指出具体菜名或诉求的问题应原样返回;
        含糊的问题(如"随便给我道菜吃")应被改写得更具体、带上明确的烹饪术语。

        Args:
            question: 用户的自然语言问题。

        Returns:
            改写后的问题;如果改写调用失败或没返回可用结果,
            则原样返回原始问题。
        """
        prompt = REWRITE_PROMPT.format(question=question)
        rewritten = self._complete(prompt, temperature=0.2)
        return rewritten if rewritten else question

    def _classify_and_rewrite(self, question: str) -> tuple[str, str]:
        """用一次 LLM 调用同时完成路由分类和问题改写。

        把 query_router() + query_rewrite() 的工作合并成一次请求而不是两次,
        因为二者看的是同一个问题,且各自都只需要一个简短的结构化判断。
        这是 ask() 在 _infer_route() 无法从关键词自信地判断出结果时才会
        走到的兜底路径——也是唯一还需要 LLM 判断的情况。

        JSON 解析用模块顶部的 _parse_json_reply():先尝试剥掉 ```json ... ```
        代码块围栏,剥不出来就当作回复本身是裸 JSON,再交给 json.loads()。

        Args:
            question: 用户的自然语言问题。

        Returns:
            (route, rewritten_question) 二元组。调用失败或回复无法解析成
            预期的 JSON 结构时,兜底为 (ROUTE_GENERAL, question)——
            与 query_router、query_rewrite 各自的兜底行为保持一致。
        """
        prompt = CLASSIFY_AND_REWRITE_PROMPT.format(question=question)
        reply = self._complete(prompt, temperature=0)
        if reply:
            try:
                data = _parse_json_reply(reply)
                route = str(data.get("route", "")).strip().lower()
                rewritten = str(data.get("rewritten", "")).strip()
            except (json.JSONDecodeError, AttributeError, TypeError):
                route, rewritten = "", ""
            if route in _VALID_ROUTES and rewritten:
                return route, rewritten
            logger.warning("Could not parse classify_and_rewrite reply: %r", reply)
        return ROUTE_GENERAL, question

    # 线上第五步 生成
    def _generate_list_answer(self, docs: list[Recipe]) -> str:
        """把菜名格式化成编号列表——不调用 LLM。

        检索到的父 Recipe 对象已经带有 dish_name,所以 "list" 类问题
        直接格式化这个元数据即可回答,既省成本,也比让 LLM 复述一份
        刚刚给它的名单更字面、更可靠。

        Args:
            docs: 要列出的父 Recipe 对象,按相关性排序。

        Returns:
            菜名编号列表;docs 为空则返回 NO_RESULTS_MESSAGE。
        """
        if not docs:
            return NO_RESULTS_MESSAGE

        seen = set()
        lines = []
        for doc in docs:
            name = doc.dish_name.title()
            if name in seen:
                continue
            seen.add(name)
            lines.append(f"{len(lines) + 1}. {name} ({doc.category}, {doc.difficulty})")

        return "Here are some recipes you might like:\n" + "\n".join(lines)

    @staticmethod
    def _step_by_step_prompt(question: str, docs: list[Recipe]) -> str:
        """构建 "detail" 路由用的结构化生成 prompt(用 STEP_BY_STEP_PROMPT 模板)。

        从 _generate_step_by_step_answer() 中抽出来,这样流式版本
        (_generate_step_by_step_answer_stream())可以复用同一份 prompt 模板,
        不用重复写一遍。
        """
        context = "\n\n".join(doc.text for doc in docs)
        return STEP_BY_STEP_PROMPT.format(
            no_results_message=NO_RESULTS_MESSAGE, context=context, question=question
        )

    def _generate_step_by_step_answer(self, question: str, docs: list[Recipe]) -> str:
        """生成结构化的分步骤菜谱回答。

        Args:
            question: 用户的原始问题(不是改写后的问题——展示给 LLM 是为了
                让它回答用户实际问的内容)。
            docs: 用于给答案提供依据的父 Recipe 对象。

        Returns:
            结构化的 Markdown 回答(overview/ingredients/steps/tips);
            docs 为空则返回 NO_RESULTS_MESSAGE;生成失败则返回
            LLM_UNAVAILABLE_MESSAGE。
        """
        if not docs:
            return NO_RESULTS_MESSAGE

        answer = self._complete(self._step_by_step_prompt(question, docs), temperature=0.3)
        return answer if answer else LLM_UNAVAILABLE_MESSAGE

    def _generate_step_by_step_answer_stream(
        self, question: str, docs: list[Recipe], state: dict
    ):
        """_generate_step_by_step_answer() 的流式版本。

        增量地把答案文本 yield 出去,而不是一次性返回。通过调用方传入的
        可变字典 `state`(原地修改)回传执行结果,因为生成器的返回值在普通
        `for` 循环里是拿不到的:
            - state["is_no_results"]: docs 为空,或流式回复完整拼起来正好
              等于 NO_RESULTS_MESSAGE 时为 True(此时什么都不会 yield,
              由调用方决定接下来要不要输出这条提示语——见 ask_stream())。
            - state["failed"]: LLM 调用本身失败(鉴权/网络/超时/API 错误),
              且没有产出任何内容时为 True。

        Args:
            question: 用户的原始问题。
            docs: 用于给答案提供依据的父 Recipe 对象。
            state: 本方法写入结果的字典(见上文)。

        Yields:
            按顺序产出的答案文本片段。如果最终 state["is_no_results"] 为
            True,则什么都不会产出。
        """
        if not docs:
            state["is_no_results"] = True
            return
        yield from self._stream_with_no_results_guard(
            self._step_by_step_prompt(question, docs), temperature=0.3, state=state
        )
        if state.get("failed"):
            yield LLM_UNAVAILABLE_MESSAGE

    @staticmethod
    def _basic_answer_prompt(question: str, docs: list[Recipe]) -> str:
        """构建 "general" 路由用的普通问答 prompt(用 BASIC_ANSWER_PROMPT 模板)。

        从 _generate_basic_answer() 中抽出来,这样流式版本
        (_generate_basic_answer_stream())可以复用同一份 prompt 模板,
        不用重复写一遍。
        """
        context = "\n\n".join(doc.text for doc in docs)
        return BASIC_ANSWER_PROMPT.format(
            no_results_message=NO_RESULTS_MESSAGE, context=context, question=question
        )

    def _generate_basic_answer(self, question: str, docs: list[Recipe]) -> str:
        """为 general(非 list、非 detail)问题生成普通的、有依据的回答。

        这是引入问题路由之前,RAGChef 原本唯一使用的那种生成 prompt。

        Args:
            question: 用户的原始问题。
            docs: 用于给答案提供依据的父 Recipe 对象。

        Returns:
            生成的回答;docs 为空则返回 NO_RESULTS_MESSAGE;生成失败则返回
            LLM_UNAVAILABLE_MESSAGE。
        """
        if not docs:
            return NO_RESULTS_MESSAGE

        answer = self._complete(self._basic_answer_prompt(question, docs), temperature=0.3)
        return answer if answer else LLM_UNAVAILABLE_MESSAGE

    def _generate_basic_answer_stream(self, question: str, docs: list[Recipe], state: dict):
        """_generate_basic_answer() 的流式版本。

        state 字典的用法与 _generate_step_by_step_answer_stream() 一致。
        """
        if not docs:
            state["is_no_results"] = True
            return
        yield from self._stream_with_no_results_guard(
            self._basic_answer_prompt(question, docs), temperature=0.3, state=state
        )
        if state.get("failed"):
            yield LLM_UNAVAILABLE_MESSAGE

    def _raw_stream_complete(self, prompt: str, temperature: float = 0.3):
        """流式发送单轮对话请求(通过 openai 客户端的
        chat.completions.create(stream=True)),边到达边 yield 出文本片段。

        _complete() 的流式版本:失败处理逻辑相同(鉴权/连接/超时/API 错误
        都会被记录并吞掉,而不是抛出),区别是不在最后一次性返回完整文本,
        而是 DeepSeek 每发来一段就 yield 一段(每个 chunk 是一个
        ChatCompletionChunk,其 .choices[0].delta.content 就是这次增量的文本)。
        如果失败发生在第一段内容到达之前,则完全不会产出任何片段;如果发生在
        响应中途,流会提前中止——仅凭这个方法本身无法区分这两种情况(见
        _stream_with_no_results_guard(),它专门跟踪了"到底有没有收到过任何内容")。

        Args:
            prompt: 作为单条用户消息发送的完整 prompt。
            temperature: 采样温度。

        Yields:
            按顺序产出的非空文本片段。
        """
        try:
            stream = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except AuthenticationError:
            logger.error("DeepSeek authentication failed - check DEEPSEEK_API_KEY.")
        except (APIConnectionError, APITimeoutError):
            logger.error("DeepSeek API connection/timeout error.")
        except APIError:
            logger.exception("DeepSeek API returned an error.")

    def _stream_with_no_results_guard(self, prompt: str, temperature: float, state: dict):
        """流式生成答案,同时按住可能是 NO_RESULTS_MESSAGE 的内容不立即输出。

        非流式的生成方法能直接判断 `answer == NO_RESULTS_MESSAGE`,
        是因为它们会等完整回复到达后再做判断。如果流式版本直接把内容原样
        转发给客户端,"没有找到相关信息"这句话就会被逐字符地流式显示出来,
        而不是作为一个整体的提示语出现。

        这个方法在不增加额外 LLM 调用的前提下堵上了这个缺口:只要缓冲区内容
        仍然可能是 NO_RESULTS_MESSAGE 的前缀,就先按住不输出;一旦流式文本
        与这句话的内容出现分歧,就立刻把缓冲区一次性吐出,之后转为直接透传——
        所以一段真正的回答最多延迟几个字符就会开始显示。只有从头到尾完全匹配
        NO_RESULTS_MESSAGE 的回复才会在整个流式过程中被全程按住。

        Args:
            prompt: 作为单条用户消息发送的完整 prompt。
            temperature: 采样温度。
            state: 本方法写入结果的字典:
                - state["is_no_results"]: 完整回复正好等于 NO_RESULTS_MESSAGE
                  时为 True(此时不会 yield 任何内容)。
                - state["failed"]: 底层流没有产出任何内容(LLM 调用直接失败)
                  时为 True。

        Yields:
            按顺序产出的答案文本片段。如果回复正好是 NO_RESULTS_MESSAGE,
            或调用直接失败,则什么都不会产出。
        """
        buffer = ""
        diverged = False
        got_any = False

        for delta in self._raw_stream_complete(prompt, temperature):
            got_any = True
            if diverged:
                yield delta
                continue
            buffer += delta
            if NO_RESULTS_MESSAGE.startswith(buffer):
                continue  # 目前仍可能是 NO_RESULTS_MESSAGE 的前缀(或已完全匹配)——继续按住
            diverged = True
            yield buffer

        if not got_any:
            state["failed"] = True
        elif not diverged:
            if buffer == NO_RESULTS_MESSAGE:
                state["is_no_results"] = True
            else:
                # 流结束时缓冲区仍是 NO_RESULTS_MESSAGE 的严格(非完全相等)前缀——
                # 属于异常截断,不算真正匹配。把按住的内容吐出来,而不是悄悄丢弃。
                yield buffer

    def ask(self, question: str) -> str:
        """对问题做路由、检索并生成回答(非流式入口)。

        流程:先分类问题(query_router)-> 对非 "list" 问题,视需要改写得更具体、
        更适合检索(query_rewrite;"list" 问题原样使用,因为它本身就是在要
        一批选项,没什么好"改写得更具体"的)-> 用(可能已改写的)查询检索父菜谱
        -> 分发到与路由匹配的生成模式:list 用纯格式化的菜名列表,
        detail 用结构化的分步骤回答,general 用普通的、有依据的回答。

        Args:
            question: 用户的自然语言问题。

        Returns:
            生成的回答;如果检索或生成没能产出可用结果,则返回
            NO_RESULTS_MESSAGE 或 LLM_UNAVAILABLE_MESSAGE 之一。本方法不会
            因为预期内的 LLM 供应商故障而抛异常——这些故障会被记录日志,
            并转成 LLM_UNAVAILABLE_MESSAGE 返回(list 类回答除外,
            它根本不调用 LLM,所以不存在这种失败)。
        """
        # 输入校验
        if not question or not question.strip():
            return "Please enter a question."

        # query 处理
        route = self._infer_route(question)
        if route == ROUTE_LIST:
            search_query = question
        elif route == ROUTE_DETAIL:
            search_query = self.query_rewrite(question)
        else:
            route, search_query = self._classify_and_rewrite(question)

        top_k = LIST_MODE_TOP_K if route == ROUTE_LIST else 2
        # 检索
        docs = self._rank_documents(search_query, top_k=top_k)

        if route == ROUTE_LIST:
            return self._generate_list_answer(docs)

        # 生成
        generate = (
            self._generate_step_by_step_answer
            if route == ROUTE_DETAIL
            else self._generate_basic_answer
        )
        answer = generate(question, docs)
        return answer

    def ask_stream(self, question: str):
        """ask() 的流式版本:增量地把答案 yield 出去。

        路由/检索/生成模式选择逻辑与 ask() 完全一致(刻意保留为一份独立实现,
        而不是让其中一个委托给另一个,这样 ask() 里非流式的 DeepSeek 调用、
        以及测试里对它的 mock,都不受影响);区别只在于所选生成模式产出结果的
        方式:纯列表回答或固定提示语作为一个整体 yield 出去(反正不调用 LLM,
        没什么好流式的),而 "detail"/"general" 的回答会随着 DeepSeek 逐步生成
        而增量 yield,使用
        _generate_step_by_step_answer_stream()/_generate_basic_answer_stream()。

        Args:
            question: 用户的自然语言问题。

        Yields:
            按顺序产出的答案文本片段;把所有片段拼起来,得到的字符串
            与同样问题下 ask() 会返回的结果完全一致。
        """
        # 输入校验
        if not question or not question.strip():
            yield "Please enter a question."
            return

        route = self._infer_route(question)
        if route == ROUTE_LIST:
            search_query = question
        elif route == ROUTE_DETAIL:
            search_query = self.query_rewrite(question)
        else:
            route, search_query = self._classify_and_rewrite(question)

        top_k = LIST_MODE_TOP_K if route == ROUTE_LIST else 2
        docs = self._rank_documents(search_query, top_k=top_k)

        if route == ROUTE_LIST:
            yield self._generate_list_answer(docs)
            return

        generate_stream = (
            self._generate_step_by_step_answer_stream
            if route == ROUTE_DETAIL
            else self._generate_basic_answer_stream
        )

        state: dict = {}
        for chunk in generate_stream(question, docs, state):
            yield chunk

        if state.get("is_no_results"):
            yield NO_RESULTS_MESSAGE
