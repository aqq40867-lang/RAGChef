"""RAGChef 的核心 RAG(检索增强生成)逻辑 —— LangChain 版本。

整体流程:
1. 加载: 递归读取知识库目录下的每个 Markdown 文件,生成一个 Recipe(父文档),
   并补全分类/菜名/难度等元数据。
2. 分块: 把每篇 Recipe 按 "#"/"##"/"###" 标题切成父块(整篇菜谱)+
   若干子块(标题/食材/步骤等)。
3. 建索引: 用 LangChain 的 HuggingFaceEmbeddings 包装 BGE 句向量模型,
   把子块编码成向量并交给 langchain_community 的 FAISS 向量库建索引
   (缓存到磁盘,避免每次启动都重新 embedding);同时用 LangChain 的
   BM25Retriever 建一份关键词索引(底层仍是 rank_bm25,但由 LangChain
   统一包装,复用同一套分词器)。
4. 检索: 对每个问题,先按分类/难度(显式传入或从问题文本推断)缩小候选范围,
   再分别用向量检索(FAISS)和关键词检索(BM25Retriever)产出子块排名,
   通过 Reciprocal Rank Fusion(RRF)融合成统一分数;然后按"子块所属的
   父文档"聚合分数,把父文档(菜谱)排序后返回。
5. 路由: 先用免费的关键词判断(_infer_route)把问题分成 list/detail/general
   三类,判不出来再退回一次 LLM 调用(_classify_and_rewrite)同时完成分类和
   问题改写;关键词判成 detail 的问题仍需单独调用 query_rewrite 做改写,
   list 问题则原样使用。所有 prompt 都用 LangChain 的 PromptTemplate 构建。
6. 生成: 按路由类型分发到对应生成模式——纯格式化的菜名列表、结构化的分步骤回答,
   或普通的问答——都要求 DeepSeek(通过 LangChain 的 ChatOpenAI,base_url
   指向 DeepSeek 的 OpenAI 兼容接口)严格依据检索到的内容作答("grounding"),
   避免模型编造信息。
7. 兜底: 如果 DeepSeek 判断检索到的本地菜谱都答不了一个 detail/general 问题,
   ask() 会现查一次 TheMealDB(免费的公开菜谱 API)作为最后一道兜底,而不是
   直接放弃,见 THEMEALDB_* 和 SimpleRAG._themealdb_fallback()。

# 中文说明: 为什么不是所有环节都用 LangChain 的现成组件?
# - 父子分块、以及"按父文档聚合子块 RRF 分数"这一步,继续手写
#   (_rrf_fuse / _rank_documents)。LangChain 的 EnsembleRetriever 也实现了
#   一样的加权 RRF 公式(默认常数同样是 60),但它只返回融合后的文档列表,
#   不对外暴露每个候选的分数;而这里需要每个子块的分数去做"父文档累计打分
#   + 按命中小节数排序 + 去重"这个本项目特有的策略,所以保留自实现,
#   FAISS/BM25 检索本身仍然换成了 LangChain 的组件。
# - TheMealDB 兜底、路由关键词预判、难度/分类的关键词推断:都是本项目的
#   业务逻辑,不是通用 RAG 能力,LangChain 不提供对应组件,继续手写。
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.output_parsers.json import parse_json_markdown
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
)

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
# 知识库规模小且用词集中,这样已经够用。传给 LangChain 的 BM25Retriever 作为
# preprocess_func,保证它与旧实现分词一致(而不是用 BM25Retriever 默认的
# 空白切分)。
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# ---------------------------------------------------------------------------
# LangChain PromptTemplate:所有跟 LLM 交互的 prompt 都用模板构建,而不是直接
# 拼 f-string——好处是模板本身可以被单独测试/复用,变量名也更清晰。
# ---------------------------------------------------------------------------

ROUTER_PROMPT = PromptTemplate.from_template(
    """Classify the user's question into exactly one category. Reply with only the category name, nothing else.

Categories:
- list: the user wants a list of dish suggestions or names (e.g. "recommend a few vegetarian dishes", "what soups do you have?")
- detail: the user wants to know how to make a specific dish (e.g. "how do I make Kung Pao Chicken?", "what ingredients do I need for dumplings?")
- general: anything else (e.g. "what's the difference between a casserole and a hotpot?", "what does al dente mean?")

User question: {question}

Category:"""
)

REWRITE_PROMPT = PromptTemplate.from_template(
    """Decide whether this recipe-assistant question needs rewriting to be more specific before searching a recipe database.

Rules:
- If the question is already specific (names a dish, an ingredient, or a clear request), return it unchanged.
- If the question is vague (e.g. "give me something to cook", "suggest a meal"), rewrite it to be more specific and search-friendly, keeping the original intent and preferring simple, easy-to-make dishes when nothing else is specified.
- Reply with only the final question text, nothing else.

Question: {question}

Rewritten question:"""
)

CLASSIFY_AND_REWRITE_PROMPT = PromptTemplate.from_template(
    """You are helping a recipe assistant prepare to answer a question. Do two things:

1. Classify the question into exactly one category:
   - list: the user wants a list of dish suggestions or names (e.g. "recommend a few vegetarian dishes", "what soups do you have?")
   - detail: the user wants to know how to make a specific dish (e.g. "how do I make Kung Pao Chicken?")
   - general: anything else (e.g. "what's the difference between a casserole and a hotpot?")
2. Produce a version of the question rewritten to be more specific and search-friendly: if the question is already specific (names a dish, an ingredient, or a clear request), repeat it unchanged; if it's vague (e.g. "give me something to cook"), rewrite it to be more specific, keeping the original intent and preferring simple, easy-to-make dishes when nothing else is specified.

Reply with ONLY a JSON object, no other text, in exactly this shape:
{{"route": "list", "rewritten": "..."}}

User question: {question}"""
)

STEP_BY_STEP_PROMPT = PromptTemplate.from_template(
    """You are a professional recipe assistant. Answer the question using ONLY the recipe content below.
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
)

BASIC_ANSWER_PROMPT = PromptTemplate.from_template(
    """You are a professional recipe assistant.

Answer the question using ONLY the recipe content provided below.
If the answer is not contained in the content, reply exactly: "{no_results_message}"

Recipe content:
{context}

User question:
{question}

Respond naturally and concisely in English.
"""
)


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
        source: 菜谱的来源路径。
        dish_name: 菜名,取自文件名。
        category: 菜谱分类,取自所在的父目录名(如 "meat_dish" -> "Meat");
            如果文件不在已知分类目录下,回退为 "Other"。
        difficulty: 难度估算("Easy"/"Medium"/"Hard"),根据 "## Steps"
            小节里的步骤数推算。
    """

    text: str
    source: str
    dish_name: str = ""
    category: str = "Other"
    difficulty: str = "Unknown"


@dataclass
class ChildChunk:
    """子块:菜谱里一个可被检索的片段(标题/食材/步骤等)。

    Attributes:
        text: 子块的 Markdown 内容,从标题行开始。
        parent_index: 指向 SimpleRAG.documents 中对应父 Recipe 的下标。
    """

    text: str
    parent_index: int


class SimpleRAG:
    """加载一次菜谱知识库,之后持续提供检索 + 生成服务"""

    def __init__(self, data_path: str):
        """初始化 RAG 流程:校验配置,加载并索引菜谱。

        流程:校验 DEEPSEEK_API_KEY -> 加载菜谱 -> 按标题切成父/子两级 ->
        加载 LangChain 的 HuggingFaceEmbeddings(包装 BGE 向量模型) ->
        构建/加载 LangChain FAISS 向量库(有磁盘缓存)-> 构建 LangChain
        BM25Retriever(不缓存,每次启动都重建)-> 创建 LangChain ChatOpenAI
        客户端(base_url 指向 DeepSeek)。这是全类里最重的一次性初始化。

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
        self.chunks, self.child_to_parent = self._build_child_chunks(self.documents)
        self.child_chunks = [chunk.text for chunk in self.chunks]

        # 加载向量模型(LangChain 的 HuggingFaceEmbeddings,底层仍是
        # sentence-transformers,但统一走 LangChain 的 Embeddings 接口,
        # 这样向量库/未来接入的其它 LangChain 组件都能直接复用它)。
        self.embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )

        # 加载/构建向量库(LangChain 的 FAISS 向量库封装)
        self.vector_store = self._build_or_load_vector_store(data_path)

        # 构建 BM25 索引(LangChain 的 BM25Retriever,复用同一套分词器)
        metadatas = [{"chunk_index": i} for i in range(len(self.child_chunks))]
        self.bm25_retriever = BM25Retriever.from_texts(
            self.child_chunks, metadatas=metadatas, preprocess_func=self._tokenize
        )

        # 初始化 LLM 客户端(LangChain 的 ChatOpenAI,base_url 指向 DeepSeek
        # 的 OpenAI 兼容接口;不同调用点各自需要的 temperature 通过
        # invoke()/stream() 的调用时参数覆盖,见 _complete/_raw_stream_complete)。
        self.model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
        self.llm = ChatOpenAI(
            model=self.model, api_key=api_key, base_url=DEEPSEEK_BASE_URL, temperature=0.3
        )

    # 第一步 加载文档
    def load_documents(self, data_path: str) -> list[Recipe]:
        """递归读取 data_path 下的每个 Markdown 文件,生成对应的 Recipe。

        Args:
            data_path: 存放菜谱文件的目录,每个 ".md" 文件对应一篇菜谱,
                可以按分类分子目录存放(如 data_path/meat_dish/kung-pao-chicken.md)。

        Returns:
            Recipe 对象列表,每个找到的 Markdown 文件一个,并已通过
            _enhance_metadata() 补全元数据。

        Raises:
            RAGConfigError: data_path 不存在或不是目录时抛出。
        """
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
            recipe = Recipe(text=content, source=str(md_file))
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

    def _build_child_chunks(
        self, documents: list[Recipe]
    ) -> tuple[list[ChildChunk], list[int]]:
        """把每篇父菜谱按标题边界切成子块。

        每个 "#"、"##" 或 "###" 标题都会开启一个新子块,直到下一个同级或更高级
        标题为止;这比"只按 ## 切"的旧做法更通用——如果某篇菜谱用了更细的
        "###" 子小节(比如简易版/进阶版两种做法),也能被切到那个更细的粒度,
        而不是被整段并入它的父 "## " 小节。

        Args:
            documents: load_documents() 返回的父菜谱文档列表。

        Returns:
            (chunks, child_to_parent) 二元组:chunks 是所有菜谱切分出的
            标题/食材/步骤/子小节,child_to_parent[j] 是 chunks[j] 所属父菜谱
            在 documents 中的下标。
        """
        chunks: list[ChildChunk] = []
        child_to_parent: list[int] = []

        for parent_index, recipe in enumerate(documents):
            for section in self._split_by_headings(recipe.text):
                chunks.append(ChildChunk(text=section, parent_index=parent_index))
                child_to_parent.append(parent_index)

        return chunks, child_to_parent

    @staticmethod
    def _split_by_headings(text: str) -> list[str]:
        """按 "#"/"##"/"###" 标题把文本切成多段,每个标题对应一段。

        每段从一个标题行开始,到下一个 1-3 级标题行之前结束(不含该标题行),
        所以嵌在 "## " 小节里的 "### " 子标题会被切成独立的一段,
        而不是并入它的父小节。

        Args:
            text: 要切分的 Markdown 文本(通常是一整篇菜谱)。

        Returns:
            按文档顺序排列的非空分段字符串列表。如果文本里没有任何标题,
            整段(去除首尾空白后)作为单独一个分段返回。
        """
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

        作为 preprocess_func 传给 LangChain 的 BM25Retriever。
        """
        return _TOKEN_RE.findall(text.lower())

    def _build_or_load_vector_store(self, data_path: str) -> FAISS:
        """构建 child_chunks 的 LangChain FAISS 向量库,或从磁盘加载已缓存的向量库。

        沿用参考项目的索引缓存思路(先 embedding 一次、存盘,后续启动直接加载,
        而不是每次都重新 embedding 所有子块),并多加了一步:缓存里记录了子块内容的
        哈希值,一旦知识库发生变化就能识别出缓存已过期并自动重建,
        而不是悄悄地继续用过期的向量。

        distance_strategy 用 MAX_INNER_PRODUCT:embedding 已经做过 L2 归一化,
        内积等价于余弦相似度,分数越高越相关(与旧版直接用 faiss.IndexFlatIP
        的语义一致)。

        Args:
            data_path: 传给 __init__ 的菜谱目录;缓存存放在其同级的
                "vector_index/" 目录下。

        Returns:
            对 self.child_chunks(已归一化)向量建立的 LangChain FAISS 向量库,
            每个文档的 metadata 里带 chunk_index,顺序与 child_chunks 一致。
        """
        index_dir = Path(data_path).parent / "vector_index"
        meta_path = index_dir / "meta.json"

        content_hash = hashlib.sha256(
            "\n".join(self.child_chunks).encode("utf-8")
        ).hexdigest()

        if index_dir.exists() and meta_path.exists():
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
                    return FAISS.load_local(
                        str(index_dir), self.embedder, allow_dangerous_deserialization=True
                    )
                except (OSError, RuntimeError, ValueError, ModuleNotFoundError):
                    # 缓存文件损坏,或来自旧版(非 LangChain)的索引格式 —— 兜底
                    # 重建,而不是让启动直接崩溃。
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
        metadatas = [{"chunk_index": i} for i in range(len(self.child_chunks))]
        vector_store = FAISS.from_texts(
            self.child_chunks,
            self.embedder,
            metadatas=metadatas,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )

        index_dir.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(index_dir))
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
        return vector_store

    def _vector_search(
        self, question: str, k: int, allowed: set[int] | None = None
    ) -> list[int]:
        """按向量余弦相似度返回最多 k 个子块下标。

        通过 LangChain FAISS 向量库的 similarity_search_with_score 对整个索引
        排序(k 传入语料库全量大小,拿到完整排名),再在 Python 里按 allowed
        过滤——FAISS 向量库本身不理解我们的分类/难度元数据过滤,所以策略与
        旧版一致:先在整个索引里搜索,再过滤,而不是把过滤下推到 ANN 搜索本身。

        Args:
            question: 查询文本。
            k: 最多返回的子块下标数量。
            allowed: 如果给定,结果只保留在这个集合中的子块下标
                (用于分类/难度过滤)。

        Returns:
            按相关性从高到低排序的子块下标列表。
        """
        total = self.vector_store.index.ntotal
        if total == 0 or k <= 0:
            return []
        docs_and_scores = self.vector_store.similarity_search_with_score(question, k=total)
        ranked = [int(doc.metadata["chunk_index"]) for doc, _ in docs_and_scores]
        if allowed is not None:
            ranked = [i for i in ranked if i in allowed]
        return ranked[:k]

    def _bm25_search(
        self, question: str, k: int, allowed: set[int] | None = None
    ) -> list[int]:
        """按 BM25 分数返回最多 k 个子块下标。

        通过 LangChain 的 BM25Retriever 对整个语料库排序(临时把 .k 设成语料库
        全量大小,拿到完整排名,不影响相对顺序,只影响截断位置),再在 Python
        里按 allowed 过滤,与 _vector_search 的过滤策略保持一致。

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
        self.bm25_retriever.k = len(self.child_chunks)
        docs = self.bm25_retriever.invoke(question)
        ranked = [int(doc.metadata["chunk_index"]) for doc in docs]
        if allowed is not None:
            ranked = [i for i in ranked if i in allowed]
        return ranked[:k]

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

    def _rank_documents(
        self,
        question: str,
        top_k: int = 2,
        category: str | None = None,
        difficulty: str | None = None,
    ) -> list[Recipe]:
        """返回与问题最相关的 top_k 篇父 Recipe 对象。

        检索发生在子块粒度上(标题/食材/步骤分别比较),融合两路互补信号:
        LangChain FAISS 向量库检索(基于 BGE 向量的语义相似度——能让"a quick
        meal"匹配上标了 Easy 难度的菜谱)和 LangChain BM25Retriever(词面
        匹配——能精确命中菜名和食材)。两路排名用 Reciprocal Rank Fusion 融合,
        避免单一信号说了算。

        如果没有显式传入 category/difficulty,会先从问题文本里启发式推断
        (见 _infer_filters)并用来缩小搜索范围;如果这个过滤条件会导致
        一篇菜谱都匹配不上,就丢弃过滤、回退到全库搜索,而不是直接返回空结果。

        父菜谱按其贡献的子块(取自融合排名前列的候选子块)RRF 分数之和排序
        (奖励在多个小节都命中的菜谱,而不是只在一个小节强匹配的菜谱;
        同时仍以匹配强度加权,避免几个弱匹配的小节反而压过一个强匹配)。
        如果候选子块覆盖不到 top_k 篇不同的父菜谱,剩余名额从完整的融合排名里
        依次补齐,确保 retrieve() 总能返回 min(top_k, 匹配到的文档数) 篇结果。

        Args:
            question: 用户的自然语言问题。
            top_k: 最多返回的父菜谱数量。
            category: 可选的显式分类过滤(如 "Vegetable"),覆盖从问题文本
                推断出的结果。
            difficulty: 可选的显式难度过滤(如 "Easy"),覆盖从问题文本
                推断出的结果。

        Returns:
            最多 top_k 篇父 Recipe 对象,按相关性从高到低排序。
        """
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

    @staticmethod
    def _themealdb_dish_query(question: str) -> str:
        """从问题里提取一个接近裸菜名的字符串,供 TheMealDB 名字搜索使用。

        TheMealDB 的 search.php?s= 是按菜名匹配,不是自由文本搜索,
        所以 "How do I make Kung Pao Chicken?" 需要先变成大致
        "Kung Pao Chicken" 这样的形式。

        Args:
            question: 用户的(可能已被改写的)问题。

        Returns:
            去掉开头提问句式(如果有)和结尾 "?" 后的问题文本。
            如果去除后什么都不剩,回退返回原始文本。
        """
        text = question.strip().rstrip("?").strip()
        cleaned = _THEMEALDB_STRIP_RE.sub("", text, count=1).strip()
        return cleaned or text

    @staticmethod
    def _themealdb_get(path: str, params: dict) -> dict | None:
        """向 TheMealDB 某个接口发 GET 请求,返回解析后的 JSON,失败则返回 None。

        永远不抛异常:网络错误、非 2xx 响应、无法解析的响应体都会被记录日志
        并当作"没有数据"处理,这样 TheMealDB 故障只会降级成已有的
        NO_RESULTS_MESSAGE 行为,而不会把 ask() 弄崩。

        Args:
            path: 接口路径,如 "/search.php"。
            params: 请求的查询参数。

        Returns:
            解析后的 JSON 响应体;请求或解析失败则返回 None。
        """
        try:
            response = httpx.get(
                THEMEALDB_BASE_URL + path, params=params, timeout=THEMEALDB_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            logger.warning("TheMealDB request to %s failed.", path, exc_info=True)
            return None

    @staticmethod
    def _themealdb_meal_to_recipe(meal: dict) -> Recipe:
        """把 TheMealDB 返回的菜谱 JSON 对象转换成本项目的 Recipe。

        重新拼出与本地知识库一致的 "# 标题" / "## Ingredients" / "## Steps"
        Markdown 结构,这样这个 Recipe 就能直接复用现有的生成 prompt,
        不用为外部数据再写一套模板。

        Args:
            meal: TheMealDB search/lookup 响应里 "meals" 列表中的一个元素。

        Returns:
            难度固定为 "Unknown"(TheMealDB 不提供难度评级),
            source 设为 "themealdb:<idMeal>" 的 Recipe。
        """
        ingredient_lines = []
        for i in range(1, 21):
            name = (meal.get(f"strIngredient{i}") or "").strip()
            if not name:
                continue
            measure = (meal.get(f"strMeasure{i}") or "").strip()
            ingredient_lines.append(f"- {measure + ' ' if measure else ''}{name}".rstrip())

        step_lines = [
            line.strip(" -")
            for line in re.split(r"\r?\n+", meal.get("strInstructions") or "")
            if line.strip(" -")
        ]
        numbered_steps = [f"{i}. {line}" for i, line in enumerate(step_lines, start=1)]

        dish_name = meal.get("strMeal") or "Unknown dish"
        text = (
            f"# {dish_name}\n\n"
            "## Ingredients\n" + "\n".join(ingredient_lines) + "\n\n"
            "## Steps\n" + "\n".join(numbered_steps)
        )

        return Recipe(
            text=text,
            source=f"themealdb:{meal.get('idMeal', '')}",
            dish_name=dish_name.lower(),
            category=THEMEALDB_CATEGORY_MAP.get(meal.get("strCategory", ""), "Other"),
            difficulty="Unknown",
        )

    def _themealdb_fallback(self, question: str, category: str | None = None) -> Recipe | None:
        """本地知识库查不到答案时,从 TheMealDB 现查一篇菜谱兜底。

        先按菜名做近似搜索(search.php?s=);如果搜不到,且问题推断出了本地分类,
        就改用该分类过滤搜索(filter.php?c=...),并对第一个候选取完整详情
        (lookup.php?i=...)。这是按需调用,不是预先抓取,所以最多只会为回答
        当前问题多发一到两个 API 请求。

        Args:
            question: 用户的(可能已被改写的)问题。
            category: 可选的本地分类标签(如 "Vegetable"),当名字搜索没结果时
                用于选取对应的 TheMealDB 分类过滤。

        Returns:
            根据 TheMealDB 响应构建的 Recipe;如果没查到、兜底被关闭,
            或 API 调用失败,则返回 None。
        """
        if not THEMEALDB_ENABLED:
            return None

        data = self._themealdb_get("/search.php", {"s": self._themealdb_dish_query(question)})
        meals = (data or {}).get("meals") or []

        if not meals:
            themealdb_category = THEMEALDB_LOCAL_TO_CATEGORY.get(category or "")
            if themealdb_category:
                filtered = self._themealdb_get("/filter.php", {"c": themealdb_category})
                candidates = (filtered or {}).get("meals") or []
                if candidates:
                    looked_up = self._themealdb_get(
                        "/lookup.php", {"i": candidates[0]["idMeal"]}
                    )
                    meals = (looked_up or {}).get("meals") or []

        if not meals:
            return None

        logger.info(
            "No local match for %r; falling back to TheMealDB (%s).",
            question,
            meals[0].get("strMeal"),
        )
        return self._themealdb_meal_to_recipe(meals[0])

    def _complete(self, prompt: str, temperature: float = 0.3) -> str | None:
        """发一次单轮对话请求给 DeepSeek(通过 LangChain 的 ChatOpenAI),吞掉
        常见的调用失败异常。

        把这个模块调用 DeepSeek 的 4 处地方(query_router、query_rewrite,
        以及两种基于 prompt 的生成模式)共用的 try/except 逻辑集中在这里,
        这样各处只需要处理"拿到 None 该怎么办",不用各写一遍相同的
        except 代码块。LangChain 的 ChatOpenAI 在底层调用失败时会原样抛出
        openai SDK 的异常(不会吞掉或包装成别的类型),所以这里捕获的异常类型
        与旧版直接用 openai.OpenAI 客户端时完全一致。

        Args:
            prompt: 作为单条用户消息发送的完整 prompt(纯文本,不是 ChatPromptTemplate
                的消息列表——ChatOpenAI.invoke() 接受纯字符串输入,会自动包装成
                一条 HumanMessage)。
            temperature: 采样温度——分类/改写这类需要稳定、字面结果的调用
                用接近 0 的低值;开放式生成用更高的值。通过 invoke() 的调用时
                参数覆盖 self.llm 构造时的默认温度。

        Returns:
            模型的回复文本;调用失败(鉴权/网络/超时/API 错误)或返回空内容
            则返回 None。调用方自行决定 None 对自己意味着什么、要不要记日志。
        """
        try:
            response = self.llm.invoke(prompt, temperature=temperature)
        except AuthenticationError:
            logger.error("DeepSeek authentication failed - check DEEPSEEK_API_KEY.")
            return None
        except (APIConnectionError, APITimeoutError):
            logger.error("DeepSeek API connection/timeout error.")
            return None
        except APIError:
            logger.exception("DeepSeek API returned an error.")
            return None

        content = response.content
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

        JSON 解析用 LangChain 的 parse_json_markdown(langchain_core.output_parsers)
        —— 它本身就是为"LLM 回复有时会把 JSON 包一层 ```json ... ``` 代码块"
        这个场景设计的,比手写正则剥代码块围栏更少踩坑。

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
                data = parse_json_markdown(reply)
                route = str(data.get("route", "")).strip().lower()
                rewritten = str(data.get("rewritten", "")).strip()
            except (json.JSONDecodeError, AttributeError, TypeError):
                route, rewritten = "", ""
            if route in _VALID_ROUTES and rewritten:
                return route, rewritten
            logger.warning("Could not parse classify_and_rewrite reply: %r", reply)
        return ROUTE_GENERAL, question

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
              这样调用方可以在任何内容到达用户之前先悄悄尝试 TheMealDB 兜底——
              见 ask_stream())。
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
        """流式发送单轮对话请求(通过 LangChain 的 ChatOpenAI.stream()),边到达
        边 yield 出文本片段。

        _complete() 的流式版本:失败处理逻辑相同(鉴权/连接/超时/API 错误
        都会被记录并吞掉,而不是抛出),区别是不在最后一次性返回完整文本,
        而是 DeepSeek 每发来一段就 yield 一段(ChatOpenAI.stream() 返回的每个
        chunk 是一个 AIMessageChunk,其 .content 就是这次增量的文本)。如果失败
        发生在第一段内容到达之前,则完全不会产出任何片段;如果发生在响应中途,
        流会提前中止——仅凭这个方法本身无法区分这两种情况(见
        _stream_with_no_results_guard(),它专门跟踪了"到底有没有收到过任何内容")。

        Args:
            prompt: 作为单条用户消息发送的完整 prompt。
            temperature: 采样温度。

        Yields:
            按顺序产出的非空文本片段。
        """
        try:
            for chunk in self.llm.stream(prompt, temperature=temperature):
                delta = chunk.content
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
        转发给客户端,就会出现"没有找到相关信息"这句话已经开始显示、
        ask_stream() 才想起来要悄悄转去用 TheMealDB 重试的尴尬情况。

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
        if not question or not question.strip():
            return "Please enter a question."

        # 快速路径:先尝试用关键词本地分类(零 LLM 调用),再考虑要不要付出一次
        # DeepSeek 往返的代价。命中 "list" 会完全跳过 query_rewrite(list 问题
        # 照旧原样使用);命中 "detail" 仍需要调用 query_rewrite(它的职责——
        # 让问题更具体——与路由分类是相互独立的)。只有两种关键词模式都判不出来的
        # 问题才会落到 LLM,而 LLM 这一次会把分类和改写合并成一次调用完成,
        # 而不是像以前那样分两次单独调用。
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
            return self._generate_list_answer(docs)

        generate = (
            self._generate_step_by_step_answer
            if route == ROUTE_DETAIL
            else self._generate_basic_answer
        )
        answer = generate(question, docs)

        # _rank_documents 总是尽力返回它认为最匹配的 top_k 篇文档——哪怕
        # 其实并不真正相关——而不是返回空列表,因为 RRF 只是对整个(过滤后的)
        # 语料库排序、取前几名。所以"本地库里没有相关内容"不会表现为 docs 为空,
        # 而是表现在这里:LLM 自己判断检索到的内容答不了这个问题(两种生成 prompt
        # 都被要求在这种情况下原样回复 NO_RESULTS_MESSAGE)。只有到这一步,
        # 才值得多花一次网络往返去试试 TheMealDB。
        if answer == NO_RESULTS_MESSAGE:
            category, _ = self._infer_filters(search_query)
            fallback_recipe = self._themealdb_fallback(search_query, category=category)
            if fallback_recipe:
                answer = generate(question, [fallback_recipe])

        return answer

    def ask_stream(self, question: str):
        """ask() 的流式版本:增量地把答案 yield 出去。

        路由/检索/生成模式选择/TheMealDB 兜底逻辑与 ask() 完全一致(刻意保留
        为一份独立实现,而不是让其中一个委托给另一个,这样 ask() 里非流式的
        DeepSeek 调用、以及测试里对它的 mock,都不受影响);区别只在于所选生成
        模式产出结果的方式:纯列表回答或固定提示语作为一个整体 yield 出去
        (反正不调用 LLM,没什么好流式的),而 "detail"/"general" 的回答会随着
        DeepSeek 逐步生成而增量 yield,使用
        _generate_step_by_step_answer_stream()/_generate_basic_answer_stream(),
        并沿用与 ask() 相同的、由 NO_RESULTS_MESSAGE 触发的 TheMealDB 兜底
        (兜底过程如何对调用方保持"无感",而不是先闪一下"没找到"再切换到真正的
        答案,见 _stream_with_no_results_guard())。

        Args:
            question: 用户的自然语言问题。

        Yields:
            按顺序产出的答案文本片段;把所有片段拼起来,得到的字符串
            与同样问题下 ask() 会返回的结果完全一致。
        """
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
            category, _ = self._infer_filters(search_query)
            fallback_recipe = self._themealdb_fallback(search_query, category=category)
            if fallback_recipe:
                fallback_state: dict = {}
                for chunk in generate_stream(question, [fallback_recipe], fallback_state):
                    yield chunk
                if fallback_state.get("is_no_results"):
                    # 兜底的这篇菜谱同样答不了问题(与 ask() 一致:不再做第三次
                    # 尝试,直接把两次都被按住的提示语放出来)。
                    yield NO_RESULTS_MESSAGE
            else:
                yield NO_RESULTS_MESSAGE
