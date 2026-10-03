# RAGChef 去 LangChain 轻量化方案

## 结论先行

LangChain 在这个项目里只是**薄封装层**，没有用到它的 Chain/Agent 这类重逻辑，所以去掉它风险很低、改动可以很局部：

- `_infer_route`、`_infer_filters`、`_rrf_fuse`、`_rank_documents`、路由分发、`ask()`/`ask_stream()` 的整体流程 —— **完全不用动**，这些本来就是手写的业务逻辑。
- 需要动的只有 6 个点：`ChatOpenAI`、`HuggingFaceEmbeddings`、`FAISS` 向量库封装、`BM25Retriever`、`PromptTemplate`、`parse_json_markdown`。
- 依赖上，`faiss-cpu`、`rank-bm25`、`sentence-transformers`、`torch` 本来就在 `requirements.txt` 里（LangChain 只是包了一层调用方式），所以卸载 LangChain 系列包本身不会砍掉任何"能力"，只是把中间这层薄封装换成直接调用底层库。

下面按文件、按改动点给出完整的替换代码。

---

## 1. `server/requirements.txt`

删掉这 5 行：

```
langchain
langchain-core
langchain-openai
langchain-community
langchain-huggingface
```

改完后：

```
fastapi
uvicorn
openai
httpx
python-dotenv

faiss-cpu
rank-bm25
sentence-transformers

--extra-index-url https://download.pytorch.org/whl/cpu
torch
```

`requirements-dev.txt` 不用改（它只是 `-r requirements.txt` + pytest + httpx）。

改完后本地执行：

```bash
cd server
pip uninstall -y langchain langchain-core langchain-openai langchain-community langchain-huggingface
pip install -r requirements.txt
```

---

## 2. `server/rag.py` —— import 部分

**删掉：**

```python
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.output_parsers.json import parse_json_markdown
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
```

**换成：**

```python
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
)
```

（`openai` 那组异常类型本来就是从 `openai` 包直接 import 的，不用动；这里只是把 `ChatOpenAI` 换成从同一个包 import 原生的 `OpenAI` 客户端。）

---

## 3. Prompt 模板：`PromptTemplate` → 纯字符串

`PromptTemplate.from_template(x).format(**kwargs)` 本质上就是 `x.format(**kwargs)`，模板里也只用了标准的 `{question}`/`{context}`/`{no_results_message}` 占位符，没用到 LangChain 模板的任何特殊功能。所以**调用点完全不用改**，只改 5 处定义，去掉 `PromptTemplate.from_template(...)` 这层包装：

```python
ROUTER_PROMPT = """Classify the user's question into exactly one category. Reply with only the category name, nothing else.
...
Category:"""

REWRITE_PROMPT = """Decide whether this recipe-assistant question needs rewriting to be more specific before searching a recipe database.
...
Rewritten question:"""

CLASSIFY_AND_REWRITE_PROMPT = """You are helping a recipe assistant prepare to answer a question. Do two things:
...
User question: {question}"""

STEP_BY_STEP_PROMPT = """You are a professional recipe assistant. Answer the question using ONLY the recipe content below.
...
"""

BASIC_ANSWER_PROMPT = """You are a professional recipe assistant.
...
"""
```

原来 `CLASSIFY_AND_REWRITE_PROMPT` 里因为要在 f-string/`.format()` 里输出字面量 `{"route": ...}`，用了 `{{`/`}}` 转义——这部分原样保留，`str.format()` 转义规则和 `PromptTemplate` 是同一套。

调用点（`ROUTER_PROMPT.format(question=question)` 等）不用改一个字。

---

## 4. `parse_json_markdown` → 自己写的小函数

在模块里新增（放在 `_TOKEN_RE` 附近即可）：

```python
# LLM 回复有时会把 JSON 包一层 ```json ... ``` 代码块，这里剥掉围栏再解析；
# 剥不出代码块就当作回复本身是裸 JSON。
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_json_reply(text: str) -> dict:
    match = _JSON_FENCE_RE.search(text)
    candidate = match.group(1) if match else text.strip()
    return json.loads(candidate)
```

`_classify_and_rewrite` 里：

```python
try:
    data = _parse_json_reply(reply)
    route = str(data.get("route", "")).strip().lower()
    rewritten = str(data.get("rewritten", "")).strip()
except (json.JSONDecodeError, AttributeError, TypeError):
    route, rewritten = "", ""
```

（原来 catch 的异常类型不变，`json.loads` 失败同样抛 `json.JSONDecodeError`。）

---

## 5. `ChatOpenAI` → 原生 `openai.OpenAI` 客户端

`__init__` 里：

```python
self.model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
self.llm = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
```

（原来 `temperature=0.3` 是构造时的默认值，`ChatOpenAI.invoke()`/`.stream()` 调用时传的 `temperature` 会覆盖它；原生客户端没有"构造时默认 temperature"这个概念，所以改成每次调用都显式传 `temperature`，行为完全一致——反正原代码里也从没依赖过"不传 temperature 时用构造时默认值"这件事，四个调用点全都显式传了。）

`_complete`：

```python
def _complete(self, prompt: str, temperature: float = 0.3) -> str | None:
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
```

`_raw_stream_complete`：

```python
def _raw_stream_complete(self, prompt: str, temperature: float = 0.3):
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
```

异常类型完全不用变：`ChatOpenAI` 底层本来就是用 `openai` 官方 SDK 发请求，失败时原样抛出 SDK 异常，换成直连客户端后异常类型不变。

---

## 6. `HuggingFaceEmbeddings` → `sentence_transformers.SentenceTransformer` 直调

`__init__` 里：

```python
self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
```

`HuggingFaceEmbeddings` 内部本来就是委托给 `sentence-transformers` 做实际编码的，换成直调之后数值结果应该是一致的（同一个模型、同样的 `normalize_embeddings=True`）。

编码文档（建索引时）：

```python
vectors = self.embedder.encode(
    self.child_chunks, normalize_embeddings=True, show_progress_bar=False
)
vectors = np.asarray(vectors, dtype="float32")
```

编码查询（检索时）：

```python
query_vec = self.embedder.encode([question], normalize_embeddings=True)
query_vec = np.asarray(query_vec, dtype="float32")
```

---

## 7. `FAISS` 向量库封装 → `faiss-cpu` 原生 API

这是改动量最大的一处，但逻辑不复杂：LangChain 的 `FAISS` 向量库内部也就是一个 `faiss.IndexFlatIP`/`IndexFlatL2` + 一份 docstore 映射；这里因为子块下标本来就是顺序 0..N-1 地加进索引的，**不需要额外的 docstore**——FAISS 返回的行号本身就是 `chunk_index`，比 LangChain 版本反而更简单。

`_build_or_load_vector_store`：

```python
def _build_or_load_vector_store(self, data_path: str) -> "faiss.Index":
    """构建/加载 child_chunks 的 faiss 向量索引（原生 faiss-cpu，无 docstore 封装）。

    子块是按顺序 0..N-1 加入索引的，所以 faiss 返回的行号就是 chunk_index，
    不需要额外维护一份 id 映射。
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
                # 缓存文件损坏，或来自旧版(LangChain 封装)的索引格式 —— 兜底
                # 重建，而不是让启动直接崩溃。
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
```

`_vector_search`：

```python
def _vector_search(
    self, question: str, k: int, allowed: set[int] | None = None
) -> list[int]:
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
```

（`self.vector_store` 这个属性名可以保留不改，反正现在只有 `_build_or_load_vector_store`/`_vector_search` 两处用到它，类型从 LangChain 的 `FAISS` 换成了原生 `faiss.Index`，外部完全无感知。）

### 关于旧缓存文件 `data/vector_index/index.faiss`

`FAISS.save_local()` 内部本来就是调用 `faiss.write_index()` 写这个 `.faiss` 文件的（`index.pkl` 才是 LangChain 自己的 docstore，你项目里目前没有这个文件，只有 `index.faiss` + `meta.json`），所以**大概率能直接用 `faiss.read_index()` 读出来，不用重建**。

保险起见：迁移后第一次启动前，建议先手动删掉 `server/data/vector_index/` 整个目录，让它自动重建一次（50 篇菜谱重新 embedding 也就几秒钟），避免历史格式细节的边角问题；`meta.json` 的 `content_hash` 机制本来就是为了自动检测"要不要重建"设计的，重建这条路径本来就有完整测试覆盖（`test_retrieve.py` 里每个测试都会触发一次 `SimpleRAG()` 构造）。

---

## 8. `BM25Retriever` → `rank_bm25.BM25Okapi` 直调

`__init__` 里，删掉：

```python
metadatas = [{"chunk_index": i} for i in range(len(self.child_chunks))]
self.bm25_retriever = BM25Retriever.from_texts(
    self.child_chunks, metadatas=metadatas, preprocess_func=self._tokenize
)
```

换成：

```python
tokenized_chunks = [self._tokenize(chunk) for chunk in self.child_chunks]
self.bm25 = BM25Okapi(tokenized_chunks)
```

`_bm25_search`：

```python
def _bm25_search(
    self, question: str, k: int, allowed: set[int] | None = None
) -> list[int]:
    if k <= 0 or not self.child_chunks:
        return []
    scores = self.bm25.get_scores(self._tokenize(question))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if allowed is not None:
        ranked = [i for i in ranked if i in allowed]
    return ranked[:k]
```

（原来 `self.bm25_retriever.k = len(self.child_chunks)` 这行是因为 LangChain 的 `BM25Retriever` 只能通过 `.k` 参数控制截断，需要临时改成全量再截断；`BM25Okapi.get_scores()` 本身就直接返回全量分数数组，不用这个 hack 了。）

---

## 9. 测试文件 `server/tests/test_ask_mocked.py`

这个文件里所有 `monkeypatch.setattr(ChatOpenAI, "invoke"/"stream", ...)` 都要改。思路：因为原生 `openai.OpenAI()` 客户端不是 pydantic 模型，可以直接在**实例**上 patch `rag.llm.chat.completions.create`，反而比之前在类上 patch 更直接。

**文件头部**，删掉：

```python
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_openai import ChatOpenAI
```

**新增 fake 响应的辅助类/函数**（替换原来的 `_fake_invoke`/`_fake_stream`）：

```python
class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _fake_create(text):
    """伪造一次非流式 chat.completions.create() 调用，返回结构与真实 openai SDK
    响应一致（.choices[0].message.content）。"""

    def create(*args, **kwargs):
        return _FakeResponse(text)

    return create


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChunk:
    def __init__(self, content):
        self.choices = [types.SimpleNamespace(delta=_FakeDelta(content))]


def _fake_stream_create(*texts):
    """伪造一次流式 chat.completions.create(stream=True) 调用，逐块吐出
    _FakeChunk（每块 .choices[0].delta.content 是这次增量的文本）。"""

    def create(*args, **kwargs):
        return (_FakeChunk(t) for t in texts)

    return create


def _fake_create_raises(error):
    """伪造一次直接抛出异常的 create() 调用（鉴权/连接失败等），
    非流式、流式场景通用——因为 _complete/_raw_stream_complete 都是把
    整个 create() 调用包在 try/except 里的。"""

    def create(*args, **kwargs):
        raise error

    return create
```

文件顶部 `import` 需要加一行 `import types`（给 `types.SimpleNamespace` 用，也可以换成自己写一个 `_FakeStreamChoice` 类，效果一样）。

**逐个测试的替换规则：**

- `monkeypatch.setattr(ChatOpenAI, "invoke", _fake_invoke("xxx"))`
  → `monkeypatch.setattr(rag.llm.chat.completions, "create", _fake_create("xxx"))`

- `monkeypatch.setattr(ChatOpenAI, "stream", _fake_stream("a", "b"))`
  → `monkeypatch.setattr(rag.llm.chat.completions, "create", _fake_stream_create("a", "b"))`

- 抛异常的 `raise_error`/`fail` 函数（原来是定义在 `ChatOpenAI.invoke`/`.stream` 签名下的方法）
  → 直接用 `_fake_create_raises(error)`，例如：
  ```python
  monkeypatch.setattr(rag.llm.chat.completions, "create", _fake_create_raises(error))
  ```

- `test_ask_passes_configured_model`（原来断言 `rag.llm.model_name == rag.model`，原生客户端没有这个属性）改成断言 `_complete()` 真的把 `rag.model` 当 `model=` 参数传给了 `create()`：
  ```python
  def test_ask_passes_configured_model(monkeypatch):
      rag = _make_rag()
      seen = {}

      def fake_create(*, model, messages, temperature, **kwargs):
          seen["model"] = model
          return _FakeResponse("ok")

      monkeypatch.setattr(rag.llm.chat.completions, "create", fake_create)
      rag._complete("irrelevant prompt")

      assert seen["model"] == rag.model
  ```

- `test_complete_passes_call_time_temperature` 里的 `fake_invoke` 同理改成捕获 `create()` 的 `temperature` 参数：
  ```python
  def test_complete_passes_call_time_temperature(monkeypatch):
      rag = _make_rag()
      seen = {}

      def fake_create(*, model, messages, temperature, **kwargs):
          seen["temperature"] = temperature
          return _FakeResponse("ok")

      monkeypatch.setattr(rag.llm.chat.completions, "create", fake_create)
      rag._complete("irrelevant prompt", temperature=0.2)

      assert seen["temperature"] == 0.2
  ```

- 需要统计调用次数的测试（`test_ask_rule_matched_detail_route_still_calls_rewrite_and_generate`）：`fake_invoke` 改成 `fake_create`，逻辑不变：
  ```python
  def fake_create(*args, **kwargs):
      call_count["n"] += 1
      return _FakeResponse("How do I make Kung Pao Chicken?")

  monkeypatch.setattr(rag.llm.chat.completions, "create", fake_create)
  ```

- 同时需要 mock invoke 和 stream 两处的测试（`test_ask_stream_detail_route_joins_to_full_answer`、`test_ask_stream_yields_no_results_message_when_llm_finds_nothing_relevant`）：现在两者都走同一个 `create()` 方法（区别只是 `stream=True` 参数），**不能像以前那样分别 patch 两个不同方法名**。改法：用 `kwargs.get("stream")` 在同一个 fake 里分流：
  ```python
  def test_ask_stream_detail_route_joins_to_full_answer(monkeypatch):
      rag = _make_rag()

      def fake_create(*args, stream=False, **kwargs):
          if stream:
              return (_FakeChunk(t) for t in ["## Overview\n", "A classic dish."])
          return _FakeResponse("How do I make Kung Pao Chicken?")

      monkeypatch.setattr(rag.llm.chat.completions, "create", fake_create)

      chunks = list(rag.ask_stream("How do I make Kung Pao Chicken?"))

      assert len(chunks) >= 1
      assert "".join(chunks) == "## Overview\nA classic dish."
  ```
  同样的模式用在 `test_ask_stream_yields_no_results_message_when_llm_finds_nothing_relevant`。

其余测试（`query_router`/`query_rewrite`/`_classify_and_rewrite`/`_infer_route` 相关的用例）本来就是直接 mock `rag._complete`，完全不涉及 `ChatOpenAI`，**不用改**。

`test_ask_detail_route_produces_structured_sections` 是普通的单次非流式 mock，按第一条规则改 `_fake_create` 即可。

---

## 10. `README.md` / `rag.py` 模块开头注释

`rag.py` 顶部那段大段中文 docstring（第 1-35 行）通篇在讲"哪些环节用了 LangChain、哪些没用"，改完之后这段话的前提就不成立了，需要重写成"哪些环节手写、哪些环节调用了 sentence-transformers/faiss-cpu/rank-bm25/openai 原生库"。`README.md` 里凡是提到 `[LangChain]`、`langchain_huggingface`、`langchain_community`、`BM25Retriever`、`ChatOpenAI`、`PromptTemplate` 的措辞（Features 部分、Structure 部分）都要同步改成对应的原生库名字。这部分是纯文档改动，不影响功能，建议放在功能改完、测试跑通之后最后处理。

---

## 11. 建议的操作顺序（每步都可独立验证，出问题好定位）

1. 替换 §3 Prompt 模板定义 —— 跑一下 `python -c "import rag"` 确认没语法错误。
2. 替换 §4 JSON 解析函数。
3. 替换 §5 LLM 客户端（`ChatOpenAI` → `OpenAI`），同步改 `_complete`/`_raw_stream_complete`。
4. 替换 §9 测试文件里跟 LLM 相关的 mock，先跑 `pytest tests/test_ask_mocked.py`，确认路由/生成/流式这条链路没问题（这一步还没碰 embedding/FAISS/BM25，`test_retrieve.py` 应该仍然全绿，因为它不 mock LLM）。
5. 删除 `server/data/vector_index/` 目录（为下一步重建做准备）。
6. 替换 §6 Embedding、§7 FAISS、§8 BM25。
7. 跑 `pytest tests/test_retrieve.py`，重点看检索相关的用例（排序、分类/难度过滤、父子分块聚合）是否还符合预期——这是最容易因为数值细节（比如相似度分数量纲）出问题的一步。
8. 全量 `pytest`，再跑一次 `python cli.py` 手动问几个问题（比如"How do I make Kung Pao Chicken?"、"recommend a few vegetarian dishes"），和迁移前的回答对比一下检索到的菜谱是否一致。
9. 更新 §1 `requirements.txt`，`pip uninstall` 掉 langchain 系列包，确认 `pip install -r requirements.txt` 后服务能正常起（`uvicorn app:app --reload` + `curl` 一下 `/ask`）。
10. 最后处理 §10 文档措辞。

## 风险点小结

- **数值差异**：`SentenceTransformer` 直调和 `HuggingFaceEmbeddings` 内部实际上走的是同一份 encode 逻辑，理论上向量应该一致；但如果 §7 步跑测试后发现排序有细微变化，大概率是 `normalize_embeddings`/`show_progress_bar` 之类的 `encode()` 参数没对齐，照抄本方案里的参数即可。
- **异常类型**：`ChatOpenAI` 本来就是 openai 官方 SDK 的薄封装，异常类型（`AuthenticationError` 等）迁移前后完全一致，`except` 块不用改类型，只需要确认 import 来源仍然是 `openai` 包（本来就是）。
- **向量索引缓存**：`index.faiss` 文件格式大概率兼容（见 §7 说明），但保险起见建议第一次迁移后删掉缓存目录重建一次，避免历史边角问题；重建对 50 篇菜谱规模来说是秒级操作。
