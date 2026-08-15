# ============================================================
# 中文导读:
# 这份测试文件覆盖 rag.py 里"路由分类 + 问题改写 + 生成 + 流式输出"这一整条
# 链路,但完全不联网、不真的调用 DeepSeek —— 所有测试都用 pytest 的
# monkeypatch 把 LangChain 的 ChatOpenAI.invoke/.stream(或更上层的
# rag._complete)替换成假方法,返回预先写好的假回复,这样测试跑得快、结果稳定
# 可重复,也能精确控制"LLM 这次回复了什么/失败了没"来触发各种分支。
#
# 为什么 patch 在 ChatOpenAI 类上,而不是 rag.llm 实例上?
# ChatOpenAI 是 pydantic 模型,实例上只能设置模型声明过的字段,不能像给普通
# 对象挂一个新属性那样直接 `rag.llm.invoke = ...`;在类上打 monkeypatch 则是
# 给这一个方法名换实现,对 rag.llm 这个实例同样生效,pytest 的 monkeypatch
# 在每个测试结束后会自动还原,不会污染后续测试。
#
# 测试大致分三类:
#   1. 普通问答 + 异常兜底(开头到 test_ask_detail_route_produces_structured_sections)
#      —— 验证 ask() 在各种 LLM 返回/失败场景下的行为符合预期。
#   2. 路由/改写相关的单元测试 —— 单独验证 query_router / query_rewrite /
#      _classify_and_rewrite / _infer_route 这些子逻辑各自的正确性。
#   3. 流式相关(文件底部 "Streaming" 分隔线之后)—— 验证 ask_stream() 和
#      它依赖的 _raw_stream_complete / _stream_with_no_results_guard,
#      尤其是"流式输出时如何按住可能是 NO_RESULTS_MESSAGE 前缀的内容,
#      不逐字显示给用户"这个比较绕的逻辑。
# ============================================================

import os

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, AuthenticationError

from rag import (
    ROUTE_DETAIL,
    ROUTE_GENERAL,
    ROUTE_LIST,
    NO_RESULTS_MESSAGE,
    SimpleRAG,
    LLM_UNAVAILABLE_MESSAGE,
)

RECIPES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recipes")


# 中文: 构建一个真实的 SimpleRAG 实例(会真的加载本地菜谱、建向量/BM25 索引),但后面每个测试会用 monkeypatch 替换掉 ChatOpenAI 的
#      invoke/stream 方法,所以不会真的联网调用 LLM。
def _make_rag():
    return SimpleRAG(RECIPES_PATH)


# 中文: 伪造一次非流式调用的假 invoke(),模拟 ChatOpenAI.invoke() 正常返回时的
#      结构(一个带 .content 的 AIMessage)。
def _fake_invoke(text):
    def invoke(self, *args, **kwargs):
        return AIMessage(content=text)

    return invoke


# 中文: 伪造一次流式调用的假 stream(),模拟 ChatOpenAI.stream() 逐块吐出
#      AIMessageChunk(每个 chunk 的 .content 是这次增量的文本)。
def _fake_stream(*texts):
    def stream(self, *args, **kwargs):
        for t in texts:
            yield AIMessageChunk(content=t)

    return stream


# 中文: 验证 ask() 会把 mock 出来的 LLM 回复原样返回,且全程没有真的发网络请求。
def test_ask_returns_llm_answer_without_hitting_network(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(
        ChatOpenAI, "invoke", _fake_invoke("Mocked answer about Kung Pao Chicken.")
    )

    answer = rag.ask("How do I make Kung Pao Chicken?")

    assert answer == "Mocked answer about Kung Pao Chicken."


# 中文: 验证 rag.llm(LangChain 的 ChatOpenAI)确实是用 rag.model(配置里读到的模型名)构造的,没有被写死或传错。
def test_ask_passes_configured_model():
    rag = _make_rag()

    assert rag.llm.model_name == rag.model


# 中文: 验证 _complete() 调用 ChatOpenAI.invoke() 时,确实把调用方指定的 temperature 当作调用时参数传了下去(而不是只用构造时的默认温度)。
def test_complete_passes_call_time_temperature(monkeypatch):
    rag = _make_rag()
    seen = {}

    def fake_invoke(self, prompt, **kwargs):
        seen.update(kwargs)
        return AIMessage(content="ok")

    monkeypatch.setattr(ChatOpenAI, "invoke", fake_invoke)
    rag._complete("irrelevant prompt", temperature=0.2)

    assert seen["temperature"] == 0.2


# 中文: 验证空白问题(只有空格)会被提前拦截,直接返回提示语,完全不会触发 LLM 调用——用一个会抛异常的假 invoke() 来断言"LLM 不该被调用"。
def test_ask_empty_question_never_calls_llm(monkeypatch):
    rag = _make_rag()

    def fail(self, *args, **kwargs):
        raise AssertionError("LLM should not be called for an empty question")

    monkeypatch.setattr(ChatOpenAI, "invoke", fail)

    answer = rag.ask("   ")

    assert "enter a question" in answer.lower()


@pytest.mark.parametrize(
    "error",
    [
        AuthenticationError(
            message="bad key",
            response=httpx.Response(401, request=httpx.Request("POST", "https://api.deepseek.com")),
            body=None,
        ),
        APIConnectionError(request=httpx.Request("POST", "https://api.deepseek.com")),
    ],
)
# 中文: 用 pytest.mark.parametrize 分别模拟"鉴权失败"和"连接失败"两种异常,验证不管哪种,ask() 都不会把异常抛出去,而是统一返回
#      LLM_UNAVAILABLE_MESSAGE 这句对用户友好的提示。LangChain 的 ChatOpenAI 在调用失败时会原样抛出这些 openai SDK 异常,不会吞掉或包装。
def test_ask_returns_friendly_message_on_llm_failure(monkeypatch, error):
    rag = _make_rag()

    def raise_error(self, *args, **kwargs):
        raise error

    monkeypatch.setattr(ChatOpenAI, "invoke", raise_error)

    answer = rag.ask("How do I make Kung Pao Chicken?")

    assert answer == LLM_UNAVAILABLE_MESSAGE


# 中文: 验证如果 LLM 回复了一个不认识的分类结果(比如乱回复的文本),query_router() 会兜底成最安全的 general,而不是报错或返回垃圾值。
def test_query_router_returns_general_on_unparseable_response(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "_complete", lambda prompt, temperature=0.3: "not a real route")

    assert rag.query_router("anything") == ROUTE_GENERAL


# 中文: 验证 LLM 调用直接失败(_complete 返回 None)时,query_router() 同样兜底成 general。
def test_query_router_returns_general_when_llm_call_fails(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "_complete", lambda prompt, temperature=0.3: None)

    assert rag.query_router("anything") == ROUTE_GENERAL


# 中文: 验证 LLM 回复里即使有多余的大小写/空格(" List "),也能被正确解析成 ROUTE_LIST。
def test_query_router_returns_classified_route(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "_complete", lambda prompt, temperature=0.3: " List ")

    # Case/whitespace in the raw LLM reply shouldn't matter.
    assert rag.query_router("recommend a few dishes") == ROUTE_LIST


# 中文: 验证问题改写调用失败时,query_rewrite() 会原样返回用户的原始问题,而不是返回空字符串或抛异常。
def test_query_rewrite_falls_back_to_original_question_on_failure(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "_complete", lambda prompt, temperature=0.3: None)

    assert rag.query_rewrite("give me something to cook") == "give me something to cook"


# 中文: 验证"list"模式格式化菜名列表时会自动去重(同一道菜出现两次只列一次)。
def test_generate_list_answer_lists_dish_names_without_duplicates():
    rag = _make_rag()
    kung_pao = next(d for d in rag.documents if d.dish_name == "kung pao chicken")
    dumplings = next(d for d in rag.documents if d.dish_name == "dumplings jiaozi")

    answer = rag._generate_list_answer([kung_pao, dumplings, kung_pao])

    assert answer.count("Kung Pao Chicken") == 1
    assert "Dumplings Jiaozi" in answer


# 中文: 验证检索结果为空时,list 模式的回答直接是 NO_RESULTS_MESSAGE,不会生成奇怪的空列表文案。
def test_generate_list_answer_empty_docs_returns_no_results_message():
    rag = _make_rag()
    assert rag._generate_list_answer([]) == NO_RESULTS_MESSAGE


# 中文: 验证"推荐几个甜品"这种能被关键词规则(_infer_route)直接命中 list 路由的问题,全程一次 LLM 都不调用(list 生成是纯 Python 格式化)。
def test_ask_rule_matched_list_route_never_calls_llm_at_all(monkeypatch):
    rag = _make_rag()

    def fail(self, *args, **kwargs):
        raise AssertionError(
            "LLM should not be called: 'recommend a few' is caught by "
            "_infer_route(), and list generation is pure Python."
        )

    monkeypatch.setattr(ChatOpenAI, "invoke", fail)

    answer = rag.ask("Recommend a few dessert recipes")

    assert answer.startswith("Here are some recipes you might like:")


# 中文: 验证"How do I make ..."这种被关键词规则直接命中 detail 路由的问题,虽然省了路由分类这一次 LLM
#      调用,但改写(query_rewrite)和最终生成各自还是要各调用一次 LLM,一共 2 次(而不是老版本的 3 次,也不是被误判成 0 次)。
def test_ask_rule_matched_detail_route_still_calls_rewrite_and_generate(monkeypatch):
    # "How do I make ..." is caught by _infer_route() as ROUTE_DETAIL, so the
    # router call is skipped, but query_rewrite and generation still each
    # need their own LLM call -- 2 calls total, not 0 and not the old 3.
    rag = _make_rag()
    call_count = {"n": 0}

    def fake_invoke(self, *args, **kwargs):
        call_count["n"] += 1
        return AIMessage(content="How do I make Kung Pao Chicken?")

    monkeypatch.setattr(ChatOpenAI, "invoke", fake_invoke)

    rag.ask("How do I make Kung Pao Chicken?")

    assert call_count["n"] == 2


# 中文: 验证包含"recommend"这类关键词的问题能被 _infer_route 正确识别成 list。
def test_infer_route_matches_list_trigger():
    rag = _make_rag()
    assert rag._infer_route("Recommend a few vegetarian dishes") == "list"


# 中文: 验证包含"how do I make"这类关键词的问题能被 _infer_route 正确识别成 detail。
def test_infer_route_matches_detail_trigger():
    rag = _make_rag()
    assert rag._infer_route("How do I make Kung Pao Chicken?") == "detail"


# 中文: 验证一个语义上既不像 list 也不像 detail 的问题,_infer_route 会老实返回 None(交给 LLM 兜底),而不是强行猜一个错误分类。
def test_infer_route_returns_none_for_ambiguous_question():
    rag = _make_rag()
    assert rag._infer_route("What's the difference between a casserole and a hotpot?") is None


# 中文: 验证 LLM 回复标准 JSON({"route":..., "rewritten":...})时能被正确解析出 route 和改写后的问题。
def test_classify_and_rewrite_parses_valid_json(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(
        rag,
        "_complete",
        lambda prompt, temperature=0: '{"route": "general", "rewritten": "what is al dente"}',
    )

    route, rewritten = rag._classify_and_rewrite("what does al dente mean?")

    assert route == ROUTE_GENERAL
    assert rewritten == "what is al dente"


# 中文: 验证 LLM 有时会把 JSON 包在 ```json ... ``` 代码块里回复,这种情况也能被 LangChain 的 parse_json_markdown 正确剥掉围栏后解析成功。
def test_classify_and_rewrite_strips_markdown_code_fence(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(
        rag,
        "_complete",
        lambda prompt, temperature=0: '```json\n{"route": "list", "rewritten": "easy dishes"}\n```',
    )

    route, rewritten = rag._classify_and_rewrite("something easy")

    assert route == ROUTE_LIST
    assert rewritten == "easy dishes"


# 中文: 验证 LLM 回复的不是合法 JSON 时,会兜底成 (ROUTE_GENERAL, 原问题),而不是抛异常崩掉。
def test_classify_and_rewrite_falls_back_on_invalid_json(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "_complete", lambda prompt, temperature=0: "not json at all")

    route, rewritten = rag._classify_and_rewrite("what does al dente mean?")

    assert route == ROUTE_GENERAL
    assert rewritten == "what does al dente mean?"


# 中文: 验证 LLM 调用本身失败(返回 None)时,同样兜底成 (ROUTE_GENERAL, 原问题)。
def test_classify_and_rewrite_falls_back_when_llm_call_fails(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "_complete", lambda prompt, temperature=0: None)

    route, rewritten = rag._classify_and_rewrite("what does al dente mean?")

    assert route == ROUTE_GENERAL
    assert rewritten == "what does al dente mean?"


# 中文: 验证一个关键词规则判断不出类型的问题,会走"分类+改写合并成一次调用"的兜底路径,而不是分开打两次 LLM,全程一共只有 2 次 LLM 调用(1 次分类改写 + 1
#      次生成)。
def test_ask_ambiguous_question_uses_combined_classify_and_rewrite_call(monkeypatch):
    # No list/detail trigger words, so _infer_route() returns None and ask()
    # must fall back to the single combined LLM call instead of two separate
    # router/rewrite calls.
    rag = _make_rag()
    call_count = {"n": 0}

    def fake_complete(prompt, temperature=0.3):
        call_count["n"] += 1
        return '{"route": "general", "rewritten": "what does al dente mean"}'

    monkeypatch.setattr(rag, "_complete", fake_complete)

    rag.ask("What does al dente mean?")

    # 1 call for the combined classify+rewrite, 1 for generation.
    assert call_count["n"] == 2


# 中文: 验证 detail 路由生成的回答确实包含 "## Ingredients""## Steps" 这些约定好的结构化小节标题,即 prompt 模板起作用了。
def test_ask_detail_route_produces_structured_sections(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "query_router", lambda question: ROUTE_DETAIL)
    monkeypatch.setattr(rag, "query_rewrite", lambda question: question)
    monkeypatch.setattr(
        ChatOpenAI,
        "invoke",
        _fake_invoke(
            "## Overview\nA classic dish.\n## Ingredients\n- chicken\n## Steps\n1. Cook it.\n## Tips\nServe hot."
        ),
    )

    answer = rag.ask("How do I make Kung Pao Chicken?")

    assert "## Ingredients" in answer
    assert "## Steps" in answer


# ---------------------------------------------------------------------------
# Streaming (_raw_stream_complete / _stream_with_no_results_guard / ask_stream)
# ---------------------------------------------------------------------------


# 中文: 验证最底层的流式调用 _raw_stream_complete 会按顺序把每个 chunk 的 .content 依次 yield 出来。
def test_raw_stream_complete_yields_deltas_in_order(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(ChatOpenAI, "stream", _fake_stream("Hello", " world"))

    assert list(rag._raw_stream_complete("irrelevant prompt")) == ["Hello", " world"]


# 中文: 验证鉴权失败时流式调用不会抛异常炸到调用方,而是安静地什么都不 yield(空列表)。
def test_raw_stream_complete_yields_nothing_on_auth_failure(monkeypatch):
    rag = _make_rag()

    def raise_error(self, *args, **kwargs):
        raise AuthenticationError(
            message="bad key",
            response=httpx.Response(401, request=httpx.Request("POST", "https://api.deepseek.com")),
            body=None,
        )
        yield  # pragma: no cover - makes this a generator function, never reached

    monkeypatch.setattr(ChatOpenAI, "stream", raise_error)

    assert list(rag._raw_stream_complete("irrelevant prompt")) == []


# 中文: 验证 _stream_with_no_results_guard 的核心行为:一旦缓冲区的内容不可能再是 NO_RESULTS_MESSAGE 的前缀了("No
#      relevant thing"和真正的消息在"thing"处就分道扬镳),就把之前攒住的内容一次性吐出来,后面的内容直接透传,不再缓冲。
def test_stream_guard_flushes_buffer_once_diverged(monkeypatch):
    rag = _make_rag()
    # "No relevant thing" stops being a possible NO_RESULTS_MESSAGE prefix
    # partway through (the real message continues "...information..."), so
    # everything buffered up to that point should flush as one chunk, then
    # the rest should stream straight through unbuffered.
    monkeypatch.setattr(
        ChatOpenAI, "stream", _fake_stream("No", " relevant", " thing", " here")
    )

    state = {}
    chunks = list(rag._stream_with_no_results_guard("prompt", 0.3, state))

    assert chunks == ["No relevant thing", " here"]
    assert state == {}


# 中文: 验证如果流式吐出来的内容从头到尾完整匹配 NO_RESULTS_MESSAGE,那么什么都不会流给调用方(chunks 为空),而是通过 state 标记
#      is_no_results=True。
def test_stream_guard_detects_exact_no_results_message(monkeypatch):
    rag = _make_rag()
    half = len(NO_RESULTS_MESSAGE) // 2
    monkeypatch.setattr(
        ChatOpenAI,
        "stream",
        _fake_stream(NO_RESULTS_MESSAGE[:half], NO_RESULTS_MESSAGE[half:]),
    )

    state = {}
    chunks = list(rag._stream_with_no_results_guard("prompt", 0.3, state))

    # Nothing should reach the caller.
    assert chunks == []
    assert state == {"is_no_results": True}


# 中文: 验证底层调用直接失败时(比如鉴权错误),state 会被标记 failed=True,且同样不会有任何内容流出去。
def test_stream_guard_marks_failed_when_call_fails(monkeypatch):
    rag = _make_rag()

    def raise_error(self, *args, **kwargs):
        raise AuthenticationError(
            message="bad key",
            response=httpx.Response(401, request=httpx.Request("POST", "https://api.deepseek.com")),
            body=None,
        )
        yield  # pragma: no cover - makes this a generator function, never reached

    monkeypatch.setattr(ChatOpenAI, "stream", raise_error)

    state = {}
    chunks = list(rag._stream_with_no_results_guard("prompt", 0.3, state))

    assert chunks == []
    assert state == {"failed": True}


# 中文: 验证 list 路由的流式接口只会产出一个 chunk(因为不需要真的流式生成),且全程不调用 LLM。
def test_ask_stream_list_route_yields_single_chunk_and_never_calls_llm(monkeypatch):
    rag = _make_rag()

    def fail(self, *args, **kwargs):
        raise AssertionError("LLM should not be called for a rule-matched list route")
        yield  # pragma: no cover

    monkeypatch.setattr(ChatOpenAI, "stream", fail)

    chunks = list(rag.ask_stream("Recommend a few dessert recipes"))

    assert len(chunks) == 1
    assert chunks[0].startswith("Here are some recipes you might like:")


# 中文: 验证 detail 路由虽然是分多个 chunk 流式返回的,但把所有 chunk 拼起来,结果和非流式生成应该产出的完整答案是完全一致的。
def test_ask_stream_detail_route_joins_to_full_answer(monkeypatch):
    rag = _make_rag()
    # query_rewrite() (non-streaming, via _complete/ChatOpenAI.invoke) and the
    # actual streamed generation (via ChatOpenAI.stream) are two independent
    # calls, so each needs its own mock -- mirroring how ask_stream() itself
    # makes one non-streaming call followed by one streaming call.
    monkeypatch.setattr(
        ChatOpenAI, "invoke", _fake_invoke("How do I make Kung Pao Chicken?")
    )
    monkeypatch.setattr(
        ChatOpenAI, "stream", _fake_stream("## Overview\n", "A classic dish.")
    )

    chunks = list(rag.ask_stream("How do I make Kung Pao Chicken?"))

    # Streamed in more than one piece, but joins to exactly what the
    # non-streaming generation would have produced from the same reply.
    assert len(chunks) >= 1
    assert "".join(chunks) == "## Overview\nA classic dish."


# 中文: 验证流式接口对空白问题的处理和非流式版本一致:直接返回提示语,不调用 LLM。
def test_ask_stream_empty_question_yields_prompt_message_without_llm_call(monkeypatch):
    rag = _make_rag()

    def fail(self, *args, **kwargs):
        raise AssertionError("LLM should not be called for an empty question")
        yield  # pragma: no cover

    monkeypatch.setattr(ChatOpenAI, "stream", fail)

    assert list(rag.ask_stream("   ")) == ["Please enter a question."]


# 中文: 验证 detail/general 路由在检索到的本地菜谱都答不了问题时,最终会把 NO_RESULTS_MESSAGE
#      正常流给用户,而不是卡住或抛异常。
def test_ask_stream_yields_no_results_message_when_llm_finds_nothing_relevant(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(ChatOpenAI, "invoke", _fake_invoke("How do I make Moussaka?"))
    monkeypatch.setattr(ChatOpenAI, "stream", _fake_stream(NO_RESULTS_MESSAGE))

    chunks = list(rag.ask_stream("How do I make Moussaka?"))

    assert "".join(chunks) == NO_RESULTS_MESSAGE
