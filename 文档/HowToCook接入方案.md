# HowToCook 知识库接入方案

## 结论

可以接入,许可证是 Unlicense(公共领域),无版权限制。但不能直接把 `dishes/` 目录扔进 `data/recipes/`——语言和格式都跟现有 pipeline 不匹配,需要动几处核心代码。

## 现状 vs HowToCook 的差异

| 项目 | RAGChef 现状 | HowToCook |
|---|---|---|
| 语言 | 英文菜谱 | 全中文 |
| Embedding 模型 | `BAAI/bge-small-en-v1.5`(英文专用) | 需要中文/多语言模型 |
| BM25 分词 | `_TOKEN_RE = [a-z0-9]+`,只认 ASCII | 中文无法被此正则切词,检索会直接失效 |
| 文件结构 | `# Title` / `## Ingredients` / `## Steps` | `# XX的做法` / `## 必备原料和工具` / `## 计算` / `## 操作`,步骤常按"简易版本""进阶版本"拆成多个 `###` 子节 |
| 难度 | 用步骤数量粗略推断 | 原文自带星级(如 ★★★★),可直接解析,更准确 |
| 分类目录 | `meat_dish` `vegetable_dish` `soup` `dessert` `staple` `aquatic` `breakfast` `other_dish` | 以上皆有,另外多出 `semi-finished`(半成品)`drink`(饮料)`condiment`(酱料),且仓库里混有 `tips/` `template/` 等非菜谱内容需要过滤 |
| 规模 | 50 道 | 500+ 道 |
| 生成侧 | Prompt/关键词触发词(如 vegetarian、dessert)按英文设计 | 需要改成中文关键词,并确认 DeepSeek 输出语言 |

## 需要改的地方(按依赖顺序)

1. **数据获取**:clone HowToCook 仓库(或用 GitHub API 拉取 `dishes/` 子树),按目录结构提取 `.md`,排除 `tips/`、`template/`、非菜谱说明文件。
2. **分类映射**:在 `rag.py` 的 `CATEGORY_LABELS` 里补上 `semi-finished`、`drink`、`condiment` 等新分类。
3. **解析器重写**:`load_documents`/`Recipe` 的字段提取逻辑要认识中文标题结构;难度直接从"预估烹饪难度:★★★★"解析,而不是数步骤数。
4. **Embedding 模型替换**:换成中文或中英双语模型,例如 `BAAI/bge-small-zh-v1.5`(纯中文)或 `BAAI/bge-m3`/`multilingual-e5-small`(中英混合场景更稳)。
5. **BM25 分词器替换**:现有正则对中文完全失效,需要引入中文分词(如 `jieba`)或退化成字符级 n-gram 分词。
6. **query_router / 关键词触发词**:`_CATEGORY_KEYWORDS`、`_DIFFICULTY_KEYWORDS` 等英文关键词要加中文版本(如"素""甜品""简单""新手")。
7. **生成 Prompt**:确认/调整 DeepSeek 的系统提示,使其按中文语境回答(如果保留双语知识库,还要处理"该用哪种语言回答"的问题)。
8. **测试**:`server/tests/` 里依赖英文样例数据的测试 fixture 需要同步更新或新增中文用例。
9. **索引重建**:数据量从 50 道涨到 500+,首次构建向量索引的时间会明显变长,但缓存机制(按内容 hash 判断是否重建)不用改。

## 关键决策点(需要你拍板)

- **中英文知识库怎么处理**:整体替换成中文,还是中英双库并存(两套 embedding 索引 + 语言路由)?双库复杂度高不少。
- **要接入多少**:全量 500+ 道,还是先挑一部分分类(比如先加"半成品加工""饮料"这两个现有库没有的类别)?
- **Chrome 插件前端**:如果知识库变中文,插件的问答体验、示例问题等文案是否也要跟着改?

## 建议的最小实施顺序

先切一个小分支验证可行性,而不是一次性全量接入:

1. 只拉取一个分类(如 `condiment` 十几道)转成本地测试数据
2. 换 embedding 模型 + 中文分词,跑通检索
3. 确认效果后再决定是否全量接入、是否需要双语支持
