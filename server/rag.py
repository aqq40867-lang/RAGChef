"""Core RAG (Retrieval-Augmented Generation) logic for RAGChef.

Pipeline: recursively load one Markdown file per recipe from the knowledge
base directory -> enhance each with metadata (category/dish name/difficulty)
-> split each recipe into a parent chunk (the whole recipe) + child chunks
(title/ingredients/steps, split on any "#"/"##"/"###" heading) -> embed the
child chunks with a BGE sentence-transformer and index them in FAISS
(cached to disk so this only happens once), and separately index them with
BM25 -> for each question, optionally narrow the search to chunks whose
parent matches a category/difficulty filter (explicit or inferred from the
question text), rank the (filtered) chunks by fusing the vector-search and
BM25 rankings with Reciprocal Rank Fusion, then rank their parent recipes by
the combined RRF score of the chunks each one contributed, and map back to
the parent recipes -> classify the question into list/detail/general, first
trying a keyword-only classifier (_infer_route) that costs no LLM call, and
falling back to a single combined DeepSeek call (_classify_and_rewrite) that
both classifies and rewrites the question only when the keywords are
inconclusive; a keyword-classified "detail" question still gets its own
rewrite call (query_rewrite), while "list" questions are used as-is either
way -- then retrieve parent recipes for that query, and answer with the
generation mode matching the route -- a plain formatted list of dish names,
a structured step-by-step answer, or a plain grounded answer -- with
DeepSeek answering strictly from the retrieved content (i.e. "grounding"),
which keeps the model from making things up. If DeepSeek decides none of the
retrieved local recipes actually answer a "detail"/"general" question, ask()
falls back to a single on-demand lookup against TheMealDB (a free public
recipe API) instead of immediately giving up -- see THEMEALDB_* and
SimpleRAG._themealdb_fallback().
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import httpx
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

# English variant of the reference project's embedding model choice
# (BAAI/bge-small-zh-v1.5): RAGChef's recipes are in English, so the
# Chinese-tuned model would be the wrong fit here.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# RRF's smoothing constant. 60 is the value used in the original RRF paper
# and in the reference implementation; it discounts the impact of rank
# position beyond the top handful of results.
RRF_K = 60

# query_router() classifies a question into one of these three routes, which
# ask() uses to pick a generation mode. Falls back to ROUTE_GENERAL if the
# classifier call fails or returns anything else.
ROUTE_LIST = "list"
ROUTE_DETAIL = "detail"
ROUTE_GENERAL = "general"
_VALID_ROUTES = {ROUTE_LIST, ROUTE_DETAIL, ROUTE_GENERAL}

# "list" questions (e.g. "recommend a few vegetarian dishes") want several
# candidate dishes to choose from, so they retrieve more parent recipes than
# the "detail"/"general" default of 2.
LIST_MODE_TOP_K = 5

# User-facing message returned when retrieval finds nothing relevant enough
# to answer the question.
NO_RESULTS_MESSAGE = "No relevant information was found in the current recipe library."

# User-facing message returned when the LLM call fails for any reason
# (auth, network, timeout, or a generic API error).
LLM_UNAVAILABLE_MESSAGE = (
    "Sorry, the recipe assistant is temporarily unavailable. Please try again in a moment."
)

# Maps the recipe directory a file lives under (server/data/recipes/<category>/)
# to a human-readable category label. Mirrors the folder layout of the
# knowledge base: server/data/recipes/meat_dish/kung-pao-chicken.md, etc.
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

# TheMealDB (themealdb.com) fallback: a free, public English recipe API used
# when the local 50-recipe library has no match at all. "1" is TheMealDB's
# published test key -- fine for personal/local use (see themealdb.com/api.php);
# set THEMEALDB_API_KEY to a supporter key if this is ever exposed publicly.
# Set THEMEALDB_ENABLED=false to disable the fallback entirely (e.g. fully
# offline use).
THEMEALDB_API_KEY = os.getenv("THEMEALDB_API_KEY", "1")
THEMEALDB_BASE_URL = f"https://www.themealdb.com/api/json/v1/{THEMEALDB_API_KEY}"
THEMEALDB_ENABLED = os.getenv("THEMEALDB_ENABLED", "true").strip().lower() not in (
    "false",
    "0",
    "no",
)
THEMEALDB_TIMEOUT = 5.0

# Maps TheMealDB's strCategory values (see /list.php?c=list) to this
# project's own category labels, so a fallback recipe plays nicely with the
# same category filtering local recipes use.
THEMEALDB_CATEGORY_MAP = {
    "Beef": "Meat",
    "Chicken": "Meat",
    "Lamb": "Meat",
    "Pork": "Meat",
    "Goat": "Meat",
    "Vegetarian": "Vegetable",
    "Vegan": "Vegetable",
    "Seafood": "Seafood",
    "Dessert": "Dessert",
    "Breakfast": "Breakfast",
    "Pasta": "Staple",
    "Side": "Other",
    "Starter": "Other",
    "Miscellaneous": "Other",
}

# Reverse-ish mapping used when the name search misses: a local category
# inferred from the question (see _infer_filters) is translated to a
# TheMealDB category to filter by. Categories with no clean TheMealDB
# equivalent (e.g. "Soup", "Other") are intentionally omitted -- the name
# search is still tried either way.
THEMEALDB_LOCAL_TO_CATEGORY = {
    "Meat": "Chicken",
    "Vegetable": "Vegetarian",
    "Dessert": "Dessert",
    "Seafood": "Seafood",
    "Breakfast": "Breakfast",
    "Staple": "Pasta",
}

# Strips common question phrasing ("how do I make ...", "recipe for ...")
# down to something closer to a bare dish name, since TheMealDB's
# search.php?s= does a name match rather than free-text search.
_THEMEALDB_STRIP_RE = re.compile(
    r"(?i)^(how (?:do|can) (?:i|you) make|how to make|what(?:'s| is) the recipe for|recipe for)\s+"
)

# Keyword triggers used to infer a category/difficulty filter from a
# question's text when the caller doesn't pass one explicitly (see
# _infer_filters). Deliberately conservative/unambiguous phrases only, so a
# query naming a specific dish that happens to contain a trigger word isn't
# needlessly narrowed (e.g. a plain dish name shouldn't accidentally filter
# anything out).
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

# Keyword triggers used by _infer_route() to classify a question as list/
# detail without an LLM call. Deliberately conservative (same philosophy as
# _CATEGORY_KEYWORDS/_DIFFICULTY_KEYWORDS above): only phrases that are
# unambiguous signals of intent, so a question that doesn't clearly match
# either falls through to the LLM classifier (see query_router /
# _classify_and_rewrite) instead of being misrouted.
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

# Every heading level we split on, from parent-document boundary (#) down to
# the finest child-chunk granularity we support (###).
_HEADING_RE = re.compile(r"(?m)^#{1,3}[ \t]+.+$")

# Steps are numbered ("1.", "2.", ...); step count is a rough proxy for
# recipe difficulty since the source data has no explicit difficulty rating
# (unlike knowledge bases that mark difficulty with a star rating in-text).
_STEP_RE = re.compile(r"(?m)^\d+\.\s")

# BM25 tokenizer: lowercase alphanumeric words. No stemming/stopword removal
# -- the knowledge base is small and specific enough that this is plenty.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class RAGConfigError(RuntimeError):
    """Raised when SimpleRAG cannot be initialized.

    Covers configuration problems such as a missing API key or an empty
    knowledge base. app.py lets this exception propagate uncaught at
    startup, so a misconfigured deployment fails fast at boot instead of
    staying up and returning 500s on every request.
    """


@dataclass
class Recipe:
    """A parent document (one whole recipe) plus its metadata.

    Attributes:
        text: Full Markdown content of the recipe, including its "# Title"
            heading and all "## "/"### " sections.
        source: Path the recipe was loaded from.
        dish_name: Recipe name, taken from the filename.
        category: Recipe category, taken from the immediate parent directory
            (e.g. "meat_dish" -> "Meat"). Falls back to "Other" if the file
            isn't inside one of the known category directories.
        difficulty: Rough difficulty estimate ("Easy"/"Medium"/"Hard"),
            derived from the number of steps in the "## Steps" section.
    """

    text: str
    source: str
    dish_name: str = ""
    category: str = "Other"
    difficulty: str = "Unknown"


@dataclass
class ChildChunk:
    """A retrievable fragment of a recipe (title, ingredients, steps, ...).

    Attributes:
        text: The chunk's Markdown content, starting at its heading line.
        parent_index: Index into SimpleRAG.documents of the parent Recipe
            this chunk was split from.
    """

    text: str
    parent_index: int


class SimpleRAG:
    """Loads the recipe knowledge base once, then serves retrieval + generation.

    An instance of this class is created once at app startup (see app.py)
    and reused across all requests, since loading the embedding model and
    building/loading the vector + BM25 indexes is the "heavy" step and
    doesn't need to be redone per request.

    Attributes:
        documents: Parent documents (Recipe objects), one per recipe.
        chunks: Child chunks (ChildChunk objects; title/ingredients/steps/
            sub-sections) across all recipes, used for retrieval.
        child_chunks: Plain-text view of chunks, in the same order; kept as
            a separate attribute because both the embedder and the BM25
            index operate on raw strings.
        child_to_parent: child_to_parent[j] is the index into documents of
            the parent recipe that child_chunks[j] belongs to. Kept in sync
            with chunks (same order, same length).
        embedder: SentenceTransformer used to embed both chunks and queries.
        vector_index: FAISS inner-product index over the (normalized)
            embeddings of child_chunks -- cosine similarity, since the
            vectors are unit-length.
        bm25: BM25Okapi index over the tokenized child_chunks.
        model: Name of the DeepSeek model used for generation.
        client: OpenAI-compatible client configured for the DeepSeek API.
    """

    def __init__(self, data_path: str):
        """Initializes the RAG pipeline: validates config, loads and indexes recipes.

        Args:
            data_path: Path to the directory containing one Markdown file
                per recipe (recursively; see load_documents()).

        Raises:
            RAGConfigError: If DEEPSEEK_API_KEY is not set, data_path
                doesn't exist, or no recipes are found under it.
        """
        # Validate config up front. Nothing downstream can work without an
        # API key, so fail here instead of waiting for the first request.
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RAGConfigError(
                "DEEPSEEK_API_KEY is not set. Add it to server/.env or the environment "
                "before starting the server."
            )

        self.documents = self.load_documents(data_path)
        if not self.documents:
            raise RAGConfigError(
                f"No recipes found under {data_path}. The knowledge base is empty."
            )

        # Parent/child split: each parent recipe (self.documents[i]) is split
        # into smaller child chunks (title/ingredients/steps/sub-sections)
        # along "#"/"##"/"###" heading boundaries. child_to_parent[j] stores
        # the index into self.documents of the parent document that
        # chunks[j] belongs to.
        self.chunks, self.child_to_parent = self._build_child_chunks(self.documents)
        self.child_chunks = [chunk.text for chunk in self.chunks]

        # Embedding model, shared for indexing chunks below and for
        # embedding queries in _vector_search(). Downloaded from the
        # HuggingFace Hub on first use and cached locally after that --
        # deployments without network access at startup need the model
        # pre-warmed into the HF cache at build/image time instead.
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Vector index: cached to disk (see _build_or_load_vector_index) so
        # only the *first* run after the knowledge base changes pays the
        # cost of re-embedding every chunk.
        self.vector_index = self._build_or_load_vector_index(data_path)

        # BM25 index: cheap enough (pure Python, no model) to rebuild on
        # every startup, so it isn't cached like the vector index is.
        self.bm25 = BM25Okapi([self._tokenize(text) for text in self.child_chunks])

        self.model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    def load_documents(self, data_path: str) -> list[Recipe]:
        """Recursively loads one Recipe per Markdown file under data_path.

        Args:
            data_path: Directory containing recipe files, one recipe per
                ".md" file, optionally grouped into category subdirectories
                (e.g. data_path/meat_dish/kung-pao-chicken.md).

        Returns:
            A list of Recipe objects, one per Markdown file found, each
            enhanced with metadata via _enhance_metadata().

        Raises:
            RAGConfigError: If data_path does not exist or is not a
                directory.
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
        """Fills in a Recipe's category/dish_name/difficulty in place.

        Args:
            recipe: The Recipe to enhance (mutated in place).
            path: Filesystem path the recipe was loaded from, used to infer
                dish_name (filename) and category (parent directory name).
        """
        # Dish name: filename, with hyphens turned back into spaces
        # ("kung-pao-chicken" -> "kung pao chicken") as a fallback in case
        # the "# Title" heading is ever missing from the content.
        recipe.dish_name = path.stem.replace("-", " ")

        # Category: inferred from the immediate parent directory, e.g.
        # .../recipes/meat_dish/kung-pao-chicken.md -> "meat_dish" -> "Meat".
        recipe.category = CATEGORY_LABELS.get(path.parent.name, "Other")

        # Difficulty: approximated from the number of steps in the "##
        # Steps" section (there's no explicit difficulty rating in the
        # source data to parse instead).
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
        """Splits each parent recipe into child chunks along heading boundaries.

        Every "#", "##", or "###" heading starts a new chunk that runs until
        the next heading of any of those levels; this generalizes the old
        "split only on ## " approach so that a recipe using deeper "###"
        sub-sections (e.g. simple vs. advanced variants) gets chunked at
        that finer granularity too, instead of being lumped into its parent
        "## " section.

        Args:
            documents: Parent recipe documents, as returned by
                load_documents().

        Returns:
            A tuple of (chunks, child_to_parent), where chunks are the
            title/ingredients/steps/sub-sections across all recipes, and
            child_to_parent[j] is the index into documents of the parent
            recipe that chunks[j] belongs to.
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
        """Splits text into one chunk per "#"/"##"/"###" heading section.

        Each returned chunk starts at a heading line and includes everything
        up to (but not including) the next heading line of level 1-3, so a
        "### " sub-heading nested inside a "## " section becomes its own
        chunk rather than being merged into its parent section.

        Args:
            text: Markdown text to split (typically a whole recipe).

        Returns:
            A list of non-empty chunk strings, in document order. If text
            contains no headings, the whole (stripped) text is returned as a
            single chunk.
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
        """Lowercase alphanumeric-word tokenizer used for BM25."""
        return _TOKEN_RE.findall(text.lower())

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Encodes texts into L2-normalized float32 embeddings.

        Normalizing lets a FAISS inner-product index double as a cosine
        similarity index.
        """
        return np.asarray(
            self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )

    def _build_or_load_vector_index(self, data_path: str) -> faiss.Index:
        """Builds a FAISS index over child_chunks, or loads a cached one from disk.

        Mirrors the reference project's index-caching approach (embed once,
        save to disk, load on subsequent startups instead of re-embedding
        every chunk), with one addition: the cache records a hash of the
        chunk contents, so if the knowledge base changes, the stale cache is
        detected and rebuilt automatically instead of silently serving
        out-of-date embeddings.

        Args:
            data_path: The recipes directory passed to __init__; the cache
                is stored in a sibling "vector_index/" directory.

        Returns:
            A FAISS IndexFlatIP over the (normalized) embeddings of
            self.child_chunks, in the same order.
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
                logger.info("Loading cached vector index from %s", index_path)
                return faiss.read_index(str(index_path))
            logger.info("Vector index cache at %s is stale; rebuilding.", index_path)

        logger.info(
            "Building vector index for %d chunks with %s...",
            len(self.child_chunks),
            EMBEDDING_MODEL_NAME,
        )
        embeddings = self._embed(self.child_chunks)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

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

    def _vector_search(
        self, question: str, k: int, allowed: set[int] | None = None
    ) -> list[int]:
        """Returns up to k chunk indices ranked by embedding cosine similarity.

        Args:
            question: The query text.
            k: Maximum number of chunk indices to return.
            allowed: If given, restricts results to chunk indices in this
                set (used for category/difficulty-filtered search).

        Returns:
            Chunk indices, ranked best-to-worst.
        """
        if self.vector_index.ntotal == 0 or k <= 0:
            return []
        query_embedding = self._embed([question])
        # FAISS has no notion of our metadata filter, so search the whole
        # index and filter in Python afterwards. Fine at this corpus size
        # (a few hundred chunks); a larger deployment would push the filter
        # into the ANN search itself (e.g. an IDSelector) instead of ranking
        # chunks that get thrown away.
        _, indices = self.vector_index.search(query_embedding, self.vector_index.ntotal)
        ranked = [int(i) for i in indices[0] if i != -1]
        if allowed is not None:
            ranked = [i for i in ranked if i in allowed]
        return ranked[:k]

    def _bm25_search(
        self, question: str, k: int, allowed: set[int] | None = None
    ) -> list[int]:
        """Returns up to k chunk indices ranked by BM25 score.

        Args:
            question: The query text.
            k: Maximum number of chunk indices to return.
            allowed: If given, restricts results to chunk indices in this
                set (used for category/difficulty-filtered search).

        Returns:
            Chunk indices, ranked best-to-worst.
        """
        if k <= 0:
            return []
        scores = self.bm25.get_scores(self._tokenize(question))
        ranked = [int(i) for i in np.argsort(scores)[::-1]]
        if allowed is not None:
            ranked = [i for i in ranked if i in allowed]
        return ranked[:k]

    @staticmethod
    def _rrf_fuse(*ranked_lists: list[int], k: int = RRF_K) -> dict[int, float]:
        """Reciprocal Rank Fusion: combines ranked index lists into one score per index.

        Args:
            *ranked_lists: One or more lists of indices, each ranked
                best-to-worst by some retriever.
            k: RRF's smoothing constant (see RRF_K).

        Returns:
            A dict mapping each index that appeared in any list to its
            summed RRF score (higher is better). An index absent from every
            list is absent from the result.
        """
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, index in enumerate(ranked):
                scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank + 1)
        return scores

    def _infer_filters(self, question: str) -> tuple[str | None, str | None]:
        """Heuristically infers a category/difficulty filter from question text.

        Deliberately conservative: only triggers on fairly unambiguous
        phrases (see _CATEGORY_KEYWORDS/_DIFFICULTY_KEYWORDS), so a question
        naming a specific dish isn't needlessly narrowed just because a
        trigger word happens to appear in it.

        Args:
            question: The user's natural-language question.

        Returns:
            A (category, difficulty) tuple; either or both may be None if
            nothing matched.
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
        """Heuristically classifies a question as list/detail without an LLM call.

        A cheap first pass in front of query_router: most everyday questions
        contain an unambiguous signal word ("recommend", "how do I make",
        ...), so classifying those locally skips a DeepSeek round trip
        entirely. Deliberately conservative -- same philosophy as
        _infer_filters -- so a question that doesn't clearly match either
        list or detail phrasing returns None and falls through to the LLM
        (see ask() / _classify_and_rewrite) rather than risk misrouting a
        "general" question.

        Args:
            question: The user's natural-language question.

        Returns:
            ROUTE_LIST or ROUTE_DETAIL if a trigger phrase matched, else
            None. Never returns ROUTE_GENERAL -- general questions have no
            reliable keyword signature, so they're always left to the LLM.
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
        """Returns the top_k parent recipe documents (as text) most relevant to the question.

        Thin wrapper around _rank_documents() for callers that only need the
        recipe text (e.g. building a grounding prompt), not the full Recipe
        object with its metadata. See _rank_documents() for the retrieval
        algorithm itself.

        Args:
            question: The user's natural-language question.
            top_k: Maximum number of parent recipes to return.
            category: Optional explicit category filter, overriding
                inference from the question text.
            difficulty: Optional explicit difficulty filter, overriding
                inference from the question text.

        Returns:
            Up to top_k parent recipe documents (as text), ordered by
            descending relevance.
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
        """Returns the top_k parent Recipe objects most relevant to the question.

        Retrieval happens at the child-chunk level (title/ingredients/steps
        compared separately), combining two complementary signals: FAISS
        vector search over BGE embeddings (semantic similarity -- catches
        "a quick meal" matching a recipe tagged Easy) and BM25 (lexical
        match -- catches exact dish names and ingredients). The two rankings
        are fused with Reciprocal Rank Fusion so neither signal dominates on
        its own.

        If category/difficulty aren't passed explicitly, they're inferred
        heuristically from the question text (see _infer_filters) and used
        to narrow the search to matching recipes; if the filter would match
        nothing, it's dropped and the search falls back to the full library
        rather than returning no results.

        Parent recipes are then ranked by the sum of RRF scores of the
        chunks they contributed to a shortlist of the best-fused chunks
        (rewarding a recipe matched on multiple sections over one matched on
        a single section, while staying weighted by match strength so a few
        weak, incidental matches can't outrank one strong match). If the
        shortlist doesn't cover top_k distinct parents, remaining slots are
        filled from the full fused ranking, so retrieve() always returns
        min(top_k, number of matching documents) results.

        Args:
            question: The user's natural-language question.
            top_k: Maximum number of parent recipes to return.
            category: Optional explicit category filter (e.g. "Vegetable"),
                overriding inference from the question text.
            difficulty: Optional explicit difficulty filter (e.g. "Easy"),
                overriding inference from the question text.

        Returns:
            Up to top_k parent Recipe objects, ordered by descending
            relevance.
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
            # If the filter matched no recipes at all, silently fall back to
            # an unfiltered search (allowed_chunks stays None) instead of
            # returning nothing.

        pool_size = len(allowed_chunks) if allowed_chunks is not None else len(self.child_chunks)
        vector_ranked = self._vector_search(question, k=pool_size, allowed=allowed_chunks)
        bm25_ranked = self._bm25_search(question, k=pool_size, allowed=allowed_chunks)

        fused_scores = self._rrf_fuse(vector_ranked, bm25_ranked)
        if not fused_scores:
            return []
        ranked_child_indices = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)

        # Shortlist of the best-fused child chunks. Sized relative to top_k
        # (with a floor of 10) so a broader ask() still has enough
        # candidates to find top_k distinct parents from the shortlist
        # alone, without scanning the whole (filtered) corpus.
        candidate_pool = min(len(ranked_child_indices), max(top_k * 5, 10))

        score: dict[int, float] = {}
        hits: dict[int, int] = {}
        ordered_parents: list[int] = []  # first-seen order among candidates
        for child_index in ranked_child_indices[:candidate_pool]:
            parent_index = self.child_to_parent[child_index]
            if parent_index not in score:
                ordered_parents.append(parent_index)
            score[parent_index] = score.get(parent_index, 0.0) + fused_scores[child_index]
            hits[parent_index] = hits.get(parent_index, 0) + 1

        # Sort by total RRF score first (rewards multiple strong matches),
        # then by hit count as a tiebreaker for near-equal scores.
        ranked_parents = sorted(
            ordered_parents, key=lambda p: (score[p], hits[p]), reverse=True
        )

        # Fall back to the full fused ranking to fill any remaining slots,
        # so retrieve() still honors top_k even when the shortlist above
        # doesn't contain that many distinct parents.
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
        """Approximates a bare dish name from a question, for TheMealDB's name search.

        TheMealDB's search.php?s= matches against meal names, not free text,
        so "How do I make Kung Pao Chicken?" needs to become roughly "Kung
        Pao Chicken" first.

        Args:
            question: The user's (possibly rewritten) question.

        Returns:
            The question with a leading question-phrase (if any) and a
            trailing "?" stripped. Falls back to the original text if
            stripping would leave nothing.
        """
        text = question.strip().rstrip("?").strip()
        cleaned = _THEMEALDB_STRIP_RE.sub("", text, count=1).strip()
        return cleaned or text

    @staticmethod
    def _themealdb_get(path: str, params: dict) -> dict | None:
        """GETs a TheMealDB endpoint, returning parsed JSON or None on any failure.

        Never raises: network errors, non-2xx responses, and unparsable
        bodies are all logged and treated as "no data", so a TheMealDB
        outage degrades to the existing NO_RESULTS_MESSAGE behavior instead
        of breaking ask().

        Args:
            path: Endpoint path, e.g. "/search.php".
            params: Query parameters for the request.

        Returns:
            The parsed JSON body, or None if the request or parsing failed.
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
        """Converts a TheMealDB "meal" JSON object into a Recipe.

        Rebuilds the same "# Title" / "## Ingredients" / "## Steps" Markdown
        shape used by the local knowledge base, so this Recipe can flow
        through the existing generation prompts unchanged.

        Args:
            meal: One element of a TheMealDB search/lookup response's
                "meals" list.

        Returns:
            A Recipe with difficulty "Unknown" (TheMealDB doesn't rate
            difficulty) and source set to "themealdb:<idMeal>".
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
        """Looks up one recipe from TheMealDB when the local library has no match.

        Tries an exact-ish name search first (search.php?s=); if that misses
        and a local category was inferred for the question, falls back to
        that category (filter.php?c=...) and fetches full details for the
        first candidate (lookup.php?i=...). This is called on demand, not
        pre-fetched, so it never downloads more than the one or two API
        responses needed to answer the current question.

        Args:
            question: The user's (possibly rewritten) question.
            category: Optional local category label (e.g. "Vegetable"),
                reused to pick a TheMealDB category filter if the name
                search misses.

        Returns:
            A Recipe built from TheMealDB's response, or None if nothing
            was found, the fallback is disabled, or the API call failed.
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
        """Sends a single-turn chat completion request, swallowing provider failures.

        Centralizes the try/except around the four places this module calls
        DeepSeek (query_router, query_rewrite, and the two prompt-based
        generation modes), so each of those methods just decides what to do
        with a None result instead of repeating the same except-blocks four
        times.

        Args:
            prompt: The full prompt to send as a single user message.
            temperature: Sampling temperature -- low (near 0) for the
                classification/rewrite calls that want a consistent, literal
                answer; higher for open-ended generation.

        Returns:
            The model's reply text, or None if the call failed (auth,
            network, timeout, API error) or returned empty content. Callers
            are responsible for logging/handling what None means for them.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
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
        """Classifies a question as a "list", "detail", or "general" query.

        - list: the user wants a set of dish suggestions/names (e.g.
          "recommend a few vegetarian dishes")
        - detail: the user wants how to make a specific dish (e.g. "how do I
          make Kung Pao Chicken?")
        - general: anything else (e.g. "what's the difference between a
          casserole and a hotpot?")

        Args:
            question: The user's natural-language question.

        Returns:
            One of ROUTE_LIST/ROUTE_DETAIL/ROUTE_GENERAL. Falls back to
            ROUTE_GENERAL (the safest default -- a plain grounded answer) if
            the classifier call fails or returns something unexpected.
        """
        prompt = f"""Classify the user's question into exactly one category. Reply with only the category name, nothing else.

Categories:
- list: the user wants a list of dish suggestions or names (e.g. "recommend a few vegetarian dishes", "what soups do you have?")
- detail: the user wants to know how to make a specific dish (e.g. "how do I make Kung Pao Chicken?", "what ingredients do I need for dumplings?")
- general: anything else (e.g. "what's the difference between a casserole and a hotpot?", "what does al dente mean?")

User question: {question}

Category:"""
        route = self._complete(prompt, temperature=0)
        if route:
            route = route.strip().lower()
        return route if route in _VALID_ROUTES else ROUTE_GENERAL

    def query_rewrite(self, question: str) -> str:
        """Rewrites a vague question into a more specific, retrieval-friendly one.

        Lets the LLM itself judge whether rewriting is needed: a question
        that already names a specific dish or is otherwise clear should come
        back unchanged; a vague one (e.g. "give me something to cook")
        should come back expanded with concrete cooking terms.

        Args:
            question: The user's natural-language question.

        Returns:
            The rewritten question, or the original question unchanged if
            the rewrite call fails or returns nothing usable.
        """
        prompt = f"""Decide whether this recipe-assistant question needs rewriting to be more specific before searching a recipe database.

Rules:
- If the question is already specific (names a dish, an ingredient, or a clear request), return it unchanged.
- If the question is vague (e.g. "give me something to cook", "suggest a meal"), rewrite it to be more specific and search-friendly, keeping the original intent and preferring simple, easy-to-make dishes when nothing else is specified.
- Reply with only the final question text, nothing else.

Question: {question}

Rewritten question:"""
        rewritten = self._complete(prompt, temperature=0.2)
        return rewritten if rewritten else question

    def _classify_and_rewrite(self, question: str) -> tuple[str, str]:
        """Classifies the route and rewrites the query in a single LLM call.

        Does the combined job of query_router() + query_rewrite() in one
        request instead of two, since both look at the same question and
        each returns a short, structured judgment. Used by ask() as the
        fallback path when _infer_route() can't confidently classify the
        question from keywords alone -- the one case that still needs the
        LLM's judgment at all.

        Args:
            question: The user's natural-language question.

        Returns:
            A (route, rewritten_question) tuple. Falls back to
            (ROUTE_GENERAL, question) -- matching query_router's and
            query_rewrite's own individual fallback behavior -- if the call
            fails or the reply isn't parseable as the expected JSON shape.
        """
        prompt = f"""You are helping a recipe assistant prepare to answer a question. Do two things:

1. Classify the question into exactly one category:
   - list: the user wants a list of dish suggestions or names (e.g. "recommend a few vegetarian dishes", "what soups do you have?")
   - detail: the user wants to know how to make a specific dish (e.g. "how do I make Kung Pao Chicken?")
   - general: anything else (e.g. "what's the difference between a casserole and a hotpot?")
2. Produce a version of the question rewritten to be more specific and search-friendly: if the question is already specific (names a dish, an ingredient, or a clear request), repeat it unchanged; if it's vague (e.g. "give me something to cook"), rewrite it to be more specific, keeping the original intent and preferring simple, easy-to-make dishes when nothing else is specified.

Reply with ONLY a JSON object, no other text, in exactly this shape:
{{"route": "list", "rewritten": "..."}}

User question: {question}"""
        reply = self._complete(prompt, temperature=0)
        if reply:
            # Strip an optional ```json ... ``` fence in case the model
            # wraps its reply in one despite the "ONLY a JSON object"
            # instruction.
            cleaned = re.sub(r"^```(?:json)?|```$", "", reply.strip(), flags=re.MULTILINE).strip()
            try:
                data = json.loads(cleaned)
                route = str(data.get("route", "")).strip().lower()
                rewritten = str(data.get("rewritten", "")).strip()
            except (json.JSONDecodeError, AttributeError):
                route, rewritten = "", ""
            if route in _VALID_ROUTES and rewritten:
                return route, rewritten
            logger.warning("Could not parse classify_and_rewrite reply: %r", reply)
        return ROUTE_GENERAL, question

    def _generate_list_answer(self, docs: list[Recipe]) -> str:
        """Builds a plain numbered list of dish names -- no LLM call needed.

        The retrieved parent Recipe objects already carry dish_name, so a
        "list" query can be answered by formatting that metadata directly,
        which is both cheaper and more literal/reliable than asking the LLM
        to summarize a list of names it was just given.

        Args:
            docs: Parent Recipe objects to list, in relevance order.

        Returns:
            A numbered list of dish names, or NO_RESULTS_MESSAGE if docs is
            empty.
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
        """Builds the "detail" route's structured generation prompt.

        Factored out of _generate_step_by_step_answer() so the streaming
        variant (_generate_step_by_step_answer_stream()) can build the exact
        same prompt without duplicating the template text.
        """
        context = "\n\n".join(doc.text for doc in docs)
        return f"""You are a professional recipe assistant. Answer the question using ONLY the recipe content below.
If the answer is not contained in the content, reply exactly: "{NO_RESULTS_MESSAGE}"

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

    def _generate_step_by_step_answer(self, question: str, docs: list[Recipe]) -> str:
        """Generates a structured, step-by-step recipe answer.

        Args:
            question: The user's original question (not the rewritten one --
                shown to the LLM so it answers what was actually asked).
            docs: Parent Recipe objects to ground the answer in.

        Returns:
            A structured Markdown answer (overview/ingredients/steps/tips),
            NO_RESULTS_MESSAGE if docs is empty, or LLM_UNAVAILABLE_MESSAGE
            if generation fails.
        """
        if not docs:
            return NO_RESULTS_MESSAGE

        answer = self._complete(self._step_by_step_prompt(question, docs), temperature=0.3)
        return answer if answer else LLM_UNAVAILABLE_MESSAGE

    def _generate_step_by_step_answer_stream(
        self, question: str, docs: list[Recipe], state: dict
    ):
        """Streaming counterpart of _generate_step_by_step_answer().

        Yields answer text incrementally instead of returning it all at
        once. Reports what happened via the caller-supplied `state` dict
        (mutated in place) rather than a return value, since a generator's
        return value isn't accessible through a plain `for` loop:
            - state["is_no_results"] = True if docs was empty, or if the
              full streamed reply turned out to be exactly
              NO_RESULTS_MESSAGE (in which case *nothing* is yielded, so a
              caller can silently try the TheMealDB fallback before
              anything reaches the end user -- see ask_stream()).
            - state["failed"] = True if the LLM call itself failed
              (auth/network/timeout/API error) before producing any
              content.

        Args:
            question: The user's original question.
            docs: Parent Recipe objects to ground the answer in.
            state: Dict this method writes its outcome into (see above).

        Yields:
            Answer text chunks, in order. Yields nothing at all if
            state["is_no_results"] ends up True.
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
        """Builds the "general" route's plain grounded-answer prompt.

        Factored out of _generate_basic_answer() so the streaming variant
        (_generate_basic_answer_stream()) can build the exact same prompt
        without duplicating the template text.
        """
        context = "\n\n".join(doc.text for doc in docs)
        return f"""You are a professional recipe assistant.

Answer the question using ONLY the recipe content provided below.
If the answer is not contained in the content, reply exactly: "{NO_RESULTS_MESSAGE}"

Recipe content:
{context}

User question:
{question}

Respond naturally and concisely in English.
"""

    def _generate_basic_answer(self, question: str, docs: list[Recipe]) -> str:
        """Generates a plain grounded answer for general (non-list, non-detail) questions.

        This is the original single-mode prompt RAGChef used before query
        routing was introduced.

        Args:
            question: The user's original question.
            docs: Parent Recipe objects to ground the answer in.

        Returns:
            The generated answer, NO_RESULTS_MESSAGE if docs is empty, or
            LLM_UNAVAILABLE_MESSAGE if generation fails.
        """
        if not docs:
            return NO_RESULTS_MESSAGE

        answer = self._complete(self._basic_answer_prompt(question, docs), temperature=0.3)
        return answer if answer else LLM_UNAVAILABLE_MESSAGE

    def _generate_basic_answer_stream(self, question: str, docs: list[Recipe], state: dict):
        """Streaming counterpart of _generate_basic_answer(). See
        _generate_step_by_step_answer_stream() for the state dict contract.
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
        """Streams a single-turn chat completion, yielding text deltas as they arrive.

        Streaming counterpart of _complete(): same failure handling (auth/
        connection/timeout/API errors are logged and swallowed rather than
        raised), but instead of returning the full text at the end, yields
        each piece of content as DeepSeek sends it. A failure produces no
        chunks at all if it happens before the first one arrives, or simply
        stops the stream early if it happens mid-response -- callers can't
        tell those two cases apart from this method alone (see
        _stream_with_no_results_guard(), which does track "got anything at
        all" for that).

        Args:
            prompt: The full prompt to send as a single user message.
            temperature: Sampling temperature.

        Yields:
            Non-empty text chunks, in order.
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
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
        """Streams a completion while withholding output that might be NO_RESULTS_MESSAGE.

        The non-streaming generation methods can check `answer ==
        NO_RESULTS_MESSAGE` only because they wait for the full reply before
        deciding anything. A naive stream-straight-to-the-client version
        would lose that: the user would see "No relevant information..."
        start to appear before ask_stream() gets a chance to quietly retry
        with TheMealDB instead.

        This closes that gap without adding another LLM call: it buffers
        output only for as long as the buffered text is still a possible
        prefix of NO_RESULTS_MESSAGE. The instant the streamed text diverges
        from that exact string, everything buffered so far is flushed and
        the rest streams straight through -- so a real answer starts
        appearing after at most a few characters of delay. Only a reply
        that matches NO_RESULTS_MESSAGE exactly, start to finish, is held
        back for the whole stream.

        Args:
            prompt: The full prompt to send as a single user message.
            temperature: Sampling temperature.
            state: Dict this method writes its outcome into:
                - state["is_no_results"] = True if the full reply was
                  exactly NO_RESULTS_MESSAGE (nothing is yielded in this
                  case).
                - state["failed"] = True if the underlying stream produced
                  no content at all (LLM call failed outright).

        Yields:
            Answer text chunks, in order. Yields nothing if the reply
            turned out to be NO_RESULTS_MESSAGE or the call failed outright.
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
                continue  # still a possible NO_RESULTS_MESSAGE prefix (or an exact match so far) -- keep withholding
            diverged = True
            yield buffer

        if not got_any:
            state["failed"] = True
        elif not diverged:
            if buffer == NO_RESULTS_MESSAGE:
                state["is_no_results"] = True
            else:
                # Stream ended while buffer was still a strict (non-equal)
                # prefix of NO_RESULTS_MESSAGE -- an unusual truncation, not
                # an actual match. Flush whatever was withheld rather than
                # silently dropping it.
                yield buffer

    def ask(self, question: str) -> str:
        """Routes, retrieves, and answers a question using the appropriate generation mode.

        Pipeline: classify the question (query_router) -> for non-"list"
        questions, optionally rewrite it to be more specific/search-friendly
        (query_rewrite; "list" questions are left as-is, since they're
        already a request for options rather than something to sharpen) ->
        retrieve parent recipes for the (possibly rewritten) query -> hand
        off to the generation mode matching the route: a plain formatted
        list of dish names for "list", a structured step-by-step answer for
        "detail", or a plain grounded answer for "general".

        Args:
            question: The user's natural-language question.

        Returns:
            The generated answer, or one of NO_RESULTS_MESSAGE /
            LLM_UNAVAILABLE_MESSAGE if retrieval or generation fails to
            produce a usable answer. This method does not raise on expected
            LLM-provider failures; failures are logged and surfaced as
            LLM_UNAVAILABLE_MESSAGE instead (except for "list" answers,
            which don't call the LLM at all and so can't fail that way).
        """
        if not question or not question.strip():
            return "Please enter a question."

        # Fast path: try to classify locally from keywords first (zero LLM
        # calls) before paying for a DeepSeek round trip. A "list" match
        # skips query_rewrite entirely (list questions are used as-is, same
        # as before); a "detail" match still needs query_rewrite (its own
        # job -- specificity -- is independent of routing). Only a question
        # neither pattern recognizes falls through to the LLM, which then
        # does classification *and* rewriting in one combined call instead
        # of the two separate ones this used to take.
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

        # _rank_documents always returns its best-effort top_k matches --
        # even ones that aren't actually relevant -- rather than an empty
        # list, since RRF just ranks the whole (filtered) corpus and takes
        # the top few. So "the local library has nothing relevant" doesn't
        # show up as empty docs; it shows up here, as the LLM's own verdict
        # that the retrieved content doesn't answer the question (both
        # generation prompts are instructed to reply with exactly
        # NO_RESULTS_MESSAGE in that case). Only then is it worth the extra
        # network round-trip to try TheMealDB before giving up for real.
        if answer == NO_RESULTS_MESSAGE:
            category, _ = self._infer_filters(search_query)
            fallback_recipe = self._themealdb_fallback(search_query, category=category)
            if fallback_recipe:
                answer = generate(question, [fallback_recipe])

        return answer

    def ask_stream(self, question: str):
        """Streaming counterpart of ask(): yields the answer incrementally.

        Mirrors ask()'s routing/retrieval/fallback logic exactly (kept as a
        separate implementation rather than having one delegate to the
        other, so ask()'s non-streaming DeepSeek calls -- and everything
        that mocks them in tests -- stay unchanged); the difference is only
        in how the chosen generation mode produces its output; a plain list
        answer or a static message is yielded as one chunk (nothing to
        stream -- these never call the LLM), while "detail"/"general"
        answers are yielded incrementally as DeepSeek generates them, using
        _generate_step_by_step_answer_stream()/_generate_basic_answer_stream()
        and the same NO_RESULTS_MESSAGE-triggered TheMealDB fallback as
        ask() (see _stream_with_no_results_guard() for how that fallback
        stays invisible to the caller instead of briefly flashing "not
        found" before the real answer streams in).

        Args:
            question: The user's natural-language question.

        Yields:
            Answer text chunks, in order; concatenating all of them
            produces the same string ask() would have returned for the same
            question.
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
                    # Fallback recipe didn't answer the question either
                    # (mirrors ask(): no third attempt, just surface the
                    # message that was withheld both times).
                    yield NO_RESULTS_MESSAGE
            else:
                yield NO_RESULTS_MESSAGE
