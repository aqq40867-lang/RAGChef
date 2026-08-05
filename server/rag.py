"""Core RAG (Retrieval-Augmented Generation) logic for RAGChef.

Pipeline: load the recipe knowledge base -> split each recipe into a parent
chunk (the whole recipe) + child chunks (title/ingredients/steps) -> build a
TF-IDF index over the child chunks -> for each question, retrieve the most
similar child chunks and map them back to their parent recipes -> have
DeepSeek answer using only the retrieved content (i.e. "grounding"), which
keeps the model from making things up.
"""

import logging
import os
import re

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

logger = logging.getLogger("ragchef")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

# User-facing message returned when retrieval finds nothing relevant enough
# to answer the question.
NO_RESULTS_MESSAGE = "No relevant information was found in the current recipe library."

# User-facing message returned when the LLM call fails for any reason
# (auth, network, timeout, or a generic API error).
LLM_UNAVAILABLE_MESSAGE = (
    "Sorry, the recipe assistant is temporarily unavailable. Please try again in a moment."
)


class RAGConfigError(RuntimeError):
    """Raised when SimpleRAG cannot be initialized.

    Covers configuration problems such as a missing API key or an empty
    knowledge base. app.py lets this exception propagate uncaught at
    startup, so a misconfigured deployment fails fast at boot instead of
    staying up and returning 500s on every request.
    """


class SimpleRAG:
    """Loads the recipe knowledge base once, then serves retrieval + generation.

    An instance of this class is created once at app startup (see app.py)
    and reused across all requests, since building the TF-IDF index is the
    only "heavy" step and doesn't need to be redone per request.

    Attributes:
        documents: Parent documents, one per recipe.
        child_chunks: Child chunks (title/ingredients/steps) across all
            recipes, used for retrieval.
        child_to_parent: Maps each index in child_chunks to the index of its
            parent document in documents.
        vectorizer: Fitted TF-IDF vectorizer over child_chunks.
        child_vectors: TF-IDF matrix for child_chunks.
        model: Name of the DeepSeek model used for generation.
        client: OpenAI-compatible client configured for the DeepSeek API.
    """

    def __init__(self, file_path: str):
        """Initializes the RAG pipeline: validates config, loads and indexes recipes.

        Args:
            file_path: Path to the recipes.md knowledge base file.

        Raises:
            RAGConfigError: If DEEPSEEK_API_KEY is not set, the file at
                file_path is missing, or no recipes are found in it.
        """
        # Validate config up front. Nothing downstream can work without an
        # API key, so fail here instead of waiting for the first request.
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RAGConfigError(
                "DEEPSEEK_API_KEY is not set. Add it to server/.env or the environment "
                "before starting the server."
            )

        self.documents = self.load_documents(file_path)
        if not self.documents:
            raise RAGConfigError(
                f"No recipes found in {file_path}. The knowledge base is empty."
            )

        # Parent/child split: each parent recipe (self.documents[i]) is split
        # into smaller child chunks (title/ingredients/steps/etc.) along
        # "## " subheadings. child_to_parent[j] stores the index into
        # self.documents of the parent document that child_chunks[j] belongs to.
        self.child_chunks, self.child_to_parent = self._build_child_chunks(self.documents)

        # Build the TF-IDF index over the child chunks rather than whole
        # parent recipes. A user question (e.g. "how much salt?") matches a
        # small, focused chunk (just the ingredients section) instead of the
        # whole recipe, so unrelated words elsewhere in the document don't
        # dilute the match.
        self.vectorizer = TfidfVectorizer()
        self.child_vectors = self.vectorizer.fit_transform(self.child_chunks)

        self.model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    def load_documents(self, file_path: str) -> list[str]:
        """Reads the knowledge base file and splits it into one document per recipe.

        Args:
            file_path: Path to the recipes.md knowledge base file.

        Returns:
            A list of parent document strings, one per recipe, split along
            top-level ("# ") headings.

        Raises:
            RAGConfigError: If file_path does not exist.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError as e:
            raise RAGConfigError(f"Recipe knowledge base not found at {file_path}") from e

        # Parent documents: whole recipes, split along "# " headings.
        parts = re.split(r"(?m)^# ", text)
        return [part.strip() for part in parts if part.strip()]

    def _build_child_chunks(
        self, documents: list[str]
    ) -> tuple[list[str], list[int]]:
        """Splits each parent recipe into child chunks along "## " headings.

        Args:
            documents: Parent recipe documents, as returned by load_documents().

        Returns:
            A tuple of (child_chunks, child_to_parent), where child_chunks
            are the title/ingredients/steps sections across all recipes, and
            child_to_parent[j] is the index into documents of the parent
            recipe that child_chunks[j] belongs to.
        """
        child_chunks = []
        child_to_parent = []

        for parent_index, parent_text in enumerate(documents):
            parts = re.split(r"(?m)^## ", parent_text)

            title_part = parts[0].strip()
            if title_part:
                child_chunks.append(title_part)
                child_to_parent.append(parent_index)

            for section in parts[1:]:
                section_text = ("## " + section).strip()
                if section_text:
                    child_chunks.append(section_text)
                    child_to_parent.append(parent_index)

        return child_chunks, child_to_parent

    def retrieve(self, question: str, top_k: int = 2) -> list[str]:
        """Returns the top_k parent recipe documents most similar to the question.

        Retrieval happens at the child-chunk level (title/ingredients/steps
        compared separately) using TF-IDF cosine similarity; each matched
        child chunk is then mapped back to its parent recipe, so the caller
        always gets complete, self-contained documents to pass to the LLM.
        If multiple matched child chunks belong to the same recipe, that
        parent recipe is only returned once (first match wins, since child
        chunks are scanned in descending order of similarity).

        Args:
            question: The user's natural-language question.
            top_k: Maximum number of parent recipes to return.

        Returns:
            Up to top_k parent recipe documents, ordered by descending
            similarity of their best-matching child chunk.
        """
        # Guard against top_k exceeding the number of documents.
        top_k = min(top_k, len(self.documents))

        query_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(query_vector, self.child_vectors)[0]
        ranked_child_indices = similarities.argsort()[::-1]

        # Deduplicate parent chunks: keep first (highest-similarity) match
        # per parent, and stop once top_k distinct parents are found.
        seen_parents = []
        for child_index in ranked_child_indices:
            parent_index = self.child_to_parent[child_index]
            if parent_index not in seen_parents:
                seen_parents.append(parent_index)
            if len(seen_parents) == top_k:
                break

        return [self.documents[i] for i in seen_parents]

    def ask(self, question: str) -> str:
        """Retrieves relevant recipes, then has the LLM answer strictly based on them.

        Args:
            question: The user's natural-language question.

        Returns:
            The generated answer, or one of NO_RESULTS_MESSAGE /
            LLM_UNAVAILABLE_MESSAGE if retrieval or generation fails to
            produce a usable answer. This method does not raise on expected
            LLM-provider failures (auth, network, timeout, API errors); it
            logs them and returns LLM_UNAVAILABLE_MESSAGE instead.
        """
        if not question or not question.strip():
            return "Please enter a question."

        results = self.retrieve(question)
        context = "\n\n".join(results)

        # Grounding prompt: explicitly tells the model to answer using only
        # the retrieved context, and to reply with the fixed message if it
        # can't, which keeps the model from hallucinating recipes.
        prompt = f"""You are a professional recipe assistant.

Answer the question using ONLY the recipe content provided below.
If the answer is not contained in the content, reply exactly: "{NO_RESULTS_MESSAGE}"

Recipe content:
{context}

User question:
{question}

Respond naturally and concisely in English.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except AuthenticationError:
            logger.error("DeepSeek authentication failed - check DEEPSEEK_API_KEY.")
            return LLM_UNAVAILABLE_MESSAGE
        except (APIConnectionError, APITimeoutError):
            logger.error("DeepSeek API connection/timeout error.")
            return LLM_UNAVAILABLE_MESSAGE
        except APIError:
            logger.exception("DeepSeek API returned an error.")
            return LLM_UNAVAILABLE_MESSAGE

        content = response.choices[0].message.content
        return content if content else LLM_UNAVAILABLE_MESSAGE
