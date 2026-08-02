import logging
import re
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError, AuthenticationError
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

load_dotenv()

logger = logging.getLogger("ragchef")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
NO_RESULTS_MESSAGE = "No relevant information was found in the current recipe library."
LLM_UNAVAILABLE_MESSAGE = (
    "Sorry, the recipe assistant is temporarily unavailable. Please try again in a moment."
)


class RAGConfigError(RuntimeError):
    """Raised when SimpleRAG cannot be initialized (missing API key, empty KB, etc.)."""


class SimpleRAG:
    def __init__(self, file_path: str):
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

        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)

        self.model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    def load_documents(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError as e:
            raise RAGConfigError(f"Recipe knowledge base not found at {file_path}") from e

        # Split on top-level "# Heading" lines only. A naive text.split("# ")
        # also matches "## Ingredients" / "## Steps" sub-headings and shreds
        # every recipe into fragments, which silently ruins retrieval.
        parts = re.split(r"(?m)^# ", text)

        docs = [part.strip() for part in parts if part.strip()]

        return docs

    def retrieve(self, question: str, top_k: int = 2):
        top_k = min(top_k, len(self.documents))
        query_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]

        top_indices = similarities.argsort()[-top_k:][::-1]

        return [self.documents[i] for i in top_indices]

    def ask(self, question: str) -> str:
        if not question or not question.strip():
            return "Please enter a question."

        results = self.retrieve(question)
        context = "\n\n".join(results)

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
