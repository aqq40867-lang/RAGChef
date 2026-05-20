from openai import OpenAI
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

load_dotenv()


class SimpleRAG:
    def __init__(self, file_path: str):
        self.documents = self.load_documents(file_path)

        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)

        self.client = OpenAI(
            api_key=os.getenv("AIHUBMIX_API_KEY"),
            base_url="https://aihubmix.com/v1"
        )

    def load_documents(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        docs = []
        parts = text.split("# ")

        for part in parts:
            if part.strip():
                docs.append(part.strip())

        return docs

    def retrieve(self, question: str, top_k: int = 2):
        query_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]

        top_indices = similarities.argsort()[-top_k:][::-1]

        return [self.documents[i] for i in top_indices]

    def ask(self, question: str):
        results = self.retrieve(question)
        context = "\n\n".join(results)

        prompt = f"""
你是一个专业中文菜谱助手。

请只根据下面提供的菜谱内容回答问题。
如果内容中没有答案，就说：“当前菜谱库中没有找到相关信息”。

菜谱内容：
{context}

用户问题：
{question}

请用自然、简洁的中文回答。
"""

        response = self.client.chat.completions.create(
            model="gpt-5.5-free",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content