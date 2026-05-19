from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
import faiss
import numpy as np

load_dotenv()


class SimpleRAG:
    def __init__(self, file_path: str):
        self.model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        self.documents = self.load_documents(file_path)
        self.index = self.build_index(self.documents)

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

    def build_index(self, documents):
        embeddings = self.model.encode(documents)
        embeddings = np.array(embeddings).astype("float32")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        return index

    def retrieve(self, question: str, top_k: int = 2):
        query_embedding = self.model.encode([question])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.documents[idx])

        return results

    def ask(self, question: str):
        results = self.retrieve(question)
        context = "\n\n".join(results)

        prompt = f"""
你是一个专业中文菜谱助手。

请根据提供的菜谱内容，
用自然、友好的中文回答用户问题。

如果是做法类问题：
请按步骤清晰回答。

如果是推荐类问题：
请简洁推荐。

不要直接复制原文。

菜谱内容：
{context}

用户问题：
{question}
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