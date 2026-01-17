import numpy as np
from openai import OpenAI
from typing import List
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

class AIService:
    def __init__(self, api_key: str, embedding_model: str, chat_model: str):
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model
        self.chat_model = chat_model

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        embeddings: List[np.ndarray] = []
        batch_size = 100
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch,
            )
            for item in response.data:
                embeddings.append(np.array(item.embedding, dtype=np.float32))
        return embeddings

    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""Use the following context to answer the user's question.
If the answer is not in the context, say that you don't know.

Context:
{context}

Question: {question}
Answer:"""
        messages: List[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant."),
            ChatCompletionUserMessageParam(role="user", content=prompt),
        ]
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content or ""
