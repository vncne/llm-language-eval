import traceback
import numpy as np
import openai
from openai import OpenAI

class EmbeddingScorer:
    """
    A class that uses OpenAI Embeddings to compute cosine similarity between two strings.
    This module focuses on generating embeddings and calculating cosine similarity.
    """
    def __init__(self, api_key: str, embedding_model: str = "text-embedding-ada-002"):
        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            print("Error initializing OpenAI client:")
            print(traceback.format_exc())
            raise e

        self.embedding_model = embedding_model

    def _get_embedding(self, text: str) -> np.ndarray:
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            # Access the embedding vector from the response using the new interface
            emb = response.data[0].embedding
            return np.array(emb, dtype=np.float32)
        except Exception:
            print("Something wild happened while fetching the embedding. Check the traceback:")
            print(traceback.format_exc())
            raise

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute the cosine similarity between two texts by obtaining their embeddings.
        """
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        sim = dot / (norm1 * norm2)
        return float(sim)
