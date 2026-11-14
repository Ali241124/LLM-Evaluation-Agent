
import requests
import numpy as np


class SemanticSimilarityEvaluator:

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text:latest"
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def get_embedding(self, text: str) -> list[float]:
        """
        Convert text into a numerical embedding using Ollama.
        """

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        url = f"{self.base_url}/api/embeddings"

        payload = {
            "model": self.model,
            "prompt": text
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=(10, 120)
            )

            response.raise_for_status()

            result = response.json()

            embedding = result.get("embedding")

            if not embedding:
                raise RuntimeError(
                    "Ollama returned an empty embedding."
                )

            return embedding

        except requests.exceptions.RequestException as error:
            raise RuntimeError(
                f"Error generating embedding: {error}"
            ) from error

    def cosine_similarity(
        self,
        embedding_a: list[float],
        embedding_b: list[float]
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        """

        vector_a = np.array(embedding_a, dtype=float)
        vector_b = np.array(embedding_b, dtype=float)

        if vector_a.shape != vector_b.shape:
            raise ValueError(
                "Embedding dimensions must be identical."
            )

        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        if norm_a == 0 or norm_b == 0:
            raise ValueError(
                "Cannot compare zero-length embeddings."
            )

        similarity = np.dot(vector_a, vector_b) / (
            norm_a * norm_b
        )

        return round(float(similarity), 4)

    def evaluate(
        self,
        predicted: str,
        expected: str
    ) -> float:
        """
        Compare the semantic meaning of two answers.
        """

        predicted_embedding = self.get_embedding(predicted)
        expected_embedding = self.get_embedding(expected)

        return self.cosine_similarity(
            predicted_embedding,
            expected_embedding
        )
