
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRetriever:

    def __init__(
        self,
        documents,
        model_name="all-MiniLM-L6-v2"
    ):

        self.model = SentenceTransformer(model_name)

        self.document_embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )

    def search(self, query, top_k=10):

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        scores = self.document_embeddings @ query_embedding

        indices = np.argsort(scores)[::-1][:top_k]

        return [
            (int(i), float(scores[i]))
            for i in indices
        ]
