
import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, documents):

        tokenized_docs = [
            doc.split() for doc in documents
        ]

        self.model = BM25Okapi(tokenized_docs)

    def search(self, query, top_k=10):

        tokens = query.split()

        scores = np.array(
            self.model.get_scores(tokens)
        )

        indices = scores.argsort()[::-1][:top_k]

        return [
            (int(i), float(scores[i]))
            for i in indices
            if scores[i] > 0
        ]
