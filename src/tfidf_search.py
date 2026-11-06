
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRetriever:

    def __init__(self, documents):

        self.vectorizer = TfidfVectorizer(
            max_features=100000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        self.matrix = self.vectorizer.fit_transform(documents)

    def search(self, query, top_k=10):

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        indices = scores.argsort()[::-1][:top_k]

        return [
            (int(i), float(scores[i]))
            for i in indices
            if scores[i] > 0
        ]
