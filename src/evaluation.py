
import math


def precision_at_k(ranked_ids, relevant_ids, k=10):

    top_results = ranked_ids[:k]

    if not top_results:
        return 0.0

    relevant_count = sum(
        1 for doc in top_results
        if doc in relevant_ids
    )

    return relevant_count / k


def recall_at_k(ranked_ids, relevant_ids, k=10):

    if not relevant_ids:
        return 0.0

    relevant_count = sum(
        1 for doc in ranked_ids[:k]
        if doc in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def average_precision(ranked_ids, relevant_ids):

    if not relevant_ids:
        return 0.0

    hits = 0
    score = 0.0

    for rank, doc in enumerate(ranked_ids, start=1):

        if doc in relevant_ids:

            hits += 1
            score += hits / rank

    return score / len(relevant_ids)


def ndcg_at_k(ranked_ids, relevant_ids, k=10):

    dcg = 0.0

    for rank, doc in enumerate(
        ranked_ids[:k], start=1
    ):

        if doc in relevant_ids:
            dcg += 1 / math.log2(rank + 1)

    ideal_count = min(len(relevant_ids), k)

    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_ranking(
    ranked_ids,
    relevant_ids,
    k=10
):

    return {
        "Precision@10": precision_at_k(
            ranked_ids, relevant_ids, k
        ),

        "Recall@10": recall_at_k(
            ranked_ids, relevant_ids, k
        ),

        "MAP": average_precision(
            ranked_ids, relevant_ids
        ),

        "nDCG@10": ndcg_at_k(
            ranked_ids, relevant_ids, k
        )
    }
