
def reciprocal_rank_fusion(
    result_lists,
    k=60,
    top_k=10
):

    fused_scores = {}

    for results in result_lists:

        for rank, (doc_id, score) in enumerate(
            results, start=1
        ):

            rrf_score = 1 / (k + rank)

            fused_scores[doc_id] = (
                fused_scores.get(doc_id, 0)
                + rrf_score
            )

    ranked_results = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_results[:top_k]
