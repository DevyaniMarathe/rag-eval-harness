def hit_at_k(retrieved_chunk_ids, relevant_chunk_ids, k):
    """
    Returns 1 if at least one relevant chunk appears
    in the top-k retrieved results, otherwise 0.
    """

    top_k = retrieved_chunk_ids[:k]

    return int(
        any(chunk_id in relevant_chunk_ids for chunk_id in top_k)
    )


def recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, k):
    """
    Measures the fraction of relevant chunks retrieved
    within the top-k results.
    """

    top_k = retrieved_chunk_ids[:k]

    if not relevant_chunk_ids:
        return 0.0

    retrieved_relevant = sum(
        chunk_id in top_k
        for chunk_id in relevant_chunk_ids
    )

    return retrieved_relevant / len(relevant_chunk_ids)


def reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids):
    """
    Returns the reciprocal rank of the first relevant result.
    """

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):

        if chunk_id in relevant_chunk_ids:
            return 1 / rank

    return 0.0


def mean_reciprocal_rank(all_reciprocal_ranks):
    """
    Calculates MRR across multiple questions.
    """

    if not all_reciprocal_ranks:
        return 0.0

    return sum(all_reciprocal_ranks) / len(all_reciprocal_ranks)