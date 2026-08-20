from evaluation.metrics.retrival_metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)


retrieved = [54, 16, 89, 27, 31]
relevant = [16]


print("Hit@5:", hit_at_k(retrieved, relevant, 5))

print("Recall@5:", recall_at_k(retrieved, relevant, 5))

print("Reciprocal Rank:", reciprocal_rank(retrieved, relevant))