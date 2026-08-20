import json

from backend.app.retriever import retrieve

from evaluation.metrics.retrival_metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)


DATASET_PATH = "evaluation/datasets/retrieval_test.json"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def main():

    dataset = load_dataset()

    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"Questions: {len(dataset)}")
    print()

    hit_1_scores = []
    hit_3_scores = []
    hit_5_scores = []

    recall_5_scores = []
    reciprocal_ranks = []

    for item in dataset:

        question = item["question"]
        relevant_chunk_ids = item["relevant_chunk_ids"]

        print("-" * 70)
        print(f"Question {item['id']}: {question}")

        # Retrieve top 5 results
        results = retrieve(question, top_k=5)

        # Extract chunk IDs returned by Qdrant
        retrieved_chunk_ids = [
            result.payload.get("chunk_id")
            for result in results
        ]

        print(f"Retrieved chunks: {retrieved_chunk_ids}")
        print(f"Relevant chunks:  {relevant_chunk_ids}")

        # Calculate metrics
        hit1 = hit_at_k(
            retrieved_chunk_ids,
            relevant_chunk_ids,
            1
        )

        hit3 = hit_at_k(
            retrieved_chunk_ids,
            relevant_chunk_ids,
            3
        )

        hit5 = hit_at_k(
            retrieved_chunk_ids,
            relevant_chunk_ids,
            5
        )

        recall5 = recall_at_k(
            retrieved_chunk_ids,
            relevant_chunk_ids,
            5
        )

        rr = reciprocal_rank(
            retrieved_chunk_ids,
            relevant_chunk_ids
        )

        hit_1_scores.append(hit1)
        hit_3_scores.append(hit3)
        hit_5_scores.append(hit5)

        recall_5_scores.append(recall5)
        reciprocal_ranks.append(rr)

        print(f"Hit@1:   {hit1}")
        print(f"Hit@3:   {hit3}")
        print(f"Hit@5:   {hit5}")
        print(f"Recall@5:{recall5:.2f}")
        print(f"RR:      {rr:.2f}")

    # Calculate averages
    total_questions = len(dataset)

    hit1_avg = sum(hit_1_scores) / total_questions
    hit3_avg = sum(hit_3_scores) / total_questions
    hit5_avg = sum(hit_5_scores) / total_questions

    recall5_avg = sum(recall_5_scores) / total_questions

    mrr = mean_reciprocal_rank(reciprocal_ranks)

    print()
    print("=" * 70)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"Questions evaluated: {total_questions}")
    print()

    print(f"Hit@1:    {hit1_avg:.3f}")
    print(f"Hit@3:    {hit3_avg:.3f}")
    print(f"Hit@5:    {hit5_avg:.3f}")
    print(f"Recall@5: {recall5_avg:.3f}")
    print(f"MRR:      {mrr:.3f}")

    print("=" * 70)


if __name__ == "__main__":
    main()