import os
import json

from backend.app.retriever import retrieve

from evaluation.metrics.retrival_metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)


DATASET_PATH = "evaluation/datasets/retrieval_test.json"
RESULTS_DIR = "evaluation/results"
REPORT_PATH = os.path.join(RESULTS_DIR, "baseline_detailed.json")


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

    # Store metrics for calculating averages
    hit_1_scores = []
    hit_3_scores = []
    hit_5_scores = []

    recall_5_scores = []
    reciprocal_ranks = []

    # Store detailed results for every question
    question_results = []

    # ---------------------------------------------------------
    # Evaluate every question
    # ---------------------------------------------------------

    for item in dataset:

        question = item["question"]
        relevant_chunk_ids = item["relevant_chunk_ids"]

        print("-" * 70)
        print(f"Question {item['id']}: {question}")

        # Retrieve top 5 results from Qdrant
        results = retrieve(
            question,
            top_k=5
        )

        # Extract chunk IDs
        retrieved_chunk_ids = [
            result.payload.get("chunk_id")
            for result in results
        ]

        # Extract similarity scores
        similarity_scores = [
            result.score
            for result in results
        ]

        print(
            f"Retrieved chunks: {retrieved_chunk_ids}"
        )

        print(
            f"Relevant chunks:  {relevant_chunk_ids}"
        )

        # -----------------------------------------------------
        # Calculate metrics
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Store aggregate metric values
        # -----------------------------------------------------

        hit_1_scores.append(hit1)
        hit_3_scores.append(hit3)
        hit_5_scores.append(hit5)

        recall_5_scores.append(recall5)
        reciprocal_ranks.append(rr)

        # -----------------------------------------------------
        # Store detailed result for this question
        # -----------------------------------------------------

        question_results.append({
            "question_id": item["id"],
            "question": question,
            "relevant_chunk_ids": relevant_chunk_ids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "similarity_scores": similarity_scores,
            "hit_at_1": hit1,
            "hit_at_3": hit3,
            "hit_at_5": hit5,
            "recall_at_5": recall5,
            "reciprocal_rank": rr
        })

        # -----------------------------------------------------
        # Print question-level metrics
        # -----------------------------------------------------

        print(f"Hit@1:    {hit1}")
        print(f"Hit@3:    {hit3}")
        print(f"Hit@5:    {hit5}")
        print(f"Recall@5: {recall5:.2f}")
        print(f"RR:       {rr:.2f}")

    # =========================================================
    # Calculate final averages
    # =========================================================

    total_questions = len(dataset)

    hit1_avg = (
        sum(hit_1_scores) / total_questions
    )

    hit3_avg = (
        sum(hit_3_scores) / total_questions
    )

    hit5_avg = (
        sum(hit_5_scores) / total_questions
    )

    recall5_avg = (
        sum(recall_5_scores) / total_questions
    )

    mrr = mean_reciprocal_rank(
        reciprocal_ranks
    )

    # =========================================================
    # Create final report
    # =========================================================

    report = {
        "experiment": "baseline",

        "questions_evaluated": total_questions,

        "retrieval_configuration": {
            "top_k": 5,
            "vector_database": "Qdrant",
            "embedding_model": "all-MiniLM-L6-v2"
        },

        "metrics": {
            "hit_at_1": hit1_avg,
            "hit_at_3": hit3_avg,
            "hit_at_5": hit5_avg,
            "recall_at_5": recall5_avg,
            "mrr": mrr
        },

        "questions": question_results
    }

    # =========================================================
    # Save detailed JSON report
    # =========================================================

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    # =========================================================
    # Print final evaluation
    # =========================================================

    print()
    print("=" * 70)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 70)

    print(
        f"Questions evaluated: {total_questions}"
    )

    print()

    print(
        f"Hit@1:    {hit1_avg:.3f}"
    )

    print(
        f"Hit@3:    {hit3_avg:.3f}"
    )

    print(
        f"Hit@5:    {hit5_avg:.3f}"
    )

    print(
        f"Recall@5: {recall5_avg:.3f}"
    )

    print(
        f"MRR:      {mrr:.3f}"
    )

    print("=" * 70)

    print()
    print("Detailed report saved to:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()