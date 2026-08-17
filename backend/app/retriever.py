from backend.app.embedding_service import generate_embedding
from backend.app.qdrant_config import client


COLLECTION_NAME = "documents"


def retrieve(
    query: str,
    top_k: int = 5
) -> list:
    """
    Retrieve the most relevant document chunks for a query.
    """

    # Convert the user's question into an embedding
    query_embedding = generate_embedding(query)

    # Search Qdrant for similar vectors
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    ).points

    return results


if __name__ == "__main__":
    query = input("\nEnter your question: ")

    results = retrieve(query)

    print(f"\nTop {len(results)} results:\n")

    for i, result in enumerate(results, start=1):
        print("=" * 70)
        print(f"Result {i}")
        print(f"Similarity score: {result.score:.4f}")
        print(f"Source: {result.payload.get('source')}")
        print(f"Chunk ID: {result.payload.get('chunk_id')}")
        print("\nText:")
        print(result.payload.get("text"))
        print()