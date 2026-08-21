from backend.app.qdrant_config import client

COLLECTION_NAME = "documents"

chunk_id = int(input("Enter chunk ID: "))

results = client.scroll(
    collection_name=COLLECTION_NAME,
    scroll_filter={
        "must": [
            {
                "key": "chunk_id",
                "match": {
                    "value": chunk_id
                }
            }
        ]
    },
    limit=1,
    with_payload=True,
)

points = results[0]

if not points:
    print(f"\nChunk {chunk_id} not found.")
else:
    point = points[0]

    print("\n" + "=" * 70)
    print(f"Chunk ID: {chunk_id}")
    print("=" * 70)
    print(f"Source: {point.payload.get('source')}")
    print("\nText:\n")
    print(point.payload.get("text"))