from backend.app.qdrant_config import client


COLLECTION_NAME = "documents"


def inspect_chunks():
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=20,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            print("=" * 80)
            print(f"Chunk ID: {point.payload.get('chunk_id')}")
            print(f"Source: {point.payload.get('source')}")
            print()
            print(point.payload.get("text"))

        if offset is None:
            break


if __name__ == "__main__":
    inspect_chunks()