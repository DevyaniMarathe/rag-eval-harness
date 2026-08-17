from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


client = QdrantClient(
    host="localhost",
    port=6333
)


def create_collection():
    client.create_collection(
        collection_name="documents",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )


if __name__ == "__main__":
    create_collection()
    print("Collection created successfully!")