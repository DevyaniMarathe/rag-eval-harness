from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def generate_embedding(text: str) -> list[float]:
    """
    Convert text into a 384-dimensional embedding vector.
    """

    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert multiple text chunks into embedding vectors.
    """

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings.tolist()


if __name__ == "__main__":
    test_text = "Plant disease detection using artificial intelligence."

    embedding = generate_embedding(test_text)

    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")