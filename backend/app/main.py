from fastapi import FastAPI

app = FastAPI(
    title="RAG Evaluation Harness",
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "message": "RAG Evaluation Harness API is running"
    }