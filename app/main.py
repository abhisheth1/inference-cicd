from fastapi import FastAPI

app = FastAPI(title="Inference API", version="1.0.0")


@app.get("/")
def root():
    return {"message": "service is running"}


@app.get("/health")
def health():
    return {"status": "ok"}