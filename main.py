import uvicorn
from fastapi import FastAPI

app = FastAPI(title="agent-claudia")


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello from agent-claudia!"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
