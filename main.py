import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

app = FastAPI(title="agent-claudia")

_DEFAULT_CWD = "/workspaces/agent-claudia"
_DEFAULT_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "Bash"]


class ChatRequest(BaseModel):
    prompt: str
    model: str = "claude-haiku-4-5"
    max_turns: int = 10


class ChatResponse(BaseModel):
    response: str
    num_turns: int
    duration_ms: float
    total_cost_usd: float | None


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello from agent-claudia!"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    """Run a prompt through the Claude Agent SDK and return the text response."""
    text_parts: list[str] = []
    result_msg: ResultMessage | None = None

    async for message in query(
        prompt=req.prompt,
        options=ClaudeAgentOptions(
            cwd=_DEFAULT_CWD,
            allowed_tools=_DEFAULT_TOOLS,
            permission_mode="acceptEdits",
            model=req.model,
            max_turns=req.max_turns,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
        elif isinstance(message, ResultMessage):
            result_msg = message

    return ChatResponse(
        response="\n".join(text_parts),
        num_turns=result_msg.num_turns if result_msg else 0,
        duration_ms=result_msg.duration_ms if result_msg else 0.0,
        total_cost_usd=result_msg.total_cost_usd if result_msg else None,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
