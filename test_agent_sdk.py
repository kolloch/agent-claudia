"""Interactive TUI chat with the Claude Agent SDK using Textual.

Also embeds a FastAPI REST API on REST_API_PORT so you can send messages
programmatically — they appear in the TUI chat log in real time:

    curl -s -X POST http://localhost:8001/chat \\
         -H "Content-Type: application/json" \\
         -d '{"prompt": "What files are in this project?"}' | python -m json.tool
"""

import asyncio
import queue
import threading
import time
import traceback
import uuid

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Footer, Header, Markdown, Select, Static, TextArea

MODELS = [
    ("Haiku 4.5 — fast & cheap", "claude-haiku-4-5"),
    ("Sonnet 4.6 — balanced", "claude-sonnet-4-6"),
    ("Opus 4.6 — most capable", "claude-opus-4-6"),
]
DEFAULT_MODEL = "claude-haiku-4-5"
REST_API_PORT = 8001

# ---------------------------------------------------------------------------
# Embedded REST API (runs in a background thread, communicates with the TUI)
# ---------------------------------------------------------------------------

_DEFAULT_CWD = "/workspaces/agent-claudia"
_DEFAULT_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "Bash"]


class _RestChatRequest(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL
    max_turns: int = 10


class _RestChatResponse(BaseModel):
    response: str
    num_turns: int
    duration_ms: float
    total_cost_usd: float | None


# Thread-safe queues bridging FastAPI ↔ TUI
_incoming: queue.Queue[tuple[str, str, str, int]] = queue.Queue()
_outgoing: dict[str, queue.Queue[_RestChatResponse]] = {}

rest_app = FastAPI(title="agent-claudia (embedded)")


@rest_app.get("/health")
async def rest_health() -> dict[str, str]:
    return {"status": "ok"}


@rest_app.post("/chat")
async def rest_chat(req: _RestChatRequest) -> _RestChatResponse:
    """Enqueue a prompt for the TUI to process and wait for the result."""
    request_id = uuid.uuid4().hex
    response_q: queue.Queue[_RestChatResponse] = queue.Queue()
    _outgoing[request_id] = response_q
    _incoming.put((request_id, req.prompt, req.model, req.max_turns))

    loop = asyncio.get_running_loop()
    try:
        result: _RestChatResponse = await loop.run_in_executor(
            None,
            response_q.get,
            True,
            600,  # 10-min timeout
        )
    except queue.Empty:
        _outgoing.pop(request_id, None)
        return _RestChatResponse(
            response="Error: query timed out after 600 s",
            num_turns=0,
            duration_ms=0.0,
            total_cost_usd=None,
        )
    finally:
        _outgoing.pop(request_id, None)

    return result


def _run_rest_server() -> None:
    """Run uvicorn in its own thread with a fresh event loop."""
    uvicorn.run(
        rest_app,
        host="0.0.0.0",
        port=REST_API_PORT,
        log_level="warning",
    )


class ChatInput(TextArea):
    """Multi-line text input that submits on Enter and inserts newlines on Shift+Enter."""

    class Submitted(Message):
        """Posted when the user presses Enter."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def _on_key(self, event: object) -> None:
        key = getattr(event, "key", "")
        if key != "enter":
            return
        shift = getattr(event, "shift", False)
        if shift:
            return  # let TextArea handle Shift+Enter as newline
        event.prevent_default()  # type: ignore[attr-defined]
        prompt = self.text.strip()
        if prompt:
            self.clear()
            self.post_message(self.Submitted(prompt))


class ProgressLine(Static):
    """A dim line showing agent progress (thinking, tool use, etc.)."""

    DEFAULT_CSS = """
    ProgressLine {
        color: $text-muted;
        margin: 0 1;
    }
    """


class UserBubble(Static):
    """A user message."""

    DEFAULT_CSS = """
    UserBubble {
        background: $primary-background;
        color: $text;
        padding: 0 1;
        margin: 1 1 0 4;
    }
    """


class AssistantMarkdown(Markdown):
    """Markdown block for assistant responses."""

    DEFAULT_CSS = """
    AssistantMarkdown {
        margin: 1 4 0 1;
    }
    """


class StatusBar(Static):
    """Shows working/ready status with elapsed time and token usage."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }
    StatusBar.working {
        color: $warning;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._start_time: float | None = None
        self._timer_handle: object | None = None
        self._input_tokens: int = 0
        self._output_tokens: int = 0

    def add_usage(self, usage: dict[str, object]) -> None:
        self._input_tokens += int(usage.get("input_tokens", 0))
        self._output_tokens += int(usage.get("output_tokens", 0))

    def set_working(self) -> None:
        self._start_time = time.monotonic()
        self.add_class("working")
        self._update_label()
        self._timer_handle = self.set_interval(1, self._update_label)

    def set_ready(self) -> None:
        if self._timer_handle is not None:
            self._timer_handle.stop()
            self._timer_handle = None
        elapsed = self._format_elapsed()
        self._start_time = None
        self.remove_class("working")
        tokens = self._format_tokens()
        if elapsed:
            self.update(f"Ready — last query took {elapsed}  {tokens}")
        else:
            self.update(f"Ready  {tokens}")

    def _update_label(self) -> None:
        elapsed = self._format_elapsed()
        tokens = self._format_tokens()
        self.update(f"⏳ Working… {elapsed}  {tokens}")

    def _format_elapsed(self) -> str:
        if self._start_time is None:
            return ""
        seconds = int(time.monotonic() - self._start_time)
        if seconds < 60:
            return f"{seconds}s"
        return f"{seconds // 60}m {seconds % 60}s"

    def _format_tokens(self) -> str:
        if self._input_tokens == 0 and self._output_tokens == 0:
            return ""
        return f"tokens: {self._format_k(self._input_tokens)} in / {self._format_k(self._output_tokens)} out"

    @staticmethod
    def _format_k(n: int) -> str:
        if n >= 1000:
            return f"{n / 1000:.1f}k"
        return str(n)


def _truncate(text: str, max_len: int = 500) -> str:
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


class RestBubble(Static):
    """A message originating from the REST API (visually distinct from typed input)."""

    DEFAULT_CSS = """
    RestBubble {
        background: $secondary-background;
        color: $text;
        padding: 0 1;
        margin: 1 1 0 4;
    }
    """


class ChatApp(App[None]):
    """Interactive Claude Agent SDK chat."""

    TITLE = "Claude Agent SDK"
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]

    CSS = """
    #chat-log {
        height: 1fr;
    }
    #input-row {
        dock: bottom;
        height: auto;
        max-height: 8;
        margin: 0 1;
    }
    #input {
        width: 1fr;
    }
    #model-select {
        width: 28;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._working = False
        self._model = DEFAULT_MODEL
        self._rest_thread: threading.Thread | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="chat-log")
        yield StatusBar()
        with Horizontal(id="input-row"):
            yield ChatInput(id="input", language=None)
            yield Select(MODELS, value=DEFAULT_MODEL, id="model-select")
        yield Footer()

    def on_mount(self) -> None:
        text_area = self.query_one("#input", ChatInput)
        text_area.show_line_numbers = False
        text_area.focus()
        self.query_one(StatusBar).set_ready()
        self.sub_title = f"REST API → http://localhost:{REST_API_PORT}/chat"
        # Start embedded REST server in a daemon thread
        self._rest_thread = threading.Thread(
            target=_run_rest_server,
            daemon=True,
            name="rest-api",
        )
        self._rest_thread.start()
        # Poll the incoming REST queue ~4 times per second
        self.set_interval(0.25, self._poll_rest_queue)

    def _poll_rest_queue(self) -> None:
        """Check whether the REST API has enqueued a chat request."""
        try:
            request_id, prompt, model, max_turns = _incoming.get_nowait()
        except queue.Empty:
            return
        self._run_rest_query(request_id, prompt, model, max_turns)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            self._model = str(event.value)

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        if self._working:
            return  # ignore submissions while a query is running
        self._working = True
        self._run_query(event.value)

    # ------------------------------------------------------------------
    # Shared query logic (used by both TUI input and REST API)
    # ------------------------------------------------------------------

    async def _stream_query(
        self,
        prompt: str,
        model: str,
        max_turns: int,
    ) -> tuple[str, ResultMessage | None]:
        """Run a query, stream progress to the TUI, and return the text + result."""
        log = self.query_one("#chat-log", VerticalScroll)
        status = self.query_one(StatusBar)
        status.set_working()

        text_parts: list[str] = []
        result_msg: ResultMessage | None = None
        last_was_progress = False

        try:
            async for message in query(
                prompt=prompt,
                options=ClaudeAgentOptions(
                    cwd=_DEFAULT_CWD,
                    allowed_tools=_DEFAULT_TOOLS,
                    permission_mode="acceptEdits",
                    model=model,
                    max_turns=max_turns,
                ),
            ):
                if isinstance(message, SystemMessage):
                    await log.mount(
                        ProgressLine(
                            Text(f"[system] {message.subtype}", style="italic")
                        )
                    )
                    last_was_progress = True

                elif isinstance(message, AssistantMessage):
                    if message.usage:
                        status.add_usage(message.usage)

                    for block in message.content:
                        if isinstance(block, ThinkingBlock):
                            thinking = _truncate(block.thinking, 200)
                            line = Text()
                            line.append("thinking ", style="bold")
                            line.append(thinking, style="italic")
                            await log.mount(ProgressLine(line))
                            last_was_progress = True

                        elif isinstance(block, ToolUseBlock):
                            tool_input = block.input
                            if isinstance(tool_input, dict):
                                details = "  ".join(
                                    f"{k}={v!r}" for k, v in tool_input.items()
                                )
                            else:
                                details = str(tool_input)
                            line = Text()
                            line.append(f"{block.name} ", style="bold yellow")
                            line.append(_truncate(details))
                            await log.mount(ProgressLine(line))
                            last_was_progress = True

                        elif isinstance(block, TextBlock):
                            text_parts.append(block.text)
                            if last_was_progress:
                                last_was_progress = False
                            await log.mount(AssistantMarkdown(block.text))

                        else:
                            block_type = getattr(block, "type", type(block).__name__)
                            content = _truncate(str(block))
                            await log.mount(
                                ProgressLine(
                                    Text(f"  [{block_type}] {content}", style="dim")
                                )
                            )
                            last_was_progress = True

                elif isinstance(message, UserMessage):
                    content_list = message.content
                    if isinstance(content_list, list):
                        for block in content_list:
                            if isinstance(block, ToolResultBlock):
                                result = (
                                    str(block.content) if block.content else "(empty)"
                                )
                                is_error = getattr(block, "is_error", False)
                                style = "red" if is_error else "green"
                                await log.mount(
                                    ProgressLine(
                                        Text(f"  → {_truncate(result)}", style=style)
                                    )
                                )
                            else:
                                await log.mount(
                                    ProgressLine(
                                        Text(f"  {_truncate(str(block))}", style="dim")
                                    )
                                )
                    elif isinstance(content_list, str):
                        await log.mount(
                            ProgressLine(
                                Text(f"  {_truncate(content_list)}", style="dim")
                            )
                        )
                    last_was_progress = True

                elif isinstance(message, ResultMessage):
                    result_msg = message
                    if message.errors:
                        for err in message.errors:
                            await log.mount(
                                ProgressLine(Text(f"  ⚠ {err}", style="bold red"))
                            )
                    if message.is_error and message.result:
                        await log.mount(
                            ProgressLine(
                                Text(f"  error: {message.result}", style="bold red")
                            )
                        )

                    parts: list[str] = []
                    if message.stop_reason:
                        parts.append(message.stop_reason)
                    parts.append(f"{message.num_turns} turns")
                    duration_s = message.duration_ms / 1000
                    parts.append(f"{duration_s:.1f}s")
                    if message.total_cost_usd is not None:
                        parts.append(f"${message.total_cost_usd:.4f}")
                    style = "bold red" if message.is_error else "italic"
                    await log.mount(
                        ProgressLine(Text(f"done ({', '.join(parts)})", style=style))
                    )

                else:
                    msg_type = type(message).__name__
                    await log.mount(
                        ProgressLine(
                            Text(f"[{msg_type}] {_truncate(str(message))}", style="dim")
                        )
                    )

                log.scroll_end(animate=False)

        except Exception as exc:
            error_type = type(exc).__name__
            error_msg = str(exc) if str(exc) else "(no message)"
            tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
            tb_str = "".join(tb_lines)
            await log.mount(
                ProgressLine(Text(f"  ❌ {error_type}: {error_msg}", style="bold red"))
            )
            await log.mount(
                ProgressLine(
                    Text(f"  Traceback:\n{_truncate(tb_str, 1500)}", style="dim red")
                )
            )
            log.scroll_end(animate=False)

        status.set_ready()
        return "\n".join(text_parts), result_msg

    # ------------------------------------------------------------------
    # TUI-typed query
    # ------------------------------------------------------------------

    @work
    async def _run_query(self, prompt: str) -> None:
        log = self.query_one("#chat-log", VerticalScroll)
        await log.mount(UserBubble(f"> {prompt}"))
        log.scroll_end(animate=False)

        await self._stream_query(prompt, self._model, max_turns=10)

        self._working = False
        self.query_one("#input", ChatInput).focus()

    # ------------------------------------------------------------------
    # REST API query (displayed in TUI, result returned to HTTP caller)
    # ------------------------------------------------------------------

    @work
    async def _run_rest_query(
        self,
        request_id: str,
        prompt: str,
        model: str,
        max_turns: int,
    ) -> None:
        log = self.query_one("#chat-log", VerticalScroll)
        await log.mount(RestBubble(f"[REST] > {prompt}"))
        log.scroll_end(animate=False)

        response_text, result_msg = await self._stream_query(prompt, model, max_turns)

        # Send the result back to the waiting REST handler
        response_q = _outgoing.get(request_id)
        if response_q is not None:
            response_q.put(
                _RestChatResponse(
                    response=response_text,
                    num_turns=result_msg.num_turns if result_msg else 0,
                    duration_ms=result_msg.duration_ms if result_msg else 0.0,
                    total_cost_usd=result_msg.total_cost_usd if result_msg else None,
                )
            )


if __name__ == "__main__":
    ChatApp().run()
