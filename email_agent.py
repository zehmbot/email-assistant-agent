"""Email Assistant: an agent that checks your inbox and drafts replies, safely.

- Authentication: until you log in, the only tool the agent has is `authenticate`.
- Dynamic tools + prompt (middleware): after login it can check the inbox and send emails.
- Human-in-the-loop: the agent pauses before sending any email so you can approve, edit or reject it.

The inbox and sending are simulated (no real emails are read or sent).

Run it in the terminal:   python email_agent.py
Run it in LangGraph Studio / Agent Chat UI: langgraph dev
"""

import asyncio
import os
import warnings
from dataclasses import dataclass, field
from typing import Callable

# Harmless warning from LangGraph when a custom context is passed (the course notebook hides it too).
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*", category=UserWarning)

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    wrap_model_call,
)
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

MODEL = "google_genai:gemini-3.1-flash-lite"


# ---------------------------------------------------------------------------
# Context and state
# ---------------------------------------------------------------------------

@dataclass
class EmailContext:
    # Demo login, read from .env (DEMO_EMAIL / DEMO_PASSWORD) with the course's defaults.
    email_address: str = field(default_factory=lambda: os.getenv("DEMO_EMAIL", "julie@example.com"))
    password: str = field(default_factory=lambda: os.getenv("DEMO_PASSWORD", "password123"))


class AuthenticatedState(AgentState):
    authenticated: bool


def _context(runtime: ToolRuntime) -> EmailContext:
    """Return the run's EmailContext, falling back to the defaults when none was passed
    (for example when chatting from LangGraph Studio or Agent Chat UI)."""
    ctx = runtime.context
    if isinstance(ctx, EmailContext):
        return ctx
    if isinstance(ctx, dict) and ctx:
        return EmailContext(**ctx)
    return EmailContext()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def check_inbox() -> str:
    """Check the inbox for recent emails"""
    return """
    Hi Julie,
    I'm going to be in town next week and was wondering if we could grab a coffee?
    - best, Jane (jane@example.com)
    """


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an response email"""
    return f"Email sent to {to} with subject {subject} and body {body}"


@tool
def authenticate(email: str, password: str, runtime: ToolRuntime) -> Command:
    """Authenticate the user with the given email and password"""
    ctx = _context(runtime)
    if email == ctx.email_address and password == ctx.password:
        return Command(update={
            "authenticated": True,
            "messages": [ToolMessage("Successfully authenticated", tool_call_id=runtime.tool_call_id)],
        })
    return Command(update={
        "authenticated": False,
        "messages": [ToolMessage("Authentication failed", tool_call_id=runtime.tool_call_id)],
    })


# ---------------------------------------------------------------------------
# Middleware: dynamic tools, dynamic prompt
# ---------------------------------------------------------------------------

@wrap_model_call
async def dynamic_tool_call(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Allow read inbox and send email tools only if user provides correct email and password"""
    if request.state.get("authenticated"):
        tools = [check_inbox, send_email]
    else:
        tools = [authenticate]
    return await handler(request.override(tools=tools))


AUTHENTICATED_PROMPT = """You are a helpful assistant that can check the inbox and send emails.
Your first step after authentication is to check the inbox."""
UNAUTHENTICATED_PROMPT = "You are a helpful assistant that can authenticate users."


@dynamic_prompt
def dynamic_prompt_func(request: ModelRequest) -> str:
    """Generate system prompt based on authentication status"""
    if request.state.get("authenticated"):
        return AUTHENTICATED_PROMPT
    return UNAUTHENTICATED_PROMPT


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def build_agent(checkpointer=None):
    return create_agent(
        model=MODEL,
        tools=[authenticate, check_inbox, send_email],
        state_schema=AuthenticatedState,
        context_schema=EmailContext,
        checkpointer=checkpointer,
        middleware=[
            dynamic_tool_call,
            dynamic_prompt_func,
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "authenticate": False,
                    "check_inbox": False,
                    "send_email": True,  # pause for human approval before sending
                }
            ),
        ],
    )


# Used by `langgraph dev` (LangGraph provides its own persistence there).
agent = build_agent()


# ---------------------------------------------------------------------------
# Terminal chat with human-in-the-loop review
# ---------------------------------------------------------------------------

def review(interrupt_value) -> list[dict]:
    """Show each email the agent wants to send and ask the human what to do."""
    decisions = []
    for action in interrupt_value["action_requests"]:
        args = action["args"]
        print("\n" + "=" * 60)
        print("✋ The assistant wants to send this email:")
        print(f"   To:      {args.get('to')}")
        print(f"   Subject: {args.get('subject')}")
        print("   Body:")
        for line in str(args.get("body", "")).splitlines():
            print(f"     {line}")
        print("=" * 60)

        while True:
            choice = input("[a]pprove, [e]dit or [r]eject? ").strip().lower()
            if choice in {"a", "approve"}:
                decisions.append({"type": "approve"})
                break
            if choice in {"e", "edit"}:
                subject = input("New subject (Enter to keep): ").strip() or args.get("subject")
                body = input("New body (Enter to keep): ").strip() or args.get("body")
                decisions.append({
                    "type": "edit",
                    "edited_action": {"name": action["name"], "args": {**args, "subject": subject, "body": body}},
                })
                break
            if choice in {"r", "reject"}:
                reason = input("Why? (optional, sent to the assistant): ").strip()
                decisions.append({"type": "reject", "message": reason or "The user rejected this email. Do not send it."})
                break
    return decisions


async def main():
    chat = build_agent(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    context = EmailContext()

    print("Email Assistant: log in first, then ask me to check your inbox or reply to emails.")
    print(f"Demo login: {context.email_address} / {context.password}   (type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue

        result = await chat.ainvoke({"messages": [HumanMessage(content=user_input)]}, config=config, context=context)
        # Human-in-the-loop: keep resuming until the agent no longer needs a decision.
        while result.get("__interrupt__"):
            decisions = review(result["__interrupt__"][0].value)
            result = await chat.ainvoke(Command(resume={"decisions": decisions}), config=config, context=context)

        print(f"\nAssistant: {result['messages'][-1].text}")


if __name__ == "__main__":
    asyncio.run(main())
