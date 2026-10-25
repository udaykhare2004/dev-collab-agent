"""Multi-step OpenAI tool-calling agent loop."""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from openai import APITimeoutError, OpenAI

from src.agent.prompts import SYSTEM_PROMPT
from src.openai_client import create_openai_client
from src.tools.agent_tools import AgentToolContext, build_tool_handlers, get_openai_tools_schema

MAX_TOOL_HOPS = 6
DEFAULT_MODEL = "gpt-4o-mini"


def run_agent(
    question: str,
    ctx: AgentToolContext,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_MODEL,
    on_tool_call: Optional[Callable[[str, dict[str, Any]], None]] = None,
) -> str:
    """
    Run autonomous agent until final assistant message or max tool hops.
    """
    openai_client = client or create_openai_client()
    handlers = build_tool_handlers(ctx)
    tools = get_openai_tools_schema()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_TOOL_HOPS):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                timeout=120.0,
            )
        except APITimeoutError as e:
            raise TimeoutError("OpenAI chat completion timed out") from e

        choice = response.choices[0]
        message = choice.message

        if message.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )
            for tc in message.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if on_tool_call:
                    on_tool_call(name, args)
                handler = handlers.get(name)
                if handler is None:
                    result = f"Unknown tool: {name}"
                else:
                    try:
                        result = handler(**args)
                    except TypeError as e:
                        result = f"Tool argument error: {e}"
                    except Exception as e:
                        result = f"Tool execution failed: {e}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
            continue

        if message.content:
            return message.content.strip()

        return "The agent did not produce a final answer."

    # Final turn without tools to force an answer
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages
            + [
                {
                    "role": "user",
                    "content": "Provide your final answer now using the evidence gathered. Cite file:line ranges.",
                }
            ],
            timeout=120.0,
        )
    except APITimeoutError as e:
        raise TimeoutError("OpenAI chat completion timed out") from e

    final = response.choices[0].message.content
    return (final or "Max tool hops reached without a complete answer.").strip()
