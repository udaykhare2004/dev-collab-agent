#!/usr/bin/env python3
"""CLI entrypoint: index and ask commands."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Ensure project root is importable when invoked as `python cli.py`
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent.core import run_agent  # noqa: E402
from src.openai_client import create_openai_client  # noqa: E402
from src.storage.index_manager import IndexManager  # noqa: E402
from src.tools.agent_tools import AgentToolContext  # noqa: E402

console = Console()
STORAGE_DIR = _PROJECT_ROOT / "storage"


def _require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[red]OPENAI_API_KEY is not set.[/red] Copy `.env.example` to `.env` and add your key."
        )
        sys.exit(1)


def cmd_index(path: str | None) -> None:
    target = Path(path or ".").expanduser().resolve()
    if not target.is_dir():
        console.print(f"[red]Directory not found:[/red] {target}")
        sys.exit(1)

    load_dotenv(_PROJECT_ROOT / ".env")
    manager = IndexManager(STORAGE_DIR)

    def progress(msg: str) -> None:
        console.print(f"[cyan]•[/cyan] {msg}")

    try:
        chunks = manager.index_repository(target, on_progress=progress)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    except TimeoutError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Indexing failed:[/red] {e}")
        sys.exit(1)

    console.print(
        Panel.fit(
            f"Indexed [green]{len(chunks)}[/green] chunks from\n[bold]{target}[/bold]",
            title="Success",
            border_style="green",
        )
    )


def cmd_ask(question: str) -> None:
    _require_api_key()
    load_dotenv(_PROJECT_ROOT / ".env")

    model = os.getenv("OPENAI_AGENT_MODEL", "gpt-4o-mini")
    client = create_openai_client()
    manager = IndexManager(STORAGE_DIR)

    try:
        repo_root = manager.load_repo_root()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    ctx = AgentToolContext(repo_root=repo_root, retriever=manager.retriever)

    def log_tool(name: str, args: dict) -> None:
        console.print(
            Panel(
                f"[bold yellow]{name}[/bold yellow]\n{args}",
                title="Tool invocation",
                border_style="yellow",
            )
        )

    console.print(Panel(question, title="Question", border_style="blue"))

    try:
        answer = run_agent(
            question=question,
            ctx=ctx,
            client=client,
            model=model,
            on_tool_call=log_tool,
        )
    except TimeoutError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Agent failed:[/red] {e}")
        sys.exit(1)

    console.print(Panel(Markdown(answer), title="Answer", border_style="green"))


def main() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Developer Collaboration RAG Agent — index codebases and ask questions."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index_p = sub.add_parser("index", help="Chunk, enrich, and index a repository")
    index_p.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Target directory (default: current directory)",
    )

    ask_p = sub.add_parser("ask", help="Ask a question about the indexed repository")
    ask_p.add_argument("question", type=str, help="Natural language question")

    args = parser.parse_args()
    if args.command == "index":
        cmd_index(args.path)
    elif args.command == "ask":
        cmd_ask(args.question)


if __name__ == "__main__":
    main()
