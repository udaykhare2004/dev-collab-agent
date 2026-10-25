"""LLM-callable tools: hybrid search, file read, git inspect."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from src.storage.hybrid_retriever import HybridRetriever

MAX_READ_LINES = 250


class AgentToolContext:
    def __init__(self, repo_root: Path, retriever: HybridRetriever) -> None:
        self.repo_root = repo_root.expanduser().resolve()
        self.retriever = retriever


def hybrid_search_code(ctx: AgentToolContext, query: str, top_k: int = 5) -> str:
    results = ctx.retriever.hybrid_search(query, top_n=top_k)
    if not results:
        return "No matching code chunks found."

    parts: list[str] = []
    for i, hit in enumerate(results, start=1):
        c = hit.chunk
        author = c.git_author or "unknown"
        commit = c.git_last_commit or "n/a"
        header = (
            f"### Result {i} ({hit.source}, RRF={hit.rrf_score:.4f})\n"
            f"**{c.relative_path}:{c.start_line}-{c.end_line}** | "
            f"Author: {author} | Commit: {commit}\n"
        )
        parts.append(f"{header}\n```\n{c.code}\n```")
    return "\n\n".join(parts)


def read_file_lines(
    ctx: AgentToolContext,
    relative_path: str,
    start_line: int,
    end_line: int,
) -> str:
    if start_line < 1 or end_line < start_line:
        return "Invalid line range: start_line must be >= 1 and end_line >= start_line."

    if end_line - start_line + 1 > MAX_READ_LINES:
        return f"Line window too large (max {MAX_READ_LINES} lines per call)."

    rel = Path(relative_path.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        return "Invalid path: must be relative to repository root without '..'."

    target = (ctx.repo_root / rel).resolve()
    try:
        target.relative_to(ctx.repo_root)
    except ValueError:
        return "Invalid path: escapes repository root."

    if not target.is_file():
        return f"File not found: {relative_path}"

    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = target.read_text(encoding="latin-1").splitlines()
    except OSError as e:
        return f"Failed to read file: {e}"

    end_line = min(end_line, len(lines))
    if start_line > len(lines):
        return f"start_line {start_line} exceeds file length ({len(lines)} lines)."

    numbered = []
    for ln in range(start_line, end_line + 1):
        numbered.append(f"{ln:5d}| {lines[ln - 1]}")
    return f"File: {relative_path} (lines {start_line}-{end_line})\n" + "\n".join(numbered)


def inspect_git_history(
    ctx: AgentToolContext,
    relative_path: str,
    line_number: int,
) -> str:
    if line_number < 1:
        return "line_number must be >= 1."

    rel = Path(relative_path.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        return "Invalid path: must be relative to repository root without '..'."

    target = (ctx.repo_root / rel).resolve()
    try:
        target.relative_to(ctx.repo_root)
    except ValueError:
        return "Invalid path: escapes repository root."

    if not target.is_file():
        return f"File not found: {relative_path}"

    try:
        result = subprocess.run(
            [
                "git",
                "blame",
                "-L",
                f"{line_number},{line_number}",
                "--porcelain",
                relative_path.replace("\\", "/"),
            ],
            cwd=ctx.repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return f"Git blame failed: {e}"

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        if "not a git repository" in err.lower():
            return "Not a git repository; git history unavailable."
        return f"Git blame error: {err or 'unknown error'}"

    stdout = result.stdout
    if stdout is None:
        return "No blame output."
    return stdout.strip() or "No blame output."


def get_openai_tools_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "hybrid_search_code",
                "description": (
                    "Search the indexed codebase using hybrid BM25 + vector retrieval. "
                    "Use for architecture, workflows, symbols, and behavior questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language or keyword query"},
                        "top_k": {
                            "type": "integer",
                            "description": "Number of chunks to return",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file_lines",
                "description": (
                    "Read a contiguous range of lines from a file in the repo (max 250 lines)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relative_path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                    },
                    "required": ["relative_path", "start_line", "end_line"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_git_history",
                "description": (
                    "Git blame for a single line: author, date, and commit for ownership / why."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relative_path": {"type": "string"},
                        "line_number": {"type": "integer"},
                    },
                    "required": ["relative_path", "line_number"],
                },
            },
        },
    ]


def build_tool_handlers(ctx: AgentToolContext) -> dict[str, Callable[..., str]]:
    return {
        "hybrid_search_code": lambda **kw: hybrid_search_code(ctx, **kw),
        "read_file_lines": lambda **kw: read_file_lines(ctx, **kw),
        "inspect_git_history": lambda **kw: inspect_git_history(ctx, **kw),
    }
