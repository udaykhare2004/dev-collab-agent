"""Git blame and commit message enrichment for code chunks."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from src.models import CodeChunk


def _run_git(repo_root: Path, *args: str, timeout: float = 30.0) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            return None
        stdout = result.stdout
        if stdout is None:
            return None
        return stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def is_git_repository(repo_root: Path) -> bool:
    out = _run_git(repo_root, "rev-parse", "--is-inside-work-tree")
    return out == "true"


def _parse_log_line(log_output: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse %h|%an|%s format."""
    if not log_output:
        return None, None, None
    first_line = log_output.split("\n")[0].strip()
    parts = first_line.split("|", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 1 and parts[0]:
        return parts[0], None, None
    return None, None, None


def _git_log_for_range(
    repo_root: Path,
    relative_path: str,
    start_line: int,
    end_line: int,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    line_log = _run_git(
        repo_root,
        "log",
        "-1",
        "-L",
        f"{start_line},{end_line}:{relative_path}",
        '--pretty=format:%h|%an|%s',
        timeout=60.0,
    )
    if line_log:
        return _parse_log_line(line_log)

    file_log = _run_git(
        repo_root,
        "log",
        "-1",
        "--pretty=format:%h|%an|%s",
        "--",
        relative_path,
    )
    return _parse_log_line(file_log or "")


def enrich_chunks_with_git(repo_root: Path, chunks: list[CodeChunk]) -> list[CodeChunk]:
    """Attach git metadata to each chunk; no-op if not a git repo."""
    repo_root = repo_root.resolve()
    if not is_git_repository(repo_root):
        return chunks

    enriched: list[CodeChunk] = []
    for chunk in chunks:
        commit_hash, author, message = _git_log_for_range(
            repo_root,
            chunk.relative_path,
            chunk.start_line,
            chunk.end_line,
        )
        enriched.append(
            chunk.model_copy(
                update={
                    "git_commit_hash": commit_hash,
                    "git_author": author,
                    "git_last_commit": message,
                }
            )
        )
    return enriched
