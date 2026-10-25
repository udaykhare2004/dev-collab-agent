"""Pydantic schemas for code chunks and search results."""

from __future__ import annotations

import hashlib
from typing import Optional

from pydantic import BaseModel, Field


def make_chunk_id(
    repo_path: str, relative_path: str, start_line: int, end_line: int
) -> str:
    raw = f"{repo_path}|{relative_path}|{start_line}|{end_line}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class CodeChunk(BaseModel):
    id: str
    relative_path: str
    language: str
    code: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    git_author: Optional[str] = None
    git_last_commit: Optional[str] = None
    git_commit_hash: Optional[str] = None


class SearchResult(BaseModel):
    chunk: CodeChunk
    rrf_score: float
    source: str = Field(description='One of "bm25", "vector", or "hybrid"')
