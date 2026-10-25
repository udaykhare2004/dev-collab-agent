"""Language-aware syntax chunking with line-accurate offsets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from src.models import CodeChunk, make_chunk_id

IGNORE_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
        ".venv",
        "venv",
        ".idea",
        ".vscode",
    }
)

IGNORE_SUFFIXES = (
    ".lock",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".min.js",
    ".min.css",
    ".map",
    ".wasm",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
)

EXTENSION_TO_LANGUAGE: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".pyw": Language.PYTHON,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".mjs": Language.JS,
    ".cjs": Language.JS,
    ".go": Language.GO,
    ".java": Language.JAVA,
    ".rs": Language.RUST,
    ".c": Language.C,
    ".h": Language.CPP,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".hpp": Language.CPP,
    ".html": Language.HTML,
    ".htm": Language.HTML,
    ".md": Language.MARKDOWN,
    ".markdown": Language.MARKDOWN,
    # CSS, JSON, YAML: no Language enum — use generic RecursiveCharacterTextSplitter
}


def _language_name(lang: Language | None) -> str:
    if lang is None:
        return "text"
    return lang.value if hasattr(lang, "value") else str(lang)


def _should_skip_file(path: Path) -> bool:
    name = path.name.lower()
    for suffix in IGNORE_SUFFIXES:
        if name.endswith(suffix):
            return True
    return False


def _iter_source_files(repo_root: Path) -> Iterator[Path]:
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        if _should_skip_file(path):
            continue
        yield path


def _line_range_for_span(full_text: str, start_char: int, end_char: int) -> tuple[int, int]:
    """Map character span to 1-indexed inclusive line numbers."""
    prefix = full_text[:start_char]
    start_line = prefix.count("\n") + 1
    chunk_text = full_text[start_char:end_char]
    end_line = start_line + chunk_text.count("\n")
    return start_line, end_line


def _split_file(
    repo_root: Path,
    file_path: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[CodeChunk]:
    rel = file_path.relative_to(repo_root).as_posix()
    ext = file_path.suffix.lower()
    language = EXTENSION_TO_LANGUAGE.get(ext)
    lang_name = _language_name(language)

    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = file_path.read_text(encoding="latin-1")
        except OSError:
            return []

    if language is not None:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    chunks: list[CodeChunk] = []
    search_from = 0
    for chunk_text in splitter.split_text(text):
        start_char = text.find(chunk_text, search_from)
        if start_char < 0:
            start_char = text.find(chunk_text)
        if start_char < 0:
            continue
        end_char = start_char + len(chunk_text)
        search_from = max(search_from, start_char + 1)
        start_line, end_line = _line_range_for_span(text, start_char, end_char)
        chunk_id = make_chunk_id(str(repo_root.resolve()), rel, start_line, end_line)
        chunks.append(
            CodeChunk(
                id=chunk_id,
                relative_path=rel,
                language=lang_name,
                code=chunk_text,
                start_line=start_line,
                end_line=end_line,
            )
        )
    return chunks


def chunk_repository(
    repo_root: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[CodeChunk]:
    """Scan repo and return all code chunks with line metadata."""
    repo_root = repo_root.expanduser().resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"Repository path does not exist: {repo_root}")

    all_chunks: list[CodeChunk] = []
    for file_path in _iter_source_files(repo_root):
        all_chunks.extend(
            _split_file(repo_root, file_path, chunk_size, chunk_overlap)
        )
    return all_chunks
