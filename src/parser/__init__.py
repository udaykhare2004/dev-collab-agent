"""Language-aware parsing and git enrichment."""

from src.parser.chunker import chunk_repository
from src.parser.git_metadata import enrich_chunks_with_git

__all__ = ["chunk_repository", "enrich_chunks_with_git"]
