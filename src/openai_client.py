"""Shared OpenAI client construction with optional OPENAI_BASE_URL."""

from __future__ import annotations

import os

from openai import OpenAI


def create_openai_client() -> OpenAI:
    """Build an OpenAI client; honor OPENAI_BASE_URL when set."""
    kwargs: dict = {}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)
