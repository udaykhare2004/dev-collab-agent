Developer Collaboration RAG Agent
Production-oriented agentic code intelligence for developer collaboration: language-aware chunking, git commit/blame enrichment, hybrid retrieval (BM25 + ChromaDB + Reciprocal Rank Fusion), and a multi-step OpenAI tool-calling agent.

Requirements
Python 3.11+
Git (optional; indexing works without it, git fields stay empty)
OpenAI-compatible API key (for ask / agent reasoning only; indexing uses local embeddings)
Setup
cd dev-collab-agent
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env     # set OPENAI_API_KEY
On Windows, chromadb>=0.5.23 pulls a prebuilt wheel (no MSVC build tools required). If install fails, run:

pip install "chromadb>=0.5.23" --only-binary=:all:
Optional environment variables:

OPENAI_BASE_URL — OpenAI-compatible API base URL for the chat agent (omitted = official OpenAI)
OPENAI_AGENT_MODEL — default gpt-4o-mini (use gpt-4o or a Gemini model name with OPENAI_BASE_URL)
Embeddings are generated locally with FastEmbed (BAAI/bge-small-en-v1.5) and do not call a remote embedding API.

Usage
Index a repository (defaults to current directory):

python cli.py index .
python cli.py index C:\path\to\your\repo
Progress is written to the console. Indexes are stored under ./storage/:

storage/chroma_db/ — ChromaDB vectors
storage/bm25.pkl — BM25 corpus
storage/config.json — active repo root for ask
Ask a question:

python cli.py ask "How does authentication work?"
The CLI logs each tool call (hybrid_search_code, read_file_lines, inspect_git_history) and prints the final Markdown answer.

Architecture
Layer	Responsibility
src/parser/chunker.py	Extension → language mapping, recursive splits, 1-based line ranges
src/parser/git_metadata.py	git log -L with file-level fallback
src/storage/hybrid_retriever.py	Local FastEmbed + BM25 + RRF merge
src/tools/agent_tools.py	Three OpenAI tools
src/agent/core.py	Up to 6 tool hops, then forced final answer
Error handling
Missing index or repo root → clear CLI message, exit code 1
Non-git repos → git metadata omitted; blame tool returns a friendly message
Missing files / path traversal → blocked in read_file_lines
OpenAI timeouts → surfaced as timeout errors
License
MIT (adjust as needed for your organization).
