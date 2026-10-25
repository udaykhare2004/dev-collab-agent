#!/usr/bin/env python3
"""Streamlit UI for the Developer Collaboration RAG Agent."""

import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent.core import run_agent
from src.openai_client import create_openai_client
from src.storage.index_manager import IndexManager
from src.tools.agent_tools import AgentToolContext

STORAGE_DIR = _PROJECT_ROOT / "storage"
load_dotenv(_PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="Developer Collaboration RAG Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for clean dark dev aesthetic
st.markdown(
    """
<style>
    .stChatMessage { border-radius: 8px; margin-bottom: 8px; }
    code { font-family: 'JetBrains Mono', 'Fira Code', monospace; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------- SIDEBAR -----------------
st.sidebar.title("⚡ Codebase Agent")
st.sidebar.caption("AST Chunking • Hybrid Search • Git Blame • Agentic Loop")

# Check API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.sidebar.error("⚠️ OPENAI_API_KEY is not set in `.env`")
else:
    st.sidebar.success("🔑 API Key configured")

# Load indexed repo info
manager = IndexManager(STORAGE_DIR)
repo_root = None
try:
    repo_root = manager.load_repo_root()
    st.sidebar.markdown(f"**Target Repository:**\n`{repo_root}`")
except Exception:
    st.sidebar.warning("⚠️ No repository indexed yet. Run `python cli.py index <path>` first.")

st.sidebar.divider()
st.sidebar.markdown("### Active Tools")
st.sidebar.markdown(
    """
- 🔍 `hybrid_search_code` (BM25 + Dense)
- 📖 `read_file_lines` (Bounded reader)
- 📜 `inspect_git_history` (Blame & commits)
"""
)

if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# ----------------- MAIN INTERFACE -----------------
st.title("💻 Developer Collaboration Agent")
if repo_root:
    st.caption(f"Reasoning over: `{Path(repo_root).name}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("tool_logs"):
            with st.expander("🛠️ View Agent Decision Trace & Tool Calls", expanded=False):
                for t_name, t_args in msg["tool_logs"]:
                    st.markdown(f"**`{t_name}`**")
                    st.json(t_args)
        st.markdown(msg["content"])

# Prompt input box
if prompt := st.chat_input("Ask about codebase architecture, security, or git ownership..."):
    if not repo_root:
        st.error("Please index a repository first using `python cli.py index <path>`.")
        st.stop()

    # Append user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent reasoning
    with st.chat_message("assistant"):
        tool_logs = []

        def log_tool(name: str, args: dict) -> None:
            tool_logs.append((name, args))

        with st.spinner("Analyzing codebase and executing agent tools..."):
            try:
                model = os.getenv("OPENAI_AGENT_MODEL", "gemini-2.5-flash")
                client = create_openai_client()
                ctx = AgentToolContext(repo_root=repo_root, retriever=manager.retriever)

                answer = run_agent(
                    question=prompt,
                    ctx=ctx,
                    client=client,
                    model=model,
                    on_tool_call=log_tool,
                )

                if tool_logs:
                    with st.expander("🛠️ View Agent Decision Trace & Tool Calls", expanded=False):
                        for t_name, t_args in tool_logs:
                            st.markdown(f"**`{t_name}`**")
                            st.json(t_args)

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "tool_logs": tool_logs,
                    }
                )

            except Exception as e:
                st.error(f"Agent failed: {e}")