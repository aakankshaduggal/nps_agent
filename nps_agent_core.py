"""
NPS Agent Core - Shared agent logic for develop and deploy stages.

This module contains the core agent implementation that is used by both:
- 1_develop/2_evaluate.ipynb (local development and evaluation)
- 2_deploy/npsagent.py (production deployment via MLflow)
"""

import os
from pathlib import Path

from openai import AsyncClient
from agents import Agent, Runner, set_default_openai_client, set_default_openai_api, set_tracing_disabled
from agents.mcp import MCPServerStdio

# Agent system prompt - shared across all environments
AGENT_INSTRUCTIONS = (
    "You are a helpful National Parks Service assistant. "
    "Use the available tools to answer questions about national parks, "
    "events, activities, campgrounds, and visitor information. "
)


def get_mcp_server_path() -> str:
    """Get the path to the MCP server relative to this module."""
    return str(Path(__file__).parent / "nps_mcp_server.py")


async def run_nps_agent(prompt: str) -> str:
    """
    Run the NPS agent with MCP tools and return the text response.
    
    This is the core agent function used by both development notebooks
    and the production deployment wrapper.
    
    Args:
        prompt: The user's question about national parks
        
    Returns:
        The agent's text response
    """
    command = "uv"
    mcp_server_path = get_mcp_server_path()
    args = ["run", "fastmcp", "run", mcp_server_path]
    env = {**os.environ, "NPS_API_KEY": os.environ.get("NPS_API_KEY", "")}
    
    async with MCPServerStdio(params={"command": command, "args": args, "env": env}) as mcp_server:
        # Configure OpenAI-compatible endpoint
        async_client = AsyncClient(
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
        set_default_openai_client(client=async_client)
        set_default_openai_api("chat_completions")
        
        # Disable OpenAI's built-in tracing (we use MLflow instead)
        set_tracing_disabled(disabled=True)
        
        # Create the agent
        agent = Agent(
            name="NPS Agent",
            instructions=AGENT_INSTRUCTIONS,
            mcp_servers=[mcp_server],
            model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o"),
        )
        
        # Run the agent
        result = await Runner.run(agent, prompt)
        return result.final_output
