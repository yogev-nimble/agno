"""
WebRouter Tools
=============================

Demonstrates web search and page extraction through WebRouter, which sends each
call to one of several search and extraction providers and falls back between them.

Prerequisites:
- Python 3.11+
- pip install "agno[webrouter]"

No API key is needed: with none set, calls go to the providers' free tiers.
A provider key set in the environment (TAVILY_API_KEY, EXA_API_KEY,
FIRECRAWL_API_KEY, ...) adds that provider to the routing, and calls it serves
are billed to that key.
"""

import asyncio

from agno.agent import Agent
from agno.tools.webrouter import WebRouterTools

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------

# Example 1: search and extract, routed across whichever providers are available
agent = Agent(tools=[WebRouterTools()], markdown=True)

# Example 2: pin each method to a provider (needs that provider's key)
# pinned_agent = Agent(
#     tools=[WebRouterTools(search_provider="exa", extract_provider="firecrawl")],
#     markdown=True,
# )

# Example 3: search only, limited to some domains, with JSON output
docs_agent = Agent(
    tools=[
        WebRouterTools(
            enable_extract=False,
            include_domains=["docs.python.org", "peps.python.org"],
            format="json",
        )
    ],
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent.print_response(
        "Search for recent developments in small language models and summarize them"
    )

    agent.print_response(
        "Read https://docs.agno.com/introduction and summarize what Agno is"
    )

    docs_agent.print_response("What changed in the Python 3.13 release?")

    # agent.aprint_response() runs the toolkit's async variants
    asyncio.run(
        agent.aprint_response(
            "Search for the Agno documentation, then read the most relevant page "
            "and list its main sections"
        )
    )
