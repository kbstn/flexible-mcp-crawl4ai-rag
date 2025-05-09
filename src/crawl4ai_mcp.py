"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
"""
import sys
import os
from pathlib import Path

# Add project root to Python path to make 'src' importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dotenv import load_dotenv
import asyncio
import logging

from crawl4ai import AsyncWebCrawler, BrowserConfig
# Updated imports for PostgreSQL/SQLModel functions and new crawler modules
from src.utils import (
    get_session,
    create_db_and_tables,
    engine,
    create_embedding, # For health check
    settings, # Import settings object
    OllamaError # For health check
)
from sqlmodel import select # For DB health check

# Import tool functions from the new module
from src.crawler import tool_definitions

# Load environment variables from the project root .env file
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Configure logging
log_level_name = os.getenv('LOG_LEVEL', 'INFO').upper() # Default to INFO if not set
log_level = getattr(logging, log_level_name, logging.INFO) # Get the logging constant

# Ensure that if an invalid level name is given, it defaults to INFO
if not isinstance(log_level, int):
    print(f"Warning: Invalid LOG_LEVEL '{log_level_name}'. Defaulting to INFO.")
    log_level = logging.INFO

logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler

@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle and performs initial checks.
    """
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False # Set to True for more detailed browser logs from Crawl4AI
    )
    
    # Initialize the crawler
    logger.info("Initializing AsyncWebCrawler...")
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    logger.info("AsyncWebCrawler initialized.")
    # server.state.crawler = crawler # Removed: FastMCP doesn't use .state like FastAPI app
    
    # Ensure engine is available and connection possible
    logger.info("Performing DB connection check...")
    try:
        # Try getting a session to ensure the engine/connection works
        with next(get_session()) as session:
             session.exec(select(1)).first() # Simple query
        logger.info("DB connection check successful.")
    except Exception as db_error:
        logger.error(f"Critical: DB connection check failed: {db_error}", exc_info=True)
        raise db_error # Fail fast if DB connection fails

    # Removed Ollama health check for faster startup during debugging

    try:
        logger.info("Yielding Crawl4AIContext...")
        yield Crawl4AIContext(crawler=crawler)
        logger.info("Crawl4AIContext yielded.") # We hope to see this now
    finally:
        # The 'crawler' variable is local to this lifespan function
        logger.info("Cleaning up AsyncWebCrawler...")
        await crawler.__aexit__(None, None, None)
        logger.info("AsyncWebCrawler cleaned up.")

# Initialize FastMCP server
# Temporary monkeypatch: ignore the `Received request before initialization was complete` race
from mcp.server.session import ServerSession

_old_received_request = ServerSession._received_request

async def _received_request(self, *args, **kwargs):
    try:
        return await _old_received_request(self, *args, **kwargs)
    except RuntimeError as e:
        if "Received request before initialization was complete" in str(e):
            logger.warning(f"Ignored expected RuntimeError during initialization race: {e}")
            return None
        raise

ServerSession._received_request = _received_request

mcp = FastMCP(
    "mcp-docs-rag",
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", "8051")),
    settings={"initialization_timeout": 10.0} # Added initialization_timeout
)

# Register tools from the tool_definitions module
mcp.tool()(tool_definitions.crawl_single_page)
mcp.tool()(tool_definitions.smart_crawl_url)
mcp.tool()(tool_definitions.get_available_sources)
mcp.tool()(tool_definitions.perform_rag_query)


async def main():
    """Main function to run the MCP server."""
    transport = os.getenv("TRANSPORT", "sse")
    logger.info(f"Starting MCP server with {transport.upper()} transport...")
    if transport == 'sse':
        await mcp.run_sse_async()
    else:
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())