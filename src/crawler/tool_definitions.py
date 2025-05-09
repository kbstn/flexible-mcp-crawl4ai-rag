"""MCP tool definitions and implementations."""
import json
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from datetime import datetime, timezone

from mcp.server.fastmcp import Context

# Import utils from the src package
from src.utils import get_session
from .web_crawler import (
    is_sitemap,
    is_txt,
    parse_sitemap,
    crawl_markdown_file,
    crawl_batch,
    crawl_recursive_internal_links,
    # smart_chunk_markdown, # Old import, will use chunk_text_according_to_settings
    chunk_text_according_to_settings # New import
)
from .metadata_extractor import extract_section_info # Needed for crawl_single_page directly
from .postgres_client import (
    store_crawled_documents,
    fetch_available_sources,
    execute_rag_query,
)

logger = logging.getLogger(__name__)

# Note: The @mcp.tool() decorator will be applied in the main crawl4ai_mcp.py file
# where the 'mcp' instance is defined. These are the raw function definitions.

async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content.

    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is chunked and stored in the database.

    Args:
        ctx: The MCP server provided context.
        url: URL of the web page to crawl.

    Returns:
        Summary of the crawling operation and storage.
    """
    try:
        # Reverted: Access crawler via request_context.lifespan_context
        # This assumes FastMCP makes the yielded lifespan context available here.
        if not hasattr(ctx, 'request_context') or not hasattr(ctx.request_context, 'lifespan_context') or \
           not hasattr(ctx.request_context.lifespan_context, 'crawler'):
            logger.error("AsyncWebCrawler not found on ctx.request_context.lifespan_context.crawler. Lifespan context might be missing or crawler not set.")
            return json.dumps({"success": False, "url": url, "error": "Crawler not initialized or context issue."}, indent=2)
        crawler = ctx.request_context.lifespan_context.crawler
        
        # Crawl the page (using the method from web_crawler.py or directly if simple enough)
        # For crawl_single_page, the logic was relatively self-contained in the original file.
        # We can replicate parts of it here, using helpers.
        from crawl4ai import CrawlerRunConfig, CacheMode # Local import for clarity
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        result = await crawler.arun(url=url, config=run_config)

        if result.success and result.markdown:
            # This part is specific to crawl_single_page's direct DB interaction
            # It's slightly different from the generalized store_crawled_documents
            chunks = await chunk_text_according_to_settings(result.markdown) # Changed to await new function
            
            db_urls: List[str] = []
            db_chunk_numbers: List[int] = []
            db_contents: List[str] = []
            db_metadatas: List[Dict[str, Any]] = []
            
            for i, chunk in enumerate(chunks):
                db_urls.append(url)
                db_chunk_numbers.append(i)
                db_contents.append(chunk)
                
                meta = extract_section_info(chunk) # from metadata_extractor
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = urlparse(url).netloc
                meta["crawl_time"] = datetime.now(timezone.utc).isoformat()
                db_metadatas.append(meta)
            
            # Create url_to_full_document mapping for contextual embeddings
            url_to_full_document = {url: result.markdown}
            
            with next(get_session()) as session:
                # Using the generic add_documents_to_db from utils
                from src.utils import add_documents_to_db
                add_documents_to_db(session, db_urls, db_chunk_numbers, db_contents, db_metadatas, url_to_full_document)
            
            return json.dumps({
                "success": True,
                "url": url,
                "chunks_stored": len(chunks),
                "content_length": len(result.markdown),
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                }
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "url": url,
                "error": result.error_message or "Crawling failed or no markdown content."
            }, indent=2)
    except Exception as e:
        logger.error(f"Error in crawl_single_page for {url}: {e}", exc_info=True)
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)


async def smart_crawl_url(
    ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10,
    chunk_size: int = 5000, follow_links: bool = False, url_pattern: str = None
) -> str:
    """
    Intelligently crawl a URL based on its type and store content.

    Detects URL type (sitemap, text file, regular webpage) and applies the appropriate crawling method.
    All crawled content is chunked and stored.

    Args:
        ctx: The MCP server provided context.
        url: URL to crawl.
        max_depth: Maximum recursion depth for regular URLs (only if follow_links is True).
        max_concurrent: Maximum number of concurrent browser sessions.
        chunk_size: Maximum size of each content chunk in characters.
        follow_links: Whether to recursively follow internal links on regular webpages (default: False).
        url_pattern: Optional regex pattern to filter which URLs to crawl when follow_links is True.
                     Example: "hosting/configuration" would only crawl URLs containing that path.

    Returns:
        JSON string with crawl summary and storage information.
    """
    try:
        # Reverted: Access crawler via request_context.lifespan_context
        if not hasattr(ctx, 'request_context') or not hasattr(ctx.request_context, 'lifespan_context') or \
           not hasattr(ctx.request_context.lifespan_context, 'crawler'):
            logger.error("AsyncWebCrawler not found on ctx.request_context.lifespan_context.crawler. Lifespan context might be missing or crawler not set.")
            return json.dumps({"success": False, "url": url, "error": "Crawler not initialized or context issue."}, indent=2)
        crawler = ctx.request_context.lifespan_context.crawler
        
        crawl_results_list: List[Dict[str, Any]] = []
        crawl_type = "webpage"
        
        if is_txt(url):
            crawl_results_list = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            sitemap_urls = await parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({"success": False, "url": url, "error": "No URLs found in sitemap"}, indent=2)
            crawl_results_list = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            if follow_links:
                crawl_results_list = await crawl_recursive_internal_links(
                    crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent,
                    url_pattern=url_pattern
                )
                crawl_type = "webpage_recursive"
            else:
                # Just crawl the single page without following links
                result = await crawl_markdown_file(crawler, url)
                crawl_results_list = result
                crawl_type = "webpage_single"
        
        if not crawl_results_list:
            return json.dumps({"success": False, "url": url, "error": "No content found or crawl failed"}, indent=2)
            
        with next(get_session()) as session:
            # store_crawled_documents is now async
            pages_processed, chunks_stored_count = await store_crawled_documents(
                session, crawl_results_list, crawl_type # Removed default_chunk_size
            )
        
        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": pages_processed,
            "chunks_stored": chunks_stored_count,
            "urls_crawled_sample": [doc['url'] for doc in crawl_results_list][:5] + \
                                (["..."] if len(crawl_results_list) > 5 else [])
        }, indent=2)
    except Exception as e:
        logger.error(f"Error in smart_crawl_url for {url}: {e}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(e)}, indent=2)


async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources (domains) from the database.

    Args:
        ctx: The MCP server provided context.

    Returns:
        JSON string with the list of available sources.
    """
    try:
        with next(get_session()) as session:
            sources = fetch_available_sources(session)
        
        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources)
        }, indent=2)
    except Exception as e:
        logger.error(f"Error in get_available_sources: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e)}, indent=2)


async def perform_rag_query(
    ctx: Context, query: str, source: Optional[str] = None, match_count: int = 5, filter_metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Perform a RAG query on the stored content.

    Searches the vector database for content relevant to the query.

    Args:
        ctx: The MCP server provided context.
        query: The search query.
        source: Optional source domain to filter results.
        match_count: Maximum number of results to return.
        filter_metadata: Optional additional metadata for filtering.

    Returns:
        JSON string with the search results.
    """
    try:
        with next(get_session()) as session:
            results = execute_rag_query(session, query, source, match_count, filter_metadata)
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source,
            "filter_metadata": filter_metadata,
            "results": results,
            "count": len(results)
        }, indent=2)
    except Exception as e:
        logger.error(f"Error in perform_rag_query for query '{query}': {e}", exc_info=True)
        return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)