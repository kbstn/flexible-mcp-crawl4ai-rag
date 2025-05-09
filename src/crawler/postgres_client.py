"""PostgreSQL client for database interactions related to crawled data."""
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from sqlmodel import select

from ..utils import add_documents_to_db, search_documents, CrawledPage, get_session
from .metadata_extractor import extract_section_info
from .web_crawler import chunk_text_according_to_settings # Changed import

logger = logging.getLogger(__name__)


async def store_crawled_documents( # Made async
    session: Session,
    crawl_results: List[Dict[str, Any]],
    crawl_type: str,
    # default_chunk_size: int = 5000, # Removed parameter
) -> tuple[int, int]:
    """
    Processes and stores crawled documents into the PostgreSQL database.

    Args:
        session: The SQLAlchemy session.
        crawl_results: A list of dictionaries, each containing 'url' and 'markdown'.
        crawl_type: The type of crawl (e.g., 'webpage', 'sitemap', 'text_file').
        # default_chunk_size: The default size for chunking markdown. # Removed parameter

    Returns:
        A tuple containing (number_of_pages_processed, total_chunks_stored).
    """
    all_urls: List[str] = []
    all_chunk_numbers: List[int] = []
    all_contents: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []
    total_chunks_stored = 0
    
    # Create url_to_full_document mapping for contextual embeddings
    url_to_full_document = {}

    for doc in crawl_results:
        source_url = doc['url']
        markdown_content = doc.get('markdown', '')
        if not markdown_content:
            logger.warning(f"No markdown content for URL: {source_url}")
            continue
            
        # Add to url_to_full_document mapping
        url_to_full_document[source_url] = markdown_content

        chunks = await chunk_text_according_to_settings(markdown_content) # Changed to await new function

        for i, chunk_text in enumerate(chunks):
            all_urls.append(source_url)
            all_chunk_numbers.append(i)
            all_contents.append(chunk_text)

            meta = extract_section_info(chunk_text)
            meta["chunk_index"] = i
            meta["url"] = source_url
            meta["source"] = urlparse(source_url).netloc
            meta["crawl_type"] = crawl_type
            meta["crawl_time"] = datetime.now(timezone.utc).isoformat()
            all_metadatas.append(meta)
            total_chunks_stored += 1

    if all_contents:
        # Batch size for DB insertion can be configured in utils or passed here
        add_documents_to_db(session, all_urls, all_chunk_numbers, all_contents, all_metadatas, url_to_full_document)
    
    return len(crawl_results), total_chunks_stored


def fetch_available_sources(session: Session) -> List[str]:
    """
    Fetches all unique source domains from the database.

    Args:
        session: The SQLAlchemy session.

    Returns:
        A sorted list of unique source strings.
    """
    # Query distinct sources from the metadata JSONB column
    # Ensure the key 'source' exists and is not null before selecting
    statement = select(CrawledPage.page_metadata['source']).where(CrawledPage.page_metadata.op('?')('source')).distinct()
    results = session.exec(statement).all()
    
    # Filter out potential None values if the distinct query includes them
    unique_sources = {source for source in results if source}
    return sorted(list(unique_sources))


def execute_rag_query(
    session: Session,
    query: str,
    source: Optional[str] = None,
    match_count: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Performs a RAG query using the search_documents utility.

    Args:
        session: The SQLAlchemy session.
        query: The search query.
        source: Optional source domain to filter results.
        match_count: Maximum number of results to return.
        filter_metadata: Additional metadata to filter by.

    Returns:
        A list of formatted search results.
    """
    combined_filter: Dict[str, Any] = {}
    if source and source.strip():
        combined_filter["source"] = source
    if filter_metadata:
        combined_filter.update(filter_metadata)

    results = search_documents(
        session=session,
        query=query,
        match_count=match_count,
        filter_metadata=combined_filter if combined_filter else None
    )

    formatted_results: List[Dict[str, Any]] = []
    for result in results:
        formatted_results.append({
            "url": result.get("url"),
            "content": result.get("content"),
            "metadata": result.get("page_metadata"), # Matches key from search_documents
            "similarity": result.get("similarity_score") # Matches key from search_documents
        })
    return formatted_results