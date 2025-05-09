"""
Utility functions for the Crawl4AI MCP server, using PostgreSQL with SQLModel/pgvector
and Ollama for embeddings.
"""
import logging
import os
import json
import time
import requests
import concurrent.futures
from enum import Enum
from typing import List, Dict, Any, Optional, Generator, Tuple

from requests.exceptions import RequestException, Timeout, HTTPError, JSONDecodeError
from pydantic import HttpUrl, PostgresDsn, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from sqlmodel import Field, SQLModel, create_engine, Session, select, JSON, Column
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import JSONB # Use JSONB for better performance and indexing
from pgvector.sqlalchemy import Vector # Import Vector type
from sqlalchemy.exc import SQLAlchemyError
import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine_distance

logger = logging.getLogger(__name__)

# --- Custom Exceptions ---
class OllamaError(Exception):
    """Custom exception for Ollama API errors."""
    pass

class ChunkStrategy(str, Enum):
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    FIXED = "fixed"
    SEMANTIC = "semantic"

# --- Application Settings ---
class Settings(BaseSettings):
    POSTGRES_URL: PostgresDsn
    OLLAMA_API_URL: HttpUrl
    OLLAMA_EMBED_MODEL: str
    OLLAMA_EMBEDDING_DIM: int = 768
    BATCH_SIZE: int = 50
    OLLAMA_MAX_RETRIES: int = 3
    OLLAMA_RETRY_DELAY_SECONDS: float = 1.0

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    CHUNK_STRATEGY: ChunkStrategy = ChunkStrategy.PARAGRAPH
    SEMANTIC_CHUNKING: bool = False

    LLM_ENABLED: bool = False
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: Optional[HttpUrl] = None
    LLM_MODEL_NAME: Optional[str] = None

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    @model_validator(mode='after')
    def check_llm_config_if_enabled(cls, values: 'Settings') -> 'Settings':
        if values.LLM_ENABLED:
            if not values.LLM_API_KEY:
                raise ValueError("LLM_API_KEY must be set when LLM_ENABLED is true.")
            if not values.LLM_BASE_URL:
                raise ValueError("LLM_BASE_URL must be set when LLM_ENABLED is true.")
            if not values.LLM_MODEL_NAME:
                raise ValueError("LLM_MODEL_NAME must be set when LLM_ENABLED is true.")
        return values

settings = Settings()

# --- Database Setup ---
connect_args = {}
engine = create_engine(str(settings.POSTGRES_URL), echo=False, connect_args=connect_args)

class CrawledPage(SQLModel, table=True):
    __tablename__ = "crawledpage"
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str = Field(index=True)
    chunk_number: int
    content: str
    page_metadata: Dict[str, Any] = Field(default={}, sa_column=Column("metadata", JSONB))
    embedding: List[float] = Field(sa_column=Column(Vector(settings.OLLAMA_EMBEDDING_DIM)))

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session

# --- Embedding Functions (Using Ollama) ---
def create_embedding(text: str, attempt: int = 1, requests_post=requests.post) -> List[float]:
    """
    Create an embedding for a single text using Ollama's API, with retry logic.
    Normalizes the embedding.
    """
    if not text or not text.strip():
        raise OllamaError("Attempted to create embedding for empty or whitespace-only string.")

    try:
        response = requests_post(
            str(settings.OLLAMA_API_URL),
            json={"model": settings.OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        embedding = result.get("embedding")
        if embedding and isinstance(embedding, list) and len(embedding) == settings.OLLAMA_EMBEDDING_DIM:
            np_embedding = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(np_embedding)
            if norm == 0:
                logger.warning("Zero vector received from Ollama before normalization. Returning as is.")
                return embedding
            normalized_embedding = np_embedding / norm
            return normalized_embedding.tolist()
        else:
            error_message = f"Ollama response missing or invalid embedding format/dimension. Expected {settings.OLLAMA_EMBEDDING_DIM} dimensions."
            logger.error(error_message)
            raise OllamaError(error_message)

    except (Timeout, ConnectionError) as e:
        error_message = f"Ollama API request failed due to Timeout/ConnectionError: {e}"
        logger.warning(f"Error (Attempt {attempt}/{settings.OLLAMA_MAX_RETRIES}): {error_message}")
        if attempt < settings.OLLAMA_MAX_RETRIES:
            delay = settings.OLLAMA_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.info(f"Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
            # Pass requests_post in recursive call
            return create_embedding(text, attempt + 1, requests_post=requests_post)
        else:
            raise OllamaError(f"Failed after {settings.OLLAMA_MAX_RETRIES} attempts: {error_message}") from e
    except HTTPError as http_err:
        error_message = f"Ollama API request failed with HTTP status code: {http_err.response.status_code}"
        try:
            error_details = http_err.response.json()
            error_message += f" - Details: {error_details}"
        except JSONDecodeError:
            error_message += " - Could not decode JSON error response from Ollama."
        
        logger.warning(f"Error (Attempt {attempt}/{settings.OLLAMA_MAX_RETRIES}): {error_message}")
        if http_err.response.status_code >= 500 and attempt < settings.OLLAMA_MAX_RETRIES:
            delay = settings.OLLAMA_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.info(f"Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
            # Pass requests_post in recursive call
            return create_embedding(text, attempt + 1, requests_post=requests_post)
        else:
            raise OllamaError(f"HTTPError (not retrying or max retries reached): {error_message}") from http_err
    except JSONDecodeError as e:
        error_message = f"Failed to decode JSON response from Ollama API: {e}"
        logger.error(f"Error: {error_message}")
        raise OllamaError(error_message) from e
    except RequestException as req_err:
        error_message = f"Ollama API request failed due to a RequestException: {req_err}"
        logger.error(f"Error: {error_message}")
        raise OllamaError(error_message) from req_err
    except Exception as e:
        error_message = f"An unexpected error occurred during embedding creation: {e}"
        logger.error(f"Error: {error_message}")
        raise OllamaError(error_message) from e

def create_embeddings_batch(texts: List[str], requests_post=requests.post) -> List[List[float]]:
    """
    Create embeddings for multiple texts using Ollama (calling sequentially).
    """
    if not texts:
        return []
    embeddings = []
    for text_item in texts:
        # Pass requests_post to create_embedding
        embedding = create_embedding(text_item, requests_post=requests_post)
        embeddings.append(embedding)
    return embeddings

# --- Contextual Embedding Functions ---
def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    """
    if not settings.LLM_ENABLED:
        logger.info("LLM_ENABLED is false. Skipping contextual embedding.")
        return chunk, False
        
    if not settings.LLM_BASE_URL or not settings.LLM_API_KEY or not settings.LLM_MODEL_NAME:
        logger.warning(
            "LLM is enabled, but one or more required configurations "
            "(LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME) are missing or empty. "
            "Skipping contextual embedding."
        )
        return chunk, False
            
    try:
        prompt = f"""<document>
{full_document[:25000]}
</document>
<chunk>
{chunk}
</chunk>
Given the document and the specific chunk, generate a concise (1-2 sentence) summary that captures the main topic of the chunk AND its relationship to the surrounding document context. Focus on keywords and concepts that would help a semantic search system understand what this chunk is about in the broader document. Do not refer to "the chunk" or "the document" in your answer.
Contextual Summary:"""

        headers = {
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": settings.LLM_MODEL_NAME,
            "prompt": prompt,
            "stream": False, # Ensure we get the full response
            "max_tokens": 150 # Adjust as needed
        }
        
        # Using requests.post directly here as it's for a different service (external LLM)
        # and not the Ollama embedding service we are primarily testing/mocking.
        # Ensure the URL ends with /chat/completions
        llm_url = str(settings.LLM_BASE_URL)
        if not llm_url.endswith('/chat/completions'):
            llm_url = llm_url.rstrip('/') + '/chat/completions'
        
        response = requests.post(llm_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        response_data = response.json()
        
        # Adjust based on actual API response structure
        # Example for OpenRouter-like /v1/chat/completions or /v1/completions
        contextual_info = ""
        if "choices" in response_data and response_data["choices"]:
            if "message" in response_data["choices"][0] and "content" in response_data["choices"][0]["message"]:
                contextual_info = response_data["choices"][0]["message"]["content"].strip()
            elif "text" in response_data["choices"][0]: # For older completion models
                 contextual_info = response_data["choices"][0]["text"].strip()
        elif "response" in response_data: # For some direct /generate endpoints
            contextual_info = response_data["response"].strip()

        print(f"DEBUG (utils): contextual_info before combining: {contextual_info}") # Added debug print
        if contextual_info:
            # Combine original chunk with contextual info for embedding
            # This strategy aims to enrich the chunk's semantic meaning.
            # Example: "Context: [Generated Context]. Original: [Chunk Content]"
            # The exact format might need tuning based on embedding model performance.
            # For now, let's prepend context.
            # Max length check to avoid overly long strings for embedding
            combined_text = f"Contextual Summary: {contextual_info}. Original Chunk: {chunk}"
            logger.debug(f"Generated contextual text (first 100 chars): {combined_text[:100]}")
            print(f"DEBUG (utils): combined_text before return: {combined_text}") # Added debug print
            print(f"DEBUG (utils): Returning combined_text[:{settings.CHUNK_SIZE * 2}]") # Added debug print
            return combined_text[:settings.CHUNK_SIZE * 2], True # Limit length
        else:
            logger.warning("LLM response did not contain expected contextual information.")
            print(f"DEBUG (utils): Returning original chunk: {chunk}") # Added debug print
            return chunk, False

    except RequestException as e:
        logger.error(f"Error calling LLM for contextual embedding: {e}")
        return chunk, False
    except Exception as e:
        logger.error(f"Unexpected error in generate_contextual_embedding: {e}")
        return chunk, False

# --- Database Operations ---
def add_documents_to_db(
    session: Session,
    urls: List[str],
    contents: List[str],
    page_metadatas: List[Dict[str, Any]],
    chunk_numbers: List[int],
    full_documents: Optional[List[str]] = None # For contextual embeddings
):
    """
    Adds a batch of documents to the PostgreSQL database.
    Deletes existing documents with the same URL before adding new ones.
    """
    if not urls or not contents or not page_metadatas or not chunk_numbers:
        print("One of the input lists (urls, contents, page_metadatas, chunk_numbers) is empty. No documents to add.")
        return

    if len(urls) != len(contents) or len(urls) != len(page_metadatas) or len(urls) != len(chunk_numbers):
        print("Input lists (urls, contents, page_metadatas, chunk_numbers) have different lengths. Aborting.")
        return
        
    if full_documents and len(urls) != len(full_documents):
        print("If full_documents is provided, it must have the same length as other input lists. Aborting.")
        return

    # Delete existing documents for the given URLs first
    # This is done once for all unique URLs in the batch.
    unique_urls_to_delete = list(set(urls))
    if unique_urls_to_delete:
        try:
            # print(f"Deleting existing documents for URLs: {unique_urls_to_delete}")
            statement = select(CrawledPage).where(CrawledPage.url.in_(unique_urls_to_delete))
            results_to_delete = session.exec(statement).all()
            for doc_to_delete in results_to_delete:
                session.delete(doc_to_delete)
            if results_to_delete: # Only commit if there was something to delete
                session.commit()
                # print(f"Successfully deleted {len(results_to_delete)} existing documents.")
        except SQLAlchemyError as e:
            print(f"Error deleting existing documents: {e}. Proceeding with adding new documents.")
            session.rollback() # Rollback deletion attempt
        except Exception as e: # Catch any other unexpected error during deletion
            print(f"Unexpected error during deletion of existing documents: {e}. Proceeding with adding new documents.")
            session.rollback()


    total_added = 0
    current_batch_size = settings.BATCH_SIZE

    for i in range(0, len(urls), current_batch_size):
        batch_urls = urls[i:i + current_batch_size]
        batch_contents = contents[i:i + current_batch_size]
        batch_page_metadatas = page_metadatas[i:i + current_batch_size]
        batch_chunk_numbers = chunk_numbers[i:i + current_batch_size]
        batch_full_documents = full_documents[i:i + current_batch_size] if full_documents else [None] * len(batch_urls)

        # Filter out items with empty content *before* attempting contextual embedding or regular embedding
        # This ensures we don't try to process invalid data.
        current_valid_urls = []
        current_valid_contents = []
        current_valid_page_metadatas = []
        current_valid_chunk_numbers = []
        current_valid_full_documents = [] # For contextual

        for idx in range(len(batch_contents)):
            if batch_contents[idx] and batch_contents[idx].strip():
                current_valid_urls.append(batch_urls[idx])
                current_valid_contents.append(batch_contents[idx])
                current_valid_page_metadatas.append(batch_page_metadatas[idx])
                current_valid_chunk_numbers.append(batch_chunk_numbers[idx])
                if full_documents: # Only append if full_documents was provided
                    current_valid_full_documents.append(batch_full_documents[idx])
            else:
                print(f"Skipping document with empty content: URL {batch_urls[idx]}, Chunk {batch_chunk_numbers[idx]}")
        
        if not current_valid_contents:
            # print(f"Batch {i//current_batch_size + 1} has no valid content after filtering. Skipping.")
            continue # Skip to next batch if all content was empty

        # Generate contextual embeddings if LLM is enabled and full_documents are available
        contextual_contents = []
        if settings.LLM_ENABLED and full_documents:
            for idx in range(len(current_valid_contents)):
                full_doc_text = current_valid_full_documents[idx]
                chunk_text = current_valid_contents[idx]
                if full_doc_text: # Ensure full_doc_text is not None
                    contextual_text, _ = generate_contextual_embedding(full_doc_text, chunk_text)
                    contextual_contents.append(contextual_text)
                else: # Fallback if full_doc_text is None for some reason
                    contextual_contents.append(chunk_text)
        else: # If LLM not enabled or no full_documents, use original content
            contextual_contents = current_valid_contents

        try:
            batch_embeddings = create_embeddings_batch(contextual_contents)
        except OllamaError as e:
            print(f"Error creating embeddings for batch {i//current_batch_size + 1}: {e}. Skipping this batch.")
            continue

        if len(batch_embeddings) != len(contextual_contents):
             print(f"Critical Error: Embedding count mismatch. Expected {len(contextual_contents)}, got {len(batch_embeddings)}. Skipping batch.")
             continue

        batch_data = []
        for j in range(len(contextual_contents)):
            chunk_size_val = len(contextual_contents[j])
            metadata_with_chunk_size = {
                "chunk_size": chunk_size_val,
                **current_valid_page_metadatas[j]
            }
            page = CrawledPage(
                url=current_valid_urls[j],
                chunk_number=current_valid_chunk_numbers[j],
                content=contextual_contents[j],
                page_metadata=metadata_with_chunk_size,
                embedding=batch_embeddings[j]
            )
            batch_data.append(page)

        if batch_data:
            try:
                session.add_all(batch_data)
                session.commit()
                total_added += len(batch_data)
            except SQLAlchemyError as e:
                print(f"Error inserting batch into PostgreSQL: {e}")
                session.rollback()
        else:
            print(f"Skipping empty batch {i//current_batch_size + 1} (after processing).")

    print(f"Finished adding documents. Total added: {total_added}")

def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculates the cosine similarity between two vectors.
    """
    try:
        np_vec1 = np.array(vec1, dtype=np.float32)
        np_vec2 = np.array(vec2, dtype=np.float32)
        if np.all(np_vec1 == 0) or np.all(np_vec2 == 0):
            return 0.0
        # scipy.spatial.distance.cosine returns cosine distance (1 - similarity)
        distance = scipy_cosine_distance(np_vec1, np_vec2)
        return float(1 - distance)
    except ValueError as ve: # Specifically catch ValueError (e.g., from dimension mismatch)
        logger.error(f"ValueError calculating cosine similarity: {ve}")
        raise # Re-raise ValueError to be caught by tests or calling code
    except Exception as e:
        logger.error(f"Unexpected error calculating cosine similarity: {e}")
        return 0.0

def search_documents(
    session: Session,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents by retrieving all pages and computing cosine similarity in Python.
    This implementation is fully Python-based for testability.
    """
    if not query or not query.strip():
        logger.warning("Search query is empty.")
        return []

    try:
        query_embedding = create_embedding(query) # Uses default requests.post
    except OllamaError as e:
        logger.error(f"Failed to create embedding for query '{query}': {e}")
        return []

    try:
        pages_query = select(CrawledPage)
        # The filter_metadata part needs to be adapted if we want to filter at DB level with SQLModel
        # For now, fetching all and filtering in Python for simplicity with current structure.
        pages = session.exec(pages_query).all()
    except Exception as e:
        logger.error(f"Failed to retrieve pages: {e}")
        return []

    # Apply metadata filter in Python if provided
    if filter_metadata:
        filtered_pages = []
        for p in pages:
            # Ensure page_metadata is a dict, handle if it's None or other types
            current_page_metadata = p.page_metadata if isinstance(p.page_metadata, dict) else {}
            match = True
            for k, v in filter_metadata.items():
                if current_page_metadata.get(k) != v:
                    match = False
                    break
            if match:
                filtered_pages.append(p)
        pages = filtered_pages

    results = []
    for p in pages:
        try:
            # Ensure p.embedding is valid before calculating similarity
            if not p.embedding or not isinstance(p.embedding, list) or not all(isinstance(x, (int, float)) for x in p.embedding):
                logger.warning(f"Skipping document ID {p.id} due to invalid or missing embedding.")
                continue
            
            sim_score = calculate_cosine_similarity(query_embedding, p.embedding)
        except Exception as e:
            logger.error(f"Error computing similarity for document ID {p.id}: {e}")
            continue # Skip this document if similarity calculation fails

        results.append({
            "id": p.id,
            "url": p.url,
            "chunk_number": p.chunk_number,
            "content": p.content,
            "page_metadata": p.page_metadata if isinstance(p.page_metadata, dict) else {},
            "similarity_score": sim_score
        })

    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    logger.debug(f"Search for '{query}' found {len(results)} potential matches, returning top {match_count}.")
    return results[:match_count]

# Example of how to run table creation (e.g., in a startup script)
# if __name__ == "__main__":
#     print("Creating database tables...")
#     create_db_and_tables()
#     print("Database tables created (if they didn't exist).")
