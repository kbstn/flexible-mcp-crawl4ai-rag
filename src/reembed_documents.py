"""
Script to re-embed all document chunks in the crawledpage table.
This is typically run after changing the embedding model or dimension.
It assumes the database schema (embedding column dimension) has already been
updated to match the new embedding dimension specified in settings.
"""
import logging
import time
import sqlalchemy as sa

from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

try:
    from src.utils import (
        get_session,
        CrawledPage,
        create_embedding,
        settings,
        OllamaError
    )
except ImportError as e:
    print(f"Error importing from src.utils: {e}")
    print("Please ensure this script is run from the project root or PYTHONPATH is set correctly.")
    exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def re_embed_all_chunks(batch_size: int = 50):
    logger.info(f"Starting re-embedding process for all chunks.")
    logger.info(f"Using embedding model: {settings.OLLAMA_EMBED_MODEL} with dimension: {settings.OLLAMA_EMBEDDING_DIM}")
    logger.info(f"Processing in batches of: {batch_size}")

    total_chunks_processed = 0
    total_chunks_failed = 0
    start_time = time.time()

    with next(get_session()) as session:
        total_chunks_query = select(sa.func.count(CrawledPage.id))
        # Corrected line:
        total_chunks_to_process = session.scalar(total_chunks_query) or 0
        
        if total_chunks_to_process == 0:
            logger.info("No chunks found in the database to re-embed.")
            return

        logger.info(f"Found {total_chunks_to_process} total chunks to re-embed.")

        offset = 0
        while True:
            logger.info(f"Fetching batch: offset={offset}, limit={batch_size}")
            statement = (
                select(CrawledPage)
                .order_by(CrawledPage.id)
                .offset(offset)
                .limit(batch_size)
            )
            chunks_batch = session.exec(statement).all()

            if not chunks_batch:
                logger.info("No more chunks to process.")
                break

            logger.info(f"Processing {len(chunks_batch)} chunks in this batch...")
            batch_processed_count = 0
            batch_failed_count = 0

            for chunk in chunks_batch:
                try:
                    if not chunk.content or not chunk.content.strip():
                        logger.warning(f"Skipping chunk ID {chunk.id} (URL: {chunk.url}, Chunk: {chunk.chunk_number}) due to empty content.")
                        continue

                    logger.debug(f"Re-embedding chunk ID {chunk.id} (URL: {chunk.url}, Chunk: {chunk.chunk_number})")
                    
                    new_embedding = create_embedding(chunk.content)
                    
                    if len(new_embedding) != settings.OLLAMA_EMBEDDING_DIM:
                        logger.error(
                            f"CRITICAL: Embedding dimension mismatch for chunk ID {chunk.id}. "
                            f"Expected {settings.OLLAMA_EMBEDDING_DIM}, got {len(new_embedding)}. Skipping update for this chunk."
                        )
                        batch_failed_count += 1
                        continue
                        
                    chunk.embedding = new_embedding
                    session.add(chunk)
                    batch_processed_count += 1

                except OllamaError as e:
                    logger.error(f"OllamaError re-embedding chunk ID {chunk.id} (URL: {chunk.url}, Chunk: {chunk.chunk_number}): {e}")
                    batch_failed_count += 1
                except Exception as e:
                    logger.error(f"Unexpected error re-embedding chunk ID {chunk.id} (URL: {chunk.url}, Chunk: {chunk.chunk_number}): {e}", exc_info=True)
                    batch_failed_count += 1
            
            try:
                if batch_processed_count > 0: 
                    session.commit()
                    logger.info(f"Committed {batch_processed_count} updated embeddings for this batch.")
                else:
                    session.rollback() 
                    logger.info("No successful embedding updates in this batch to commit.")

            except Exception as e:
                logger.error(f"Error committing batch to database: {e}", exc_info=True)
                session.rollback()
                total_chunks_failed += len(chunks_batch) 
            
            total_chunks_processed += batch_processed_count # Moved this line here
            total_chunks_failed += batch_failed_count      # Moved this line here

            logger.info(f"Batch complete. Processed in batch: {batch_processed_count}, Failed in batch: {batch_failed_count}")
            logger.info(f"Total processed so far: {total_chunks_processed}/{total_chunks_to_process}, Total failed so far: {total_chunks_failed}")

            offset += batch_size
            
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Re-embedding process finished in {duration:.2f} seconds.")
    logger.info(f"Total chunks successfully re-embedded: {total_chunks_processed}")
    logger.info(f"Total chunks failed to re-embed: {total_chunks_failed}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    if load_dotenv(): 
        logger.info(".env file loaded successfully.")
    else:
        logger.warning(".env file not found or not loaded. Using existing environment variables or defaults.")

    configured_batch_size = settings.BATCH_SIZE if hasattr(settings, 'BATCH_SIZE') else 50
    re_embed_all_chunks(batch_size=configured_batch_size)