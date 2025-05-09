"""Core crawling logic for web pages."""
import logging
import re  # Ensure re is imported for regex pattern matching
import json
from typing import List, Dict, Any
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree

import httpx
import nltk
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher

from ..utils import settings, ChunkStrategy


logger = logging.getLogger(__name__)

# Ensure NLTK 'punkt' tokenizer is available for sentence chunking
# This is a common practice, but might be better in a setup script or Dockerfile
def __init__(self, package, message):
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.ErrorMessage(message, Exception):
        logger.info("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt', quiet=True)


def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.

    Args:
        url: URL to check

    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path


def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.

    Args:
        url: URL to check

    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith('.txt')


async def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs asynchronously.

    Args:
        sitemap_url: URL of the sitemap

    Returns:
        List of URLs found in the sitemap
    """
    urls = []
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(sitemap_url)
            resp.raise_for_status()  # Raise an exception for bad status codes

            # ElementTree.fromstring expects bytes
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc') if loc.text]  # Ensure loc.text is not None
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching sitemap {sitemap_url}: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Request error fetching sitemap {sitemap_url}: {e}")
    except ElementTree.ParseError as e:
        logger.error(f"Error parsing sitemap XML from {sitemap_url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error parsing sitemap {sitemap_url}: {e}")

    return urls


# --- Start of New Chunking Helper Functions ---

def _fixed_char_chunking(text: str, size: int, overlap: int) -> List[str]:
    """
    Chunks text into fixed character sizes with overlap.

    Args:
        text: The text to chunk.
        size: The target size of each chunk in characters.
        overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    if not text:
        return []
    chunks = []
    # DEBUG print inside _fixed_char_chunking
    print(f"DEBUG_FIXED_CHUNK: input text len={len(text)}, size={size}, overlap={overlap}")
    start = 0
    text_length = len(text)
    # Ensure text_length is what's expected for the specific call
    if len(text) == 108 and size == 100: # Specific to the failing case
        print(f"DEBUG_FIXED_CHUNK: para2_text case detected. text_length={text_length}")

    while start < text_length:
        end = start + size
        chunk = text[start:end]
        if chunk.strip(): # Ensure non-empty chunk after stripping
            chunks.append(chunk)
        if end >= text_length:
            break
        # Prevent infinite loop if overlap is >= size
        increment = size - overlap
        if increment <= 0:
            # If no progress can be made, break after the first chunk
            # (or handle as an error, but breaking is safer for now)
            break
        start += increment
        if start >= text_length: # Ensure start doesn't go beyond text_length
            break
    return chunks

def _paragraph_chunking(text: str, size: int, overlap: int) -> List[str]:
    """
    Chunks text by paragraphs, trying to respect a max size.
    If a paragraph is larger than 'size', it's split using _fixed_char_chunking.
    Overlap is handled by including parts of subsequent paragraphs if needed or by fixed overlap.

    Args:
        text: The text to chunk.
        size: The approximate maximum size of each chunk.
        overlap: The character overlap between chunks (primarily for when splitting large paragraphs).

    Returns:
        A list of text chunks.
    """
    if not text:
        return []
    
    paragraphs = text.split('\n\n')
    chunks = []

    for p_text in paragraphs:
        p_stripped = p_text.strip()
        if not p_stripped:
            continue

        if len(p_stripped) <= size:
            chunks.append(p_stripped)
        else:
            # Paragraph is larger than size, use fixed_char_chunking for this paragraph
            # The overlap parameter is primarily used by _fixed_char_chunking here.
            sub_chunks = _fixed_char_chunking(p_stripped, size, overlap)
            chunks.extend(sub_chunks)


    # If no specific inter-paragraph overlap logic is added here, return `chunks`.
    # If chunks were modified for overlap, return `final_chunks_with_overlap`.
    # For now, returning `chunks` as the primary logic is paragraph splitting,
    # and `_fixed_char_chunking` handles overlap for oversized paragraphs.
    return chunks


def _sentence_chunking(text: str, size: int, overlap: int) -> List[str]:
    """
    Chunks text by sentences, trying to respect a max size.
    Overlap is handled by including sentences from the previous/next chunk.

    Args:
        text: The text to chunk.
        size: The approximate maximum size of each chunk.
        overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    if not text:
        return []
    
    try:
        sentences = nltk.sent_tokenize(text)
        
        # Special case: if there's a single sentence and it's longer than the chunk size,
        # return it as a single chunk instead of falling back to fixed char chunking
        if len(sentences) == 1 and len(sentences[0]) > size:
            logger.info(f"Single sentence exceeding chunk size ({len(sentences[0])} > {size}). Returning as one chunk.")
            return [sentences[0]]
            
    except Exception as e:
        logger.error(f"NLTK sent_tokenize failed: {e}. Falling back to paragraph chunking for this text.")
        # Fallback to paragraph if sentence tokenization fails
        return _paragraph_chunking(text, size, overlap)

    chunks = []
    current_chunk_sentences = []
    current_chunk_len = 0
    
    sentence_cumulative_len = 0
    sentence_start_indices = []
    for s in sentences:
        sentence_start_indices.append(sentence_cumulative_len)
        sentence_cumulative_len += len(s) + 1 # +1 for space if rejoining

    idx = 0
    while idx < len(sentences):
        # Start a new chunk
        chunk_text = ""
        chunk_char_count = 0
        
        # Determine the actual start index for this chunk, considering overlap
        # If this isn't the first chunk, try to include `overlap` chars from previous content.
        start_char_index_for_current_chunk = sentence_start_indices[idx]
        if chunks and overlap > 0:
            # Find the sentence that contains the character `overlap` positions before current sentence.
            # This is complex. A simpler way: add previous N sentences to ensure overlap.
            # Or, more directly, when a chunk is formed, the *next* chunk should start `overlap` chars *before* its natural start.
            
            # Let's try a direct approach: when forming a chunk, once it's full,
            # the next chunk will start from sentences that ensure `overlap` with the end of the current one.
            pass


        # Greedily add sentences until chunk size is met or exceeded
        temp_chunk_sentences = []
        temp_chunk_len = 0
        
        # Effective start for this iteration, considering overlap from previous chunk
        iter_idx = idx
        
        # If we have previous chunks and need overlap, try to step back some sentences
        if chunks and overlap > 0:
            # How many sentences to step back? Estimate based on average sentence length or fixed number.
            # Let's try stepping back by a few sentences or until overlap char count is met.
            # This needs to be done carefully to avoid excessive re-processing.
            
            # Simpler: the `idx` for the next chunk should be set such that it creates an overlap.
            # When a chunk is finalized (e.g., `chunk_A`), the next chunk (`chunk_B`)
            # should start at an `idx` such that `sentences[idx:...]` overlaps with `chunk_A` by `overlap` chars.
            
            # Let's adjust `idx` for the *start* of the current chunk based on the *end* of the *previous* chunk.
            # This is what `start_idx_for_next_chunk` below tries to manage.
            pass


        for i in range(idx, len(sentences)):
            s = sentences[i]
            s_len = len(s)
            
            if temp_chunk_len > 0: # If not the first sentence in this potential chunk
                s_len += 1 # Account for space

            if temp_chunk_len + s_len <= size:
                temp_chunk_sentences.append(s)
                temp_chunk_len += s_len
            else:
                # Sentence makes chunk too large. If chunk is empty, add this sentence anyway.
                if not temp_chunk_sentences:
                    temp_chunk_sentences.append(s)
                    temp_chunk_len += s_len
                    idx = i + 1 # Move to next sentence for next chunk
                else:
                    # Current chunk is full enough with previous sentences
                    idx = i # This sentence will start the next chunk
                break
        else: # All remaining sentences fit
            idx = len(sentences)

        if temp_chunk_sentences:
            chunk_str = " ".join(temp_chunk_sentences)
            if chunk_str.strip():
                 chunks.append(chunk_str.strip())

        # Determine the starting sentence index for the *next* chunk to achieve overlap
        if idx < len(sentences) and overlap > 0 and chunks:
            last_added_chunk_text = chunks[-1]
            # We want the next chunk to start such that it includes `overlap` characters
            # that were at the end of `last_added_chunk_text`.
            # Find which sentence `idx` should point to.
            
            # Iterate backwards from `idx-1` (last sentence of current chunk)
            # until `overlap` characters are covered.
            current_overlap_achieved = 0
            start_idx_for_next_chunk = idx # Default if no suitable overlap found
            
            # Iterate from the sentence *before* the one `idx` currently points to
            for prev_s_idx in range(idx - 1, -1, -1):
                current_overlap_achieved += len(sentences[prev_s_idx]) + 1 # +1 for space
                if current_overlap_achieved >= overlap:
                    start_idx_for_next_chunk = prev_s_idx
                    break
            else: # Loop finished without achieving overlap (e.g., text too short)
                # If overlap is larger than current chunk, start from beginning of current chunk
                if idx > 0 : # and current_overlap_achieved < overlap: #This condition is implied by else
                    start_idx_for_next_chunk = idx - len(temp_chunk_sentences) if temp_chunk_sentences else idx
                    start_idx_for_next_chunk = max(0, start_idx_for_next_chunk)


            idx = start_idx_for_next_chunk
            # Safety: ensure idx does not go backward indefinitely if overlap is very large
            # This logic needs to be robust. If idx is set too far back, it can loop.
            # The `idx` should advance. The overlap means the *next* chunk re-includes some sentences.
            # So, after forming a chunk ending at sentence `k`, the next chunk should start at sentence `j` where `j <= k`.
            # The `idx` variable tracks the main progression.
            # Let's simplify: the main `idx` progresses. The overlap is achieved by how `temp_chunk_sentences` are selected.
            # The current `idx` marks the *start* of sentences not yet firmly assigned to a chunk.

    # Refined loop for sentence chunking with overlap:
    # The core idea: each new chunk starts, and we try to prepend sentences from the *end* of the *previous* chunk
    # to satisfy the overlap requirement, without exceeding the `size` for the new chunk.

    refined_chunks = []
    sentence_idx = 0
    while sentence_idx < len(sentences):
        current_chunk_text_list = []
        current_length = 0
        
        # Determine effective start for this chunk to achieve overlap with the previous chunk
        # This means current_chunk_text_list might start with sentences already in the previous chunk.
        start_build_idx = sentence_idx
        if refined_chunks and overlap > 0:
            # Try to find a sentence index `s_idx <= sentence_idx` such that
            # text from `sentences[s_idx ... sentence_idx-1]` is about `overlap` chars.
            temp_overlap_len = 0
            for i in range(sentence_idx - 1, -1, -1):
                temp_overlap_len += len(sentences[i]) + (1 if temp_overlap_len > 0 else 0)
                if temp_overlap_len >= overlap:
                    start_build_idx = i
                    break
            else: # Overlap is larger than all preceding text, so start from beginning
                start_build_idx = 0
        
        # Add sentences, starting from `start_build_idx` (for overlap)
        # but the main progression is `sentence_idx`
        
        # Add sentences for overlap first
        for i in range(start_build_idx, sentence_idx):
            s_text = sentences[i]
            s_len_with_space = len(s_text) + (1 if current_length > 0 else 0)
            if current_length + s_len_with_space <= size:
                if current_length > 0: current_chunk_text_list.append(" ")
                current_chunk_text_list.append(s_text)
                current_length += s_len_with_space
            else:
                # Overlap part itself is too big, this shouldn't happen if size > overlap
                # Or, chunk is already full from overlap.
                break
                
        # Add new sentences
        chunk_ended_naturally = False
        for i in range(sentence_idx, len(sentences)):
            s_text = sentences[i]
            s_len_with_space = len(s_text) + (1 if current_length > 0 else 0)

            if current_length + s_len_with_space <= size:
                if current_length > 0: current_chunk_text_list.append(" ")
                current_chunk_text_list.append(s_text)
                current_length += s_len_with_space
                if i == len(sentences) -1: # Last sentence
                    chunk_ended_naturally = True
            else:
                # Current sentence makes it too large. Chunk ends *before* this sentence.
                # The next main chunk will start at sentence `i`.
                sentence_idx = i
                chunk_ended_naturally = True
                break
        else: # All remaining sentences processed
            sentence_idx = len(sentences)
            chunk_ended_naturally = True

        if current_chunk_text_list:
            final_chunk_str = "".join(current_chunk_text_list).strip()
            if final_chunk_str:
                refined_chunks.append(final_chunk_str)

        if not chunk_ended_naturally and sentence_idx < len(sentences):
            # This case implies a chunk was formed by overlap + some new sentences,
            # but it hit `size` limit before processing all sentences up to `sentence_idx`.
            # This needs careful handling of `sentence_idx` advancement.
            # The `sentence_idx` should point to the first sentence NOT included in the *main part* of the current chunk.
            # The overlap logic effectively re-reads previous sentences.
            # The `sentence_idx` must advance based on the *new* sentences added.
            # Ensure we always make progress to prevent infinite loops
            if sentence_idx > 0 and sentence_idx < len(sentences):
                # If we're stuck on the same sentence and it's too big for the chunk size,
                # just increment sentence_idx to make progress
                logger.warning(f"Forced sentence_idx advancement to prevent infinite loop")
                sentence_idx += 1

        if sentence_idx >= len(sentences):
            break
            
    return refined_chunks


async def _semantic_chunking(text: str, fallback_size: int, fallback_overlap: int) -> List[str]:
    """
    Chunks text using an LLM to identify semantic boundaries.
    Falls back to paragraph chunking if LLM is not configured or fails.
    Sub-chunks overly large LLM-generated chunks.

    Args:
        text: The text to chunk.
        fallback_size: Chunk size to use for fallback and sub-chunking.
        fallback_overlap: Chunk overlap to use for fallback and sub-chunking.

    Returns:
        A list of text chunks.
    """
    if not settings.LLM_ENABLED or not settings.LLM_BASE_URL or not settings.LLM_API_KEY or not settings.LLM_MODEL_NAME:
        logger.warning("Semantic chunking enabled but LLM is not configured. Falling back to paragraph chunking.")
        return _paragraph_chunking(text, fallback_size, fallback_overlap)

    # Limit input text to avoid overly long prompts and potential high costs/latency
    # Max 25k chars, roughly 6k tokens. Adjust if needed.
    text_to_send = text[:25000]
    if len(text) > 25000:
        logger.warning(f"Semantic chunking input text truncated to 25000 characters from original {len(text)} characters.")


    prompt = f"""You are an expert text analyst. Your task is to segment the following document into semantically coherent chunks.
Each chunk should represent a distinct topic or a logical unit of information.
Avoid making chunks too small (e.g., single sentences unless they are a self-contained point) or too large (e.g., spanning multiple major topics).
The goal is to create chunks that are ideal for semantic search and retrieval.
Return the chunks as a JSON list of strings. For example: ["chunk 1 text...", "chunk 2 text...", ...].
Ensure the output is ONLY the JSON list of strings, with no other text before or after it.

Document to chunk:
---
{text_to_send}
---
JSON list of chunks:
"""
    try:
        # Assuming settings.LLM_BASE_URL might be just the base, and specific endpoint needs to be appended.
        # For OpenAI compatible, it's usually /v1/chat/completions.
        # Ensure the URL is correctly formed. If LLM_BASE_URL includes the full path, this is fine.
        # Otherwise, it might need: f"{str(settings.LLM_BASE_URL).rstrip('/')}/v1/chat/completions"
        # For now, assume LLM_BASE_URL is the full endpoint path if it's not an Ollama-like base.
        # The example in utils.py for contextual embeddings uses /chat/completions.
        
        llm_api_endpoint = str(settings.LLM_BASE_URL)
        # A common pattern is that BASE_URL is like "https://api.openai.com/v1" and then you add "/chat/completions"
        # Or for local Ollama, it might be "http://localhost:11434/api/chat"
        # Let's assume it's an OpenAI compatible endpoint for now as per planning doc hints.
        # If settings.LLM_BASE_URL is truly just a base, one might need:
        # if not llm_api_endpoint.endswith(("/chat/completions", "/api/chat")): # Add other common patterns
        #     llm_api_endpoint = f"{llm_api_endpoint.rstrip('/')}/v1/chat/completions" # Example for OpenAI

        async with httpx.AsyncClient(timeout=120.0) as client: # Increased timeout for potentially long LLM calls
            logger.debug(f"Sending request to LLM for semantic chunking: {llm_api_endpoint}")
            response = await client.post(
                llm_api_endpoint,
                headers={
                    "Authorization": f"Bearer {settings.LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.LLM_MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": "You are an assistant that segments text into semantic chunks and returns a JSON list of strings. Output ONLY the JSON list."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1, # Very low temperature for deterministic chunking
                    "max_tokens": 3000, # Max tokens for the response (the chunks themselves)
                    # "response_format": {"type": "json_object"} # This is great if LLM supports it
                },
            )
            response.raise_for_status()
            result = response.json()
            
            llm_response_content = ""
            if "choices" in result and result["choices"] and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                llm_response_content = result["choices"][0]["message"]["content"]
            # Add other potential extraction paths if necessary based on LLM provider
            else:
                logger.error(f"Could not extract chunk list from LLM response structure: {json.dumps(result)[:500]}")
                raise ValueError("LLM response does not contain expected 'choices...content' structure.")

            try:
                # The LLM might return a string that IS the JSON list.
                # Or it might return a JSON object where one key contains the stringified JSON list.
                # The prompt asks for "ONLY the JSON list".
                semantic_chunks = json.loads(llm_response_content)
                if not isinstance(semantic_chunks, list) or not all(isinstance(c, str) for c in semantic_chunks):
                    logger.error(f"LLM returned malformed JSON or not a list of strings: {llm_response_content[:500]}")
                    # Try to find a list of strings if the top level is a dict containing it
                    if isinstance(semantic_chunks, dict):
                        for key, value in semantic_chunks.items():
                            if isinstance(value, list) and all(isinstance(c, str) for c in value):
                                logger.info(f"Found list of strings under key '{key}' in LLM response.")
                                semantic_chunks = value
                                break
                        else: # No suitable list found in dict
                            raise ValueError("LLM did not return a valid JSON list of strings, nor a dict containing one.")
                    else: # Not a list, and not a dict containing a list
                        raise ValueError("LLM did not return a valid JSON list of strings.")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from LLM response for semantic chunking: {e}. Response: {llm_response_content[:1000]}")
                # Regex fallback as a last resort
                try:
                    logger.info("Attempting regex fallback for chunk extraction due to JSONDecodeError.")
                    # This regex is very basic, looks for quoted strings. May need refinement.
                    matches = re.findall(r'"((?:\\.|[^"\\])*)"', llm_response_content)
                    if matches:
                        # Check if the content looks like a list of these matches
                        # e.g. if the raw response was `["chunk1", "chunk2", "oops not json`
                        # This is risky, as it might pick up random quoted strings.
                        # A better regex might look for list-like structures.
                        # For now, accept if found, but log warning.
                        logger.warning(f"Regex fallback extracted {len(matches)} potential chunks. This is a heuristic.")
                        semantic_chunks = matches
                    else:
                        raise # Re-raise JSONDecodeError if regex also fails
                except Exception as regex_e:
                    logger.error(f"Regex fallback for chunk extraction also failed: {regex_e}")
                    raise json.JSONDecodeError(e.msg, e.doc, e.pos) # Re-raise original error

        final_chunks = []
        # Allow LLM chunks to be a bit larger than rule-based, but not excessively so.
        max_llm_chunk_size = fallback_size * 2.0 # Increased from 1.5 to give LLM more room

        for chunk_text in semantic_chunks:
            stripped_chunk = chunk_text.strip()
            if not stripped_chunk:
                continue
            if len(stripped_chunk) > max_llm_chunk_size:
                logger.info(f"Semantic chunk from LLM exceeds max size ({len(stripped_chunk)} > {int(max_llm_chunk_size)}). Sub-chunking with fixed_char_chunking.")
                # Use smaller overlap for sub-chunking to keep semantic units more distinct
                sub_chunks = _fixed_char_chunking(stripped_chunk, fallback_size, fallback_overlap // 2 if fallback_overlap > 0 else 0)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(stripped_chunk)
        
        if not final_chunks:
            logger.warning("Semantic chunking (LLM) resulted in no usable chunks after processing. Falling back to paragraph chunking.")
            return _paragraph_chunking(text, fallback_size, fallback_overlap)
            
        logger.info(f"Semantic chunking successful, {len(final_chunks)} chunks generated.")
        return final_chunks

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during semantic chunking: {e.response.status_code} - {e.response.text[:500]}. Falling back to paragraph chunking.")
    except httpx.RequestError as e: # Includes Timeout, ConnectError, etc.
        logger.error(f"Request error during semantic chunking: {e}. Falling back to paragraph chunking.")
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Data parsing/validation error during semantic chunking: {e}. Content: {llm_response_content[:500]}. Falling back to paragraph chunking.")
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error during semantic chunking: {e}. Falling back to paragraph chunking.", exc_info=True)
    
    # Fallback call
    return _paragraph_chunking(text, fallback_size, fallback_overlap)


async def chunk_text_according_to_settings(text: str) -> List[str]: # Was smart_chunk_markdown
    """
    Chunks text based on the strategy defined in application settings.
    Prioritizes semantic chunking if enabled.

    Args:
        text: The text to chunk.

    Returns:
        A list of text chunks.
    """
    if not text or not text.strip():
        logger.debug("chunk_text_according_to_settings received empty or whitespace-only text. Returning empty list.")
        return []

    chunk_size = settings.CHUNK_SIZE
    chunk_overlap = settings.CHUNK_OVERLAP
    
    # Ensure overlap is not greater than or equal to size, which would lead to issues.
    if chunk_overlap >= chunk_size and chunk_size > 0 : # check chunk_size > 0 to avoid issues with 0
        logger.warning(f"CHUNK_OVERLAP ({chunk_overlap}) >= CHUNK_SIZE ({chunk_size}). Setting overlap to CHUNK_SIZE / 2 = {chunk_size // 2}.")
        chunk_overlap = chunk_size // 2
    elif chunk_overlap < 0:
        logger.warning(f"CHUNK_OVERLAP ({chunk_overlap}) is negative. Setting overlap to 0.")
        chunk_overlap = 0


    if settings.SEMANTIC_CHUNKING:
        logger.info(f"Attempting SEMANTIC chunking. Effective fallback size: {chunk_size}, overlap: {chunk_overlap}")
        chunks = await _semantic_chunking(text, chunk_size, chunk_overlap)
        if chunks: # _semantic_chunking should return a list, even if its internal fallback was used.
            return chunks
        else:
            # This implies _semantic_chunking (and its internal fallback) returned an empty list.
            # This could be due to the input text being too short or problematic for all strategies.
            logger.warning("Semantic chunking (including its internal fallbacks) returned no chunks. This might be due to very short or unusual input text. Returning empty list.")
            return [] # Return empty list if all attempts failed.

    # Rule-based chunking if semantic chunking is not enabled or failed at a higher level (which it shouldn't with current logic)
    strategy = settings.CHUNK_STRATEGY
    logger.info(f"Using rule-based chunking strategy: {strategy}. Size: {chunk_size}, Overlap: {chunk_overlap}")

    if strategy == ChunkStrategy.FIXED:
        return _fixed_char_chunking(text, chunk_size, chunk_overlap)
    elif strategy == ChunkStrategy.SENTENCE:
        return _sentence_chunking(text, chunk_size, chunk_overlap)
    elif strategy == ChunkStrategy.PARAGRAPH:
        return _paragraph_chunking(text, chunk_size, chunk_overlap)
    else:
        logger.warning(f"Unknown chunk strategy '{strategy}', defaulting to paragraph chunking.")
        return _paragraph_chunking(text, chunk_size, chunk_overlap)

# --- End of New Chunking Helper Functions ---



async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.

    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig()

    try:
        result = await crawler.arun(url=url, config=crawl_config)
    except Exception as e:
        logger.error(f"Crawl markdown file error for {url}: {e}")
        return []
    if result.success and result.markdown:
        return [{'url': url, 'markdown': result.markdown}]
    else:
        # Using logger instead of print
        logger.warning(f"Failed to crawl {url}: {result.error_message}")
        return []


async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.

    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    try:
        results = await crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
    except Exception as e:
        logger.error(f"Crawl batch error for urls {urls}: {e}")
        return []
    return [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]


async def crawl_recursive_internal_links(
    crawler: AsyncWebCrawler,
    start_urls: List[str],
    max_depth: int = 3,
    max_concurrent: int = 10,
    url_pattern: str = None
) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.

    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions
        url_pattern: Optional regex pattern to filter URLs. Only URLs matching this pattern will be crawled.
                     Example: "hosting/configuration" would only crawl URLs containing that path.

    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()

    def normalize_url(url_to_normalize: str) -> str: # Renamed variable to avoid conflict
        return urldefrag(url_to_normalize)[0]

    current_urls_set = set([normalize_url(u) for u in start_urls]) # Renamed variable
    results_all = []

    for depth in range(max_depth):
        urls_to_crawl = [url for url in current_urls_set if url not in visited] # Use normalized urls directly
        if not urls_to_crawl:
            break

        try:
            results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
        except Exception as e:
            logger.error(f"Recursive crawl error at depth {depth} for urls {urls_to_crawl}: {e}")
            break
        next_level_urls = set()

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append({'url': result.url, 'markdown': result.markdown})
                for link in result.links.get("internal", []):
                    next_url = normalize_url(link["href"])
                    if next_url not in visited:
                        # If url_pattern is specified, only add URLs that match the pattern
                        if url_pattern is None or re.search(url_pattern, next_url):
                            next_level_urls.add(next_url)

        current_urls_set = next_level_urls

    return results_all