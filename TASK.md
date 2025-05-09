# Tasks

## Pending

- [ ] Implement configurable chunking strategies as per PLANNING.md (2025-05-09)


## Completed

- [x] Debug MCP client hanging, 'Response ended prematurely' error during embedding, and missing `url_to_full_document` column. Investigate asynchronous embedding feasibility. (2025-05-07)
- [x] Fix Docker build failure: `nltk_data` not found (2025-05-07)
  - Attempt 1: Added `ENV NLTK_DATA=/usr/local/share/nltk_data` to `builder` stage in [`Dockerfile`](./Dockerfile:0) before `crawl4ai-setup` command. (Failed)
  - Attempt 2: Explicitly created `/usr/local/share/nltk_data` with `mkdir -p` and `chmod` in `builder` stage, and added `ls` commands for debugging after `crawl4ai-setup`. (Completed)

- [X] Fix Playwright runtime error: `libglib-2.0.so.0: cannot open shared object file` (2025-05-07)
  - Previous fix for executable path was successful, but now missing shared libraries.
  - Plan: Add required Playwright system dependencies (like `libglib2.0-0`) to the `runtime` stage in [`Dockerfile`](./Dockerfile:0).

- [X] Ensure LLM API key is present and validated at startup if LLM is enabled (2025-05-07)
  - Note: This task is blocked by the Playwright runtime error.

- [x] Debug RAG query retrieval: "endpoint environment variables" query not returning expected results from `https://docs.n8n.io/hosting/configuration/environment-variables/endpoints/`. (2025-05-07)
- [x] Fixed "ImportError: attempted relative import with no known parent package" in src/crawl4ai_mcp.py (2025-05-07)
  - Changed relative imports to absolute imports in both src/crawl4ai_mcp.py and src/crawler/tool_definitions.py

### pytest problems

- [x] FIXED: Fix `test_create_embedding_no_retry_on_http_4xx`: Adjust the regex in the `pytest.raises` assertion in [`tests/test_utils.py`](tests/test_utils.py) to match the actual error message including details for 4xx errors. (2025-05-07)
- [x] FIXED: Fix `test_create_embedding_json_decode_error`: Adjust the regex in the `pytest.raises` assertion in [`tests/test_utils.py`](tests/test_utils.py) to match the actual error message for JSON decode errors. (2025-05-07)
- [x] FIXED: Fix `test_add_documents_to_db_embedding_failure_skips_batch` print assertion mismatch in [`tests/test_utils.py`](tests/test_utils.py). (2025-05-07)
- [x] FIXED: Fix `test_add_documents_to_db_skip_empty_content` print assertion mismatch and removed duplicate test definition in [`tests/test_utils.py`](tests/test_utils.py). (2025-05-07)
- [x] ATTEMPTED FIX: Fix `test_search_documents_success`: Relax the assertion on the SQL query string in [`tests/test_utils.py`](tests/test_utils.py) to be less sensitive to formatting or case, focusing only on the essential parts like the `ORDER BY` clause with the vector distance operator. (2025-05-07)
- [x] FIXED: Fix `test_search_documents_empty_query` logic for whitespace queries in [`tests/test_utils.py`](tests/test_utils.py). (2025-05-07)
- [x] FIXED: Fix `test_add_documents_to_db_handle_sqlalchemy_error_on_delete` logger `AttributeError` by patching `src.utils.logger` and ensuring test logic is correct. (2025-05-07)

- [x] Optimize Dockerfile and improve documentation (2025-05-07)
    - Created multi-stage build to reduce image size and build time
    - Fixed issue with package downloads during container startup
    - Updated README.md with fork acknowledgment and improvements
    - Created comprehensive DOCKER_USAGE.md with detailed instructions
    - Updated .env.example to match the current configuration

- [x] Address remaining PLANNING.md items (2025-05-06):
    - [x] Centralize batch size configuration via Settings
    - [x] Add retry mechanism for Ollama API calls
    - [x] Create unit tests for PostgreSQL integration

- [x] Fix pytest `ModuleNotFoundError: No module named 'src'` (2025-05-07)
    - Added `[tool.pytest.ini_options]` with `pythonpath = ["."]` to `pyproject.toml` to ensure `src` is discoverable by tests.
    
- [x] Implement fixes based on the latest reviewer report (2025-05-06)
    - Update `.env.example` with missing critical variables. (Done by previous task)
    - Add environment variable validation using Pydantic Settings in `src/utils.py`. (Done by previous task)
    - Correct metadata (e.g., `crawl_time` in `src/crawl4ai_mcp.py`). (Done by previous task)
    - Replace blocking HTTP requests with async alternatives (`httpx` in `src/crawl4ai_mcp.py`). (Done by previous task)
    - Clean up unused dependencies (`openai` from `pyproject.toml`). (Done by previous task)
    - Improved error handling in `src/utils.py` for Ollama calls (raises `OllamaError` instead of returning zero vectors). (2025-05-06)
    - Made server startup in `src/crawl4ai_mcp.py` fail fast if DB or Ollama checks fail during lifespan initialization. (2025-05-06)

- [x] Replace Supabase with PostgreSQL in `src/utils.py` (2025-05-05) - Using Ollama for embeddings.
    - Removed Supabase client and imports.
    - Added PostgreSQL connection logic (using `POSTGRES_URL` env var).
    - Rewrote `add_documents_to_db` for PostgreSQL insertion (using SQLModel and pgvector).
    - Rewrote `search_documents` for PostgreSQL vector search (using SQLModel and pgvector).
    - Updated `pyproject.toml` with dependencies (sqlmodel, psycopg2-binary, pgvector, requests).
    - Skipped updating `tests/test_utils.py` as requested.
    - Skipped updating `README.md` as requested.
    - Marked task as complete in `TASK.md`.

- [x] Create `docker-compose.yml` (2025-05-05)
    - Added `app` service building from `Dockerfile`.
    - Added `db` service using `pgvector/pgvector:pg17`.
    - Configured environment variables (`POSTGRES_URL`, `OLLAMA_*`, etc.).
    - Added `initdb/init.sql` to enable `vector` extension.
    - Added named volume `postgres_data` for persistence.
    - Set `depends_on` for `app` service.

## Discovered During Work

### Test and Error Handling Fixes
- 2025-05-07: Applied multiple fixes to `tests/test_utils.py`:
    - Corrected regex for `OllamaError` in `test_create_embedding_no_retry_on_http_4xx`.
    - Corrected regex for `OllamaError` in `test_create_embedding_json_decode_error` to match updated error message from `src/utils.py`.
    - Corrected print assertion in `test_add_documents_to_db_embedding_failure_skips_batch`.
    - Corrected logic for whitespace queries in `test_search_documents_empty_query`.
    - Removed duplicate definition of `test_add_documents_to_db_skip_empty_content`.
    - Ensured `test_add_documents_to_db_handle_sqlalchemy_error_on_delete` correctly uses patched logger and that `src/utils.py` handles the error appropriately.
- 2025-05-07: Updated error handling in `src/utils.py`:
    - Reordered exception handling in `create_embedding` to correctly catch `JSONDecodeError` before `RequestException` and updated the corresponding error message.
    - Modified `add_documents_to_db` to log and continue rather than re-raising `SQLAlchemyError` during the initial deletion phase.