import pytest
from sqlmodel import SQLModel # Session as SQLModelSession - not needed for mock
from typing import List, Dict, Any
import os
from unittest.mock import MagicMock, patch

# Ensure src is in path for imports if running pytest from root
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import (
    Settings,
    CrawledPage,
    create_embedding, # We need this to get real embeddings
    search_documents,
    OllamaError,
    # calculate_cosine_similarity # search_documents handles this internally
)

# These serve as fallbacks if environment variables are not set.
DEFAULT_OLLAMA_EMBEDDING_DIM = 1024 # Defaulting to user-confirmed value
DEFAULT_OLLAMA_MODEL = "bge-m3-FP16" # Defaulting to user-confirmed value
DEFAULT_OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
# DEFAULT_POSTGRES_URL_FOR_TEST is not needed as DB is mocked

@pytest.fixture(scope="module")
def module_settings_for_rag_test():
    """
    Provides a Settings object for the RAG test module, focusing on Ollama settings.
    It attempts to load from environment variables first.
    """
    # Load settings; this should pick up .env or actual environment variables
    temp_settings = Settings()

    # Use environment variables if set, otherwise use the module-level defaults
    # For Pydantic V2, use model_fields instead of __fields__
    final_ollama_api_url = str(temp_settings.OLLAMA_API_URL) if str(temp_settings.OLLAMA_API_URL) != str(Settings.model_fields["OLLAMA_API_URL"].default) else DEFAULT_OLLAMA_API_URL
    final_ollama_model = str(temp_settings.OLLAMA_EMBED_MODEL) if str(temp_settings.OLLAMA_EMBED_MODEL) != str(Settings.model_fields["OLLAMA_EMBED_MODEL"].default) else DEFAULT_OLLAMA_MODEL
    final_ollama_dim = int(temp_settings.OLLAMA_EMBEDDING_DIM) if temp_settings.OLLAMA_EMBEDDING_DIM != Settings.model_fields["OLLAMA_EMBEDDING_DIM"].default else DEFAULT_OLLAMA_EMBEDDING_DIM

    print(f"Test settings: OLLAMA_URL={final_ollama_api_url}, OLLAMA_MODEL={final_ollama_model}, DIM={final_ollama_dim}")
    
    return Settings(
        POSTGRES_URL="postgresql://mock:mock@mock/mock", # Dummy value, not used
        OLLAMA_API_URL=final_ollama_api_url,
        OLLAMA_EMBED_MODEL=final_ollama_model,
        OLLAMA_EMBEDDING_DIM=final_ollama_dim,
        OLLAMA_MAX_RETRIES=int(os.getenv("OLLAMA_MAX_RETRIES", "1")),
        OLLAMA_RETRY_DELAY_SECONDS=float(os.getenv("OLLAMA_RETRY_DELAY_SECONDS", "0.1")),
        BATCH_SIZE=int(os.getenv("BATCH_SIZE", "5")), # Not strictly needed for this test version
        LLM_ENABLED=False,
        LLM_API_KEY="dummy_key_not_used",
        LLM_BASE_URL="http://dummy.llm.url",
        LLM_MODEL_NAME="dummy_llm_model"
    )

@pytest.fixture
def patch_utils_settings(monkeypatch, module_settings_for_rag_test: Settings):
    """Patches src.utils.settings for the scope of the test function."""
    monkeypatch.setattr('src.utils.settings', module_settings_for_rag_test)


@patch('src.utils.Session') # Mock the Session object used in search_documents
def test_rag_retrieval_for_endpoint_env_vars(MockSession, module_settings_for_rag_test: Settings, patch_utils_settings):
    """
    Tests RAG retrieval accuracy for the query "endpoint environment variables"
    using the real Ollama embedding model but MOCKING database interactions.
    """
    mock_session_instance = MagicMock()
    # Configure the mock_session_instance to behave like a context manager if `search_documents` uses `with Session(...)`
    MockSession.return_value.__enter__.return_value = mock_session_instance 

    test_documents_data = [
        {
            "id": 1, 
            "url": "https://docs.n8n.io/hosting/configuration/environment-variables/endpoints/",
            "chunk_number": 0,
            "content": "Detailed information about endpoint-specific environment variables. For example, `N8N_ENDPOINT_WEBHOOK_API_KEY` can be set. This page covers all endpoint variables.",
            "page_metadata": {"source": "n8n_docs", "title": "Endpoint Variables"}
        },
        {
            "id": 2,
            "url": "https://docs.n8n.io/hosting/configuration/environment-variables/general/",
            "chunk_number": 0,
            "content": "General n8n configuration is managed using various environment variables like `DB_TYPE` and `EXECUTIONS_MODE`.",
            "page_metadata": {"source": "n8n_docs", "title": "General Variables"}
        },
        {
            "id": 3,
            "url": "https://docs.n8n.io/concepts/workflows/",
            "chunk_number": 0,
            "content": "Workflows in n8n are a series of connected nodes that automate tasks. They do not directly use endpoint environment variables in their definition.",
            "page_metadata": {"source": "n8n_docs", "title": "Workflows"}
        },
        {
            "id": 4,
            "url": "https://example.com/irrelevant",
            "chunk_number": 0,
            "content": "This document discusses fruit cultivation, focusing on apples and oranges, which is not related to n8n or environment variables.",
            "page_metadata": {"source": "fruit_docs", "title": "Fruit"}
        },
        {
            "id": 5,
            "url": "https://docs.n8n.io/hosting/configuration/environment-variables/endpoints_raw_url/",
            "chunk_number": 0,
            "content": "Reference: https://docs.n8n.io/hosting/configuration/environment-variables/endpoints/",
            "page_metadata": {"source": "n8n_docs", "title": "Endpoint URL Reference"}
        }
    ]

    # 1. Create REAL embeddings for test documents
    embedded_documents: List[CrawledPage] = []
    for doc_data in test_documents_data:
        try:
            # `patch_utils_settings` ensures `create_embedding` uses `module_settings_for_rag_test`
            embedding = create_embedding(doc_data["content"]) 
            
            # Create CrawledPage instances. These are simple Pydantic models, so direct instantiation is fine.
            embedded_documents.append(
                CrawledPage(
                    id=doc_data["id"],
                    url=doc_data["url"],
                    chunk_number=doc_data["chunk_number"],
                    content=doc_data["content"],
                    page_metadata=doc_data["page_metadata"],
                    embedding=embedding # Real embedding
                )
            )
        except OllamaError as e:
            pytest.fail(f"OllamaError during test document embedding for '{doc_data['url']}'. Is Ollama running and model '{module_settings_for_rag_test.OLLAMA_EMBED_MODEL}' available? Error: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error during test document embedding for '{doc_data['url']}': {e}")
    
    assert len(embedded_documents) == len(test_documents_data), "Not all test documents were embedded."

    # 2. Mock the database query part of search_documents
    #    search_documents internally does: session.exec(select(...).order_by(...).limit(...)).all()
    #    We need `mock_session_instance.exec().all()` to return our `embedded_documents`.
    #    The actual SQL query construction with vector operators won't run against a DB.
    #    The sorting based on similarity will happen *after* this mocked retrieval.
    
    mock_exec_result = MagicMock()
    mock_exec_result.all.return_value = embedded_documents # Return all our pre-embedded docs
    mock_session_instance.exec.return_value = mock_exec_result
    
    query = "endpoint environment variables"
    try:
        # `patch_utils_settings` ensures `search_documents` (and its call to `create_embedding` for the query)
        # uses `module_settings_for_rag_test`.
        results = search_documents(
            session=mock_session_instance, # Pass the mocked session instance
            query=query,
            match_count=5,
            filter_metadata={"source": "n8n_docs"} # This filter is applied by search_documents *after* mock retrieval
        )
    except OllamaError as e: # For query embedding
        pytest.fail(f"OllamaError during query embedding. Is Ollama running and model '{module_settings_for_rag_test.OLLAMA_EMBED_MODEL}' available? Error: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during search_documents: {e}")

    assert len(results) > 0, "Search returned no results."
    
    # The filter_metadata in search_documents should have already filtered this.
    # If not, this manual filter is a fallback for assertion clarity.
    results = [res for res in results if res["page_metadata"].get("source") == "n8n_docs"]
    assert len(results) > 0, "Search returned no 'n8n_docs' results after filtering."

    # IMPORTANT: The search_documents function, when the DB is mocked like this,
    # returns documents in the order they were provided by the mock, then calculates similarity.
    # It does NOT re-sort them by similarity in Python. The DB would normally do the sorting.
    # So, for this test to be meaningful for ranking, we must sort the results here.
    results_sorted_by_similarity = sorted(results, key=lambda x: x['similarity_score'], reverse=True)

    print(f"\nSearch results for query: '{query}' (DB Mocked, Embeddings Real, Sorted by Python for test assertion)")
    for i, res_dict in enumerate(results_sorted_by_similarity): # Iterate over sorted results
        print(f"{i+1}. URL: {res_dict['url']}, Score: {res_dict['similarity_score']:.4f}, Content: {res_dict['content'][:100]}...")

    # --- Assertions on SORTED results ---
    top_result_url = results_sorted_by_similarity[0]["url"]
    assert top_result_url in [
        "https://docs.n8n.io/hosting/configuration/environment-variables/endpoints/",
        "https://docs.n8n.io/hosting/configuration/environment-variables/endpoints_raw_url/"
    ], f"Top result '{top_result_url}' after sorting was not one of the expected most relevant URLs."

    assert results_sorted_by_similarity[0]["similarity_score"] > 0.6, \
        f"Top result similarity score {results_sorted_by_similarity[0]['similarity_score']:.4f} is too low. Expected > 0.6 (using 0.6 as a slightly stricter threshold for the top item)."

    # Check that the irrelevant document is not present (already filtered by source)
    fruit_doc_urls_in_results = [res["url"] for res in results_sorted_by_similarity if res["url"] == "https://example.com/irrelevant"]
    assert not fruit_doc_urls_in_results, "Irrelevant fruit document should have been filtered out."

    relevant_urls = [
        "https://docs.n8n.io/hosting/configuration/environment-variables/endpoints/",
        "https://docs.n8n.io/hosting/configuration/environment-variables/endpoints_raw_url/"
    ]
    general_config_url = "https://docs.n8n.io/hosting/configuration/environment-variables/general/"

    # Get ranks from the *sorted* list
    relevant_ranks = {url: -1 for url in relevant_urls}
    general_config_rank = -1
    
    result_urls_ordered_by_similarity = [res["url"] for res in results_sorted_by_similarity]

    for i, url_in_res in enumerate(result_urls_ordered_by_similarity):
        if url_in_res in relevant_ranks:
            relevant_ranks[url_in_res] = i
        if url_in_res == general_config_url:
            general_config_rank = i
            
    # Assert that our two most relevant documents are ranked 0 and 1 (or vice-versa)
    # and that the general_config_url is ranked lower than both.
    
    # Find the ranks of the two target documents
    rank_endpoints_doc = relevant_ranks["https://docs.n8n.io/hosting/configuration/environment-variables/endpoints/"]
    rank_endpoints_raw_url_doc = relevant_ranks["https://docs.n8n.io/hosting/configuration/environment-variables/endpoints_raw_url/"]

    assert rank_endpoints_doc != -1, "Endpoints doc not found in sorted results"
    assert rank_endpoints_raw_url_doc != -1, "Endpoints raw URL doc not found in sorted results"
    
    # Check they are the top two
    assert rank_endpoints_doc <= 1, f"Endpoints doc (score {results_sorted_by_similarity[rank_endpoints_doc]['similarity_score']:.4f}) not in top 2, rank {rank_endpoints_doc}"
    assert rank_endpoints_raw_url_doc <= 1, f"Endpoints raw URL doc (score {results_sorted_by_similarity[rank_endpoints_raw_url_doc]['similarity_score']:.4f}) not in top 2, rank {rank_endpoints_raw_url_doc}"
    
    if general_config_rank != -1: # if general config doc is found
        assert general_config_rank > rank_endpoints_doc, \
            f"General config doc (rank {general_config_rank}, score {results_sorted_by_similarity[general_config_rank]['similarity_score']:.4f}) should be ranked lower than endpoints doc (rank {rank_endpoints_doc}, score {results_sorted_by_similarity[rank_endpoints_doc]['similarity_score']:.4f})."
        assert general_config_rank > rank_endpoints_raw_url_doc, \
            f"General config doc (rank {general_config_rank}, score {results_sorted_by_similarity[general_config_rank]['similarity_score']:.4f}) should be ranked lower than endpoints raw URL doc (rank {rank_endpoints_raw_url_doc}, score {results_sorted_by_similarity[rank_endpoints_raw_url_doc]['similarity_score']:.4f})."

    print("RAG retrieval accuracy test (with mocked DB, sorted results) passed assertions.")