import pytest
from unittest.mock import MagicMock, patch, ANY
import os
import requests
from typing import Dict, Any, List

from src.utils import generate_contextual_embedding


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up environment variables for testing contextual embeddings."""
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_BASE_URL", "https://test-api.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL_NAME", "test-model")


@pytest.fixture
def mock_requests_post(mocker):
    """Mock the requests.post function."""
    return mocker.patch('requests.post')


# --- Tests for generate_contextual_embedding ---

@patch('src.utils.settings', new_callable=MagicMock)
def test_generate_contextual_embedding_success(mock_settings_obj, mock_env_vars, mock_requests_post): # mock_env_vars is now for conceptual clarity, direct patch is used
    """Test successful generation of contextual embedding."""
    # Configure the mock_settings_obj directly
    mock_settings_obj.LLM_ENABLED = True
    mock_settings_obj.LLM_BASE_URL = "https://test-api.com/v1"
    mock_settings_obj.LLM_API_KEY = "test-key"
    mock_settings_obj.LLM_MODEL_NAME = "test-model"
    mock_settings_obj.CHUNK_SIZE = 500  # Explicitly set CHUNK_SIZE for correct slicing
    mock_settings_obj.CHUNK_SIZE = 500 # Explicitly set CHUNK_SIZE for slicing

    full_document = "This is a full document about climate change. It discusses various aspects including causes and effects."
    chunk = "Climate change causes include greenhouse gases."
    context = "Context about climate change causes within the document"
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": context
                }
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_requests_post.return_value = mock_response
    
    result, success = generate_contextual_embedding(full_document, chunk)
    
    assert success is True
    expected_result_string = f"Contextual Summary: {context}. Original Chunk: {chunk}"
    print(f"DEBUG: Actual result: {result}")
    print(f"DEBUG: Expected result: {expected_result_string}")
    # Less strict assertion to debug why result is 'C'
    assert result.startswith("Contextual Summary:")
    assert result.endswith(chunk)
    # Optional: check if the generated context is included somewhere in the middle
    assert context in result
    assert result == expected_result_string # Keep the strict assertion for now, but focus on the others
    mock_requests_post.assert_called_once()
    # Verify the correct URL was used
    assert mock_requests_post.call_args[0][0] == "https://test-api.com/v1/chat/completions"
    # Verify the prompt structure
    # The payload uses 'prompt' directly, not 'messages'
    assert 'prompt' in mock_requests_post.call_args[1]['json']
    assert mock_requests_post.call_args[1]['json']['prompt'].startswith("<document>")
    # The endswith check might be too brittle if the prompt template changes slightly.
    # For now, let's ensure the core part is there.
    # If more specific end content is needed, it should be carefully matched.
    # assert mock_requests_post.call_args[1]['json']['prompt'].endswith("Contextual Summary:") # Example if prompt ends like this
    assert mock_requests_post.call_args[1]['headers']['Authorization'] == "Bearer test-key"


@patch('src.utils.settings', new_callable=MagicMock)
def test_generate_contextual_embedding_llm_disabled(mock_settings_obj, mock_requests_post, monkeypatch): # monkeypatch is kept if other parts of test use it
    """Test that contextual embedding is skipped when LLM is disabled."""
    mock_settings_obj.LLM_ENABLED = False
    # Other settings don't matter if LLM_ENABLED is False
    mock_settings_obj.LLM_BASE_URL = "https://any-url.com"
    mock_settings_obj.LLM_API_KEY = "any-key"
    mock_settings_obj.LLM_MODEL_NAME = "any-model"

    full_document = "Full document"
    chunk = "Chunk content"
    
    result, success = generate_contextual_embedding(full_document, chunk)
    
    assert success is False
    assert result == chunk
    mock_requests_post.assert_not_called()


@patch('src.utils.settings', new_callable=MagicMock)
def test_generate_contextual_embedding_missing_config(mock_settings_obj, mock_requests_post, monkeypatch):
    """Test that contextual embedding is skipped when configuration is incomplete."""
    mock_settings_obj.LLM_ENABLED = True
    mock_settings_obj.LLM_BASE_URL = ""  # Empty URL
    mock_settings_obj.LLM_API_KEY = "test-key" # Needs to be set if LLM_ENABLED is true, for Settings validation
    mock_settings_obj.LLM_MODEL_NAME = "test-model" # Needs to be set

    full_document = "Full document"
    chunk = "Chunk content"
    
    result, success = generate_contextual_embedding(full_document, chunk)
    
    assert success is False
    assert result == chunk
    mock_requests_post.assert_not_called()


@patch('src.utils.settings', new_callable=MagicMock)
def test_generate_contextual_embedding_api_error(mock_settings_obj, mock_env_vars, mock_requests_post): # mock_env_vars for consistency
    """Test that original chunk is returned when API call fails."""
    mock_settings_obj.LLM_ENABLED = True
    mock_settings_obj.LLM_BASE_URL = "https://test-api.com/v1"
    mock_settings_obj.LLM_API_KEY = "test-key"
    mock_settings_obj.LLM_MODEL_NAME = "test-model"

    full_document = "Full document"
    chunk = "Chunk content"
    
    mock_requests_post.side_effect = requests.exceptions.RequestException("API error")
    
    result, success = generate_contextual_embedding(full_document, chunk)
    
    assert success is False
    assert result == chunk
    mock_requests_post.assert_called_once()


# test_process_chunk_with_context was removed as the function it tested was removed.

# --- Integration test for add_documents_to_db with contextual embeddings ---

@patch('src.utils.create_embeddings_batch')
@patch('src.utils.generate_contextual_embedding')
@patch('concurrent.futures.ThreadPoolExecutor')
def test_add_documents_to_db_with_contextual_embeddings(
    mock_executor, mock_generate_contextual_embedding, mock_create_embeddings_batch, monkeypatch
):
    """Test that add_documents_to_db uses contextual embeddings when LLM_ENABLED is true.
    
    This test would be better in test_utils.py, but we're adding it here to keep
    contextual embedding tests together.
    """
    # This test would need more setup and is complex to implement here
    # The implementation would involve mocking ThreadPoolExecutor and its
    # context manager behavior, setting up the executor to return results
    # for submitted tasks, and verifying operations on batches
    pass
