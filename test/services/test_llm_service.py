import src.configs.config                    
import pytest
import os
from loguru import logger
from langchain_core.messages import AIMessage, BaseMessageChunk
from src.services.llm_service import LLMService
from src.llm.gemini_chat_model import get_gemini_llm

# Explicitly mark this module for pytest-asyncio auto mode handling
pytestmark = pytest.mark.asyncio

@pytest.fixture
async def llm_service() -> LLMService:
    """
    Fixture to provide a real LLMService instance initialized with the
    GeminiChatModel.
    """
    try:
        # Check if API key is present before attempting initialization
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY environment variable not set. Skipping E2E tests.")
            
        gemini_model = get_gemini_llm()
        return LLMService(llm=gemini_model)
    except Exception as e:
        pytest.skip(f"Failed to initialize Gemini LLM: {e}")

async def test_ainvoke_e2e(llm_service: LLMService):
    """
    End-to-end test for LLMService.ainvoke() using real Gemini API.
    """
    logger.info("Running test_ainvoke_e2e")
    prompt = "Hello! In one short sentence, tell me who you are."
    
    try:
        response = await llm_service.ainvoke(prompt)
    except Exception as e:
        pytest.fail(f"LLMService.ainvoke() failed: {e}")

    logger.info(f"Response: {response.content}")
    
    assert response is not None
    assert isinstance(response, AIMessage) or isinstance(response, BaseMessageChunk)
    assert len(response.content) > 0
    logger.info("test_ainvoke_e2e passed")

async def test_astream_e2e(llm_service: LLMService):
    """
    End-to-end test for LLMService.astream() using real Gemini API.
    """
    logger.info("Running test_astream_e2e")
    prompt = "Count from 1 to 5."
    
    collected_content = ""
    chunk_count = 0
    
    try:
        async for chunk in llm_service.astream(prompt):
            chunk_count += 1
            collected_content += chunk.content
            # logger.debug(f"Chunk {chunk_count}: {chunk.content}")
    except Exception as e:
        pytest.fail(f"LLMService.astream() failed: {e}")

    logger.info(f"Total collected content: {collected_content}")
    
    assert chunk_count > 0, "Should receive at least one chunk"
    assert len(collected_content) > 0, "Total content should not be empty"
    # Basic check to see if it actually followed instructions (optional)
    assert "1" in collected_content and "5" in collected_content
    logger.info("test_astream_e2e passed")
