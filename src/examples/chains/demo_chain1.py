import src.configs.config 
from loguru import logger
from src.llm.gemini_chat_model import get_gemini_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import asyncio

# Initialize model and template
llm_model = get_gemini_llm()

prompt_template = PromptTemplate(
    input_variables=["topic", "language"],
    template="Tell a story in {language} about the following topics :{topic}"
)

# ------------------------------------------------------------------
# Demo 1: Synchronous invoke
# ------------------------------------------------------------------
logger.info("--- Demo 1: Basic Chain (Synchronous Invoke) ---")
chain = prompt_template | llm_model

# Call invoke directly, skipping asyncio loop
# Note: This blocks current thread until completion
result = chain.invoke({"topic": "sea", "language": "Chinese"})

logger.info(f"type of result: {type(result)}") 
if hasattr(result, 'response_metadata'):
    logger.info(f"result's meta data : {result.response_metadata}")
logger.info(f"result: {result}")


# ------------------------------------------------------------------
# Demo 2 & 3: Async operations must be wrapped in async function
# ------------------------------------------------------------------
async def async_main():
    # Demo 2: Chain with OutputParser (Async ainvoke)
    logger.info("\n--- Demo 2: Chain with OutputParser (Async) ---")
    chain_with_parser = prompt_template | llm_model | StrOutputParser()
    
    result_str = await chain_with_parser.ainvoke({"topic": "mountain", "language": "English"})
    
    logger.info(f"type of result: {type(result_str)}") 
    logger.info(f"result: {result_str[:100]}...") 

    # Demo 3: Streaming Output (Typewriter Effect)
    logger.info("\n--- Demo 3: Streaming Output (Typewriter Effect) ---")
    print("\n[Start Streaming]\n")
    
    # async for iterates through chunks as soon as they are generated.
    # It does NOT wait for the entire response to be ready.
    # This enables the real-time typewriter effect.
    async for chunk in chain_with_parser.astream({"topic": "coding", "language": "Chinese"}):
        print(chunk, end="", flush=True)
    
    print("\n\n[End Streaming]\n")

if __name__ == "__main__":
    # Run async part
    asyncio.run(async_main())
