import asyncio
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.llm.gemini_chat_model import get_gemini_llm
from loguru import logger

async def main():
    llm = get_gemini_llm()

    # ------------------------------------------------------------------
    # 1. Define Sub-Chains (The destinations)
    # ------------------------------------------------------------------
    
    # Chain A: For Math questions
    math_chain = (
        PromptTemplate.from_template("You are a mathematician. Solve this: {query}")
        | llm
        | StrOutputParser()
    )

    # Chain B: For History questions
    history_chain = (
        PromptTemplate.from_template("You are a historian. Explain this event: {query}")
        | llm
        | StrOutputParser()
    )

    # Chain C: Default (General)
    general_chain = (
        PromptTemplate.from_template("You are a helpful assistant. Answer this: {query}")
        | llm
        | StrOutputParser()
    )

    # ------------------------------------------------------------------
    # 2. Define Routing Logic (The Switch)
    # ------------------------------------------------------------------
    # RunnableBranch takes a list of (condition, runnable) pairs, 
    # plus a default runnable.
    
    branch = RunnableBranch(
        (lambda x: "math" in x["topic"].lower(), math_chain),
        (lambda x: "history" in x["topic"].lower(), history_chain),
        general_chain # Default branch
    )

    # ------------------------------------------------------------------
    # 3. Compose the Full Chain
    # ------------------------------------------------------------------
    # Input: {"topic": "...", "query": "..."}
    full_chain = branch

    # ------------------------------------------------------------------
    # 4. Execute
    # ------------------------------------------------------------------
    
    # Test 1: Math
    logger.info("--- Test 1: Math Topic ---")
    result1 = await full_chain.ainvoke({"topic": "math", "query": "1 + 1"})
    logger.info(f"Result: {result1}")

    # Test 2: History
    logger.info("\n--- Test 2: History Topic ---")
    result2 = await full_chain.ainvoke({"topic": "history", "query": "WWII"})
    logger.info(f"Result: {result2}")

    # Test 3: General (Default)
    logger.info("\n--- Test 3: General Topic ---")
    result3 = await full_chain.ainvoke({"topic": "cooking", "query": "How to boil an egg?"})
    logger.info(f"Result: {result3}")

if __name__ == "__main__":
    asyncio.run(main())
