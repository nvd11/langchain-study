import asyncio
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.llm.gemini_chat_model import get_gemini_llm
from loguru import logger

async def main():
    llm = get_gemini_llm()

    # ------------------------------------------------------------------
    # 1. Define The Router Chain (The Classifier)
    # ------------------------------------------------------------------
    # This chain's job is ONLY to classify the intent.
    
    router_prompt = PromptTemplate.from_template(
        """
        Given the user question below, classify it into one of the following categories:
        - math
        - history
        - general
        
        Do not answer the question. Just return the category name (lowercase).
        
        <Question>
        {query}
        </Question>
        
        Category:
        """
    )
    
    # Input: {"query": "..."} -> Output: "math" (str)
    router_chain = router_prompt | llm | StrOutputParser()

    # ------------------------------------------------------------------
    # 2. Define Sub-Chains (The Experts)
    # ------------------------------------------------------------------
    
    math_chain = (
        PromptTemplate.from_template("You are a mathematician. Solve this: {query}")
        | llm
        | StrOutputParser()
    )

    history_chain = (
        PromptTemplate.from_template("You are a historian. Explain this event: {query}")
        | llm
        | StrOutputParser()
    )

    general_chain = (
        PromptTemplate.from_template("You are a helpful assistant. Answer this: {query}")
        | llm
        | StrOutputParser()
    )

    # ------------------------------------------------------------------
    # 3. Define Branching Logic based on Router Output
    # ------------------------------------------------------------------
    # The input to this branch will be a dict: {"query": "...", "topic": "math"}
    
    branch = RunnableBranch(
        (lambda x: "math" in x["topic"].lower(), math_chain),
        (lambda x: "history" in x["topic"].lower(), history_chain),
        general_chain # Default
    )

    # ------------------------------------------------------------------
    # 4. Compose the Full Chain
    # ------------------------------------------------------------------
    # Step A: Pass input to Router to get 'topic'
    # Step B: Pass {query, topic} to Branch
    
    full_chain = (
        RunnablePassthrough.assign(topic=router_chain) 
        | branch
    )

    # ------------------------------------------------------------------
    # 5. Execute Tests (Semantic Routing)
    # ------------------------------------------------------------------
    
    # Case 1: Math (No keyword 'math' needed!)
    # The Router should infer this is a math problem.
    query1 = "What is the square root of 144?"
    logger.info(f"--- Test 1: '{query1}' ---")
    
    # Let's peek at the classification first (optional, for demo)
    topic1 = await router_chain.ainvoke({"query": query1})
    logger.info(f"Router classified as: [{topic1.strip()}]")
    
    result1 = await full_chain.ainvoke({"query": query1})
    logger.info(f"Result: {result1}\n")


    # Case 2: History (Complex sentence)
    query2 = "Tell me about the fall of the Roman Empire."
    logger.info(f"--- Test 2: '{query2}' ---")
    
    topic2 = await router_chain.ainvoke({"query": query2})
    logger.info(f"Router classified as: [{topic2.strip()}]")
    
    result2 = await full_chain.ainvoke({"query": query2})
    logger.info(f"Result: {result2}\n")


    # Case 3: General
    query3 = "How to make a sandwich?"
    logger.info(f"--- Test 3: '{query3}' ---")
    
    topic3 = await router_chain.ainvoke({"query": query3})
    logger.info(f"Router classified as: [{topic3.strip()}]")
    
    result3 = await full_chain.ainvoke({"query": query3})
    logger.info(f"Result: {result3}\n")

if __name__ == "__main__":
    asyncio.run(main())
