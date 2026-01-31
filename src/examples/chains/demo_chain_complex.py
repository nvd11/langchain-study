import asyncio
from operator import itemgetter
import src.configs.config 
from src.llm.gemini_chat_model import get_gemini_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from loguru import logger

async def main():
    # Initialize LLM
    llm = get_gemini_llm()

    logger.info("Building Complex LCEL Chain (Chinese Output)...")

    # ------------------------------------------------------------------
    # Step 1: Define Parallel Branches
    # ------------------------------------------------------------------
    
    # Branch A: History Expert
    # Gets the city name, generates a brief history
    history_chain = (
        PromptTemplate.from_template("Provide a brief history of {city} in Chinese. Keep it under 100 words.")
        | llm
        | StrOutputParser()
    )

    # Branch B: Tour Guide
    # Gets the city name, lists top attractions
    attractions_chain = (
        PromptTemplate.from_template("List 3 top attractions in {city} in Chinese. Keep it concise.")
        | llm
        | StrOutputParser()
    )

    # ------------------------------------------------------------------
    # Step 2: Combine with RunnableParallel
    # ------------------------------------------------------------------
    # RunnableParallel runs 'history' and 'attractions' chains concurrently.
    # RunnablePassthrough() passes the original input 'city' through to the next step.
    # The output will be a dictionary: {'history': ..., 'attractions': ..., 'city': ...}
    map_chain = RunnableParallel(
        history=history_chain,
        attractions=attractions_chain,
        city=RunnablePassthrough() 
    )

    # ------------------------------------------------------------------
    # Step 3: Define Final Synthesis Chain
    # ------------------------------------------------------------------
    final_prompt = PromptTemplate.from_template(
        """
        Write a travel proposal email for {city} in Chinese.
        
        Historical Context:
        {history}
        
        Must-see Attractions:
        {attractions}
        
        Tone: Professional yet exciting.
        """
    )

    # ------------------------------------------------------------------
    # Step 4: Compose the Full Chain
    # ------------------------------------------------------------------
    # Logic: Input -> Parallel Processing -> Dictionary -> Final Prompt -> LLM -> String
    full_chain = map_chain | final_prompt | llm | StrOutputParser()

    # ------------------------------------------------------------------
    # Step 5: Execute
    # ------------------------------------------------------------------
    city = "Kyoto"
    logger.info(f"Start processing for city: {city}")
    logger.info("Running parallel chains (History & Attractions) and combining results...")
    
    # Using astream to show the typewriter effect on the final result
    print("\n" + "="*50)
    print(f" TRAVEL PROPOSAL FOR {city.upper()} ")
    print("="*50 + "\n")

    async for chunk in full_chain.astream(city):
        print(chunk, end="", flush=True)
    
    print("\n\n" + "="*50 + "\n")
    logger.info("Chain execution completed.")

if __name__ == "__main__":
    asyncio.run(main())
