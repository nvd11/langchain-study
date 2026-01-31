import asyncio
import requests
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.llm.gemini_chat_model import get_gemini_llm
from loguru import logger

# ------------------------------------------------------------------
# 1. Define the Tool Function (Plain Python Function)
# ------------------------------------------------------------------
def get_exchange_rate(inputs: dict) -> str:
    """
    Explicitly fetches exchange rate.
    Expects input dict with 'base' and 'target' keys.
    """
    base = inputs["base"].upper()
    target = inputs["target"].upper()
    
    logger.info(f"⚡ [Explicit Tool] Fetching rate for {base} -> {target}...")
    
    url = f"https://api.exchangerate-api.com/v4/latest/{base}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        rates = data.get('rates', {})
        if target in rates:
            return str(rates[target])
        return "Unknown"
    except Exception as e:
        return "Error"

async def main():
    llm = get_gemini_llm()

    # ------------------------------------------------------------------
    # 2. Build the Explicit Chain
    # ------------------------------------------------------------------
    # Logic: 
    # 1. Receive input (base, target)
    # 2. FORCE execution of get_exchange_rate (No LLM decision here!)
    # 3. Pass result to Prompt -> LLM
    
    # Step A: Define the prompt that EXPECTS the rate as context
    prompt = PromptTemplate.from_template(
        """
        Current exchange rate from {base} to {target} is: {rate}.
        
        Please give me a short financial advice based on this rate.
        """
    )

    # Step B: Construct the chain using RunnableLambda for the tool
    # RunnablePassthrough.assign() adds the 'rate' key to the input dictionary
    chain = (
        RunnablePassthrough.assign(rate=RunnableLambda(get_exchange_rate)) 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    # ------------------------------------------------------------------
    # 3. Execute
    # ------------------------------------------------------------------
    # Notice: We are NOT asking a natural language question.
    # We are providing structured arguments because the flow is deterministic.
    input_data = {"base": "USD", "target": "JPY"}
    
    logger.info(f"Input: {input_data}")
    logger.info("Starting explicit chain...")
    
    result = await chain.ainvoke(input_data)
    
    logger.info(f"Final Output:\n{result}")

if __name__ == "__main__":
    asyncio.run(main())
