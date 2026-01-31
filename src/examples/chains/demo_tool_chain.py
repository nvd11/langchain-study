import asyncio
import requests
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from src.llm.gemini_chat_model import get_gemini_llm
from loguru import logger

# ------------------------------------------------------------------
# 1. Define Tools (Real Exchange Rate API)
# ------------------------------------------------------------------

@tool
def get_exchange_rate(base_currency: str, target_currency: str) -> str:
    """
    Get the LIVE exchange rate between two currencies using a real public API.
    Args:
        base_currency: The currency code to convert from (e.g., 'USD', 'CNY').
        target_currency: The currency code to convert to (e.g., 'EUR', 'JPY').
    """
    base_currency = base_currency.upper()
    target_currency = target_currency.upper()
    
    logger.info(f"API CALL: Fetching real rate for {base_currency} -> {target_currency}...")
    
    url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        rates = data.get('rates', {})
        if target_currency in rates:
            rate = rates[target_currency]
            return f"Current exchange rate: 1 {base_currency} = {rate} {target_currency} (Date: {data.get('date')})"
        else:
            return f"Error: Target currency '{target_currency}' not found in API response."
            
    except Exception as e:
        logger.error(f"API Request Failed: {e}")
        return f"Error fetching exchange rate: {str(e)}"

async def main():
    logger.info("Initializing LLM with Real Exchange Rate Tool...")
    llm = get_gemini_llm()
    
    # ------------------------------------------------------------------
    # 2. Bind Tools to LLM
    # ------------------------------------------------------------------
    tools = [get_exchange_rate]
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    prompt = ChatPromptTemplate.from_template("Answer the user query: {query}")
    
    # ------------------------------------------------------------------
    # 3. Add Parser: JsonOutputToolsParser
    # ------------------------------------------------------------------
    # This parser automatically extracts tool calls from the AIMessage 
    # and returns them as a clean list of dictionaries.
    # No need to manually check result.tool_calls!
    parser = JsonOutputToolsParser()
    
    chain = prompt | llm_with_tools | parser

    # ------------------------------------------------------------------
    # Test: Query that requires Exchange Rate
    # ------------------------------------------------------------------
    query = "How much is 100 US dollars in Chinese Yuan today?"
    logger.info(f"\nUser Query: {query}")
    logger.info("Sending query to LLM...")
    
    # Invoke the chain
    # The result will now be a LIST of tool calls (dictionaries)
    tool_calls = await chain.ainvoke({"query": query})
    
    logger.info(f"Parsed Result Type: {type(tool_calls)}")
    logger.info(f"Parsed Result: {tool_calls}")
    
    # ------------------------------------------------------------------
    # 4. Execute Parsed Tool Calls
    # ------------------------------------------------------------------
    if tool_calls:
        logger.info(f"Tool Calls Detected: {len(tool_calls)}")
        
        for call in tool_calls:
            # JsonOutputToolsParser returns: {'type': 'tool_name', 'args': {...}}
            name = call["type"]
            args = call["args"]
            
            logger.info(f"-> Model requests Tool: '{name}'")
            logger.info(f"-> Arguments: {args}")
            
            # Execute the tool
            if name in tool_map:
                tool_instance = tool_map[name]
                tool_output = tool_instance.invoke(args)
                logger.info(f"<- Tool Output: {tool_output}")
            else:
                logger.warning(f"Unknown tool: {name}")
    else:
        logger.info("No tool calls generated.")

if __name__ == "__main__":
    asyncio.run(main())
