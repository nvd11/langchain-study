import asyncio
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from src.llm.gemini_chat_model import get_gemini_llm
from loguru import logger

# ------------------------------------------------------------------
# 1. Define Tools
# ------------------------------------------------------------------
@tool
def get_magic_number(input_text: str) -> int:
    """Returns a magic number based on the input."""
    logger.info(f"🛠️ Tool 'get_magic_number' called with input: {input_text}")
    return 42

# ------------------------------------------------------------------
# 2. Implement Custom Agent Constructor
# ------------------------------------------------------------------
def create_custom_tool_calling_agent(llm, tools, prompt):
    """
    A custom implementation of create_tool_calling_agent.
    It binds tools to the LLM and connects it to the prompt.
    """
    logger.info("🔧 Creating Custom Tool Calling Agent...")
    
    # 1. Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # 2. Construct the Agent (Runnable)
    # Logic: Input -> Prompt -> LLM with Tools
    agent = prompt | llm_with_tools
    return agent

async def main():
    llm = get_gemini_llm()
    tools = [get_magic_number]
    tool_map = {t.name: t for t in tools}

    # ------------------------------------------------------------------
    # 3. Create the Agent using our custom function
    # ------------------------------------------------------------------
    
    # Standard Agent Prompt
    prompt = ChatPromptTemplate.from_template(
        "You MUST use the 'get_magic_number' tool to answer this: {input}"
    )
    
    # Create the Agent!
    agent_runnable = create_custom_tool_calling_agent(llm, tools, prompt)

    # ------------------------------------------------------------------
    # 4. Define Executor (The Runtime)
    # ------------------------------------------------------------------
    
    def execute_agent(input_dict):
        # 1. Run the Agent (Reasoning)
        agent_response = agent_runnable.invoke(input_dict)
        
        # 2. Check for Tool Calls
        if not agent_response.tool_calls:
            return "No tool called."
            
        # 3. Execute Tool (Acting)
        call = agent_response.tool_calls[0]
        name = call["name"]
        args = call["args"]
        
        logger.info(f"🤖 Agent decided to call: {name}")
        
        if name in tool_map:
            result = str(tool_map[name].invoke(args))
            logger.info(f"✅ Tool Result: {result}")
            return result
        return "Unknown tool"

    # Encapsulate the execution logic into a Chain
    # This 'agent_executor_chain' mimics the behavior of AgentExecutor
    agent_executor_chain = RunnableLambda(execute_agent)

    # ------------------------------------------------------------------
    # 5. Embed Agent in a Larger LCEL Chain
    # ------------------------------------------------------------------
    
    preprocess_chain = (
        ChatPromptTemplate.from_template("Translate this to English: {user_query}")
        | llm
        | StrOutputParser()
    )
    
    postprocess_chain = (
        ChatPromptTemplate.from_template(
            "The magic number is {number}. Write a short poem about this number."
        )
        | llm
        | StrOutputParser()
    )

    full_chain = (
        preprocess_chain 
        | (lambda x: {"input": x}) 
        | agent_executor_chain      # The Agent runs here!
        | (lambda x: {"number": x}) 
        | postprocess_chain
    )

    # ------------------------------------------------------------------
    # 6. Execute
    # ------------------------------------------------------------------
    query = "Quel est le nombre magique?" # French
    logger.info(f"User Query: {query}")
    logger.info("Running chain: Translate -> Agent -> Poem...")
    
    result = await full_chain.ainvoke({"user_query": query})
    
    logger.info(f"\nFinal Result:\n{result}")

if __name__ == "__main__":
    asyncio.run(main())
