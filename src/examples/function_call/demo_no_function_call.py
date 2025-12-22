import src.configs.config
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from src.llm.gemini_chat_model import get_gemini_llm

# 1. 定义工具 (Tools)
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

def main():
    # 2. 初始化 LLM
    llm = get_gemini_llm()

    # 3. 定义 ReAct 风格的 Prompt (手动教 LLM 如何调用工具)
    # 我们不使用 bind_tools，而是把工具描述写在 Prompt 里
    react_system_prompt = """
    You are a helpful assistant. You have access to the following tools:

    1. multiply: Multiplies two integers. Input should be two numbers separated by a comma.

    To use a tool, please use the following format exactly:

    Thought: Do I need to use a tool? Yes
    Action: multiply
    Action Input: 5, 4
    Observation: [Tool output will be placed here]

    If you do not need to use a tool, just answer the question directly.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", react_system_prompt),
        ("user", "{input}")
    ])

    chain = prompt | llm

    # 4. 执行
    logger.info("=== Demo: Manual Tool Usage (No Function Call) ===")
    query = "What is 123 multiplied by 456?"
    logger.info(f"User Question: {query}")

    response = chain.invoke({"input": query})
    content = response.content
    logger.info(f"LLM Response (Raw Content):\n{content}")

    # 5. 手动解析并执行 (模拟 Agent 的工作)
    # 这是一个非常简化的解析器
    if isinstance(content, str) and "Action: multiply" in content:
        try:
            # Extract input (Assuming format "Action Input: x, y")
            lines = content.split('\n')
            action_input_line = next(line for line in lines if line.strip().startswith("Action Input:"))
            input_str = action_input_line.split(":")[1].strip()
            args = [int(x.strip()) for x in input_str.split(",")]
            
            logger.info(f"Detected Tool Call: multiply with args {args}")
            
            # Execute tool
            result = multiply(args[0], args[1])
            logger.info(f"Tool Execution Result: {result}")
            
            # (Optional) We could feed this back to LLM to generate final answer, 
            # but for this demo we just show that we executed the tool manually.
        except Exception as e:
            logger.error(f"Failed to parse or execute tool: {e}")
    else:
        logger.info("No tool call detected or format incorrect.")

if __name__ == "__main__":
    main()
