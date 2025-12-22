import src.configs.config
from loguru import logger
from langchain_core.tools import tool
from src.llm.gemini_chat_model import get_gemini_llm

# 1. 定义工具 (使用 @tool 装饰器)
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

# 2. 初始化 LLM
llm = get_gemini_llm()

# 3. 绑定工具 (Native Function Calling)
# 这会把工具的 schema 转换为 Gemini API 能理解的格式 (Function Declaration)
llm_with_tools = llm.bind_tools([multiply])

# 4. 执行
logger.info("=== Demo: Native Function Calling (bind_tools) ===")
query = "What is 123 multiplied by 456?"
logger.info(f"User Question: {query}")

response = llm_with_tools.invoke(query)

logger.info(f"LLM Response Type: {type(response)}")
logger.info(f"LLM Response Content: {response.content}")

# 5. 检查是否触发了 Function Call
if response.tool_calls:
    logger.info("Tool Call Detected!")
    for tool_call in response.tool_calls:
        logger.info(f"Tool Name: {tool_call['name']}")
        logger.info(f"Arguments: {tool_call['args']}")
        
        # 执行工具 (可选)
        if tool_call['name'] == 'multiply':
            result = multiply.invoke(tool_call['args'])
            logger.info(f"Tool Execution Result: {result}")
else:
    logger.info("No tool call detected.")
