import src.configs.config
from loguru import logger
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from src.llm.gemini_chat_model import get_gemini_llm

# 1. 确保环境变量已加载 (.env)
# 必须包含:
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=...

# 2. 定义一个简单的链
prompt = PromptTemplate(
     template = "List 3 {things}",
     input_variables=["things"]
)
model =  get_gemini_llm()
parser = StrOutputParser()
list_parser = CommaSeparatedListOutputParser()

chain = prompt | model | parser
chain2 = prompt | model | list_parser

# 3. 运行链
# 这次运行会自动被 LangSmith 记录，因为环境变量开关已打开
logger.info("正在生成列表...")

print("\n=== 生成结果 ===")
input_vars = {"things": "sports that don't use balls"}
# **input_vars: The double asterisks unpack the dictionary into keyword arguments.
# Example: {"things": "..."} becomes the argument things="..."
# A single asterisk (*) is used for unpacking Lists (positional arguments).
print(f"--- Prompt Value ---\n{prompt.format(**input_vars)}\n--------------------")
response = chain.invoke(input_vars)
logger.info(f"type of response: {type(response)}\n response: {response}")
print("==================")
response = chain2.invoke(input_vars)
logger.info(f"type of response: {type(response)}\n response: {response}")
print("请去 LangSmith 控制台查看本次运行的 Trace 详情。")
