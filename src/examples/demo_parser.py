import src.configs.config
from loguru import logger
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

chain = prompt | model | parser

# 3. 运行链
# 这次运行会自动被 LangSmith 记录，因为环境变量开关已打开
logger.info("正在生成列表...")

print("\n=== 生成结果 ===")
input_vars = {"things": "sports that don't use balls"}
print(f"--- Prompt Value ---\n{prompt.format(**input_vars)}\n--------------------")
response = chain.invoke(input_vars)
print(response)
print("==================")
print("请去 LangSmith 控制台查看本次运行的 Trace 详情。")
