import src.configs.config
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.llm.gemini_chat_model import get_gemini_llm

# 1. 确保环境变量已加载 (.env)
# 必须包含:
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=...

# 2. 定义一个简单的链
prompt = ChatPromptTemplate.from_template("请用{language}写一首关于{topic}的四行诗。")
model =  get_gemini_llm()
parser = StrOutputParser()

chain = prompt | model | parser

# 3. 运行链
# 这次运行会自动被 LangSmith 记录，因为环境变量开关已打开
print("正在生成诗歌...")
response = chain.invoke({"language": "中文", "topic": "人工智能"})

print("\n=== 生成结果 ===")
print(response)
print("==================")
print("请去 LangSmith 控制台查看本次运行的 Trace 详情。")
