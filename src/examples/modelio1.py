import src.configs.config  
from loguru import logger
from openai import OpenAIError
from src.llm.gemini_chat_model import get_gemini_llm
from src.llm.deepseek_chat_model import get_deepseek_llm
import asyncio

from langchain_openai import ChatOpenAI


logger.info("lession 1 - Model IO")

# ChatOpenAI is a subclass of BaseChatModel
# openai.OpenAIError: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable
llm_openai = None
llm_gemini = None

# 
# 
# 
# 
try:
    llm_openai= ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
except OpenAIError as e:
    logger.error(e)


llm_gemini = get_gemini_llm()
llm_deepseek = get_deepseek_llm()


prompt = "how to be an ai engineer?"
response = asyncio.run(llm_gemini.ainvoke(prompt))
logger.info("response of gemini:" + str(response.content))

response = asyncio.run(llm_deepseek.ainvoke(prompt))
logger.info("response of deepseek:" + str(response.content))
logger.info("done")
