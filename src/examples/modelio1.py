import src.configs.config  
from loguru import logger

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




logger.info("done")