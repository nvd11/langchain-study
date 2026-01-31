import  src.configs.config 
from loguru import logger
from src.llm.gemini_chat_model import get_gemini_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import asyncio
# create a llm model ( gemini 2.5 pro)
llm_model = get_gemini_llm()



#define a prompt template
prompt_template = PromptTemplate(
    input_variables=["topic", "language"],
    template="Tell a story in {language} about the following topics :{topic}"
)


# use LCEL - LangChain Expression Languagev
chain = prompt_template | llm_model

# execute the chain
result = chain.invoke({"topic": "sea", "language": "Chinese"})
logger.info(f"type of result: {type(result)}") #return of llm model invoke() is a AIMesseage object
logger.info(f"result's meta data :")ssss
logger.info(result.response_metadata)
logger.info(result.usage_metadata)
logger.info(f"result: {result}")

chain = prompt_template | llm_model | StrOutputParser()
result = asyncio.run(chain.ainvoke({"topic": "sea", "language": "Chinese"}))
logger.info(f"type of result: {type(result)}") # return of StrOutputParser is a str object
logger.info(f"result: {result}")    
