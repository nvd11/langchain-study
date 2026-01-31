import src.configs.config
from loguru import logger
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from src.llm.gemini_chat_model import get_gemini_llm

llm = get_gemini_llm()

# ==========================================
# 1. StrOutputParser (最基础：输出为字符串)
# ==========================================
logger.info("--- Demo 1: StrOutputParser ---")
prompt1 = PromptTemplate.from_template("List 3 {things}")
chain1 = prompt1 | llm | StrOutputParser()
result1 = chain1.invoke({"things": "colors"})
logger.info(f"Result type: {type(result1)}") # <class 'str'>
logger.info(f"Result: {result1}")


# ==========================================
# 2. CommaSeparatedListOutputParser (输出为列表)
# ==========================================
logger.info("\n--- Demo 2: CommaSeparatedListOutputParser ---")
prompt2 = PromptTemplate.from_template("List 3 {things}. Return as comma separated list.")
chain2 = prompt2 | llm | CommaSeparatedListOutputParser()
result2 = chain2.invoke({"things": "fruits"})
logger.info(f"Result type: {type(result2)}") # <class 'list'>
logger.info(f"Result: {result2}")


# ==========================================
# 3. JsonOutputParser (输出为字典)
# ==========================================
logger.info("\n--- Demo 3: JsonOutputParser ---")
# 定义期望的数据结构
prompt3 = PromptTemplate.from_template(
    """
    Return a JSON object with two fields: 'name' (string) and 'population' (int) for the city {city}.
    Do not wrap in markdown code blocks.
    """
)
chain3 = prompt3 | llm | JsonOutputParser()
result3 = chain3.invoke({"city": "Tokyo"})
logger.info(f"Result type: {type(result3)}") # <class 'dict'>
logger.info(f"Result: {result3}")


# ==========================================
# 4. PydanticOutputParser (最强大：输出为强类型对象)
# ==========================================
logger.info("\n--- Demo 4: PydanticOutputParser ---")

# 定义数据模型
class Country(BaseModel):
    name: str = Field(description="name of the country")
    capital: str = Field(description="capital city of the country")
    population: int = Field(description="approximate population")

parser = PydanticOutputParser(pydantic_object=Country)

# 将格式说明注入 Prompt
prompt4 = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n\nQuery: {query}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain4 = prompt4 | llm | parser

result4 = chain4.invoke({"query": "Tell me about France."})
logger.info(f"Result type: {type(result4)}") # <class '__main__.Country'>
logger.info(f"Result: {result4}")
logger.info(f"Accessing field 'capital': {result4.capital}")
