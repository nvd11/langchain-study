import src.configs.config
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from loguru import logger

logger.info("Loading memory")

system_message = """
You are a helpful assistant. your name is gemini-boy.
"""

llm = get_gemini_llm()

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

