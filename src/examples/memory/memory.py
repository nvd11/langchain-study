import src.configs.config
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from loguru import logger
from src.llm.gemini_chat_model import get_gemini_llm

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

    # 关键参数解释：return_messages=True
    #
    # 1. 默认情况 (False): load_memory_variables 返回一个纯字符串 (String)。
    #    格式如: "Human: Hi\nAI: Hello"
    #    适用场景: PromptTemplate (Text Completion Model)
    #
    # 2. 开启情况 (True): load_memory_variables 返回一个消息对象列表 (List[BaseMessage])。
    #    格式如: [HumanMessage(content="Hi"), AIMessage(content="Hello")]
    #    适用场景: ChatPromptTemplate + MessagesPlaceholder (Chat Model)
    #    这能确保消息结构不丢失，直接以 Object 形式传给模型 API。
#/workspace/src/examples/memory/memory.py:33: LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
# 为什么会有 Warning：您使用的是 langchain 或 langchain_community 中的旧版 Memory 组件。官方为了推动架构升级（从单体 Chain 到 LCEL/LangGraph），标记了这些组件为 Deprecated（已弃用），并计划在未来的 1.0 版本中移除。
conversation_buffer_memory = ConversationBufferMemory(return_messages=True)
history = conversation_buffer_memory.load_memory_variables({})
# 2026-01-31 16:50:19.875 | INFO     | __main__:<module>:36 - History: {'history': []
logger.info(f"History: {history}")


chat_with_conversation_buffer_memory = (
    RunnablePassthrough(
         history=RunnableLambda(
    )
)