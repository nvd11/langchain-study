import src.configs.config
from loguru import logger
from src.llm.gemini_chat_model import get_gemini_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# 1. 定义存储 ChatHistory 的字典
# 在实际生产中，通常使用 Redis 或数据库来持久化存储
store = {}

# 2. 定义获取 session history 的工厂函数
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        logger.info(f"Creating new history for session: {session_id}")
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def main():
    # 3. 初始化 LLM
    try:
        llm = get_gemini_llm()
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return

    # 4. 创建包含历史记录占位符的 Prompt
    # MessagesPlaceholder(variable_name="history") 将自动填充历史消息
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # 5. 创建基础链 (LCEL)
    chain = prompt | llm

    # 6. 使用 RunnableWithMessageHistory 包装基础链
    # 参数详解：
    # 1. runnable (chain): 
    #    被包装的基础链 (Prompt | LLM)。它负责处理单次交互，不知道历史记录的存在。
    #
    # 2. get_session_history: 
    #    工厂函数。接收 session_id，返回对应的 ChatMessageHistory 实例。
    #    系统会自动调用它来加载旧记录和保存新记录。
    #
    # 3. input_messages_key="input": 
    #    告诉系统，用户的"当前问题"在输入字典的哪个 key 中。
    #    例如 invoke({"input": "你好"})，这里就是 "input"。
    #
    # 4. history_messages_key="history":
    #    告诉系统，加载出来的"历史记录"应该填入 Prompt 的哪个变量中。
    #    这必须与 Prompt 中的 MessagesPlaceholder(variable_name="history") 对应。
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    logger.info("--- Testing Session 1 ---")
    
    # 第一轮对话
    logger.info("User: Hi! My name is Bob.")
    response1 = with_message_history.invoke(
        # 参数1: Input
        # 常见问题：为何这里不需要传入 "history"？
        # 答：因为 RunnableWithMessageHistory 会自动去后台加载历史记录，
        # 并自动填充到 Prompt 中的 MessagesPlaceholder(variable_name="history")。
        # 你只需要传入新产生的变量（如 "input"）。
        {"input": "Hi! My name is Bob."},

        # 参数2: Config
        # 常见问题：config 的作用是什么？
        # 答：它是 LangChain 传递元数据（Metadata）的标准方式。
        # RunnableWithMessageHistory 需要知道"当前是谁在聊天"，以便去数据库加载正确的记录。
        # 这里的 session_id="session_1" 会被传给上面的 get_session_history(session_id) 函数。
        config={"configurable": {"session_id": "session_1"}}
    )
    logger.info(f"AI: {response1.content}")

    # 第二轮对话 (应该能记住名字)
    logger.info("User: What is my name?")
    response2 = with_message_history.invoke(
        {"input": "What is my name?"},
        config={"configurable": {"session_id": "session_1"}}
    )
    logger.info(f"AI: {response2.content}")

    logger.info("\n--- Testing Session 2 (New Session) ---")
    
    # 第三轮对话 (新的 session_id，应该不记得名字)
    logger.info("User: What is my name?")
    response3 = with_message_history.invoke(
        {"input": "What is my name?"},
        config={"configurable": {"session_id": "session_2"}}
    )
    logger.info(f"AI: {response3.content}")

if __name__ == "__main__":
    main()
