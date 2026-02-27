import src.configs.config
from loguru import logger
from src.llm.gemini_chat_model import get_gemini_llm

# 注意：在您的环境中，经典组件位于 langchain_classic 中
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain

def main():
    # 1. 初始化 LLM
    try:
        llm = get_gemini_llm()
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return

    # 2. 初始化 Memory
    # ConversationBufferMemory 的作用是将所有对话历史直接存储在内存中（Raw Text）
    memory = ConversationBufferMemory()

    # 3. 创建 ConversationChain
    # 这是一个经典的链，预置了默认的 Prompt 模板，专门用于简单的闲聊
    conversation = ConversationChain(
        llm=llm, 
        memory=memory,
        verbose=True # 开启 verbose 可以看到链的思考过程
    )

    logger.info("--- Round 1 ---")
    # 开始对话
    response1 = conversation.predict(input="Hi, my name is Alice.")
    logger.info(f"User: Hi, my name is Alice.")
    logger.info(f"AI: {response1}")

    logger.info("\n--- Round 2 ---")
    # 测试记忆能力
    response2 = conversation.predict(input="What is my name?")
    logger.info(f"User: What is my name?")
    logger.info(f"AI: {response2}")

    logger.info("\n--- Memory Inspection ---")
    # 查看 Memory 内部到底存了什么
    # load_memory_variables 会返回包含历史记录的字典
    buffer_content = memory.load_memory_variables({})
    logger.info(f"Buffer Content (Raw String - Default):\n{buffer_content['history']}")

    # ------------------------------------------------------------------
    # 演示 return_messages=True 的作用
    # ------------------------------------------------------------------
    logger.info("\n--- Demo: return_messages=True ---")
    
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
    chat_memory = ConversationBufferMemory(return_messages=True)
    
    # 模拟保存一些对话
    chat_memory.save_context({"input": "Hi"}, {"output": "Hello"})
    chat_memory.save_context({"input": "Who are you?"}, {"output": "I am AI"})

    # 查看内容
    chat_buffer = chat_memory.load_memory_variables({})
    logger.info(f"Buffer Content (Message List):\n{chat_buffer['history']}")
    # 输出将是: [HumanMessage(...), AIMessage(...)]

if __name__ == "__main__":
    main()
