from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from src.llm.gemini_chat_model import get_gemini_llm
from loguru import logger

# 模拟一个检索器 (Retriever)
# 在真实场景中，这通常是 vector_store.as_retriever()
def fake_retriever(query: str):
    logger.info(f"Retrieving documents for: {query}")
    return f"这里是关于 '{query}' 的一些背景知识..."

def main():
    llm = get_gemini_llm()

    # -----------------------------------------------------------
    # 场景 1: 基础用法 - 原样透传
    # -----------------------------------------------------------
    logger.info("--- Demo 1: Basic Passthrough ---")
    
    # 这里的 RunnablePassthrough() 就像一个占位符，它把 invoke 传入的 "Hello" 原封不动地传给下一步
    # 虽然在这个简单的例子里看起来没用，但在复杂的字典构造中非常关键
    chain = RunnablePassthrough() 
    result = chain.invoke("Hello World")
    logger.info(f"Result: {result}") # Output: Hello World


    # -----------------------------------------------------------
    # 场景 2: RAG (检索增强生成) - 最经典用法
    # -----------------------------------------------------------
    logger.info("\n--- Demo 2: RAG Scenario ---")
    
    prompt = ChatPromptTemplate.from_template(
        "基于以下上下文回答问题:\n\n上下文: {context}\n\n问题: {question}"
    )

    # 我们构建一个并行运行的 Map (RunnableParallel)
    # 1. "context" 键：把用户输入传给 retriever，获取上下文
    # 2. "question" 键：我们需要把用户原始输入填到这里。
    #    如果不加 RunnablePassthrough()，我们就没法在这里引用“原始输入”了。
    
    # 提示：这里的 RunnableLambda(fake_retriever) 其实可以简化为直接写 fake_retriever
    # LangChain 会自动把函数转换为 RunnableLambda。
    rag_chain = (
        {
            "context": fake_retriever,  # <--- 自动隐式转换为 RunnableLambda(fake_retriever)
            "question": RunnablePassthrough() 
        }
        | prompt
        | llm
    )

    # 调用链
    # 用户输入 "什么是 LangChain?"
    # 1. fake_retriever("什么是 LangChain?") -> 填充 context
    # 2. RunnablePassthrough() 接收 "什么是 LangChain?" 并原样返回 -> 填充 question
    response = rag_chain.invoke("什么是 LangChain?")
    logger.info(f"AI Response: {response.content}")


    # -----------------------------------------------------------
    # 场景 3: .assign() - 添加新字段
    # -----------------------------------------------------------
    logger.info("\n--- Demo 3: RunnablePassthrough.assign() ---")
    
    # 假设输入是一个字典
    input_data = {"num": 10}

    # 我们想在不丢失原始数据 ("num") 的基础上，计算一个新字段 "squared"
    # assign 会把函数的返回值合并到原始字典中
    chain_with_assign = RunnablePassthrough.assign(
        squared=lambda x: x["num"] * x["num"]
    )

    result_assign = chain_with_assign.invoke(input_data)
    logger.info(f"Result with assign: {result_assign}") 
    # Output: {'num': 10, 'squared': 100}

if __name__ == "__main__":
    main()
