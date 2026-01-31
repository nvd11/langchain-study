from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 这个示例专门用于演示 MessagesPlaceholder 的核心作用：
# 将一个"消息列表"展开并无缝嵌入到 Prompt 中。

def main():
    # 1. 准备数据：假设这是从内存或数据库中取出的对话历史
    history_messages = [
        HumanMessage(content="我叫小明。"),
        AIMessage(content="你好小明，很高兴认识你！"),
        HumanMessage(content="我是一名程序员。")
    ]

    # ------------------------------------------------------------------
    # 场景 A: 使用 MessagesPlaceholder (正确做法)
    # ------------------------------------------------------------------
    print("--- 场景 A: 使用 MessagesPlaceholder ---")
    print("Prompt 结构: System -> [History Placeholder] -> Human")
    
    prompt_with_placeholder = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手。"),
        # 这里的 variable_name="history" 对应 invoke 时传入的 key
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{input}")
    ])

    # 格式化 Prompt (模拟 Chain 运行时的第一步)
    final_messages_a = prompt_with_placeholder.format_messages(
        history=history_messages,
        input="我刚才说了我是做什么的？"
    )

    # 打印结果：可以看到 history_messages 被"展开"了
    print(f"\n生成的最终消息列表 (共 {len(final_messages_a)} 条):")
    for i, msg in enumerate(final_messages_a):
        print(f"[{i}] {type(msg).__name__}: {msg.content}")

    # 结果分析：
    # [0] SystemMessage: 你是一个有用的助手。
    # [1] HumanMessage: 我叫小明。          <--- 历史记录直接作为独立消息嵌入
    # [2] AIMessage: 你好小明...            <--- 历史记录直接作为独立消息嵌入
    # [3] HumanMessage: 我是一名程序员。      <--- 历史记录直接作为独立消息嵌入
    # [4] HumanMessage: 我刚才说了我是做什么的？

    
    # ------------------------------------------------------------------
    # 场景 B: 不使用 MessagesPlaceholder (常见错误)
    # ------------------------------------------------------------------
    print("\n\n--- 场景 B: 错误尝试 (试图把 List 塞进字符串变量) ---")
    try:
        # 如果我们尝试像处理字符串一样处理 history
        prompt_bad = ChatPromptTemplate.from_messages([
            ("system", "你是一个有用的助手。"),
            ("human", "之前的对话:\n{history}"), # <--- 错误用法
            ("human", "{input}")
        ])
        
        final_messages_b = prompt_bad.format_messages(
            history=history_messages,
            input="我刚才说了我是做什么的？"
        )
        
        print(f"生成的最终消息列表 (共 {len(final_messages_b)} 条):")
        for i, msg in enumerate(final_messages_b):
            print(f"[{i}] {type(msg).__name__}: {msg.content}")
            
    except Exception as e:
        print(f"发生错误: {e}")
        # 在某些版本中，这会将 List[Message] 强行转换为字符串表示，导致模型困惑。
        # 转换后的字符串可能长这样: "[HumanMessage(content='...'), ...]" 
        # 这不再是结构化的消息，而是一段乱糟糟的文本。

if __name__ == "__main__":
    main()
