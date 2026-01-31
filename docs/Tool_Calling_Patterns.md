# LangChain LCEL 工具调用实战：从确定性链到动态 Agent

## 1. 引言

大型语言模型 (LLM) 的真正威力在于连接外部世界。在 LangChain 中，"Tool Calling"（工具调用）是将 LLM 与 API、数据库或自定义函数连接的核心机制。

然而，并不是所有的工具调用都需要复杂的 Agent。根据业务场景的**确定性 (Determinism)**，我们可以将其分为三种整合范式：

1.  **确定性显式调用 (Explicit Chain)**：业务流程固定，无需 LLM 决策。
2.  **半动态绑定 (Bind Tools)**：由 LLM 决定是否调用及参数，但流程是线性的。
3.  **高度动态 Agent**：需要多步推理、循环和自我纠错。

本文将结合实战代码，详细解析前两种 LCEL 原生模式，并简要介绍第三种进阶方案。

---

## 2. 模式一：确定性显式调用 (Deterministic Explicit Chain)

### 适用场景
当您的业务逻辑非常固定，必须先执行某个动作才能生成回答时。例如：
*   用户进入理财页面，系统必须先查询当前汇率，再生成建议。
*   用户请求生成日报，系统必须先从数据库拉取昨日数据。

在这种场景下，**不需要** LLM 去思考“我要不要查数据”，我们通过代码**强制**它查。

### 核心技术
*   `RunnableLambda`: 将工具函数包装为链的一部分。
*   `RunnablePassthrough`: 数据流透传与注入。

### 实战代码解析 (`src/examples/chains/demo_explicit_tool.py`)

在这个示例中，我们构建了一个“理财顾问”。无论用户问什么，系统都会先强制调用 `get_exchange_rate`。

```python
# 1. 定义工具函数 (普通 Python 函数)
def get_exchange_rate(inputs: dict) -> str:
    # ... 实际 API 调用 ...
    return "1 USD = 158.17 JPY"

# 2. 构建显式链路
# 关键点：使用 RunnablePassthrough.assign 强制执行工具
# 这发生在 Prompt 之前，LLM 根本不知道这个过程
chain = (
    RunnablePassthrough.assign(rate=RunnableLambda(get_exchange_rate)) 
    | prompt 
    | llm 
    | StrOutputParser()
)

# 3. Prompt 模板
# 直接使用 {rate} 变量，就像它是用户输入的一部分一样
prompt = PromptTemplate.from_template(
    "Current exchange rate is: {rate}. Give me financial advice."
)
```

**优势**：
*   **极致稳定**：工具调用绝对不会失败或被跳过。
*   **低延迟**：省去了 LLM 进行意图识别的步骤。
*   **零 Token 消耗**：路由逻辑由代码控制，不消耗 LLM Token。

---

## 3. 模式二：半动态绑定 (Semi-dynamic Bind Tools)

### 适用场景
当用户的意图不确定，或者需要从自然语言中提取参数时。例如：
*   用户问“查汇率”，需要调工具。
*   用户问“讲个笑话”，不需要调工具。
*   用户问“100美元换多少人民币”，需要提取 `USD` 和 `CNY`。

这种场景下，我们需要利用 LLM 的**推理能力**来进行路由和参数提取。

### 核心技术
*   `@tool`: 定义工具 Schema。
*   `llm.bind_tools()`: 将工具描述注入 LLM。
*   `JsonOutputToolsParser`: 解析 LLM 的结构化输出。

### 实战代码解析 (`src/examples/chains/demo_tool_chain.py`)

在这个示例中，我们将汇率查询功能“教”给了 LLM，让它自己决定什么时候用。

```python
# 1. 定义工具 (带类型提示和文档)
@tool
def get_exchange_rate(base_currency: str, target_currency: str) -> str:
    """Get the LIVE exchange rate between two currencies..."""
    # ...

# 2. 绑定工具
# 这赋予了 LLM 使用该工具的能力
llm_with_tools = llm.bind_tools([get_exchange_rate])

# 3. 构建自动解析链路
# JsonOutputToolsParser 会自动提取工具调用请求，返回 List[Dict]
chain = prompt | llm_with_tools | JsonOutputToolsParser()

# 4. 执行与自动执行
# 当用户问 "How much is 100 USD in CNY?"
# Chain 输出: [{'type': 'get_exchange_rate', 'args': {'base_currency': 'USD', 'target_currency': 'CNY'}}]
```

**关键点**：
*   如果是简单的意图，我们可以像上面那样只运行到 Parser。
*   如果想实现“自动执行”，我们可以在 Chain 末尾接一个 `RunnableLambda(execute_tools)`，这就构成了一个完整的 **Tool Chain**。

**优势**：
*   **灵活**：能够处理多变的自然语言输入。
*   **智能**：利用 LLM 强大的参数提取能力，比正则表达式强大得多。

---

## 4. 模式三：高度动态 Agent (进阶)

### 为什么前两种还不够？
前两种模式都是**单向**的（DAG）。
*   **模式一**：Code -> LLM
*   **模式二**：LLM -> Code -> (End)

如果工具执行报错了怎么办？如果查到的汇率需要进一步计算怎么办？
这就需要 **Loop (循环)**：
`LLM -> Tool -> Result -> LLM (思考) -> Tool (重试/下一步) -> ...`

这就是 **Agent** 的定义。

### 核心技术：LangGraph
在 LangChain 生态中，构建这种包含循环的 Agent 的最佳实践是使用 **LangGraph**。它允许定义有环图（Cyclic Graph），完美模拟 Agent 的思考-行动循环。

虽然 `bind_tools` 是 Agent 的基石，但要实现完整的 Agent，我们需要将其放入 LangGraph 的 `StateGraph` 中进行编排。

---

## 5. 总结与选型指南

| 模式 | 核心特征 | 适用场景 | 复杂度 | 稳定性 |
| :--- | :--- | :--- | :--- | :--- |
| **显式调用** | 代码控制路由 | 报表、固定工作流、预加载数据 | 低 | ⭐⭐⭐⭐⭐ |
| **Bind Tools** | LLM 控制路由 (单步) | 意图识别、简单查询、参数提取 | 中 | ⭐⭐⭐⭐ |
| **Agent** | LLM 控制循环 (多步) | 复杂任务解决、自我纠错、多工具协作 | 高 | ⭐⭐⭐ |

**建议**：始终从最简单的模式开始。如果模式一能解决问题，不要用模式二；如果模式二够用，不要上 Agent。
