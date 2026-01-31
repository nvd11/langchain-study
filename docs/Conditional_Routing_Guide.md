# LangChain LCEL 进阶：深入解析 RunnableBranch 与语义路由

## 1. 引言

在构建复杂的 LLM 应用时，"One prompt fits all"（一个提示词解决所有问题）往往是不现实的。处理数学问题需要严谨的推理，而处理创意写作则需要发散的思维。

因此，我们需要一种机制，能够根据用户的输入**动态选择**最合适的处理链路。这就是 **条件路由 (Conditional Routing)**。

在 LangChain LCEL 中，`RunnableBranch` 是实现这一机制的核心原语。本文将通过两个循序渐进的实战案例，带您掌握从简单的关键词路由到高级的语义路由的实现方法。

---

## 2. RunnableBranch 基础原理

`RunnableBranch` 的逻辑非常直观，它等同于编程语言中的 `if-elif-else` 结构。

它接收一个包含 `(condition, runnable)` 元组的列表，以及一个默认的 `default_runnable`。

```python
branch = RunnableBranch(
    (condition_a, chain_a),  # if condition_a is True, run chain_a
    (condition_b, chain_b),  # elif condition_b is True, run chain_b
    default_chain            # else, run default_chain
)
```

---

## 3. 实战一：基于关键词的简单路由

在某些简单场景下，我们可以通过检测输入中是否包含特定关键词来进行路由。

### 代码解析 (`src/examples/chains/demo_branch.py`)

在这个例子中，我们要区分“数学问题”和“历史问题”。

```python
# 1. 定义分支逻辑
# 使用 lambda 函数检查输入字典中的 "topic" 字段
branch = RunnableBranch(
    (lambda x: "math" in x["topic"].lower(), math_chain),
    (lambda x: "history" in x["topic"].lower(), history_chain),
    general_chain # 默认分支
)

# 2. 执行
# 输入必须包含明确的 topic
await branch.ainvoke({"topic": "math", "query": "1+1"})
```

**局限性**：这种方式依赖于外部显式地提供 `topic`，或者输入中必须包含特定的关键词。如果用户直接问 "1+1 等于几？" 而不包含 "math" 这个词，简单的关键词匹配就会失效。

---

## 4. 实战二：基于 LLM 的语义路由 (Semantic Routing)

为了处理自然语言的模糊性，我们需要更智能的路由方式：**让 LLM 先读懂意图，再决定去哪里**。这就是语义路由。

### 架构设计
1.  **Router Chain**: 一个专门的 LLM 链，负责将用户问题分类（如输出 "math", "history"）。
2.  **Branch**: 根据 Router 的输出结果进行分发。

### 代码解析 (`src/examples/chains/demo_router_chain.py`)

#### 第一步：构建分类器 (The Classifier)

```python
router_prompt = PromptTemplate.from_template(
    """
    Classify the question into: math, history, or general.
    Return ONLY the category name.
    Question: {query}
    """
)
router_chain = router_prompt | llm | StrOutputParser()
```

#### 第二步：构建完整链路

这里我们使用了 `RunnablePassthrough.assign` 将 Router 的分类结果注入到数据流中。

```python
full_chain = (
    # 1. 先调用 Router 获取 topic，并保留原始 query
    # Output: {"query": "...", "topic": "math"}
    RunnablePassthrough.assign(topic=router_chain) 
    
    # 2. 根据 topic 进行路由
    | RunnableBranch(
        (lambda x: "math" in x["topic"].lower(), math_chain),
        (lambda x: "history" in x["topic"].lower(), history_chain),
        general_chain
    )
)
```

#### 第三步：智能执行

现在，即使用户的提问中没有关键词，系统也能正确路由：

*   **Input**: "What is the square root of 144?"
*   **Router**: 识别出这是数学计算，输出 `topic: "math"`。
*   **Branch**: 路由到 `math_chain`（数学专家）。
*   **Result**: 得到专业的数学解答。

---

## 5. 架构延伸：从 Router Chain 到多 Agent 系统

我们在 Demo 中展示的“语义路由”模式，实际上是现代 **多 Agent 系统 (Multi-Agent System)** 架构的微缩版。

### 架构映射 (Pattern Mapping)

| Demo 组件 | 企业级架构对应组件 | 职责 |
| :--- | :--- | :--- |
| **Router Chain** | **Main Agent / Supervisor / Routing Agent** | 负责意图识别、任务拆解与分发。它是系统的“大脑”。 |
| **Math Chain** | **GitHub Agent / DevOps Agent** | 垂直领域的专家。比如专门负责查询代码、操作 Git。 |
| **History Chain** | **Jira Agent / HR Agent** | 另一个领域的专家。比如专门负责查询工单、处理人事流程。 |
| **RunnableBranch** | **Orchestrator / Dispatcher** | 负责具体的流量转发逻辑。 |

### 实际应用场景：DevOps 助手

想象一个企业级的 DevOps 助手，它的工作流程与我们的 Demo 如出一辙：

1.  **用户输入**："帮我修复 main 分支的构建错误，并更新 Jira 票据。"
2.  **Main Agent (Router)**：分析意图，发现需要两个动作。
    *   首先调用 **GitHub Agent** 查看构建日志。
    *   然后调用 **Jira Agent** 更新状态。
3.  **分发与执行**：系统根据 Router 的指令，依次激活对应的专业 Agent 进行处理。

**意义**：通过 `RunnableBranch` 和语义路由，我们实现了**关注点分离 (Separation of Concerns)**。Main Agent 只需要懂“分发”，而具体 Agent 只需要懂“执行”，这使得系统极易扩展和维护。

---

## 6. 总结

*   **RunnableBranch** 是 LCEL 中实现逻辑分叉的关键组件。
*   **关键词路由**：简单、快速，适用于规则明确的场景。
*   **语义路由**：利用 LLM 的理解能力作为“路由器”，是构建智能系统的基石。
*   **架构价值**：掌握了语义路由，就等于掌握了构建复杂 **多 Agent 协作系统** 的核心钥匙。
