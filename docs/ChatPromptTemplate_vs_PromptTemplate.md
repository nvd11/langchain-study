# LangChain 核心对比：ChatPromptTemplate vs PromptTemplate

在 LangChain 开发中，开发者经常面临一个基础选择：是使用 `PromptTemplate` 还是 `ChatPromptTemplate`？这两个组件虽然都用于构建 Prompt，但它们服务于完全不同的模型范式。本文将从底层原理、应用场景和代码实现三个维度进行深度对比。

## 1. 核心差异概览

| 特性 | PromptTemplate | ChatPromptTemplate |
| :--- | :--- | :--- |
| **面向模型类型** | **Text Model (Completion)**<br>例：GPT-3 DaVinci, Llama 2 Base | **Chat Model (Conversation)**<br>例：GPT-4, Claude 3, Gemini |
| **底层数据结构** | **单一字符串 (String)** | **消息列表 (List[BaseMessage])** |
| **思维模型** | “文本续写” (Next Token Prediction) | “角色扮演与对话” (Role-based Interaction) |
| **变量替换能力** | 仅支持字符串插值 (String Interpolation) | 支持字符串插值 + **列表扩展 (List Expansion)** |
| **最佳实践** | 用于简单任务、指令生成、老旧模型 | 用于对话系统、Agent、复杂指令遵循 |

## 2. 深度解析

### 2.1 PromptTemplate：文本世界的遗留者

`PromptTemplate` 是大语言模型早期（2020-2022）的产物。当时的主流模型（如 GPT-3）接口接收的是纯文本。开发者需要手动设计文本格式（Prompt Engineering）。

**工作原理**：它本质上是一个 Python f-string 的高级封装。

```python
from langchain_core.prompts import PromptTemplate

# 定义模板：本质上就是一段文本
template = PromptTemplate.from_template(
    "User: {input}\nAI:"
)

# 渲染结果：一个长字符串
formatted = template.format(input="Hello")
# 输出: "User: Hello\nAI:"
```

### 2.2 ChatPromptTemplate：结构化交互的未来

`ChatPromptTemplate` 是为了适应 ChatGPT 及其后续模型（RLHF 训练的对话模型）而设计的。这些模型在训练时就区分了 System（系统指令）、User（用户）和 Assistant（AI）的角色。

**工作原理**：它构建的是一个由 `BaseMessage` 对象组成的列表。

```python
from langchain_core.prompts import ChatPromptTemplate

# 定义模板：由角色和内容组成的列表
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

# 渲染结果：一个消息对象列表
formatted = template.format_messages(input="Hello")
# 输出: [SystemMessage(content="..."), HumanMessage(content="Hello")]
```

## 3. 常见疑问：为什么不直接用 `string.format` (f-string)？

**用户提问：** `PromptTemplate` 可以用 `string.format()` 代替吗？

**回答：** **在简单场景下可以，但在工程化场景下不推荐。**

如果你只是写一个简单的脚本，`f"Hello {name}"` 确实比 `PromptTemplate.from_template("Hello {name}")` 更简洁。但在复杂的 LangChain 应用中，`PromptTemplate` 提供了 Python 原生字符串无法比拟的优势：

1.  **部分填充 (Partial Formatting)**：
    你可以先填充一部分变量（例如由另一个 Chain 生成的变量），保留另一部分变量留给后续步骤填充。这是构建复杂 Chain 的基础。
    ```python
    template = PromptTemplate(template="{foo} {bar}", input_variables=["foo", "bar"])
    partial = template.partial(foo="Hello")
    # partial 现在是一个只等待 "bar" 的新模板
    print(partial.format(bar="World")) # -> "Hello World"
    ```

2.  **输入验证 (Validation)**：
    LangChain 会自动检查变量名是否匹配，防止因为拼写错误导致的运行时 Bug。

3.  **序列化与加载 (Serialization)**：
    `PromptTemplate` 可以被保存为 JSON/YAML 文件并重新加载，这对于 Prompt 的版本管理和从外部配置加载 Prompt 至关重要。

4.  **与其他组件集成**：
    它实现了 `Runnable` 接口，可以直接接入 LCEL 管道 (`chain = prompt | model`)，而原生字符串不能直接作为 Runnable 使用。

## 4. 总结

随着 LLM 产业向 Chat Model 全面转移，**`ChatPromptTemplate` 已经成为事实上的标准**。除非有极其特殊的理由使用 Text Completion 模型，否则应始终优先选择 `ChatPromptTemplate`，以获得更好的结构化支持和安全性。
