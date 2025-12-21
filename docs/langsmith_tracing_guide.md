# LangSmith 自动追踪 (Tracing) 实战指南

本文档介绍如何利用 **LangChain** 的环境变量机制，实现**零代码侵入**的 LangSmith 自动追踪。

---

## 1. 核心原理

LangChain 框架内置了对 LangSmith 的支持。你不需要在代码里显式初始化 Client 或写日志，只需设置一个环境变量开关：

```bash
LANGCHAIN_TRACING_V2=true
```

只要这个开关打开，你的所有 Chain、LLM、Retriever 的运行细节都会被自动发送到 LangSmith 云端。

---

## 2. 环境配置

确保你的 `.env` 文件包含以下核心配置：

```env
# 1. 总开关 (必须 true)
LANGCHAIN_TRACING_V2=true

# 2. 身份认证 (你的 Key)
LANGCHAIN_API_KEY="ls__xxxxxx"

# 3. 区域端点 (如果你是 EU 账号，必须加这行；US 账号不用加)
# LANGCHAIN_ENDPOINT="https://eu.api.smith.langchain.com"

# 4. 项目名称 (可选，如果不填，默认存入 "default" 项目)
# LANGCHAIN_PROJECT="Mydemo"
```

---

## 3. 实战演示

我们已经准备好了一个测试脚本：`src/examples/trace_demo.py`。

### 代码一览
这个脚本非常简单，就是一个普通的 LangChain 对话链，没有任何与 LangSmith 相关的代码。

```python
import src.configs.config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.llm.gemini_chat_model import get_gemini_llm

# 1. 确保环境变量已加载 (.env)
# 必须包含:
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=...

# 2. 定义一个简单的链
prompt = ChatPromptTemplate.from_template("请用{language}写一首关于{topic}的四行诗。")
model =  get_gemini_llm()
parser = StrOutputParser()

chain = prompt | model | parser

# 3. 运行链
# 这次运行会自动被 LangSmith 记录，因为环境变量开关已打开
print("正在生成诗歌...")
response = chain.invoke({"language": "中文", "topic": "人工智能"})

print("\n=== 生成结果 ===")
print(response)
print("==================")
print("请去 LangSmith 控制台查看本次运行的 Trace 详情。")

```

### 运行脚本
在终端执行：
```bash
python src/examples/trace_demo.py
```

---

## 4. 如何查看 Trace Report (追踪报告)

脚本运行结束后，请按照以下步骤查看“上帝视角”的运行记录：

1.  **登录控制台**：访问 [LangSmith (EU)](https://eu.smith.langchain.com/) 或 [LangSmith (US)](https://smith.langchain.com/)。
2.  **进入项目**：
    *   在左侧侧边栏，点击 **Projects (项目)** 图标。
    *   点击列表中的 **"default"** (如果你没设置 `LANGCHAIN_PROJECT` 变量)。
3.  **查看 Trace**：
    *   你会看到列表中最新出现的一行记录，Name 列通常显示为 `RunnableSequence` 或 `ChatOpenAI`。
    *   **点击这行记录**。
4.  **深度透视**：
    *   在右侧弹出的详情页中，你可以看到完整的**调用链路树**。
    *   **Prompt**: 点击 `ChatPromptTemplate` 节点，查看填入变量后的完整提示词。
    *   **LLM**: 点击 `ChatOpenAI` 节点，查看发给 API 的原始 Payload 和 Token 消耗。
    *   **Output**: 查看最终输出结果和耗时。

### 看到的数据价值
*   **Latency (耗时)**: 哪个步骤拖慢了速度？
*   **Token Usage**: 这次问答花了多少钱？
*   **Debug**: 模型到底是因为 Prompt 没写好，还是逻辑错了？

---

## 总结
**Zero Code Instrumentation (零代码插桩)** 是 LangSmith 最强大的特性之一。你只管写业务逻辑，监控和日志交给环境变量去处理。
