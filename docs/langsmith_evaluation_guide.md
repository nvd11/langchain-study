# LangSmith 模型评估 (Evaluation) 完整指南

本文档将手把手教你如何使用 **LangSmith** 对 LLM（如 Gemini, DeepSeek, GPT）进行自动化评估和对比。

---

## 1. 注册与准备

### 1.1 注册账号
1.  访问 [LangSmith 官网](https://smith.langchain.com/)。
2.  使用 GitHub 或 Google 账号登录。
3.  **注意**：LangSmith 分为 **US (美国)** 和 **EU (欧洲)** 两个数据中心。注册时请留意你所在的区域（或者由系统自动分配）。

### 1.2 获取 API Key
1.  登录后，点击左下角的 **Settings (设置)** 图标。
2.  选择 **API Keys** 选项卡。
3.  点击 **Create API Key**。
4.  给 Key 起个名字（如 "Test Key"），然后**复制保存**（只显示一次）。

### 1.3 确认 Endpoint (关键步骤！)
如果你的账号被分配到了 **EU (欧洲)** 节点（URL 是 `eu.smith.langchain.com`），你必须显式配置 Endpoint，否则会报 `403 Forbidden`。

*   **US 节点**（默认）：`https://api.smith.langchain.com`
*   **EU 节点**：`https://eu.api.smith.langchain.com`

---

## 2. 环境配置

### 2.1 安装依赖
你需要安装 `langsmith` 和 `langchain` 相关库。
```bash
pip install langsmith langchain langchain-openai
```

### 2.2 配置环境变量 (.env)
在项目根目录创建 `.env` 文件，填入以下内容：

```env
# 开启 Tracing (可选，但推荐)
LANGCHAIN_TRACING_V2=true

# 你的 API Key
LANGCHAIN_API_KEY="ls__your_api_key_here"

# 如果你是 EU 账号，必须加这一行！US 账号可忽略
LANGCHAIN_ENDPOINT="https://eu.api.smith.langchain.com"

# 你的模型 Key (用于调用模型)
GEMINI_API_KEY="AIza..."
DEEPSEEK_API_KEY="sk-..."
```

---

## 3. 实战代码：模型对比评估

我们将编写一个脚本，对比 **Gemini** 和 **DeepSeek** 在回答同一组问题时的表现。

### 完整代码 (`compare_model.py`)

```python
import os
from langsmith import Client, evaluate
from src.llm.gemini_chat_model import get_gemini_llm
from src.llm.deepseek_chat_model import get_deepseek_llm

# ================= 1. 初始化客户端 =================
client = Client()

# ================= 2. 准备数据集 (Dataset) =================
# 数据集名称
dataset_name = "AI_Interview_Questions"

# 检查数据集是否存在，不存在则创建
if not client.has_dataset(dataset_name=dataset_name):
    print(f"创建新数据集: {dataset_name}")
    dataset = client.create_dataset(
        dataset_name=dataset_name, 
        description="用于测试模型的基础问答能力"
    )
    
    # 写入测试用例 (Inputs)
    # 可以在这里添加标准答案 (Outputs) 用于自动打分，这里仅做生成测试
    client.create_examples(
        inputs=[
            {"prompt": "什么是 RAG (Retrieval-Augmented Generation)？"},
            {"prompt": "用 Python 写一个快排算法。"},
            {"prompt": "解释量子纠缠，像我只有5岁一样。"},
        ],
        dataset_id=dataset.id,
    )
else:
    print(f"使用现有数据集: {dataset_name}")

# ================= 3. 准备模型 (Target Functions) =================
# 初始化 LangChain 模型对象
gemini = get_gemini_llm()
deepseek = get_deepseek_llm()

# 定义包装函数
# LangSmith 会把数据集里的 inputs (如 {"prompt": "..."}) 传给这个函数
def predict_gemini(inputs: dict):
    # 调用模型
    response = gemini.invoke(inputs["prompt"])
    # 返回结果，key 可以是 "output" 或 "answer"
    return {"output": response.content}

def predict_deepseek(inputs: dict):
    response = deepseek.invoke(inputs["prompt"])
    return {"output": response.content}

# ================= 4. 运行评估 (Run Evaluation) =================
print("开始评估 Gemini...")
evaluate(
    predict_gemini,
    data=dataset_name,
    experiment_prefix="gemini-v1", # 实验名称前缀
    description="Gemini Pro 基础测试"
)

print("开始评估 DeepSeek...")
evaluate(
    predict_deepseek,
    data=dataset_name,
    experiment_prefix="deepseek-v1",
    description="DeepSeek Chat 基础测试"
)
```

---

## 4. 代码深度解析

### Step 1: `client.create_dataset`
*   **作用**：在云端创建一个持久化的数据集。
*   **特性**：数据集只需创建一次。之后你可以反复使用它来测试不同的模型，或者测试同一个模型的不同版本（Prompt 迭代）。

### Step 2: `client.create_examples`
*   **Inputs**：模型的输入（Prompt）。
*   **Outputs (可选)**：标准答案（Ground Truth）。如果提供了 Output，你可以使用“正确性评估器”来自动判断模型回答得对不对。

### Step 3: `predict_wrapper` (包装函数)
*   `evaluate` 函数需要一个可调用的对象（函数）。
*   这个函数接收 `inputs` 字典，必须返回一个字典（通常包含 `output`）。
*   你可以在这里进行 Prompt 组装、解析 JSON 等预处理/后处理逻辑。

### Step 4: `evaluate` (核心)
这是 LangSmith 的魔法所在。它会：
1.  拉取数据集中的每一条例子。
2.  并发调用你的 `predict` 函数。
3.  将 Input, Output, Latency (耗时), Token Usage 等信息全部上传到云端。
4.  生成一个唯一的 **Experiment (实验)** 链接。

---

## 5. 查看结果

1.  运行脚本后，控制台会输出一个 URL。
2.  点击进入 LangSmith 网页。
3.  你可以看到一个**对比视图**：
    *   每一行是一个测试用例（Prompt）。
    *   每一列是一个实验（Gemini vs DeepSeek）。
4.  你可以直观地看到：
    *   哪个模型回答得更准确？
    *   哪个模型速度更快（Latency）？
    *   哪个模型更啰嗦？

通过这种方式，原本凭感觉的“模型好坏”，变成了可视化、可量化的数据。
