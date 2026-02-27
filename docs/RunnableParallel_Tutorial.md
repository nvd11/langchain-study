# LangChain 核心机制：RunnableParallel 与隐式转换

在 LCEL (LangChain Expression Language) 中，你经常会看到类似这样的代码：

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | llm
)
```

很多初学者会疑惑：**“为什么可以在 Chain 中直接写一个字典？它不是应该全都是 Runnable 对象吗？”**

本文将为你揭示这个语法背后的机制：**`RunnableParallel`** 以及 LangChain 的 **隐式转换 (Coercion)**。

## 1. 什么是 RunnableParallel？

`RunnableParallel` 是一个用于**并行执行**多个 Runnable 的组件。
它接收一个输入，将其同时传递给多个分支（Runnables），并将所有分支的输出组合成一个字典返回。

### 显式用法

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

chain = RunnableParallel({
    "branch_a": RunnableLambda(lambda x: x + 1),
    "branch_b": RunnableLambda(lambda x: x * 2)
})

chain.invoke(10)
# 输出: {'branch_a': 11, 'branch_b': 20}
```

## 2. 魔法字典：隐式转换 (Coercion)

LangChain 的设计哲学之一是“让常见模式变得简单”。
由于并行执行并返回字典（特别是在构建 Prompt 输入时）是一个极高频的操作，LangChain 允许你在 Chain 中**直接使用 Python 字典**来代表 `RunnableParallel`。

### 转换规则

当你使用管道操作符 `|` 构建 Chain 时（例如 `step1 | step2`），`RunnableSequence` 会检查每一个步骤：

*   如果它是一个 **字典 (Dict)**：
    *   系统会自动调用 `RunnableParallel(dict)` 将其包装。
    *   字典中的每个 Value 也会被递归地转换为 Runnable（例如 lambda 函数会被转为 `RunnableLambda`）。

这就是为什么你可以写：

```python
{
    "context": retriever, 
    "question": RunnablePassthrough() 
}
```

而不需要写繁琐的：

```python
RunnableParallel({
    "context": retriever,
    "question": RunnablePassthrough()
})
```

## 3. 实战验证

为了证明这一点，我们编写了一个验证脚本 (`src/examples/chains/demo_runnable_parallel_coercion.py`)。

### 3.1 并行性验证

我们定义了两个耗时 1 秒的任务 `task_a` 和 `task_b`。

*   **如果串行执行**：总耗时应为 1s + 1s = 2s。
*   **如果并行执行**：总耗时应接近 1s。

**运行结果**：
```text
--- Demo 1: Explicit RunnableParallel ---
Task A processing: Input 1
Task B processing: Input 1
Explicit Result: {'output_a': 'Result A(Input 1)', 'output_b': 'Result B(Input 1)'}
Time taken: 1.00s  <-- 证明是并行的
```

### 3.2 字典语法验证

我们直接定义了一个字典 `chain_implicit`，并把它放入 Chain 中：

```python
chain_implicit = {
    "output_a": RunnableLambda(task_a),
    "output_b": RunnableLambda(task_b)
}
final_chain = chain_implicit | RunnableLambda(...)
```

**运行结果**：
```text
--- Demo 2: Implicit Dictionary Coercion ---
Task A processing: Input 2
Task B processing: Input 2
Implicit Result: Combined: {'output_a': 'Result A(Input 2)', 'output_b': 'Result B(Input 2)'}
Time taken: 1.00s  <-- 字典也被并行执行了！
Type of chain_implicit: <class 'dict'>
```

注意最后一行：虽然在 Python 层面它仍然是一个 `dict`，但在 LCEL 的执行流中，它被动态转换成了并行执行逻辑。

## 4. 总结

*   **语法糖**：在 Chain 中看到的字典 `{key: value}` 等价于 `RunnableParallel({key: value})`。
*   **作用**：
    1.  **并行执行**：同时运行多个任务，优化性能。
    2.  **数据格式化**：将多个来源的数据组合成一个字典，这是构建 Prompt 输入（Prompt 往往需要字典输入）的标准范式。

掌握了这个概念，你就能读懂绝大多数复杂的 LCEL 代码了。
