from langchain_core.runnables import RunnableParallel, RunnableLambda
from loguru import logger
import time

def task_a(x):
    logger.info(f"Task A processing: {x}")
    time.sleep(1) # Simulate work
    return f"Result A({x})"

def task_b(x):
    logger.info(f"Task B processing: {x}")
    time.sleep(1) # Simulate work
    return f"Result B({x})"

def main():
    # -----------------------------------------------------------
    # 演示 1: 显式定义 RunnableParallel
    # -----------------------------------------------------------
    logger.info("--- Demo 1: Explicit RunnableParallel ---")
    chain_explicit = RunnableParallel({
        "output_a": RunnableLambda(task_a),
        "output_b": RunnableLambda(task_b)
    })
    
    # 两个任务并行执行，总耗时应接近 1秒 而不是 2秒
    start = time.time()
    result_explicit = chain_explicit.invoke("Input 1")
    end = time.time()
    
    logger.info(f"Explicit Result: {result_explicit}")
    logger.info(f"Time taken: {end - start:.2f}s")


    # -----------------------------------------------------------
    # 演示 2: 隐式转换 (Magic Dictionary)
    # -----------------------------------------------------------
    logger.info("\n--- Demo 2: Implicit Dictionary Coercion ---")
    
    # 这里的字典会被 LangChain 自动检测并转换为 RunnableParallel
    # 这就是为什么你可以在 Chain 中直接写字典
    chain_implicit = {
        "output_a": RunnableLambda(task_a),
        "output_b": RunnableLambda(task_b)
    }
    
    # 为了验证，我们可以把它包装进一个简单的 Chain 中，看看行为是否一致
    # 注意：直接对字典调用 invoke 是不行的，字典本身没有 invoke 方法。
    # 这种“隐式转换”发生在 Chain 的构建过程中（例如使用 | 管道符时），
    # 或者当你显式使用 RunnableSequence 时。
    
    # 为了演示，我们把它放在一个管道中
    final_chain = chain_implicit | RunnableLambda(lambda x: f"Combined: {x}")
    
    start = time.time()
    result_implicit = final_chain.invoke("Input 2")
    end = time.time()
    
    logger.info(f"Implicit Result: {result_implicit}")
    logger.info(f"Time taken: {end - start:.2f}s")
    
    # 验证类型：虽然 python 层面它还是 dict，但在 Chain 运行时它被当作 RunnableParallel 处理
    logger.info(f"Type of chain_implicit: {type(chain_implicit)}")
    
    # 实际上，LangChain 的 RunnableSequence ( | ) 会在初始化时检查每一步
    # 如果发现是 dict，就会调用 RunnableParallel(dict) 进行转换。

if __name__ == "__main__":
    main()
