import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. 加载环境变量
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env")
    exit(1)

# 2. 定义工具函数
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

# =============================================================================
# SDK 差异说明 (SDK Migration Note)
# -----------------------------------------------------------------------------
# 旧版 SDK (google-generativeai):
#   通常使用 `model = genai.GenerativeModel('gemini-pro')` 来初始化模型对象，
#   然后调用 `model.generate_content(...)`。
#
# 新版 SDK (google.genai V1):
#   采用以 Client 为中心的设计。你需要先初始化 `client = genai.Client(...)`，
#   然后通过 `client.models.generate_content(model='...', ...)` 来调用。
#   它不再强制要求你先实例化一个 "Model" 对象。
#
# 为了模拟旧版的 "封装 Model" 风格（复用配置），我们可以自定义一个简单的 Wrapper 类。
# =============================================================================

class GeminiModelWrapper:
    """
    一个简单的包装器，用于模拟旧版 SDK 的 'Model' 对象风格。
    它可以预先存储 model_name 和 config，避免每次调用都重复传参。
    """
    def __init__(self, client: genai.Client, model_name: str, config: types.GenerateContentConfig):
        self.client = client
        self.model_name = model_name
        self.config = config

    def generate(self, prompt: str):
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.config
        )

def main():
    # 3. 初始化 Client
    client = genai.Client(api_key=api_key)

    # 4. 配置模型参数 (预定义配置)
    my_config = types.GenerateContentConfig(
        tools=[multiply],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY # 强制模型使用工具
            )
        )
    )

    # 5. 实例化封装后的 Model
    print("=== Demo: Native Gemini SDK (Wrapped Style) ===")
    my_model = GeminiModelWrapper(client, "gemini-2.5-pro", my_config)

    # 6. 调用
    query = "What is 123456 multiplied by 6854321?"
    print(f"User Question: {query}")

    response = my_model.generate(query)

    # 7. 解析结果
    print(f"\nResponse Text: {response.text}")

    # 检查 Function Calls
    if (response.candidates
        and response.candidates[0].content
        and response.candidates[0].content.parts):
        for part in response.candidates[0].content.parts:
            if part.function_call:
                fc = part.function_call
                print("\nFunction Call Detected:")
                print(f"  Name: {fc.name}")
                print(f"  Args: {fc.args}")

                # 增加对 fc.args 的非空检查以满足静态类型检查
                args = fc.args
                if fc.name == "multiply" and args is not None:
                    result = multiply(int(args['a']), int(args['b']))
                    print(f"  Execution Result: {result}")

if __name__ == "__main__":
    main()
