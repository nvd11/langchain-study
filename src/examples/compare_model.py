import src.configs.config
from loguru import logger
from langsmith import Client, evaluate
from src.llm.gemini_chat_model import get_gemini_llm
from src.llm.deepseek_chat_model import get_deepseek_llm

# 1. Initialize Client
client = Client()

# 2. Define Dataset Name
dataset_name = "Model_Comparison_Quickstart"

# 3. Create Dataset (if not exists)
if not client.has_dataset(dataset_name=dataset_name):
    logger.info(f"Creating dataset: {dataset_name}")
    dataset = client.create_dataset(dataset_name=dataset_name, description="A quick comparison of models")
    
    # Add examples to the dataset
    client.create_examples(
        inputs=[
            {"prompt": "How to become an AI engineer?"},
            {"prompt": "Explain Quantum Computing in 5 words."},
            {"prompt": "Write a python function to reverse a string."},
        ],
        dataset_id=dataset.id,
    )
else:
    logger.info(f"Dataset {dataset_name} already exists. Using existing data.")

# 4. Initialize Models
logger.info("Initializing models...")
llm_gemini = get_gemini_llm()
llm_deepseek = get_deepseek_llm()

# 5. Define Evaluation Logic (Wrappers)
def predict_gemini(inputs: dict):
    """Invokes Gemini model"""
    prompt_text = inputs["prompt"]
    response = llm_gemini.invoke(prompt_text)
    return {"output": response.content}

def predict_deepseek(inputs: dict):
    """Invokes DeepSeek model"""
    prompt_text = inputs["prompt"]
    response = llm_deepseek.invoke(prompt_text)
    return {"output": response.content}

# 6. Run Evaluation
logger.info("Starting evaluation for Gemini...")
evaluate(
    predict_gemini,
    data=dataset_name,
    experiment_prefix="gemini-comparison",
    description="Testing Gemini on standard prompts"
)

logger.info("Starting evaluation for DeepSeek...")
evaluate(
    predict_deepseek,
    data=dataset_name,
    experiment_prefix="deepseek-comparison",
    description="Testing DeepSeek on standard prompts"
)

logger.success("Evaluation complete! Check your LangSmith dashboard to see the comparison.")
