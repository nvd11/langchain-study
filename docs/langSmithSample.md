Get started with evaluation
Run an experiment

Python

TypeScript
Build more reliable AI applications with evaluations.
1
New key created:
<your-api-key>


Copy
2
Install Dependencies
pip install -U langsmith openevals openai

3
Configure environment
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
export LANGSMITH_API_KEY=<your-api-key>
export OPENAI_API_KEY=<your-openai-api-key>

4
Import dependencies
from langsmith import Client, wrappers
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from openai import OpenAI

5
```python
client = Client()

dataset = client.create_dataset(
    dataset_name="ds-gargantuan-fender-81", description="A sample dataset in LangSmith."
)
examples = [
    {
        "inputs": {"question": "Which country is Mount Kilimanjaro located in?"},
        "outputs": {"answer": "Mount Kilimanjaro is located in Tanzania."},
    },
    {
        "inputs": {"question": "What is Earth's lowest point?"},
        "outputs": {"answer": "Earth's lowest point is The Dead Sea."},
    },
]
client.create_examples(dataset_id=dataset.id, examples=examples)

dataset = client.create_dataset(
    dataset_name="ds-gargantuan-fender-81", description="A sample dataset in LangSmith."
)
examples = [
    {
        "inputs": {"question": "Which country is Mount Kilimanjaro located in?"},
        "outputs": {"answer": "Mount Kilimanjaro is located in Tanzania."},
    },
    {
        "inputs": {"question": "What is Earth's lowest point?"},
        "outputs": {"answer": "Earth's lowest point is The Dead Sea."},
    },
]
client.create_examples(dataset_id=dataset.id, examples=examples)
```
6
Define what you're evaluating
⌄
⌄
# Wrap the OpenAI client for LangSmith tracing
openai_client = wrappers.wrap_openai(OpenAI())


# Define the application logic to evaluate.
# Dataset inputs are automatically sent to this target function.
def target(inputs: dict) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer the following question accurately"},
            {"role": "user", "content": inputs["question"]},
        ],
    )
    return {"answer": response.choices[0].message.content}

7
Define evaluator
⌄
# Define an LLM-as-a-judge evaluator to evaluate correctness of the output
def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:o3-mini",
        feedback_key="correctness",
    )
    eval_result = evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    )
    return eval_result

8
Run experiment
experiment_results = client.evaluate(
    target,
    data="ds-gargantuan-fender-81",
    evaluators=[correctness_evaluator],
    experiment_prefix="experiment-quickstart-upbeat-mass-4",
    max_concurrency=2,
)


Skip
