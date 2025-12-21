try:
    from langchain.model_laboratory import ModelLaboratory
    print("Found in langchain.model_laboratory")
except ImportError:
    print("Not in langchain.model_laboratory")

try:
    from langchain_experimental.model_laboratory import ModelLaboratory
    print("Found in langchain_experimental.model_laboratory")
except ImportError:
    print("Not in langchain_experimental.model_laboratory")
